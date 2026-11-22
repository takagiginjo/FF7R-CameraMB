// Camera motion blur for Final Fantasy VII Remake (Unreal Engine 4, Direct3D 11),
// as a ReShade add-on, next to Luma.
//
// The game has Unreal's motion blur and it blurs objects only: the velocity
// buffer holds the velocity of things that move by themselves and is 0 for the
// rest, and the camera's own motion, which Unreal normally works out from the
// depth for those pixels, comes out as nothing in this game.
//
// This add-on:
//   1. Finds the velocity flatten compute shader, the first pass of the motion
//      blur, by the constants it decodes the velocity with.
//   2. Just before that shader runs, writes the camera's motion into the
//      velocity buffer wherever it is 0, worked out from the depth and the
//      ClipToPrevClip matrix in the View constants, as Unreal's temporal AA
//      does it. Things that have a velocity of their own keep it, scaled.
//   3. Things near the camera, the player's character among them, are kept
//      sharp: their velocity fades to 0 below a distance. No velocity is
//      longer than a limit.
//   4. The game's own motion blur does the rest, with its own settings.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#define ImTextureID ImU64
#include <imgui.h>
#include <reshade.hpp>

#include <d3d11_1.h>
#include <wrl/client.h>

#include <inject.h> // generated from shaders/inject.hlsl: inject_cs

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_set>

using Microsoft::WRL::ComPtr;
using namespace reshade::api;

extern "C" __declspec(dllexport) const char *NAME = "Camera Motion Blur (FF7 Remake)";
extern "C" __declspec(dllexport) const char *DESCRIPTION =
    "Puts the camera's motion into the velocity buffer, so the game's own motion blur blurs it too.";

static const char *SECTION = "CameraBlur";

//----------------------------------------------------------------------------
// Settings

struct settings
{
    std::atomic<bool> enabled{true};
    std::atomic<float> strength{2.0f};   // camera motion, 1 = as Unreal works it out
    std::atomic<float> objects{1.0f};    // objects' own velocity, 1 = untouched
    std::atomic<float> max_length{0.04f}; // longest blur, share of the screen height
    std::atomic<float> near_start{4.0f}; // things nearer than this (meters) are kept sharp
    std::atomic<float> near_fade{2.0f};  // meters from sharp to fully blurred
    std::atomic<float> near_plane{10.0f}; // the near plane, in the game's units (cm)
    std::atomic<bool> debug{false};      // show where things are kept sharp
};
static settings g_settings;

static void load_settings()
{
    int enabled = 1, strength = 200, objects = 100, max_length = 40, near_start = 4, near_fade = 2, near_plane = 10,
        debug = 0;
    reshade::get_config_value(nullptr, SECTION, "Enabled", enabled);
    reshade::get_config_value(nullptr, SECTION, "CameraBlurPercent", strength);
    reshade::get_config_value(nullptr, SECTION, "ObjectBlurPercent", objects);
    reshade::get_config_value(nullptr, SECTION, "MaxLength", max_length); // tenths of a percent of the height
    reshade::get_config_value(nullptr, SECTION, "KeepNearSharp", near_start); // meters
    reshade::get_config_value(nullptr, SECTION, "NearFade", near_fade);       // meters
    reshade::get_config_value(nullptr, SECTION, "NearPlaneCm", near_plane);
    reshade::get_config_value(nullptr, SECTION, "Debug", debug);
    g_settings.enabled = enabled != 0;
    g_settings.strength = float(std::clamp(strength, 0, 500)) / 100.0f;
    g_settings.objects = float(std::clamp(objects, 0, 300)) / 100.0f;
    g_settings.max_length = float(std::clamp(max_length, 5, 150)) / 1000.0f;
    g_settings.near_start = float(std::clamp(near_start, 0, 20));
    g_settings.near_fade = float(std::clamp(near_fade, 1, 10));
    g_settings.near_plane = float(std::clamp(near_plane, 1, 100));
    g_settings.debug = debug != 0;
}

//----------------------------------------------------------------------------
// Log, next to the add-on. Small on purpose.

static FILE *g_log = nullptr;
static std::mutex g_log_mutex;
static int g_log_lines = 0;

static void mb_log(const char *fmt, ...)
{
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (!g_log || g_log_lines >= 300)
        return;
    ++g_log_lines;
    char text[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(text, sizeof(text), fmt, args);
    va_end(args);
    fprintf(g_log, "%s\n", text);
    fflush(g_log);
}

static void open_log(HMODULE module)
{
    wchar_t path[MAX_PATH];
    const DWORD n = GetModuleFileNameW(module, path, MAX_PATH);
    if (n == 0 || n >= MAX_PATH)
        return;
    std::wstring p(path, n);
    const size_t dot = p.find_last_of(L'.');
    if (dot != std::wstring::npos)
        p.resize(dot);
    p += L".log";
    g_log = _wfopen(p.c_str(), L"w");
}

//----------------------------------------------------------------------------
// The velocity flatten compute shader, found when the game creates it.

static std::unordered_set<void *> g_flatten_cs;
static std::shared_mutex g_flatten_mutex;
static unsigned g_flatten_found = 0;

static bool is_flatten_cs(void *cs)
{
    std::shared_lock<std::shared_mutex> lock(g_flatten_mutex);
    return g_flatten_cs.count(cs) != 0;
}

static unsigned fnv1a(const void *data, size_t size)
{
    unsigned h = 2166136261u;
    const unsigned char *p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < size; ++i)
        h = (h ^ p[i]) * 16777619u;
    return h;
}

// True when the shader holds a 32-bit float close to `value`.
static bool has_float(const unsigned char *code, size_t size, float value)
{
    for (size_t i = 0; i + 4 <= size; ++i)
    {
        float f;
        memcpy(&f, code + i, 4);
        if (std::fabs(f - value) < 2.0e-5f)
            return true;
    }
    return false;
}

static void on_init_pipeline(device *dev, pipeline_layout, uint32_t count, const pipeline_subobject *subobjects,
                             pipeline pipe)
{
    if (dev->get_api() != device_api::d3d11 || pipe.handle == 0)
        return;
    for (uint32_t i = 0; i < count; ++i)
    {
        if (subobjects[i].type != pipeline_subobject_type::compute_shader || subobjects[i].count == 0)
            continue;
        const shader_desc *desc = static_cast<const shader_desc *>(subobjects[i].data);
        if (!desc || !desc->code || desc->code_size < 200 || desc->code_size > 20000)
            continue;
        const unsigned char *code = static_cast<const unsigned char *>(desc->code);

        // Unreal decodes its velocity with (enc - 0.499992) * 4.008016. The one
        // compute shader that does it and keeps a group-shared tile is the
        // flatten pass; its hash in this game build is AFC9976E.
        const bool by_hash = fnv1a(code, desc->code_size) == 0xAFC9976Eu;
        const bool by_constants = has_float(code, desc->code_size, 4.008016f) &&
                                  has_float(code, desc->code_size, -0.499992f);
        if (!by_hash && !by_constants)
            continue;
        {
            std::unique_lock<std::shared_mutex> lock(g_flatten_mutex);
            g_flatten_cs.insert(reinterpret_cast<void *>(pipe.handle));
            ++g_flatten_found;
        }
        mb_log("velocity flatten compute shader %p, %zu bytes (%s)", reinterpret_cast<void *>(pipe.handle),
               desc->code_size, by_hash ? "by hash" : "by its constants");
    }
}

static void on_destroy_pipeline(device *, pipeline pipe)
{
    std::unique_lock<std::shared_mutex> lock(g_flatten_mutex);
    g_flatten_cs.erase(reinterpret_cast<void *>(pipe.handle));
}

//----------------------------------------------------------------------------
// Direct3D 11 objects

static std::mutex g_state_mutex; // the add-on's own state

static ComPtr<ID3D11Device> g_dev;
static ComPtr<ID3D11ComputeShader> g_cs;
static ComPtr<ID3D11Buffer> g_cb;        // the settings
static ComPtr<ID3D11Buffer> g_view_copy; // ClipToPrevClip and the jitter, copied from the View constants
static ComPtr<ID3D11Texture2D> g_scratch;
static ComPtr<ID3D11UnorderedAccessView> g_scratch_uav;
static UINT g_width = 0, g_height = 0;
static DXGI_FORMAT g_tex_format = DXGI_FORMAT_UNKNOWN;
static bool g_unsupported = false;

// The View constants' vectors 114 to 118, as bytes.
static const UINT VIEW_OFFSET = 114 * 16;
static const UINT VIEW_BYTES = 5 * 16;

struct statistics
{
    unsigned flatten_dispatches = 0, injected = 0, skipped = 0;
};
static statistics g_stats;
static unsigned g_frame = 0;
static unsigned g_reasons_logged = 0;

static void release_resources()
{
    g_scratch_uav.Reset();
    g_scratch.Reset();
    g_cs.Reset();
    g_cb.Reset();
    g_view_copy.Reset();
    g_dev.Reset();
    g_width = g_height = 0;
    g_tex_format = DXGI_FORMAT_UNKNOWN;
    g_unsupported = false;
}

static bool ensure_resources(ID3D11Device *dev, const D3D11_TEXTURE2D_DESC &desc)
{
    if (g_unsupported)
        return false;
    // Only a change of device drops what was made; the first call notes the device.
    if (g_dev.Get() != dev)
    {
        if (g_dev)
            release_resources();
        g_dev = dev;
    }

    if (!g_cs)
    {
        if (FAILED(dev->CreateComputeShader(inject_cs, sizeof(inject_cs), nullptr, g_cs.GetAddressOf())))
        {
            mb_log("could not create the compute shader");
            g_unsupported = true;
            return false;
        }
        D3D11_BUFFER_DESC bd = {};
        bd.ByteWidth = 32;
        bd.Usage = D3D11_USAGE_DYNAMIC;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        D3D11_BUFFER_DESC cd = {};
        cd.ByteWidth = VIEW_BYTES;
        cd.Usage = D3D11_USAGE_DEFAULT;
        cd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        if (FAILED(dev->CreateBuffer(&bd, nullptr, g_cb.GetAddressOf())) ||
            FAILED(dev->CreateBuffer(&cd, nullptr, g_view_copy.GetAddressOf())))
        {
            mb_log("could not create the constant buffers");
            g_unsupported = true;
            return false;
        }
    }

    if (g_scratch && g_width == desc.Width && g_height == desc.Height && g_tex_format == desc.Format)
        return true;

    g_scratch_uav.Reset();
    g_scratch.Reset();

    if (desc.ArraySize != 1 || desc.MipLevels != 1 || desc.SampleDesc.Count != 1)
    {
        mb_log("the velocity texture is an array, has mips or is multisampled (%u/%u/%u), not supported",
               desc.ArraySize, desc.MipLevels, desc.SampleDesc.Count);
        g_unsupported = true;
        return false;
    }

    D3D11_FEATURE_DATA_FORMAT_SUPPORT2 fs2 = {};
    fs2.InFormat = DXGI_FORMAT_R16G16_UNORM;
    if (FAILED(dev->CheckFeatureSupport(D3D11_FEATURE_FORMAT_SUPPORT2, &fs2, sizeof(fs2))) ||
        !(fs2.OutFormatSupport2 & D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE))
    {
        mb_log("R16G16_UNORM cannot be written by a compute shader on this GPU");
        g_unsupported = true;
        return false;
    }

    D3D11_TEXTURE2D_DESC ud = desc;
    ud.Usage = D3D11_USAGE_DEFAULT;
    ud.CPUAccessFlags = 0;
    ud.MiscFlags = 0;
    ud.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav = {};
    uav.Format = DXGI_FORMAT_R16G16_UNORM;
    uav.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    if (FAILED(dev->CreateTexture2D(&ud, nullptr, g_scratch.GetAddressOf())) ||
        FAILED(dev->CreateUnorderedAccessView(g_scratch.Get(), &uav, g_scratch_uav.GetAddressOf())))
    {
        mb_log("could not create the working texture (%ux%u, format %u)", desc.Width, desc.Height,
               unsigned(desc.Format));
        g_scratch_uav.Reset();
        g_scratch.Reset();
        g_unsupported = true;
        return false;
    }

    g_width = desc.Width;
    g_height = desc.Height;
    g_tex_format = desc.Format;
    mb_log("working texture ready: %ux%u, texture format %u", g_width, g_height, unsigned(g_tex_format));
    return true;
}

//----------------------------------------------------------------------------
// The injection

// Says why nothing was done, the first few times.
static void skip(const char *reason)
{
    ++g_stats.skipped;
    if (g_reasons_logged < 6)
    {
        ++g_reasons_logged;
        mb_log("nothing done at the velocity flatten pass: %s", reason);
    }
}

static void inject(ID3D11DeviceContext *ctx)
{
    ++g_stats.flatten_dispatches;

    ID3D11ShaderResourceView *raw[2] = {};
    ctx->CSGetShaderResources(0, 2, raw);
    ComPtr<ID3D11ShaderResourceView> vel_srv, depth_srv;
    vel_srv.Attach(raw[0]);
    depth_srv.Attach(raw[1]);
    if (!vel_srv || !depth_srv)
        return skip("the velocity or depth texture is not bound where expected");

    ComPtr<ID3D11Resource> res;
    vel_srv->GetResource(res.GetAddressOf());
    ComPtr<ID3D11Texture2D> vel_tex;
    if (!res || FAILED(res.As(&vel_tex)))
        return skip("the velocity texture is not a 2D texture");
    D3D11_TEXTURE2D_DESC td = {};
    vel_tex->GetDesc(&td);
    if (td.Format != DXGI_FORMAT_R16G16_UNORM && td.Format != DXGI_FORMAT_R16G16_TYPELESS)
        return skip("the velocity texture is not R16G16");

    ComPtr<ID3D11Device> dev;
    ctx->GetDevice(dev.GetAddressOf());
    if (!ensure_resources(dev.Get(), td) || !g_cs)
        return skip("the working resources could not be made");

    // The View constants, vectors 114 to 118, copied now: the copy is recorded
    // in the command stream at this dispatch.
    ComPtr<ID3D11DeviceContext1> ctx1;
    if (FAILED(ctx->QueryInterface(IID_PPV_ARGS(ctx1.GetAddressOf()))))
        return skip("no ID3D11DeviceContext1");
    ComPtr<ID3D11Buffer> view;
    UINT first = 0, num = 0;
    ctx1->CSGetConstantBuffers1(1, 1, view.GetAddressOf(), &first, &num);
    if (!view)
        return skip("the View constants are not bound at slot 1");
    D3D11_BUFFER_DESC vbd = {};
    view->GetDesc(&vbd);
    const UINT start = first * 16 + VIEW_OFFSET;
    if (start + VIEW_BYTES > vbd.ByteWidth)
        return skip("the View constants are smaller than expected");
    D3D11_BOX box = {start, 0, 0, start + VIEW_BYTES, 1, 1};
    ctx->CopySubresourceRegion(g_view_copy.Get(), 0, 0, 0, 0, view.Get(), 0, &box);

    const float constants[8] = {g_settings.strength,
                                g_settings.max_length,
                                g_settings.objects,
                                g_settings.near_plane,
                                g_settings.near_start * 100.0f, // meters to the game's centimeters
                                g_settings.near_fade * 100.0f,
                                g_settings.debug ? 1.0f : 0.0f,
                                0.0f};
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    if (FAILED(ctx->Map(g_cb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
        return skip("could not update the constant buffer");
    memcpy(mapped.pData, constants, sizeof(constants));
    ctx->Unmap(g_cb.Get(), 0);

    // Save what the compute stage has bound; the game's own state must come back
    // untouched.
    ComPtr<ID3D11ComputeShader> old_cs;
    ctx->CSGetShader(old_cs.GetAddressOf(), nullptr, nullptr);
    ID3D11ShaderResourceView *old_srv[2] = {};
    ctx->CSGetShaderResources(0, 2, old_srv);
    ID3D11UnorderedAccessView *old_uav[2] = {};
    ctx->CSGetUnorderedAccessViews(0, 2, old_uav);
    ID3D11Buffer *old_cb[2] = {};
    ctx->CSGetConstantBuffers(0, 2, old_cb);

    ID3D11ShaderResourceView *srvs[2] = {vel_srv.Get(), depth_srv.Get()};
    ID3D11UnorderedAccessView *uavs[2] = {g_scratch_uav.Get(), nullptr};
    ID3D11Buffer *cbs[2] = {g_cb.Get(), g_view_copy.Get()};
    const UINT keep[2] = {0xFFFFFFFF, 0xFFFFFFFF};
    ctx->CSSetShader(g_cs.Get(), nullptr, 0);
    ctx->CSSetShaderResources(0, 2, srvs);
    ctx->CSSetUnorderedAccessViews(0, 2, uavs, keep);
    ctx->CSSetConstantBuffers(0, 2, cbs);
    ctx->Dispatch((td.Width + 7) / 8, (td.Height + 7) / 8, 1);

    ctx->CSSetShader(old_cs.Get(), nullptr, 0);
    ctx->CSSetShaderResources(0, 2, old_srv);
    ctx->CSSetUnorderedAccessViews(0, 2, old_uav, keep);
    ctx->CSSetConstantBuffers(0, 2, old_cb);
    for (ID3D11ShaderResourceView *s : old_srv)
        if (s)
            s->Release();
    for (ID3D11UnorderedAccessView *u : old_uav)
        if (u)
            u->Release();
    for (ID3D11Buffer *b : old_cb)
        if (b)
            b->Release();

    // The velocity buffer, with the camera's motion in it, goes back; the game's
    // own dispatch, next, reads it.
    ctx->CopyResource(vel_tex.Get(), g_scratch.Get());

    if (++g_stats.injected == 1)
        mb_log("first injection: velocity %ux%u, format %u, camera blur %d%%", td.Width, td.Height, unsigned(td.Format),
               int(std::lround(g_settings.strength * 100.0f)));
}

//----------------------------------------------------------------------------
// Events

static bool on_dispatch(command_list *cmd, uint32_t, uint32_t, uint32_t)
{
    if (!g_settings.enabled || cmd->get_device()->get_api() != device_api::d3d11)
        return false;
    ID3D11DeviceContext *ctx = reinterpret_cast<ID3D11DeviceContext *>(cmd->get_native());
    if (!ctx)
        return false;

    ComPtr<ID3D11ComputeShader> cs;
    ctx->CSGetShader(cs.GetAddressOf(), nullptr, nullptr);
    if (!cs || !is_flatten_cs(cs.Get()))
        return false;

    std::lock_guard<std::mutex> lock(g_state_mutex);
    inject(ctx);
    return false; // the game's own dispatch still runs
}

static void on_init_device(device *dev)
{
    if (dev->get_api() == device_api::d3d11)
        mb_log("Direct3D 11 device %p created", reinterpret_cast<void *>(dev->get_native()));
}

static void on_destroy_device(device *dev)
{
    // Other Direct3D 11 devices may come and go. Only our own resources, made on
    // one device, go with it.
    std::lock_guard<std::mutex> lock(g_state_mutex);
    if (dev->get_api() == device_api::d3d11 && g_dev && reinterpret_cast<void *>(dev->get_native()) == g_dev.Get())
    {
        mb_log("the device our resources were made on was destroyed");
        release_resources();
    }
}

static void on_init_effect_runtime(effect_runtime *)
{
    load_settings();
    mb_log("settings: %s, camera blur %d%%, objects %d%%, longest blur %.1f%%, sharp nearer than %.0f m (fade %.0f m), "
           "near plane %.0f cm, debug %d",
           g_settings.enabled ? "on" : "off", int(std::lround(g_settings.strength * 100.0f)),
           int(std::lround(g_settings.objects * 100.0f)), g_settings.max_length * 100.0f, g_settings.near_start.load(),
           g_settings.near_fade.load(), g_settings.near_plane.load(), g_settings.debug ? 1 : 0);
}

static void on_present(effect_runtime *)
{
    std::lock_guard<std::mutex> lock(g_state_mutex);
    ++g_frame;
    if (g_frame == 300 || g_frame == 1800 || g_frame % 7200 == 0)
        mb_log("frame %u: %u flatten shaders known, flatten pass dispatched %u times, camera motion added %u, "
               "nothing done %u",
               g_frame, g_flatten_found, g_stats.flatten_dispatches, g_stats.injected, g_stats.skipped);
}

//----------------------------------------------------------------------------
// The overlay window

static void draw_window(effect_runtime *)
{
    bool enabled = g_settings.enabled;
    if (ImGui::Checkbox("Enabled", &enabled))
    {
        g_settings.enabled = enabled;
        reshade::set_config_value(nullptr, SECTION, "Enabled", enabled ? 1 : 0);
    }

    int blur = int(std::lround(g_settings.strength * 100.0f));
    if (ImGui::SliderInt("Camera blur", &blur, 0, 500, "%d%%"))
    {
        g_settings.strength = float(blur) / 100.0f;
        reshade::set_config_value(nullptr, SECTION, "CameraBlurPercent", blur);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("How much of the camera's motion goes to the game's motion blur. 100%% is its true motion. "
                          "The game's own Motion Blur option has to stay on, and its strength applies on top.");

    int objects = int(std::lround(g_settings.objects * 100.0f));
    if (ImGui::SliderInt("Object blur", &objects, 0, 300, "%d%%"))
    {
        g_settings.objects = float(objects) / 100.0f;
        reshade::set_config_value(nullptr, SECTION, "ObjectBlurPercent", objects);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Scales the motion of things that move by themselves. 100%% leaves it as the game has it.");

    int length = int(std::lround(g_settings.max_length * 1000.0f));
    if (ImGui::SliderInt("Longest blur", &length, 5, 150, "%d (tenths of %% of height)"))
    {
        g_settings.max_length = float(length) / 1000.0f;
        reshade::set_config_value(nullptr, SECTION, "MaxLength", length);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("The longest streak any motion can make, as a share of the screen height. 40 is 4%%. "
                          "The game's own limit still applies after this.");

    int near_start = int(std::lround(g_settings.near_start));
    if (ImGui::SliderInt("Keep near things sharp", &near_start, 0, 20, "%d m"))
    {
        g_settings.near_start = float(near_start);
        reshade::set_config_value(nullptr, SECTION, "KeepNearSharp", near_start);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Things nearer than this get no blur, your character among them. 0 blurs everything. "
                          "Use the check box below to see where the limit falls.");

    int near_fade = int(std::lround(g_settings.near_fade));
    if (ImGui::SliderInt("Sharp to blurred over", &near_fade, 1, 10, "%d m"))
    {
        g_settings.near_fade = float(near_fade);
        reshade::set_config_value(nullptr, SECTION, "NearFade", near_fade);
    }

    bool debug = g_settings.debug;
    if (ImGui::Checkbox("Show the sharp zone (debug)", &debug))
    {
        g_settings.debug = debug;
        reshade::set_config_value(nullptr, SECTION, "Debug", debug ? 1 : 0);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Blurs everything sideways, as long as the longest blur allows, except what is kept sharp. "
                          "If the zone is too big or small for the meters above, change NearPlaneCm in ReShade.ini "
                          "(CameraBlur section, 10 by default).");

    ImGui::TextDisabled("camera motion added %u times", g_stats.injected);
}

//----------------------------------------------------------------------------

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID)
{
    switch (reason)
    {
    case DLL_PROCESS_ATTACH:
        if (!reshade::register_addon(module))
            return FALSE;
        open_log(module);
        mb_log("Camera Motion Blur (FF7 Remake) v2 loaded");
        reshade::register_event<reshade::addon_event::init_device>(on_init_device);
        reshade::register_event<reshade::addon_event::destroy_device>(on_destroy_device);
        reshade::register_event<reshade::addon_event::init_pipeline>(on_init_pipeline);
        reshade::register_event<reshade::addon_event::destroy_pipeline>(on_destroy_pipeline);
        reshade::register_event<reshade::addon_event::dispatch>(on_dispatch);
        reshade::register_event<reshade::addon_event::init_effect_runtime>(on_init_effect_runtime);
        reshade::register_event<reshade::addon_event::reshade_present>(on_present);
        reshade::register_overlay("Camera Blur", draw_window);
        break;
    case DLL_PROCESS_DETACH:
        reshade::unregister_addon(module);
        if (g_log)
            fclose(g_log);
        break;
    }
    return TRUE;
}
