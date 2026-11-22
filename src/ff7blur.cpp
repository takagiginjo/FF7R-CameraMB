// Camera motion blur for Final Fantasy VII Remake (Unreal Engine 4, Direct3D 11),
// as a ReShade add-on, next to Luma.
//
// The game has Unreal's motion blur and it blurs objects only: its first pass,
// the velocity flatten compute shader, works out the camera's motion from
// constants that come out as nothing in this game. This add-on takes that pass
// over, as the Assassin's Creed add-ons take over their game's blur draw:
//   1. It finds the pass by its hash, or by the constants it decodes the
//      velocity with, when the game creates the shader.
//   2. On that dispatch, it runs shaders/flatten.hlsl instead, the game's own
//      shader written back as HLSL, with the camera's motion worked out from the
//      depth and ClipToPrevClip in the View constants, and with a strength, a
//      longest blur and near things kept sharp. It writes what the game's pass
//      writes, into the game's own textures, and the game's dispatch is skipped.
//   3. The game's own blur passes then read it as usual. If anything is missing,
//      the game's pass runs as if this add-on were not there.
// The game's own Motion Blur option has to stay on, as that is what makes it run.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#define ImTextureID ImU64
#include <imgui.h>
#include <reshade.hpp>

#include <d3d11_1.h>
#include <wrl/client.h>

#include <flatten.h> // generated from shaders/flatten.hlsl: flatten_cs

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
    std::atomic<bool> debug{false};      // show where things are kept sharp
};
static settings g_settings;

static void load_settings()
{
    int enabled = 1, strength = 200, objects = 100, max_length = 40, near_start = 4, near_fade = 2, debug = 0;
    reshade::get_config_value(nullptr, SECTION, "Enabled", enabled);
    reshade::get_config_value(nullptr, SECTION, "CameraBlurPercent", strength);
    reshade::get_config_value(nullptr, SECTION, "ObjectBlurPercent", objects);
    reshade::get_config_value(nullptr, SECTION, "MaxLength", max_length); // tenths of a percent of the height
    reshade::get_config_value(nullptr, SECTION, "KeepNearSharp", near_start); // meters
    reshade::get_config_value(nullptr, SECTION, "NearFade", near_fade);       // meters
    reshade::get_config_value(nullptr, SECTION, "Debug", debug);
    g_settings.enabled = enabled != 0;
    g_settings.strength = float(std::clamp(strength, 0, 500)) / 100.0f;
    g_settings.objects = float(std::clamp(objects, 0, 300)) / 100.0f;
    g_settings.max_length = float(std::clamp(max_length, 5, 150)) / 1000.0f;
    g_settings.near_start = float(std::clamp(near_start, 0, 20));
    g_settings.near_fade = float(std::clamp(near_fade, 1, 10));
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
static bool g_unsupported = false;

// The View constants' vectors 114 to 118, as bytes.
static const UINT VIEW_OFFSET = 114 * 16;
static const UINT VIEW_BYTES = 5 * 16;

struct statistics
{
    unsigned flatten_dispatches = 0, replaced = 0, left_to_game = 0;
};
static statistics g_stats;
static unsigned g_frame = 0;
static unsigned g_reasons_logged = 0;
static bool g_constants_logged = false;

static void release_resources()
{
    g_cs.Reset();
    g_cb.Reset();
    g_view_copy.Reset();
    g_dev.Reset();
    g_unsupported = false;
}

static bool ensure_resources(ID3D11Device *dev)
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
    if (g_cs)
        return true;

    if (FAILED(dev->CreateComputeShader(flatten_cs, sizeof(flatten_cs), nullptr, g_cs.GetAddressOf())))
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
    return true;
}

//----------------------------------------------------------------------------
// The game's flatten dispatch, taken over

// Says why the game's own pass was left to run, the first few times.
static bool give_back(const char *reason)
{
    ++g_stats.left_to_game;
    if (g_reasons_logged < 6)
    {
        ++g_reasons_logged;
        mb_log("the game's own flatten pass runs: %s", reason);
    }
    return false;
}

// Reads `count` vectors of the constant buffer bound at a compute slot, from a
// vector offset in the bound range. Stalls the GPU: only for the log, once.
static bool read_constants(ID3D11DeviceContext *ctx, ID3D11DeviceContext1 *ctx1, UINT slot, UINT first_vector,
                           UINT count, float *out)
{
    ComPtr<ID3D11Device> dev;
    ctx->GetDevice(dev.GetAddressOf());
    ComPtr<ID3D11Buffer> buf;
    UINT first = 0, num = 0;
    ctx1->CSGetConstantBuffers1(slot, 1, buf.GetAddressOf(), &first, &num);
    if (!buf)
        return false;
    D3D11_BUFFER_DESC bd = {};
    buf->GetDesc(&bd);
    const UINT start = (first + first_vector) * 16;
    if (start + count * 16 > bd.ByteWidth)
        return false;
    D3D11_BUFFER_DESC sd = {};
    sd.ByteWidth = count * 16;
    sd.Usage = D3D11_USAGE_STAGING;
    sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Buffer> staging;
    if (FAILED(dev->CreateBuffer(&sd, nullptr, staging.GetAddressOf())))
        return false;
    D3D11_BOX box = {start, 0, 0, start + count * 16, 1, 1};
    ctx->CopySubresourceRegion(staging.Get(), 0, 0, 0, 0, buf.Get(), 0, &box);
    D3D11_MAPPED_SUBRESOURCE m = {};
    if (FAILED(ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &m)))
        return false;
    memcpy(out, m.pData, count * 16);
    ctx->Unmap(staging.Get(), 0);
    return true;
}

// The game's constants for this pass, once, to check the shader's reading of them.
static void log_game_constants(ID3D11DeviceContext *ctx, ID3D11DeviceContext1 *ctx1)
{
    if (g_constants_logged || ctx->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE)
        return;
    g_constants_logged = true;
    float v[5 * 4] = {};
    if (read_constants(ctx, ctx1, 0, 15, 5, v))
    {
        unsigned r[4];
        memcpy(r, v, sizeof(r));
        mb_log("b0[15] view rect %u %u %u %u", r[0], r[1], r[2], r[3]);
        mb_log("b0[18] %g %g %g %g (b0[18].x is read as the height/width, .w/.y as the longest blur)", v[12], v[13],
               v[14], v[15]);
        mb_log("b0[19] %g %g %g %g (.w mixes the objects' length with their motion against the camera)", v[16],
               v[17], v[18], v[19]);
    }
    float c[5 * 4] = {};
    if (read_constants(ctx, ctx1, 2, 0, 5, c))
        mb_log("b2 camera constants, which this add-on does not use: %g %g %g %g | %g %g %g %g | %g %g %g %g | %g %g "
               "%g %g | %g %g %g %g",
               c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15],
               c[16], c[17], c[18], c[19]);
    float z[4] = {};
    if (read_constants(ctx, ctx1, 1, 57, 1, z))
        mb_log("b1[57] depth to distance: %g %g %g %g", z[0], z[1], z[2], z[3]);
    float m[5 * 4] = {};
    if (read_constants(ctx, ctx1, 1, 114, 5, m))
        mb_log("View 114..118: %g %g %g %g | %g %g %g %g | %g %g %g %g | %g %g %g %g | jitter %g %g %g %g", m[0], m[1],
               m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15], m[16], m[17],
               m[18], m[19]);
}

// Returns true when the game's dispatch should be skipped.
static bool replace_flatten(ID3D11DeviceContext *ctx, UINT gx, UINT gy, UINT gz)
{
    ++g_stats.flatten_dispatches;

    ComPtr<ID3D11Device> dev;
    ctx->GetDevice(dev.GetAddressOf());
    if (!ensure_resources(dev.Get()) || !g_cs)
        return give_back("the working resources could not be made");

    // What the game's shader reads and writes has to be where the port expects it:
    // velocity and depth in t0 and t1, the two outputs in u0 and u1.
    ID3D11ShaderResourceView *srv[2] = {};
    ctx->CSGetShaderResources(0, 2, srv);
    const bool have_srv = srv[0] && srv[1];
    for (ID3D11ShaderResourceView *s : srv)
        if (s)
            s->Release();
    ID3D11UnorderedAccessView *uav[2] = {};
    ctx->CSGetUnorderedAccessViews(0, 2, uav);
    const bool have_uav = uav[0] && uav[1];
    for (ID3D11UnorderedAccessView *u : uav)
        if (u)
            u->Release();
    if (!have_srv || !have_uav)
        return give_back("the velocity, depth or output textures are not bound where expected");

    // The View constants, vectors 114 to 118, copied now: the copy is recorded in
    // the command stream at this dispatch.
    ComPtr<ID3D11DeviceContext1> ctx1;
    if (FAILED(ctx->QueryInterface(IID_PPV_ARGS(ctx1.GetAddressOf()))))
        return give_back("no ID3D11DeviceContext1");
    ComPtr<ID3D11Buffer> view;
    UINT first = 0, num = 0;
    ctx1->CSGetConstantBuffers1(1, 1, view.GetAddressOf(), &first, &num);
    if (!view)
        return give_back("the View constants are not bound at slot 1");
    D3D11_BUFFER_DESC vbd = {};
    view->GetDesc(&vbd);
    const UINT start = first * 16 + VIEW_OFFSET;
    if (start + VIEW_BYTES > vbd.ByteWidth)
        return give_back("the View constants are smaller than expected");
    D3D11_BOX box = {start, 0, 0, start + VIEW_BYTES, 1, 1};
    ctx->CopySubresourceRegion(g_view_copy.Get(), 0, 0, 0, 0, view.Get(), 0, &box);

    const float constants[8] = {g_settings.strength,
                                g_settings.max_length,
                                g_settings.objects,
                                g_settings.near_start * 100.0f, // meters to the game's centimeters
                                g_settings.near_fade * 100.0f,
                                g_settings.debug ? 1.0f : 0.0f,
                                0.0f,
                                0.0f};
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    if (FAILED(ctx->Map(g_cb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
        return give_back("could not update the constant buffer");
    memcpy(mapped.pData, constants, sizeof(constants));
    ctx->Unmap(g_cb.Get(), 0);

    log_game_constants(ctx, ctx1.Get());

    // Our shader takes the game's place; everything the game bound stays. Only
    // the shader and our two constant buffers change, and come back after.
    ComPtr<ID3D11ComputeShader> old_cs;
    ctx->CSGetShader(old_cs.GetAddressOf(), nullptr, nullptr);
    ID3D11Buffer *old_cb[2] = {};
    ctx->CSGetConstantBuffers(3, 2, old_cb);

    ID3D11Buffer *cbs[2] = {g_cb.Get(), g_view_copy.Get()};
    ctx->CSSetShader(g_cs.Get(), nullptr, 0);
    ctx->CSSetConstantBuffers(3, 2, cbs);
    ctx->Dispatch(gx, gy, gz);

    ctx->CSSetShader(old_cs.Get(), nullptr, 0);
    ctx->CSSetConstantBuffers(3, 2, old_cb);
    for (ID3D11Buffer *b : old_cb)
        if (b)
            b->Release();

    if (++g_stats.replaced == 1)
        mb_log("first flatten pass taken over: %ux%ux%u groups, camera blur %d%%", gx, gy, gz,
               int(std::lround(g_settings.strength * 100.0f)));
    return true;
}

//----------------------------------------------------------------------------
// Events

static bool on_dispatch(command_list *cmd, uint32_t gx, uint32_t gy, uint32_t gz)
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
    return replace_flatten(ctx, gx, gy, gz); // true: the game's own dispatch is skipped
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
           "debug %d",
           g_settings.enabled ? "on" : "off", int(std::lround(g_settings.strength * 100.0f)),
           int(std::lround(g_settings.objects * 100.0f)), g_settings.max_length * 100.0f, g_settings.near_start.load(),
           g_settings.near_fade.load(), g_settings.debug ? 1 : 0);
}

static void on_present(effect_runtime *)
{
    std::lock_guard<std::mutex> lock(g_state_mutex);
    ++g_frame;
    if (g_frame == 300 || g_frame == 1800 || g_frame % 7200 == 0)
        mb_log("frame %u: %u flatten shaders known, flatten pass dispatched %u times, taken over %u, left to the game "
               "%u",
               g_frame, g_flatten_found, g_stats.flatten_dispatches, g_stats.replaced, g_stats.left_to_game);
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
                          "The distance is the game's own, in meters.");

    ImGui::TextDisabled("flatten pass dispatched %u times, taken over %u", g_stats.flatten_dispatches, g_stats.replaced);
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
        mb_log("Camera Motion Blur (FF7 Remake) v3 loaded");
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
