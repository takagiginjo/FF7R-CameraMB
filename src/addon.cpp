// FF7 Remake Intergrade / Rebirth - full camera motion blur (ReShade add-on, DX12)
//
// How it works:
//  1. Hook NVSDK_NGX_D3D12_EvaluateFeature. DLSS needs a complete motion vector buffer
//     (camera + objects), so the data the game's own blur lacks is already there.
//  2. After DLSS runs, read Output / MotionVectors / Depth / MV scale from the NGX parameter block.
//  3. At the next ReShade command-list event (same thread), run a compute pass that blurs
//     the DLSS output along the motion vectors. This is before tonemap/UI in UE4.
//
// Config (ReShade.ini):
//   [CameraMotionBlur]
//   Enabled=1  Strength=0.5  MaxBlur=0.04  Samples=16  ReverseZ=1

#include <d3d12.h>
#include <d3dcompiler.h>
#include <windows.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <reshade.hpp>

#include <MinHook.h>

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_params.h>

using namespace reshade::api;

extern "C" __declspec(dllexport) const char *NAME = "FF7 Camera Motion Blur";
extern "C" __declspec(dllexport) const char *DESCRIPTION =
    "Full camera motion blur using the DLSS motion vectors (DX12).";

// ---------------------------------------------------------------------------------------------
// Logging (own file, independent of the ReShade log API version)
// ---------------------------------------------------------------------------------------------
static void LOG(const char *fmt, ...)
{
    char buf[512];
    va_list va;
    va_start(va, fmt);
    vsnprintf(buf, sizeof(buf), fmt, va);
    va_end(va);
    OutputDebugStringA("[ff7_cam_mb] ");
    OutputDebugStringA(buf);
    OutputDebugStringA("\n");
    FILE *f = nullptr;
    if (fopen_s(&f, "ff7_camera_motion_blur.log", "a") == 0 && f) {
        fprintf(f, "%s\n", buf);
        fclose(f);
    }
}

// ---------------------------------------------------------------------------------------------
// Shader (compiled at runtime with D3DCompile, cs_5_0 / DXBC, accepted by D3D12)
// ---------------------------------------------------------------------------------------------
static const char *kShader = R"HLSL(
cbuffer P : register(b0)
{
    float2 mvScale;
    float2 outSize;
    float2 mvSize;
    float  strength;   // shutter fraction of the per-frame motion
    float  maxBlur;    // max blur length, fraction of screen width
    uint   samples;
    uint   reverseZ;
    uint   frame;
    uint   pad;
};

Texture2D<float4>   Src   : register(t0);  // copy of the DLSS output
Texture2D<float4>   MV    : register(t1);  // DLSS motion vectors (pixels * mvScale)
Texture2D<float4>   Depth : register(t2);
RWTexture2D<float4> Dst   : register(u0);  // the DLSS output, written in place

float IGN(float2 p, uint f)
{
    p += 5.588238 * (f & 63);
    return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715))));
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= (uint2)outSize)) return;

    float4 c0 = Src.Load(int3(id.xy, 0));
    float2 uv = (id.xy + 0.5) / outSize;

    int2 mc = clamp((int2)(uv * mvSize), 0, (int2)mvSize - 1);
    float2 mv = MV.Load(int3(mc, 0)).xy * mvScale / mvSize;   // UV space
    mv *= strength;

    float len = length(mv);
    if (len > maxBlur) mv *= maxBlur / len;
    if (length(mv * outSize) < 0.75) { Dst[id.xy] = c0; return; }   // sub-pixel: skip

    float cd = Depth.Load(int3(mc, 0)).x;
    float noise = IGN((float2)id.xy, frame);

    float3 acc = c0.rgb;
    float  ws  = 1.0;

    [loop]
    for (uint i = 0; i < samples; i++)
    {
        float t = ((i + noise) / samples) - 0.5;
        float2 suv = uv + mv * t;
        if (any(suv < 0.0) || any(suv > 1.0)) continue;

        int2 sm = clamp((int2)(suv * mvSize), 0, (int2)mvSize - 1);
        float sd = Depth.Load(int3(sm, 0)).x;

        // Do not smear nearer geometry (foreground) over the center pixel
        bool nearer = reverseZ ? (sd > cd * 1.02) : (sd < cd * 0.98);
        if (nearer) continue;

        int2 sp = clamp((int2)(suv * outSize), 0, (int2)outSize - 1);
        acc += Src.Load(int3(sp, 0)).rgb;
        ws  += 1.0;
    }

    Dst[id.xy] = float4(acc / ws, c0.a);
}
)HLSL";

struct Params {
    float mvScaleX, mvScaleY;
    float outW, outH;
    float mvW, mvH;
    float strength, maxBlur;
    uint32_t samples, reverseZ, frame, pad;
};
static_assert(sizeof(Params) == 12 * 4);

// ---------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------
struct Settings {
    int   enabled   = 1;
    float strength  = 0.5f;
    float maxBlur   = 0.04f;
    int   samples   = 16;
    int   reverseZ  = 1;
} g_cfg;

struct Pending {
    bool valid = false;
    ID3D12Resource *out = nullptr, *mv = nullptr, *depth = nullptr;
    float mvScaleX = 1.f, mvScaleY = 1.f;
};
static thread_local Pending g_pending;
static thread_local bool    g_busy = false;

static std::mutex g_mutex;
static std::unordered_map<uint64_t, resource_view> g_srv, g_uav;
static pipeline_layout g_layout = {};
static pipeline        g_pipe = {};
static resource        g_tmp = {};
static uint32_t        g_tmp_w = 0, g_tmp_h = 0;
static format          g_tmp_fmt = format::unknown;
static uint32_t        g_frame = 0;
static bool            g_pipe_failed = false;
static std::vector<std::pair<resource, uint32_t>> g_deferred;   // old tmp textures, destroyed a few frames later

static format srv_format(format f)
{
    switch (f) {
    case format::d32_float:
    case format::r32_typeless:         return format::r32_float;
    case format::r32_g8_typeless:
    case format::d32_float_s8_uint:    return format::r32_float_x8_uint;
    case format::r24_g8_typeless:
    case format::d24_unorm_s8_uint:    return format::r24_unorm_x8_uint;
    case format::r16_typeless:
    case format::d16_unorm:            return format::r16_unorm;
    default:                           return format_to_default_typed(f);
    }
}

static resource_view get_view(device *dev, resource res, bool uav)
{
    auto &map = uav ? g_uav : g_srv;
    auto it = map.find(res.handle);
    if (it != map.end()) return it->second;

    const resource_desc d = dev->get_resource_desc(res);
    resource_view v = {};
    const format f = uav ? format_to_default_typed(d.texture.format) : srv_format(d.texture.format);
    if (!dev->create_resource_view(res, uav ? resource_usage::unordered_access : resource_usage::shader_resource,
                                   resource_view_desc(f), &v))
        return {};
    map[res.handle] = v;
    return v;
}

static bool ensure_pipeline(device *dev)
{
    if (g_pipe.handle) return true;
    if (g_pipe_failed) return false;

    ID3DBlob *blob = nullptr, *err = nullptr;
    HRESULT hr = D3DCompile(kShader, strlen(kShader), "ff7_mb", nullptr, nullptr, "main", "cs_5_0",
                            D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &err);
    if (FAILED(hr)) {
        LOG("shader compile failed: %s", err ? (const char *)err->GetBufferPointer() : "?");
        g_pipe_failed = true;
        return false;
    }

    pipeline_layout_param params[3];
    params[0] = constant_range{ 0, 0, 0, 12, shader_stage::compute };
    params[1] = descriptor_range{ 0, 0, 0, 3, shader_stage::compute, 1, descriptor_type::texture_shader_resource_view };
    params[2] = descriptor_range{ 0, 0, 0, 1, shader_stage::compute, 1, descriptor_type::texture_unordered_access_view };
    if (!dev->create_pipeline_layout(3, params, &g_layout)) {
        LOG("create_pipeline_layout failed");
        g_pipe_failed = true;
        return false;
    }

    shader_desc sd = {};
    sd.code = blob->GetBufferPointer();
    sd.code_size = blob->GetBufferSize();
    pipeline_subobject so = { pipeline_subobject_type::compute_shader, 1, &sd };
    if (!dev->create_pipeline(g_layout, 1, &so, &g_pipe)) {
        LOG("create_pipeline failed");
        g_pipe_failed = true;
        return false;
    }
    blob->Release();
    LOG("pipeline created");
    return true;
}

static bool ensure_tmp(device *dev, uint32_t w, uint32_t h, format f)
{
    if (g_tmp.handle && g_tmp_w == w && g_tmp_h == h && g_tmp_fmt == f) return true;
    if (g_tmp.handle) {
        // the GPU may still be using it: destroy a few frames later (see on_present)
        g_deferred.emplace_back(g_tmp, g_frame);
        g_tmp = {};
    }
    const resource_desc d(w, h, 1, 1, f, 1, memory_heap::gpu_only,
                          resource_usage::copy_dest | resource_usage::shader_resource);
    if (!dev->create_resource(d, nullptr, resource_usage::copy_dest, &g_tmp)) {
        LOG("create tmp failed");
        return false;
    }
    g_tmp_w = w; g_tmp_h = h; g_tmp_fmt = f;
    return true;
}

static void run_pending(command_list *cl)
{
    if (!g_pending.valid || g_busy) return;
    const Pending p = g_pending;
    g_pending.valid = false;
    g_busy = true;

    do {
        std::lock_guard<std::mutex> lock(g_mutex);
        device *dev = cl->get_device();
        if (!ensure_pipeline(dev)) break;

        const resource out   = { reinterpret_cast<uintptr_t>(p.out) };
        const resource mv    = { reinterpret_cast<uintptr_t>(p.mv) };
        const resource depth = { reinterpret_cast<uintptr_t>(p.depth) };

        const resource_desc od = dev->get_resource_desc(out);
        const resource_desc md = dev->get_resource_desc(mv);
        const format tf = format_to_default_typed(od.texture.format);
        if (!ensure_tmp(dev, od.texture.width, od.texture.height, tf)) break;

        resource_view views[3] = { get_view(dev, g_tmp, false), get_view(dev, mv, false), get_view(dev, depth, false) };
        resource_view uav = get_view(dev, out, true);
        if (!views[0].handle || !views[1].handle || !views[2].handle || !uav.handle) break;

        // out (UAV) -> tmp copy
        cl->barrier(out, resource_usage::unordered_access, resource_usage::copy_source);
        cl->copy_resource(out, g_tmp);
        cl->barrier(out, resource_usage::copy_source, resource_usage::unordered_access);
        cl->barrier(g_tmp, resource_usage::copy_dest, resource_usage::shader_resource);

        Params c = {};
        c.mvScaleX = p.mvScaleX; c.mvScaleY = p.mvScaleY;
        c.outW = (float)od.texture.width;  c.outH = (float)od.texture.height;
        c.mvW  = (float)md.texture.width;  c.mvH  = (float)md.texture.height;
        c.strength = g_cfg.strength;
        c.maxBlur  = g_cfg.maxBlur;
        c.samples  = (uint32_t)g_cfg.samples;
        c.reverseZ = (uint32_t)g_cfg.reverseZ;
        c.frame    = g_frame;

        cl->bind_pipeline(pipeline_stage::all_compute, g_pipe);
        cl->push_constants(shader_stage::compute, g_layout, 0, 0, 12, &c);

        descriptor_table_update u = {};
        u.binding = 0; u.array_offset = 0; u.count = 3;
        u.type = descriptor_type::texture_shader_resource_view;
        u.descriptors = views;
        cl->push_descriptors(shader_stage::compute, g_layout, 1, u);

        u.count = 1;
        u.type = descriptor_type::texture_unordered_access_view;
        u.descriptors = &uav;
        cl->push_descriptors(shader_stage::compute, g_layout, 2, u);

        cl->dispatch((od.texture.width + 7) / 8, (od.texture.height + 7) / 8, 1);

        cl->barrier(g_tmp, resource_usage::shader_resource, resource_usage::copy_dest);
    } while (false);

    g_busy = false;
}

// ---------------------------------------------------------------------------------------------
// NGX hook
// ---------------------------------------------------------------------------------------------
using EvalFn = NVSDK_NGX_Result(__cdecl *)(ID3D12GraphicsCommandList *, const NVSDK_NGX_Handle *,
                                           const NVSDK_NGX_Parameter *, PFN_NVSDK_NGX_ProgressCallback);
static EvalFn g_orig = nullptr;
static std::atomic<bool> g_hooked{ false };

static NVSDK_NGX_Result __cdecl EvalDetour(ID3D12GraphicsCommandList *cl, const NVSDK_NGX_Handle *h,
                                           const NVSDK_NGX_Parameter *cp, PFN_NVSDK_NGX_ProgressCallback cb)
{
    const NVSDK_NGX_Result r = g_orig(cl, h, cp, cb);
    if (r != NVSDK_NGX_Result_Success || !cp || !g_cfg.enabled) return r;

    NVSDK_NGX_Parameter *p = const_cast<NVSDK_NGX_Parameter *>(cp);
    Pending n = {};
    ID3D12Resource *out = nullptr, *mv = nullptr, *depth = nullptr;
    if (NVSDK_NGX_FAILED(p->Get(NVSDK_NGX_Parameter_Output, &out)) || !out) return r;
    if (NVSDK_NGX_FAILED(p->Get(NVSDK_NGX_Parameter_MotionVectors, &mv)) || !mv) return r;
    if (NVSDK_NGX_FAILED(p->Get(NVSDK_NGX_Parameter_Depth, &depth)) || !depth) return r;
    float sx = 1.f, sy = 1.f;
    p->Get(NVSDK_NGX_Parameter_MV_Scale_X, &sx);
    p->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &sy);

    n.valid = true; n.out = out; n.mv = mv; n.depth = depth; n.mvScaleX = sx; n.mvScaleY = sy;
    g_pending = n;   // consumed by the next ReShade event on this thread
    return r;
}

static void hook_thread()
{
    MH_Initialize();
    static const wchar_t *mods[] = { L"_nvngx.dll", L"nvngx.dll", L"nvngx_dlss.dll" };
    for (int i = 0; i < 1200 && !g_hooked; i++) {   // ~10 minutes
        for (const wchar_t *m : mods) {
            HMODULE mod = GetModuleHandleW(m);
            if (!mod) continue;
            void *fn = (void *)GetProcAddress(mod, "NVSDK_NGX_D3D12_EvaluateFeature");
            if (!fn) { continue; }
            if (MH_CreateHook(fn, (void *)&EvalDetour, (void **)&g_orig) == MH_OK && MH_EnableHook(fn) == MH_OK) {
                g_hooked = true;
                LOG("hooked NVSDK_NGX_D3D12_EvaluateFeature in module %ls", m);
                break;
            }
        }
        if (!g_hooked) Sleep(500);
    }
    if (!g_hooked) LOG("NGX export not found. Hook point needs adapting (see README).");
}

// ---------------------------------------------------------------------------------------------
// ReShade events: any of these fires *before* the game's next command executes,
// so inserting our pass here is safe with respect to the game's own state setup.
// ---------------------------------------------------------------------------------------------
static void on_barrier(command_list *cl, uint32_t, const resource *, const resource_usage *, const resource_usage *) { run_pending(cl); }
static void on_bind_pipeline(command_list *cl, pipeline_stage, pipeline) { run_pending(cl); }
static void on_bind_rt(command_list *cl, uint32_t, const resource_view *, resource_view) { run_pending(cl); }
static void on_present(command_queue *, swapchain *sc, const rect *, const rect *, uint32_t, const rect *)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    g_frame++;
    for (size_t i = 0; i < g_deferred.size();) {
        if (g_frame - g_deferred[i].second >= 8) {
            sc->get_device()->destroy_resource(g_deferred[i].first);
            g_deferred.erase(g_deferred.begin() + i);
        } else {
            i++;
        }
    }
}

static void on_destroy_resource(device *dev, resource res)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto *map : { &g_srv, &g_uav }) {
        auto it = map->find(res.handle);
        if (it != map->end()) { dev->destroy_resource_view(it->second); map->erase(it); }
    }
}

static void on_destroy_device(device *dev)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto *map : { &g_srv, &g_uav }) {
        for (auto &kv : *map) dev->destroy_resource_view(kv.second);
        map->clear();
    }
    for (auto &d : g_deferred) dev->destroy_resource(d.first);
    g_deferred.clear();
    if (g_tmp.handle)    { dev->destroy_resource(g_tmp); g_tmp = {}; }
    if (g_pipe.handle)   { dev->destroy_pipeline(g_pipe); g_pipe = {}; }
    if (g_layout.handle) { dev->destroy_pipeline_layout(g_layout); g_layout = {}; }
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID)
{
    switch (reason) {
    case DLL_PROCESS_ATTACH:
        if (!reshade::register_addon(hModule)) return FALSE;

        reshade::get_config_value(nullptr, "CameraMotionBlur", "Enabled",  g_cfg.enabled);
        reshade::get_config_value(nullptr, "CameraMotionBlur", "Strength", g_cfg.strength);
        reshade::get_config_value(nullptr, "CameraMotionBlur", "MaxBlur",  g_cfg.maxBlur);
        reshade::get_config_value(nullptr, "CameraMotionBlur", "Samples",  g_cfg.samples);
        reshade::get_config_value(nullptr, "CameraMotionBlur", "ReverseZ", g_cfg.reverseZ);

        reshade::register_event<reshade::addon_event::barrier>(on_barrier);
        reshade::register_event<reshade::addon_event::bind_pipeline>(on_bind_pipeline);
        reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind_rt);
        reshade::register_event<reshade::addon_event::present>(on_present);
        reshade::register_event<reshade::addon_event::destroy_resource>(on_destroy_resource);
        reshade::register_event<reshade::addon_event::destroy_device>(on_destroy_device);

        std::thread(hook_thread).detach();
        break;
    case DLL_PROCESS_DETACH:
        reshade::unregister_addon(hModule);
        break;
    }
    return TRUE;
}
