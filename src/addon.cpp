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

#include <d3d11.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <windows.h>
#include <psapi.h>

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
    int   hookOnly  = 0;   // 1 = only hook + log, never draw (for bisecting crashes)
} g_cfg;

static std::atomic<int> g_dbg{ 0 };
#define DBG(...) do { if (g_dbg.fetch_add(1) < 60) LOG(__VA_ARGS__); } while (0)

struct Pending {
    bool valid = false;
    void *out = nullptr, *mv = nullptr, *depth = nullptr;
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
    if (g_cfg.hookOnly) { DBG("hook-only: pass skipped"); return; }
    g_busy = true;
    DBG("run_pending: begin");

    do {
        std::lock_guard<std::mutex> lock(g_mutex);
        device *dev = cl->get_device();
        DBG("run_pending: api=%d", (int)dev->get_api());
        if (!ensure_pipeline(dev)) break;
        DBG("run_pending: pipeline ok");

        const resource out   = { reinterpret_cast<uintptr_t>(p.out) };
        const resource mv    = { reinterpret_cast<uintptr_t>(p.mv) };
        const resource depth = { reinterpret_cast<uintptr_t>(p.depth) };

        const resource_desc od = dev->get_resource_desc(out);
        const resource_desc md = dev->get_resource_desc(mv);
        const format tf = format_to_default_typed(od.texture.format);
        DBG("run_pending: out %ux%u fmt=%d, mv %ux%u fmt=%d", od.texture.width, od.texture.height, (int)od.texture.format,
            md.texture.width, md.texture.height, (int)md.texture.format);
        if (!ensure_tmp(dev, od.texture.width, od.texture.height, tf)) break;

        resource_view views[3] = { get_view(dev, g_tmp, false), get_view(dev, mv, false), get_view(dev, depth, false) };
        resource_view uav = get_view(dev, out, true);
        DBG("run_pending: views %d %d %d uav %d", views[0].handle != 0, views[1].handle != 0, views[2].handle != 0, uav.handle != 0);
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

        DBG("run_pending: dispatch");
        cl->dispatch((od.texture.width + 7) / 8, (od.texture.height + 7) / 8, 1);
        DBG("run_pending: dispatched");

        if (dev->get_api() == device_api::d3d11) {
            // D3D11 state is sticky: unbind our views so the game's next pass sees no hazards
            resource_view nulls[3] = {};
            u.count = 3;
            u.type = descriptor_type::texture_shader_resource_view;
            u.descriptors = nulls;
            cl->push_descriptors(shader_stage::compute, g_layout, 1, u);
            u.count = 1;
            u.type = descriptor_type::texture_unordered_access_view;
            cl->push_descriptors(shader_stage::compute, g_layout, 2, u);
        }

        cl->barrier(g_tmp, resource_usage::shader_resource, resource_usage::copy_dest);
    } while (false);

    g_busy = false;
}

// ---------------------------------------------------------------------------------------------
// NGX hook (D3D11 for Remake + Luma, D3D12 for Rebirth)
// ---------------------------------------------------------------------------------------------
using EvalFn11 = NVSDK_NGX_Result(__cdecl *)(ID3D11DeviceContext *, const NVSDK_NGX_Handle *,
                                             const NVSDK_NGX_Parameter *, PFN_NVSDK_NGX_ProgressCallback);
using EvalFn12 = NVSDK_NGX_Result(__cdecl *)(ID3D12GraphicsCommandList *, const NVSDK_NGX_Handle *,
                                             const NVSDK_NGX_Parameter *, PFN_NVSDK_NGX_ProgressCallback);
static EvalFn11 g_orig11 = nullptr;
static EvalFn12 g_orig12 = nullptr;
static std::atomic<bool> g_hooked11{ false }, g_hooked12{ false };

template <class Res>
static void capture_impl(NVSDK_NGX_Parameter *p)
{
    Res *out = nullptr, *mv = nullptr, *depth = nullptr;
    if (NVSDK_NGX_FAILED(p->Get(NVSDK_NGX_Parameter_Output, &out)) || !out) return;
    if (NVSDK_NGX_FAILED(p->Get(NVSDK_NGX_Parameter_MotionVectors, &mv)) || !mv) return;
    if (NVSDK_NGX_FAILED(p->Get(NVSDK_NGX_Parameter_Depth, &depth)) || !depth) return;
    float sx = 1.f, sy = 1.f;
    p->Get(NVSDK_NGX_Parameter_MV_Scale_X, &sx);
    p->Get(NVSDK_NGX_Parameter_MV_Scale_Y, &sy);

    DBG("capture: out=%p mv=%p depth=%p mvscale=%f,%f", (void *)out, (void *)mv, (void *)depth, sx, sy);
    Pending n = {};
    n.valid = true; n.out = out; n.mv = mv; n.depth = depth; n.mvScaleX = sx; n.mvScaleY = sy;
    g_pending = n;   // consumed by the next ReShade event on this thread
}

template <class Res>
static void capture(NVSDK_NGX_Parameter *p)
{
    __try {
        capture_impl<Res>(p);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        LOG("exception while reading NGX parameters (code 0x%08lx): add-on disabled", GetExceptionCode());
        g_cfg.enabled = 0;
    }
}

static void *g_target11 = nullptr, *g_target12 = nullptr;
static thread_local int g_depth = 0;
static std::atomic<int> g_api{ -1 };   // reshade::api::device_api of the running game

static void break_recursion()
{
    // g_orig leads back into our detour (seen in a crash dump): unhook everything and fall back to the real function
    LOG("re-entrancy detected (orig11=%p orig12=%p target11=%p target12=%p): hooks removed", (void *)g_orig11,
        (void *)g_orig12, g_target11, g_target12);
    MH_DisableHook(MH_ALL_HOOKS);
    g_cfg.enabled = 0;
}

static NVSDK_NGX_Result __cdecl EvalDetour11(ID3D11DeviceContext *cl, const NVSDK_NGX_Handle *h,
                                             const NVSDK_NGX_Parameter *cp, PFN_NVSDK_NGX_ProgressCallback cb)
{
    if (g_depth >= 4) {
        break_recursion();
        return ((EvalFn11)g_target11)(cl, h, cp, cb);   // now unhooked: calls the real function
    }
    ++g_depth;
    const NVSDK_NGX_Result r = g_orig11(cl, h, cp, cb);
    --g_depth;
    if (r == NVSDK_NGX_Result_Success && cp && g_cfg.enabled)
        capture<ID3D11Resource>(const_cast<NVSDK_NGX_Parameter *>(cp));
    return r;
}

static NVSDK_NGX_Result __cdecl EvalDetour12(ID3D12GraphicsCommandList *cl, const NVSDK_NGX_Handle *h,
                                             const NVSDK_NGX_Parameter *cp, PFN_NVSDK_NGX_ProgressCallback cb)
{
    if (g_depth >= 4) {
        break_recursion();
        return ((EvalFn12)g_target12)(cl, h, cp, cb);
    }
    ++g_depth;
    const NVSDK_NGX_Result r = g_orig12(cl, h, cp, cb);
    --g_depth;
    if (r == NVSDK_NGX_Result_Success && cp && g_cfg.enabled)
        capture<ID3D12Resource>(const_cast<NVSDK_NGX_Parameter *>(cp));
    return r;
}

static void log_bytes(const char *what, void *p)
{
    char hex[64] = {};
    __try {
        for (int i = 0; i < 16; i++) sprintf_s(hex + i * 3, sizeof(hex) - i * 3, "%02x ", ((unsigned char *)p)[i]);
    } __except (EXCEPTION_EXECUTE_HANDLER) { strcpy_s(hex, "?"); }
    LOG("%s at %p: %s", what, p, hex);
}

static void hook_thread()
{
    // Wait for the game's graphics API, then hook ONLY the matching NGX export
    for (int i = 0; i < 600 && g_api < 0; i++) Sleep(100);   // init_device event normally fires first
    if (g_api < 0) {   // fallback if the device was created before this add-on loaded
        g_api = GetModuleHandleW(L"d3d11.dll") ? (int)device_api::d3d11 : (int)device_api::d3d12;
        LOG("init_device not seen, guessing API %d from loaded modules", (int)g_api);
    }
    const bool dx11 = g_api == (int)device_api::d3d11;
    const bool dx12 = g_api == (int)device_api::d3d12;
    if (!dx11 && !dx12) { LOG("unsupported graphics API %d", (int)g_api); return; }
    const char *exportName = dx11 ? "NVSDK_NGX_D3D11_EvaluateFeature" : "NVSDK_NGX_D3D12_EvaluateFeature";

    // Hooks outlive this DLL if ReShade unloads and reloads it (seen in logs): never hook twice in one process.
    // The mutex is intentionally never closed, so it lives as long as the process.
    CreateMutexW(nullptr, FALSE, L"Local\\ff7_camera_motion_blur_hooked");
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        LOG("hooks already installed by an earlier load of this add-on, skipping (add-on disabled in this instance)");
        g_cfg.enabled = 0;
        return;
    }

    MH_Initialize();
    static const wchar_t *mods[] = { L"_nvngx.dll", L"nvngx.dll", L"nvngx_dlss.dll" };
    for (int i = 0; i < 1200 && !g_hooked11 && !g_hooked12; i++) {   // ~10 minutes
        for (const wchar_t *m : mods) {
            HMODULE mod = GetModuleHandleW(m);
            if (!mod) continue;
            void *fn = (void *)GetProcAddress(mod, exportName);
            if (!fn) continue;
            LOG("%ls: %s resolves to %p (module base %p)", m, exportName, fn, (void *)mod);
            log_bytes("  prologue", fn);

            void *detour = dx11 ? (void *)&EvalDetour11 : (void *)&EvalDetour12;
            void **orig = dx11 ? (void **)&g_orig11 : (void **)&g_orig12;
            const MH_STATUS st = MH_CreateHook(fn, detour, orig);
            if (st != MH_OK) { LOG("  MH_CreateHook failed: %s", MH_StatusToString(st)); continue; }
            (dx11 ? g_target11 : g_target12) = fn;
            LOG("  trampoline=%p detour=%p", *orig, detour);
            if (*orig == detour) { LOG("  trampoline == detour, aborting"); MH_RemoveHook(fn); continue; }
            if (MH_EnableHook(fn) != MH_OK) { LOG("  MH_EnableHook failed"); continue; }
            (dx11 ? g_hooked11 : g_hooked12) = true;
            LOG("hooked %s in %ls", exportName, m);
            break;
        }
        if (!g_hooked11 && !g_hooked12) Sleep(500);
    }
    if (!g_hooked11 && !g_hooked12) LOG("NGX evaluate export not found/hooked.");
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

static void on_init_device(device *dev) { g_api = (int)dev->get_api(); }

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
    {
        // Pin: our detours live in this DLL, so it must never be unloaded (FreeLibrary becomes a no-op)
        HMODULE pinned = nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                           reinterpret_cast<LPCWSTR>(&DllMain), &pinned);

        if (!reshade::register_addon(hModule)) return FALSE;

        reshade::get_config_value(nullptr, "CameraMotionBlur", "Enabled",  g_cfg.enabled);
        reshade::get_config_value(nullptr, "CameraMotionBlur", "Strength", g_cfg.strength);
        reshade::get_config_value(nullptr, "CameraMotionBlur", "MaxBlur",  g_cfg.maxBlur);
        reshade::get_config_value(nullptr, "CameraMotionBlur", "Samples",  g_cfg.samples);
        reshade::get_config_value(nullptr, "CameraMotionBlur", "ReverseZ", g_cfg.reverseZ);
        reshade::get_config_value(nullptr, "CameraMotionBlur", "HookOnly", g_cfg.hookOnly);

        reshade::register_event<reshade::addon_event::barrier>(on_barrier);
        reshade::register_event<reshade::addon_event::bind_pipeline>(on_bind_pipeline);
        reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(on_bind_rt);
        reshade::register_event<reshade::addon_event::init_device>(on_init_device);
        reshade::register_event<reshade::addon_event::present>(on_present);
        reshade::register_event<reshade::addon_event::destroy_resource>(on_destroy_resource);
        reshade::register_event<reshade::addon_event::destroy_device>(on_destroy_device);

        std::thread(hook_thread).detach();
        break;
    }
    case DLL_PROCESS_DETACH:
        reshade::unregister_addon(hModule);
        break;
    }
    return TRUE;
}
