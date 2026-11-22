// Motion blur for Final Fantasy VII Remake (Unreal Engine 4, Direct3D 11).
//
// It stands in for the game's own motion blur filter, and is built the way the
// ones in the Assassin's Creed add-ons are, so the same settings mean the same
// thing. It reads the scene colour (t0), the velocity buffer (t1, R16G16_UNORM,
// with the camera's motion added by the other shader of this add-on) and the
// depth (t2):
//
//  - Each pixel is averaged along its own velocity, from half a blur behind to
//    half ahead, as if the shutter had stayed open for `shutter` of a frame,
//    with taps spread by noise.
//  - The blur is never longer than `max_length` of the screen height.
//  - Things near the camera are kept sharp, and do not lend their colour to the
//    blur of things behind them: the player's character, when the camera turns
//    round it, is the reason. The distance comes from the depth the way
//    Unreal's shaders do it, with View.InvDeviceZToWorldZTransform (b1, vector
//    57 of the View constants): d * x + y + 1 / (d * z - w), in centimetres.
//
// The velocity is decoded the way Unreal's shaders do: (enc - 0.499992) *
// 4.008016, in clip space (2 across the screen, y up), and is the current
// position minus the previous one.

cbuffer constants : register(b0)
{
    float shutter;    // the part of a frame's motion that is blurred
    float max_length; // longest blur, as a fraction of the screen height
    float samples;    // taps along the blur
    float frame;      // changes every frame, moves the noise
    float near_cm;    // distance below which things are kept sharp
    float debug;      // 1 = show the velocity instead of the image
    float padding0;
    float padding1;
};

cbuffer depth_params : register(b1)
{
    float4 inv_z; // View.InvDeviceZToWorldZTransform
};

Texture2D<float4> color_tex : register(t0);
Texture2D<float4> vel_tex : register(t1);
Texture2D<float4> depth_tex : register(t2);
SamplerState linear_clamp : register(s0);
RWTexture2D<float4> target : register(u0);

float smooth01(float x)
{
    x = saturate(x);
    return x * x * (3.0 - 2.0 * x);
}

// Distance from the camera, in centimetres, from the depth at a UV.
float distance_at(float2 uv)
{
    uint dw, dh;
    depth_tex.GetDimensions(dw, dh);
    int2 p = clamp(int2(uv * float2(dw, dh)), int2(0, 0), int2(int(dw) - 1, int(dh) - 1));
    float d = depth_tex.Load(int3(p, 0)).x;
    return abs(d * inv_z.x + inv_z.y + 1.0 / (d * inv_z.z - inv_z.w));
}

// Interleaved gradient noise: spreads the taps so a few of them don't band.
float noise(uint2 p, float f)
{
    return frac(52.9829189 * frac(dot(float2(p) + f * 5.588238, float2(0.06711056, 0.00583715))));
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    target.GetDimensions(width, height);
    if (id.x >= width || id.y >= height)
        return;

    float2 size = float2(width, height);
    float2 uv = (float2(id.xy) + 0.5) / size;

    uint vw, vh;
    vel_tex.GetDimensions(vw, vh);
    int2 vp = int2(min(uint2(uv * float2(vw, vh)), uint2(vw - 1, vh - 1)));
    float2 enc = vel_tex.Load(int3(vp, 0)).xy;
    float2 v_clip = (enc - 0.499992) * 4.008016;
    // Clip space to UV: half as long, and y runs the other way.
    float2 mv = float2(v_clip.x * 0.5, v_clip.y * -0.5);

    float4 here = color_tex.SampleLevel(linear_clamp, uv, 0.0);

    if (debug > 0.5)
    {
        // Red and green: how far the pixel moved sideways and up or down.
        target[id.xy] = float4(saturate(abs(mv) * 20.0), 0.0, 1.0);
        return;
    }

    // The blur's length: the velocity, the shutter, and the fade that keeps
    // near things sharp.
    float fade = smooth01((distance_at(uv) - near_cm) / (0.4 * near_cm + 1.0));
    float2 v = mv * shutter * fade;

    float aspect = size.x / size.y;
    float len = length(v * float2(aspect, 1.0));
    if (len > max_length)
    {
        v *= max_length / len;
        len = max_length;
    }

    // Less than a pixel of motion: nothing to blur.
    if (len * size.y < 1.0)
    {
        target[id.xy] = here;
        return;
    }

    int n = clamp(int(samples), 2, 32);
    float jitter = noise(id.xy, frame) - 0.5;
    float3 sum = float3(0.0, 0.0, 0.0);
    float weight = 0.0;
    [loop]
    for (int i = 0; i < n; ++i)
    {
        // From half a blur behind to half ahead, each tap moved a little by the
        // noise inside its own slot.
        float t = (float(i) + 0.5 + jitter) / float(n) - 0.5;
        float2 tap_uv = uv + v * t;
        // A near tap does not count: it would drag the near thing into the
        // blur of what is behind it.
        float w = smooth01((distance_at(tap_uv) - near_cm) / (3.0 * near_cm + 1.0));
        sum += color_tex.SampleLevel(linear_clamp, tap_uv, 0.0).rgb * w;
        weight += w;
    }
    // Every tap near: nothing to blur with.
    float3 color = weight > 0.0 ? sum / weight : here.rgb;
    target[id.xy] = float4(color, here.a);
}
EOF
python3 - <<'EOF'
p='/tmp/ff7/CMakeLists.txt'
c=open(p).read()
c=c.replace('''# The leading 00- makes ReShade load the add-on first.''','''set(blur_header ${CMAKE_CURRENT_BINARY_DIR}/shaders/blur.h)
add_custom_command(
    OUTPUT ${blur_header}
    COMMAND ${FXC} /nologo /T cs_5_0 /E main /O3 /Vn blur_cs /Fh ${blur_header} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/blur.hlsl
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/blur.hlsl
    COMMENT "Compiling blur.hlsl")

# The leading 00- makes ReShade load the add-on first.''')
c=c.replace('add_library(ff7blur SHARED src/ff7blur.cpp ${header})','add_library(ff7blur SHARED src/ff7blur.cpp ${header} ${blur_header})')
open(p,'w').write(c)
print('cmake ok')
EOF
p='src/ff7blur.cpp'
s=open(p).read()
def sub(old,new,cnt=1):
    global s
    assert s.count(old)==cnt,(s.count(old),old[:90])
    s=s.replace(old,new)

# clean the odd lambda: create the buffer with the others
a=s.index('    // Vector 57 of the View constants turns a depth into a distance; the blur')
b=s.index('    const bool own = g_settings.own_blur;') if False else s.index('    const UINT zstart = first * 16 + 57 * 16;')
# find the end of that block: the closing of the if(zstart...) and the outer if
end_marker='''            ctx->CopySubresourceRegion(g_zconv_copy.Get(), 0, 0, 0, 0, view.Get(), 0, &zbox);
        }
    }
'''
e=s.index(end_marker)+len(end_marker)
s=s[:a]+'''    // Vector 57 of the View constants turns a depth into a distance; the blur
    // needs it later, when the View constants may not be bound.
    const UINT zstart = first * 16 + 57 * 16;
    if (g_zconv_copy && zstart + 16 <= vbd.ByteWidth)
    {
        D3D11_BOX zbox = {zstart, 0, 0, zstart + 16, 1, 1};
        ctx->CopySubresourceRegion(g_zconv_copy.Get(), 0, 0, 0, 0, view.Get(), 0, &zbox);
    }
'''+s[e:]

# create zconv buffer + blur cb + sampler along with the others
sub('''        if (FAILED(dev->CreateBuffer(&bd, nullptr, g_cb.GetAddressOf())) ||
            FAILED(dev->CreateBuffer(&cd, nullptr, g_view_copy.GetAddressOf())))''','''        D3D11_BUFFER_DESC zd = {};
        zd.ByteWidth = 16;
        zd.Usage = D3D11_USAGE_DEFAULT;
        zd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        if (FAILED(dev->CreateBuffer(&bd, nullptr, g_cb.GetAddressOf())) ||
            FAILED(dev->CreateBuffer(&cd, nullptr, g_view_copy.GetAddressOf())) ||
            FAILED(dev->CreateBuffer(&zd, nullptr, g_zconv_copy.GetAddressOf())))''')

# blur resources + replace_filter before the Events section
sub('''//----------------------------------------------------------------------------
// Events
''','''//----------------------------------------------------------------------------
// The motion blur filter, taken over

static bool give_back(const char *reason)
{
    ++g_stats.filters_given_back;
    g_filter_pending = false;
    if (g_reasons_logged < 8)
    {
        ++g_reasons_logged;
        mb_log("the game's own blur runs: %s", reason);
    }
    return false;
}

static bool ensure_blur_resources(ID3D11Device *dev, const D3D11_TEXTURE2D_DESC &desc, DXGI_FORMAT view_format)
{
    if (g_blur_unsupported)
        return false;
    if (!g_blur_cs)
    {
        D3D11_BUFFER_DESC bd = {};
        bd.ByteWidth = 32;
        bd.Usage = D3D11_USAGE_DYNAMIC;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        D3D11_SAMPLER_DESC sd = {};
        sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        sd.MaxLOD = D3D11_FLOAT32_MAX;
        if (FAILED(dev->CreateComputeShader(blur_cs, sizeof(blur_cs), nullptr, g_blur_cs.GetAddressOf())) ||
            FAILED(dev->CreateBuffer(&bd, nullptr, g_blur_cb.GetAddressOf())) ||
            FAILED(dev->CreateSamplerState(&sd, g_sampler.GetAddressOf())))
        {
            mb_log("could not create the blur's compute shader, constant buffer or sampler");
            g_blur_unsupported = true;
            return false;
        }
    }
    if (g_blur_dst && g_blur_w == desc.Width && g_blur_h == desc.Height && g_blur_format == desc.Format)
        return true;

    g_blur_uav.Reset();
    g_blur_dst.Reset();
    if (desc.ArraySize != 1 || desc.MipLevels != 1 || desc.SampleDesc.Count != 1)
    {
        mb_log("the scene colour is an array, has mips or is multisampled, not supported");
        g_blur_unsupported = true;
        return false;
    }
    D3D11_FEATURE_DATA_FORMAT_SUPPORT2 fs2 = {};
    fs2.InFormat = view_format;
    if (FAILED(dev->CheckFeatureSupport(D3D11_FEATURE_FORMAT_SUPPORT2, &fs2, sizeof(fs2))) ||
        !(fs2.OutFormatSupport2 & D3D11_FORMAT_SUPPORT2_UAV_TYPED_STORE))
    {
        mb_log("format %u cannot be written by a compute shader on this GPU", unsigned(view_format));
        g_blur_unsupported = true;
        return false;
    }
    D3D11_TEXTURE2D_DESC ud = desc;
    ud.Usage = D3D11_USAGE_DEFAULT;
    ud.CPUAccessFlags = 0;
    ud.MiscFlags = 0;
    ud.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav = {};
    uav.Format = view_format;
    uav.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    if (FAILED(dev->CreateTexture2D(&ud, nullptr, g_blur_dst.GetAddressOf())) ||
        FAILED(dev->CreateUnorderedAccessView(g_blur_dst.Get(), &uav, g_blur_uav.GetAddressOf())))
    {
        mb_log("could not create the blur's working texture");
        g_blur_uav.Reset();
        g_blur_dst.Reset();
        g_blur_unsupported = true;
        return false;
    }
    g_blur_w = desc.Width;
    g_blur_h = desc.Height;
    g_blur_format = desc.Format;
    mb_log("blur working texture ready: %ux%u, format %u", g_blur_w, g_blur_h, unsigned(desc.Format));
    return true;
}

static DXGI_FORMAT typed_format(DXGI_FORMAT f)
{
    switch (f)
    {
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case DXGI_FORMAT_R11G11B10_FLOAT: return DXGI_FORMAT_R11G11B10_FLOAT;
    default: return f;
    }
}

// The draw after the flatten pass that reads what it wrote: the filter. Returns
// true when it was taken over and the game's draw is to be skipped.
static bool replace_filter(ID3D11DeviceContext *ctx)
{
    ComPtr<ID3D11RenderTargetView> rtv;
    ctx->OMGetRenderTargets(1, rtv.GetAddressOf(), nullptr);
    if (!rtv)
        return false;
    ComPtr<ID3D11Resource> rres;
    rtv->GetResource(rres.GetAddressOf());
    ComPtr<ID3D11Texture2D> tex;
    if (!rres || FAILED(rres.As(&tex)))
        return false;
    D3D11_TEXTURE2D_DESC td = {};
    tex->GetDesc(&td);
    if (td.Width < 960)
        return false; // the small tile passes read it too

    // What the draw reads: the flatten pass's output marks it as the filter.
    ID3D11ShaderResourceView *raw[16] = {};
    ctx->PSGetShaderResources(0, 16, raw);
    ComPtr<ID3D11ShaderResourceView> srvs[16];
    for (int i = 0; i < 16; ++i)
        srvs[i].Attach(raw[i]);
    bool reads_flat = false;
    int color_slot = -1;
    for (int i = 0; i < 16; ++i)
    {
        if (!srvs[i])
            continue;
        ComPtr<ID3D11Resource> r;
        srvs[i]->GetResource(r.GetAddressOf());
        if (r.Get() == g_flat_res.Get())
        {
            reads_flat = true;
            continue;
        }
        ComPtr<ID3D11Texture2D> t;
        if (color_slot < 0 && r && SUCCEEDED(r.As(&t)))
        {
            D3D11_TEXTURE2D_DESC d = {};
            t->GetDesc(&d);
            if (d.Width == td.Width && d.Height == td.Height && d.Format == td.Format && t.Get() != tex.Get())
                color_slot = i;
        }
    }
    if (!reads_flat)
        return false;

    // From here on this is the filter.
    if (g_stats.filters_replaced == 0 && g_stats.filters_given_back < 2)
    {
        mb_log("the filter draw: render target %ux%u format %u; reads:", td.Width, td.Height, unsigned(td.Format));
        for (int i = 0; i < 16; ++i)
        {
            if (!srvs[i])
                continue;
            ComPtr<ID3D11Resource> r;
            srvs[i]->GetResource(r.GetAddressOf());
            ComPtr<ID3D11Texture2D> t;
            D3D11_TEXTURE2D_DESC d = {};
            if (r && SUCCEEDED(r.As(&t)))
                t->GetDesc(&d);
            mb_log("  t%d: %ux%u format %u%s%s", i, d.Width, d.Height, unsigned(d.Format),
                   r.Get() == g_flat_res.Get() ? " (the flatten pass's output)" : "",
                   i == color_slot ? " (taken as the scene colour)" : "");
        }
    }
    if (color_slot < 0)
        return give_back("no scene colour found among what the filter reads");
    if (!g_vel_srv || !g_depth_srv || !g_zconv_copy)
        return give_back("the velocity, depth or distance constants were not kept");

    ComPtr<ID3D11Device> dev;
    ctx->GetDevice(dev.GetAddressOf());
    const DXGI_FORMAT view_format = typed_format(td.Format);
    if (!ensure_blur_resources(dev.Get(), td, view_format) || !g_blur_cs)
        return give_back("the blur's resources could not be made");

    const float shutter = g_settings.strength * g_settings.scale * TOP_SHUTTER;
    const float constants[8] = {shutter,
                                float(g_settings.limit.load()) / 100.0f,
                                float(SAMPLES),
                                float(g_frame % 64),
                                float(g_settings.near_m.load()) * 100.0f,
                                g_settings.debug ? 1.0f : 0.0f,
                                0.0f,
                                0.0f};
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    if (FAILED(ctx->Map(g_blur_cb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
        return give_back("could not update the blur's constant buffer");
    memcpy(mapped.pData, constants, sizeof(constants));
    ctx->Unmap(g_blur_cb.Get(), 0);

    ComPtr<ID3D11ComputeShader> old_cs;
    ctx->CSGetShader(old_cs.GetAddressOf(), nullptr, nullptr);
    ID3D11ShaderResourceView *old_srv[3] = {};
    ctx->CSGetShaderResources(0, 3, old_srv);
    ID3D11UnorderedAccessView *old_uav = nullptr;
    ctx->CSGetUnorderedAccessViews(0, 1, &old_uav);
    ID3D11Buffer *old_cb[2] = {};
    ctx->CSGetConstantBuffers(0, 2, old_cb);
    ID3D11SamplerState *old_smp = nullptr;
    ctx->CSGetSamplers(0, 1, &old_smp);

    ID3D11ShaderResourceView *in[3] = {srvs[color_slot].Get(), g_vel_srv.Get(), g_depth_srv.Get()};
    ID3D11UnorderedAccessView *uav = g_blur_uav.Get();
    ID3D11Buffer *cbs[2] = {g_blur_cb.Get(), g_zconv_copy.Get()};
    ID3D11SamplerState *smp = g_sampler.Get();
    const UINT keep = 0xFFFFFFFF;
    ctx->CSSetShader(g_blur_cs.Get(), nullptr, 0);
    ctx->CSSetShaderResources(0, 3, in);
    ctx->CSSetUnorderedAccessViews(0, 1, &uav, &keep);
    ctx->CSSetConstantBuffers(0, 2, cbs);
    ctx->CSSetSamplers(0, 1, &smp);
    ctx->Dispatch((td.Width + 7) / 8, (td.Height + 7) / 8, 1);

    ctx->CSSetShader(old_cs.Get(), nullptr, 0);
    ctx->CSSetShaderResources(0, 3, old_srv);
    ctx->CSSetUnorderedAccessViews(0, 1, &old_uav, &keep);
    ctx->CSSetConstantBuffers(0, 2, old_cb);
    ctx->CSSetSamplers(0, 1, &old_smp);
    for (ID3D11ShaderResourceView *s : old_srv)
        if (s)
            s->Release();
    if (old_uav)
        old_uav->Release();
    for (ID3D11Buffer *b : old_cb)
        if (b)
            b->Release();
    if (old_smp)
        old_smp->Release();

    // The result goes where the game's filter would have written it.
    ctx->CopyResource(tex.Get(), g_blur_dst.Get());
    g_filter_pending = false;
    if (++g_stats.filters_replaced == 1)
        mb_log("first blur ran: %ux%u, shutter %d%%, longest blur %d%%, keep sharp nearer than %d m", td.Width, td.Height,
               int(std::lround(shutter * 100.0f)), g_settings.limit.load(), g_settings.near_m.load());
    return true;
}

//----------------------------------------------------------------------------
// Events

static bool on_draw_any(command_list *cmd)
{
    if (!g_filter_pending || !g_settings.enabled || !g_settings.own_blur ||
        cmd->get_device()->get_api() != device_api::d3d11)
        return false;
    ID3D11DeviceContext *ctx = reinterpret_cast<ID3D11DeviceContext *>(cmd->get_native());
    if (!ctx)
        return false;
    std::lock_guard<std::mutex> lock(g_state_mutex);
    if (!g_filter_pending)
        return false;
    return replace_filter(ctx);
}

static bool on_draw(command_list *cmd, uint32_t, uint32_t, uint32_t, uint32_t)
{
    return on_draw_any(cmd);
}

static bool on_draw_indexed(command_list *cmd, uint32_t, uint32_t, uint32_t, int32_t, uint32_t)
{
    return on_draw_any(cmd);
}
''')

# present: stats and reset pending
sub('''    ++g_frame;
    if (g_frame == 300 || g_frame == 1800 || g_frame % 7200 == 0)
        mb_log("frame %u: %u flatten shaders known, flatten pass dispatched %u times, camera motion added %u, "
               "nothing done %u",
               g_frame, g_flatten_found, g_stats.flatten_dispatches, g_stats.injected, g_stats.skipped);''','''    ++g_frame;
    g_filter_pending = false; // nothing carries over from one frame to the next
    if (g_frame == 300 || g_frame == 1800 || g_frame % 7200 == 0)
        mb_log("frame %u: %u flatten shaders known, flatten pass dispatched %u times, camera motion added %u, "
               "nothing done %u; blur filter taken over %u times, left to the game %u",
               g_frame, g_flatten_found, g_stats.flatten_dispatches, g_stats.injected, g_stats.skipped,
               g_stats.filters_replaced, g_stats.filters_given_back);''')
sub('mb_log("settings: %s, camera blur %d%%, longest blur %d%%", g_settings.enabled ? "on" : "off",\n           int(std::lround(g_settings.strength * 100.0f)), g_settings.limit.load());','mb_log("settings: %s, own blur %s, motion blur %d%%, scale %d%%, longest blur %d%%, keep sharp nearer than %d m",\n           g_settings.enabled ? "on" : "off", g_settings.own_blur ? "on" : "off",\n           int(std::lround(g_settings.strength * 100.0f)), int(std::lround(g_settings.scale * 100.0f)),\n           g_settings.limit.load(), g_settings.near_m.load());')

# registrations
sub('''        reshade::register_event<reshade::addon_event::dispatch>(on_dispatch);''','''        reshade::register_event<reshade::addon_event::dispatch>(on_dispatch);
        reshade::register_event<reshade::addon_event::draw>(on_draw);
        reshade::register_event<reshade::addon_event::draw_indexed>(on_draw_indexed);''')
sub('mb_log("Camera Motion Blur (FF7 Remake) v1 loaded");','mb_log("Camera Motion Blur (FF7 Remake) v2 loaded");')

# overlay
a=s.index('    int blur = int(std::lround(g_settings.strength * 100.0f));')
b=s.index('    ImGui::TextDisabled("camera motion added %u times", g_stats.injected);')
s=s[:a]+'''    bool own = g_settings.own_blur;
    if (ImGui::Checkbox("Use this add-on's blur", &own))
    {
        g_settings.own_blur = own;
        reshade::set_config_value(nullptr, SECTION, "OwnBlur", own ? 1 : 0);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("On: this add-on does the blur, camera and objects, with the settings below. "
                          "Off: the game's own blur, with the camera's motion added to it. "
                          "The game's Motion Blur option has to stay on either way.");

    int blur = int(std::lround(g_settings.strength * 100.0f));
    if (ImGui::SliderInt("Motion blur", &blur, 0, 100, "%d%%"))
    {
        g_settings.strength = float(blur) / 100.0f;
        reshade::set_config_value(nullptr, SECTION, "BlurAmount", blur);
    }
    int scale = int(std::lround(g_settings.scale * 100.0f));
    if (ImGui::SliderInt("Overall scale", &scale, 10, 100, "%d%%"))
    {
        g_settings.scale = float(scale) / 100.0f;
        reshade::set_config_value(nullptr, SECTION, "BlurScale", scale);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Multiplies the Motion blur slider.");

    int limit = std::clamp(g_settings.limit.load(), 1, 3) - 1;
    if (ImGui::Combo("Longest blur", &limit, "Low\\0Medium\\0High\\0"))
    {
        g_settings.limit = limit + 1;
        reshade::set_config_value(nullptr, SECTION, "BlurLength", limit + 1);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("The longest streak a fast camera turn can make, as a share of the screen height. "
                          "Higher settings smear more.");

    int near_m = g_settings.near_m;
    if (ImGui::SliderInt("Keep near things sharp", &near_m, 0, 10, "%d m"))
    {
        g_settings.near_m = near_m;
        reshade::set_config_value(nullptr, SECTION, "KeepNearSharp", near_m);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Things nearer than this many metres stay sharp, and do not lend their colour to the blur "
                          "of what is behind them. This is what keeps your character sharp when the camera turns. "
                          "0 blurs everything.");

    bool debug = g_settings.debug;
    if (ImGui::Checkbox("Show motion (debug)", &debug))
    {
        g_settings.debug = debug;
        reshade::set_config_value(nullptr, SECTION, "Debug", debug ? 1 : 0);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Replaces the image with the velocity: red where things move sideways, green where they "
                          "move up or down.");

'''+s[b:]
sub('ImGui::TextDisabled("camera motion added %u times", g_stats.injected);','ImGui::TextDisabled("camera motion added %u times, blur taken over %u times", g_stats.injected,\n                        g_stats.filters_replaced);')
open(p,'w').write(s)
import re
print('braces',s.count('{'),s.count('}'),'parens',s.count('('),s.count(')'))
used=set(re.findall(r'\bg_[a-z_0-9]+\b',s))
declared=set()
for m in re.finditer(r'^static [^\n;(]*;',s,re.M):
    for n in re.findall(r'\bg_[a-z_0-9]+\b',m.group(0)): declared.add(n)
print('undeclared:',sorted(u for u in used if u not in declared))
EOF
grep -n "g_sampler\|release_resources()" src/ff7blur.cpp | head -20
Sortie

braces 102 102 parens 523 523
undeclared: []
222:static ComPtr<ID3D11SamplerState> g_sampler;
245:static void release_resources()
253:    g_sampler.Reset();
279:            release_resources();
530:            FAILED(dev->CreateSamplerState(&sd, g_sampler.GetAddressOf())))
700:    ID3D11SamplerState *smp = g_sampler.Get();
793:        release_resources();
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

    // Vector 57 of the View constants turns a depth into a distance; the blur
    // needs it later, when the View constants may not be bound.
    const UINT zstart = first * 16 + 57 * 16;
    if (g_zconv_copy && zstart + 16 <= vbd.ByteWidth)
    {
        D3D11_BOX zbox = {zstart, 0, 0, zstart + 16, 1, 1};
        ctx->CopySubresourceRegion(g_zconv_copy.Get(), 0, 0, 0, 0, view.Get(), 0, &zbox);
    }

    // When this add-on's own blur follows, the camera's true motion goes in and
    // the blur limits it. When the game's own blur runs, the settings apply here.
    const bool own = g_settings.own_blur;
    const float constants[4] = {own ? 1.0f : g_settings.strength * g_settings.scale * 3.0f,
                                own ? 0.15f : float(g_settings.limit.load()) / 100.0f, 0.0f, 0.0f};
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
    if (old_uav[0])
    {
        ComPtr<ID3D11Resource> flat;
        old_uav[0]->GetResource(flat.GetAddressOf());
        g_flat_res = flat;
    }
    for (ID3D11UnorderedAccessView *u : old_uav)
        if (u)
            u->Release();
    for (ID3D11Buffer *b : old_cb)
        if (b)
            b->Release();

    // The velocity buffer, with the camera's motion in it, goes back; the game's
    // own dispatch, next, reads it.
    ctx->CopyResource(vel_tex.Get(), g_scratch.Get());

    // What the filter draw that follows will need.
    g_vel_srv = vel_srv;
    g_depth_srv = depth_srv;
    g_filter_pending = own;

    if (++g_stats.injected == 1)
        mb_log("first injection: velocity %ux%u, format %u, camera blur %d%%", td.Width, td.Height, unsigned(td.Format),
               int(std::lround(g_settings.strength * 100.0f)));
}

//----------------------------------------------------------------------------