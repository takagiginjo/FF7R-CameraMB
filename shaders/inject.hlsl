// Camera motion for Final Fantasy VII Remake (Unreal Engine 4, Direct3D 11).
//
// The game has Unreal's motion blur, and it blurs objects but not the camera's
// own motion. Unreal's velocity buffer (R16G16_UNORM) only holds the velocity
// of things that move by themselves, drawn by their own shaders; everything
// else is left at 0, and the shaders that read it work out the camera's motion
// from the depth for those pixels. The velocity flatten pass that starts the
// motion blur (a compute shader) uses a camera motion that comes out as nothing
// in this game.
//
// This runs just before that pass and rewrites the velocity buffer:
//
//  - Where nothing wrote a velocity, it writes the camera's motion, worked out
//    the way Unreal's temporal AA does it: the pixel's clip position and depth
//    taken through ClipToPrevClip (the View constants' vectors 114 to 117, only
//    x, y and w used), and the difference of the two positions.
//  - Where something did (objects, the player's character), the velocity is
//    kept, scaled by `object_scale`.
//  - Things near the camera are kept sharp, as in the Assassin's Creed add-ons:
//    every velocity is faded to 0 below `near_start` and back to full over
//    `near_fade`. The distance comes from the depth, assuming Unreal's reversed
//    infinite-far projection: distance = near plane / depth.
//  - No velocity is longer than `max_length` of the screen height.
//
// The velocity is encoded as   enc = v * 0.2495 + 0.499992   with v in clip
// space (2 across the screen, y up): Unreal's shaders decode it with
// (enc - 0.499992) * 4.008016. b1 holds the View constants' vectors 114 to 118.

cbuffer constants : register(b0)
{
    float strength;      // camera motion, 1 = as Unreal works it out
    float max_length;    // longest blur, as a fraction of the screen height
    float object_scale;  // objects' own velocity, 1 = untouched
    float near_plane;    // distance of the near plane, in the game's units (cm)
    float near_start;    // below this distance things are kept sharp (cm)
    float near_fade;     // how far it takes to go from sharp to fully blurred (cm)
    float debug;         // 1 = blur everything sideways except the sharp zone
    float padding0;
};

cbuffer view_params : register(b1)
{
    float4 clip_to_prev[4];
    float4 jitter;
};

Texture2D<float4> vel_tex : register(t0);
Texture2D<float4> depth_tex : register(t1);
RWTexture2D<unorm float2> vel_out : register(u0);

float smooth01(float x)
{
    x = saturate(x);
    return x * x * (3.0 - 2.0 * x);
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    vel_out.GetDimensions(width, height);
    if (id.x >= width || id.y >= height)
        return;

    float2 size = float2(width, height);
    float2 uv = (float2(id.xy) + 0.5) / size;
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    uint dw, dh;
    depth_tex.GetDimensions(dw, dh);
    int2 dp = int2(min(uint2(uv * float2(dw, dh)), uint2(dw - 1, dh - 1)));
    float d = depth_tex.Load(int3(dp, 0)).x;

    // 0 where things are near and kept sharp, 1 where they blur in full.
    float dist = near_plane / max(d, 1.0e-7);
    float sharp = smooth01((dist - near_start) / max(near_fade, 1.0));

    float2 enc = vel_tex.Load(int3(id.xy, 0)).xy;
    bool written = enc.x != 0.0 || enc.y != 0.0;

    float2 v;
    if (written)
    {
        // Something wrote a velocity here: it keeps it, with its own scale.
        v = (enc - 0.499992) * 4.008016 * object_scale;
    }
    else
    {
        // Where this pixel was last frame, in clip space.
        float3 prev = ndc.x * clip_to_prev[0].xyw + ndc.y * clip_to_prev[1].xyw + d * clip_to_prev[2].xyw +
                      clip_to_prev[3].xyw;
        v = float2(0.0, 0.0);
        if (abs(prev.z) > 1.0e-6)
            v = ndc - prev.xy / prev.z;
        v *= strength;
    }
    v *= sharp;

    // Never longer than max_length of the screen height.
    float aspect = size.x / size.y;
    float len = length(v * 0.5 * float2(aspect, 1.0));
    bool clamped = len > max_length;
    if (clamped)
        v *= max_length / len;

    if (debug > 0.5)
    {
        // Sideways blur of the longest length everywhere, except in the sharp
        // zone: shows where things are kept sharp.
        v = float2(2.0 * max_length / aspect, 0.0) * sharp;
        clamped = true;
    }

    // Untouched objects keep their exact bits.
    if (written && !clamped && object_scale == 1.0 && sharp >= 0.9999)
    {
        vel_out[id.xy] = enc;
        return;
    }

    vel_out[id.xy] = saturate(v * 0.2495 + 0.499992);
}
