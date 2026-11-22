// Velocity flatten for Final Fantasy VII Remake (Unreal Engine 4, Direct3D 11).
//
// This takes the place of the game's own velocity flatten compute shader, the
// first pass of its motion blur (shader hash AFC9976E, 16x16 threads). It is
// the game's shader written back as HLSL from its disassembly, line for line,
// with these changes:
//
//  - The camera's motion is not the game's (its camera constants come out as
//    nothing here), but worked out from the depth and ClipToPrevClip, taken from
//    the View constants, the way the game's temporal AA does it. That fixes
//    two things at once: static pixels get a camera blur, and objects are
//    measured against the camera's real motion, so a character that stands
//    still while the camera turns has no blur.
//  - Camera motion, object motion, near things kept sharp and the longest blur
//    are settings, as in the Assassin's Creed add-ons.
//
// What the pass writes is unchanged, so the game's blur passes after it read it
// as they always do:
//  u0, per pixel: x = the length of the blur, y = the distance from the camera.
//  u1, per 16x16 tile: xy = the shortest blur in the tile, as a vector, zw = the
//      longest.
// The game's constant buffers stay bound where they were:
//  b0 = the blur's constants, b1 = the View constants (a window of them), b2 = the
//  game's camera constants (unused here). b3 and b4 are ours.

cbuffer game_blur : register(b0)
{
    float4 cb0[20];
};

cbuffer game_view : register(b1)
{
    float4 cb1[58];
};

cbuffer game_camera : register(b2)
{
    float4 cb2[5];
};

cbuffer settings : register(b3)
{
    float strength;      // the camera's motion, 1 = as it is
    float max_length;    // longest blur, as a fraction of the screen height
    float object_scale;  // the objects' motion, 1 = as the game has it
    float near_start;    // below this distance, things are kept sharp (cm)
    float near_fade;     // how far it takes from sharp to fully blurred (cm)
    float debug;         // 1 = blur sideways as long as allowed, except what is kept sharp
    float padding0;
    float padding1;
};

cbuffer view_extra : register(b4)
{
    float4 clip_to_prev[4]; // the View constants' vectors 114 to 117
    float4 jitter;          // vector 118, not used
};

Texture2D<float4> vel_tex : register(t0);   // R16G16_UNORM, enc = v * 0.2495 + 0.499992
Texture2D<float4> depth_tex : register(t1);
RWTexture2D<float4> out_pixel : register(u0);
RWTexture2D<float4> out_tile : register(u1);

groupshared float4 tile[256];

float smooth01(float x)
{
    x = saturate(x);
    return x * x * (3.0 - 2.0 * x);
}

// The pair of blurs of two threads, as the game keeps them: xy the shorter,
// zw the longer, each a length and an angle.
float4 merge(float4 a, float4 b)
{
    float4 r;
    r.xy = (a.x < b.x) ? a.xy : b.xy;
    r.zw = (b.z < a.z) ? a.zw : b.zw;
    return r;
}

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    int4 rect = asint(cb0[15]);                 // the view's rectangle: x0, y0, x1, y1
    int2 p = int2(tid.xy) + rect.xy;
    float2 size = float2(rect.zw - rect.xy);
    bool inside = all(uint2(p) < uint2(rect.zw));

    float2 raw = vel_tex.Load(int3(p, 0)).xy;
    float d = depth_tex.Load(int3(p, 0)).x;
    float2 uv = (float2(tid.xy) + 0.5) / size;

    // Distance from the camera, as the game works it out (UE ConvertFromDeviceZ).
    float dist = d * cb1[57].x + cb1[57].y + rcp(d * cb1[57].z - cb1[57].w);
    if (d <= 1.0e-7)
        dist = 1.0e9; // the sky
    float sharp = smooth01((dist - near_start) / max(near_fade, 1.0));

    // The camera's motion in clip space (2 across the screen, y up), current
    // position minus previous, from the depth and ClipToPrevClip.
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float3 prev = ndc.x * clip_to_prev[0].xyw + ndc.y * clip_to_prev[1].xyw + d * clip_to_prev[2].xyw +
                  clip_to_prev[3].xyw;
    float2 cam = float2(0.0, 0.0);
    if (abs(prev.z) > 1.0e-6)
        cam = ndc - prev.xy / prev.z;

    // Something wrote a velocity: an object. The rest is only the camera.
    bool written = dot(raw, raw) > 0.0;
    float2 obj = (raw - 0.499992) * 4.008016;

    // A is the motion the blur follows, B the motion of the object against
    // the camera. The game measures both with y flipped and brought to the
    // width's units by cb0[18].x.
    float2 A = written ? obj : cam;
    float2 B = written ? obj - cam : float2(0.0, 0.0);
    float2 Ap = float2(A.x, A.y * -cb0[18].x);
    float2 Bp = float2(B.x, B.y * -cb0[18].x);

    float len1 = length(Ap);
    float len2 = length(Bp);
    float len;
    if (written)
        len = (len1 + cb0[19].w * (min(len2, len1) - len1)) * object_scale; // the game's own mix
    else
        len = len1 * strength; // the camera's motion, in full
    len *= sharp;

    // The longest blur, as a share of the screen height, in the game's units.
    float aspect = size.x / size.y;
    float2 Af = 0.5 * float2(A.x * aspect, A.y);
    float units_per_height = len1 / max(length(Af), 1.0e-9);
    float angle = atan2(Ap.y, Ap.x);
    if (debug > 0.5)
    {
        // The longest blur sideways everywhere, except where things are kept sharp.
        units_per_height = 2.0 * abs(cb0[18].x);
        len = max_length * units_per_height * sharp;
        angle = 0.0;
    }
    len = min(len, max_length * units_per_height);

    if (inside)
        out_pixel[p] = float4(len, dist, 0.0, 0.0);

    // The shortest and the longest blur of the 16x16 tile, found by halving.
    tile[gi] = float4(inside ? len : 2.0, angle, inside ? len : 0.0, angle);
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint step = 0; step < 8; ++step)
    {
        const uint stride = 128u >> step; // 128, 64, ... 1
        if (gi < stride)
            tile[gi] = merge(tile[gi], tile[gi + stride]);
        GroupMemoryBarrierWithGroupSync();
    }

    if (gi == 0)
    {
        float4 t = tile[0];
        float2 shortest = t.x * float2(cos(t.y), sin(t.y));
        float2 longest = t.z * float2(cos(t.w), sin(t.w));
        out_tile[gid.xy] = float4(shortest, longest);
    }
}
