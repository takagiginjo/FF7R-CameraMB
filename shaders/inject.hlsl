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
// This runs just before that pass and writes the camera's motion into the
// velocity buffer where it is 0. The game's own blur then blurs it, with its own
// settings, together with the objects' motion. Objects, the player's character
// among them, have written a velocity of their own, and keep it.
//
// The velocity is encoded as   enc = v * 0.2495 + 0.499992   with v in clip
// space (2 across the screen, y up): Unreal's shaders decode it with
// (enc - 0.499992) * 4.008016. The camera's velocity is worked out the way
// Unreal's temporal AA does it for pixels without one: the pixel's clip
// position and depth, taken through ClipToPrevClip, the View constants'
// vectors 114 to 117 (only x, y and w are used), and the difference of the two
// positions. b1 holds the View constants' vectors 114 to 118.

cbuffer constants : register(b0)
{
    float strength;    // 1 = the camera's motion as Unreal works it out
    float max_length;  // longest blur, as a fraction of the screen height
    float padding0;
    float padding1;
};

cbuffer view_params : register(b1)
{
    float4 clip_to_prev[4];
    float4 jitter;
};

Texture2D<float4> vel_tex : register(t0);
Texture2D<float4> depth_tex : register(t1);
RWTexture2D<unorm float2> vel_out : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    vel_out.GetDimensions(width, height);
    if (id.x >= width || id.y >= height)
        return;

    float2 enc = vel_tex.Load(int3(id.xy, 0)).xy;

    // Something wrote a velocity here: it keeps it.
    if (enc.x != 0.0 || enc.y != 0.0)
    {
        vel_out[id.xy] = enc;
        return;
    }

    float2 size = float2(width, height);
    float2 uv = (float2(id.xy) + 0.5) / size;
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    uint dw, dh;
    depth_tex.GetDimensions(dw, dh);
    int2 dp = int2(min(uint2(uv * float2(dw, dh)), uint2(dw - 1, dh - 1)));
    float d = depth_tex.Load(int3(dp, 0)).x;

    // Where this pixel was last frame, in clip space.
    float3 prev = ndc.x * clip_to_prev[0].xyw + ndc.y * clip_to_prev[1].xyw + d * clip_to_prev[2].xyw +
                  clip_to_prev[3].xyw;
    float2 v = float2(0.0, 0.0);
    if (abs(prev.z) > 1.0e-6)
        v = ndc - prev.xy / prev.z;
    v *= strength;

    // Never longer than max_length of the screen height.
    float aspect = size.x / size.y;
    float len = length(v * 0.5 * float2(aspect, 1.0));
    if (len > max_length)
        v *= max_length / len;

    vel_out[id.xy] = saturate(v * 0.2495 + 0.499992);
}
