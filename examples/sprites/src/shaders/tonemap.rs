//! Post-process pass: exposure, tone map, sRGB encode.

use synaga_shader::*;

pub const LUMA: Vec3 = vec3(0.2126, 0.7152, 0.0722);

pub struct PostParams {
    pub exposure: f32,
    pub needs_srgb: u32,
}

pub static post_params: Uniform<PostParams> = group(0).binding(0);
pub static hdr: Texture2D<f32> = group(0).binding(1);
pub static ldr: TextureStorage2D<Rgba8Unorm, Write> = group(0).binding(2);

pub fn encode_srgb(linear: Vec3) -> Vec3 {
    let low = 12.92 * linear;
    let high = 1.055 * pow(max(linear, Vec3::ZERO), Vec3::splat(1.0 / 2.4)) - 0.055;
    select(high, low, linear.cmple(Vec3::splat(0.0031308)))
}

/// The side of the square of pixels a workgroup covers, which the host divides
/// the image by to dispatch.
pub const TILE: u32 = 8;

#[entry_point(compute, threads(TILE, TILE))]
pub fn tonemap(#[builtin(global_invocation_id)] gid: Vec3<u32>) {
    let size = hdr.dimensions();
    if gid.x >= size.x || gid.y >= size.y {
        return;
    }
    let coord = Vec2::<i32>::from(gid.xy());
    let raw = hdr.load(coord, 0);
    let mapped = raw.xyz() * post_params.exposure / (dot(raw.xyz(), LUMA) + 1.0);
    let encoded = select(mapped, encode_srgb(mapped), post_params.needs_srgb != 0);
    ldr.store(coord, encoded.extend(raw.w));
}
