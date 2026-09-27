//! Post-process pass: exposure, tone map, sRGB encode.

use synaga_shader::*;

pub const LUMA: vec3 = vec3(0.2126, 0.7152, 0.0722);

pub struct PostParams {
    pub exposure: f32,
    pub needs_srgb: u32,
}

pub static post_params: Uniform<PostParams> = binding();
pub static hdr: texture_2d<f32> = binding();
pub static ldr: texture_storage_2d<Rgba8Unorm, Write> = binding();

pub fn encode_srgb(linear: vec3) -> vec3 {
    let low = 12.92 * linear;
    let high = 1.055 * pow(max(linear, vec3::ZERO), vec3::splat(1.0 / 2.4)) - 0.055;
    select(high, low, linear.cmple(vec3::splat(0.0031308)))
}

#[entry_point(compute, threads(8, 8))]
pub fn tonemap(#[builtin(global_invocation_id)] gid: vec3u) {
    let size = hdr.dimensions();
    if gid.x >= size.x || gid.y >= size.y {
        return;
    }
    let coord = vec2i::from(gid.xy());
    let raw = hdr.load(coord, 0);
    let mapped = raw.xyz() * post_params.exposure / (dot(raw.xyz(), LUMA) + 1.0);
    let encoded = select(mapped, encode_srgb(mapped), post_params.needs_srgb != 0);
    ldr.store(coord, encoded.extend(raw.w));
}
