//! Shared by every shader module here, in place of a WGSL `#include`.

use synaga_shader::*;

pub struct Globals {
    pub mvp_transform: mat4,
    pub sprite_size: vec2,
}

/// The bind group every sprite shader shares, set once a frame. The host can
/// name it by this constant too.
pub const FRAME: u32 = 0;

pub static globals: Uniform<Globals> = group(FRAME).binding(0);

pub fn unpack_color(raw: u32) -> vec4 {
    let bytes = (vec4u::splat(raw) >> vec4u(0, 8, 16, 24)) & vec4u::splat(0xFF);
    vec4::from(bytes) / 255.0
}
