//! Shared by every shader module here, in place of a WGSL `#include`.

use synaga_shader::*;

/// Filled in and uploaded by the host, which `#[repr(C)]` says. The build
/// checks that the GPU reads each field where Rust puts it: without `_pad`,
/// the GPU's struct would be 8 bytes longer than Rust's.
#[repr(C)]
#[derive(Shared)]
pub struct Globals {
    pub mvp_transform: Mat4,
    pub sprite_size: Vec2,
    pub _pad: Vec2,
}

/// The bind group every sprite shader shares, set once a frame. The host can
/// name it by this constant too.
pub const FRAME: u32 = 0;

pub static globals: Uniform<Globals> = group(FRAME).binding(0);

pub fn unpack_color(raw: u32) -> Vec4 {
    let bytes = (Vec4::splat(raw) >> vec4(0, 8, 16, 24)) & Vec4::splat(0xFF);
    Vec4::from(bytes) / 255.0
}
