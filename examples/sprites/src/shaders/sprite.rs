//! Instanced sprite drawing.

use synaga_shader::*;

use super::common::{globals, unpack_color};

/// One sprite, as the host moves it and uploads it; only the host reads
/// `velocity`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Locals {
    pub position: Vec2,
    pub velocity: Vec2,
    pub color: u32,
    pub _pad: u32,
}

/// The bind group each batch of sprites sets.
pub const BATCH: u32 = 1;

pub static locals: Uniform<Locals> = group(BATCH).binding(0);
pub static sprite_texture: Texture2D<f32> = group(BATCH).binding(1);
pub static sprite_sampler: Sampler = group(BATCH).binding(2);

#[derive(Io)]
pub struct Vertex {
    #[location(0)]
    pub pos: Vec2,
}

#[derive(Io)]
pub struct VertexOutput {
    #[builtin(position)]
    pub position: Vec4,
    #[location(0)]
    pub tex_coords: Vec2,
    #[location(1)]
    pub color: Vec4,
}

#[entry_point(vertex)]
pub fn vs_main(vertex: Vertex) -> VertexOutput {
    let tc = vertex.pos;
    let offset = tc * globals.sprite_size;
    VertexOutput {
        position: globals.mvp_transform * (locals.position + offset).extend(0.0).extend(1.0),
        tex_coords: tc,
        color: unpack_color(locals.color),
    }
}

#[entry_point(fragment)]
pub fn fs_main(vertex: VertexOutput) -> Vec4 {
    vertex.color * sprite_texture.sample_level(&sprite_sampler, vertex.tex_coords, 0.0)
}
