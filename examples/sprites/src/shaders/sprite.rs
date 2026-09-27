//! Instanced sprite drawing.

use synaga_shader::*;

use super::common::{globals, unpack_color};

pub struct Locals {
    pub position: vec2,
    // The host uploads the whole sprite, velocity included; only it reads this.
    #[allow(dead_code)]
    pub velocity: vec2,
    pub color: u32,
}

pub static locals: Uniform<Locals> = binding();
pub static sprite_texture: texture_2d<f32> = binding();
pub static sprite_sampler: sampler = binding();

pub struct Vertex {
    pub pos: vec2,
}

#[derive(Io)]
pub struct VertexOutput {
    #[builtin(position)]
    pub position: vec4,
    #[location(0)]
    pub tex_coords: vec2,
    #[location(1)]
    pub color: vec4,
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
#[output(location(0))]
pub fn fs_main(vertex: VertexOutput) -> vec4 {
    vertex.color * textureSampleLevel(&sprite_texture, &sprite_sampler, vertex.tex_coords, 0.0)
}
