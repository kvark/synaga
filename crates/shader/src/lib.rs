//! Rust types for shaders written in the [synaga] dialect, so that a shader
//! module is an ordinary Rust module: `rustc` checks it, `cargo fmt` formats
//! it, and rust-analyzer understands it.
//!
//! ```ignore
//! use synaga_shader::*;
//!
//! #[derive(Clone, Copy, Io)]
//! struct VsOut {
//!     #[builtin(position)] clip: Vec4,
//!     #[location(0)] uv: Vec2,
//! }
//!
//! static camera: Uniform<Mat4> = group(0).binding(0);
//!
//! #[entry_point(vertex)]
//! fn vs(#[location(0)] pos: Vec3, #[location(1)] uv: Vec2) -> VsOut {
//!     VsOut { clip: *camera * pos.extend(1.0), uv }
//! }
//! ```
//!
//! # What runs on the CPU
//!
//! The math: the vectors, the matrices and the builtins, so a shader's pure
//! helpers can be called from Rust and tested there. Each means on the CPU
//! what it means in the shader. A method or operator named as Rust names it
//! is Rust's, so `v.fract()` is `v - v.trunc()` and `+` on a `Vec3<u32>`
//! panics on overflow under overflow checks, as `+` on a `u32` does. A free
//! function keeps WGSL's meaning, so `fract(v)` is `v - floor(v)`. The atomics
//! are real atomics, and the [`ir`] module decodes what the build step wrote.
//!
//! What needs the GPU panics: resources, textures and samplers, ray queries,
//! barriers and `discard()`. Running those needs a runtime that runs the
//! shader's invocations, with their memory, which this crate does not have.
//! The types are there to be *checked*.
//!
//! # Sharing a struct with the host
//!
//! A struct that says `#[repr(C)]` is one the host shares: it fills one in and
//! uploads its bytes. So the host uses the shader's definition rather than a
//! copy that has to be kept the same by hand. The build step checks that the
//! GPU reads every field where Rust puts it, which is the one thing that can
//! go silently wrong, and says how much padding lines up a field that is not.
//!
//! ```ignore
//! #[repr(C)]
//! #[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
//! pub struct Globals {
//!     pub view_proj: Mat4,
//!     pub sprite_size: Vec2,
//!     pub _pad: Vec2,
//! }
//! ```
//!
//! With the `bytemuck` feature the vectors and matrices are `Pod`, so a shared
//! struct can derive it. They convert from arrays, a matrix column by column,
//! and with the `mint` feature from mint's types, which most math crates
//! convert to. `vec3(x, y, z)` and the other constructors work on the CPU too.
//!
//! # Where this differs from WGSL
//!
//! What Rust cannot express the way WGSL does:
//!
//! - **Names follow Rust.** Types are WGSL's words, capitalized: `vec3<i32>` is
//!   [`Vec3<i32>`], `mat4x4f` is [`Mat4`], `texture_2d<f32>` is
//!   [`Texture2D<f32>`]. A bare `Vec3` is `Vec3<f32>`, and `vec3u(1, 2, 3)` is
//!   `vec3::<u32>(1, 2, 3)`. Builtins are snake_case: `workgroup_barrier()`.
//! - **Swizzles are methods.** `v.x` is a field, but `v.xyz` would need a
//!   hundred overlapping names for one piece of memory, so it is `v.xyz()`.
//!   `.r`/`.g`/`.b`/`.a` are methods for the same reason.
//! - **Constructors are fixed-arity.** `vec3(x, y, z)` is a function, so the
//!   other WGSL forms get their own names: [`Vec3::splat`], [`Vec2::extend`],
//!   and `From` for joining two vectors.
//! - **Vector comparisons are methods.** `a < b` yields one `bool` in Rust and
//!   one per lane in a shader, so the lane-wise forms are `cmplt`, `cmple` and
//!   the rest, as glam spells them.
//!
//! And what it expresses better: texture and ray-query operations are methods
//! on the types they apply to, math is methods named as `f32`'s are, and
//! atomics are the standard ones without an `Ordering`.
//!
//! [synaga]: https://github.com/kvark/synaga

// The bodies that need the GPU never run on the CPU, so their parameters are
// unused by construction.
#![allow(unused_variables)]
#![allow(clippy::too_many_arguments, clippy::needless_lifetimes)]

pub mod atomic;
pub mod builtins;
pub mod ir;
pub mod matrix;
pub mod resource;
pub mod texture;
pub mod vector;

pub use atomic::*;
pub use builtins::*;
pub use matrix::*;
pub use resource::*;
pub use synaga_macros::{entry_point, Io};
pub use texture::*;
pub use vector::*;

/// A struct of bound shader inputs or outputs. `#[derive(Io)]` implements it.
pub trait Io {
    /// Touches every field. Nothing calls it: its body is how `rustc` learns
    /// that the fields are read, by a stage or the rasterizer it cannot see.
    #[doc(hidden)]
    fn read_every_field(&self);
}

/// The body of what only the GPU runs: resources, textures, ray queries and
/// barriers.
///
/// These types are here so `rustc` can check the source that describes the
/// shader. Reaching one of these at runtime means something called a shader
/// function that uses one on the CPU.
#[inline]
#[track_caller]
pub fn unimplemented_on_cpu<T>() -> T {
    panic!(
        "this needs the GPU: synaga-shader runs a shader's math on the CPU, \
         but not its resources, textures, ray queries or barriers"
    )
}

/// The ray flags and intersection kinds WGSL predeclares.
pub mod ray {
    pub const RAY_FLAG_NONE: u32 = 0;
    pub const RAY_FLAG_FORCE_OPAQUE: u32 = 1;
    pub const RAY_FLAG_FORCE_NO_OPAQUE: u32 = 2;
    pub const RAY_FLAG_TERMINATE_ON_FIRST_HIT: u32 = 4;
    pub const RAY_FLAG_SKIP_CLOSEST_HIT_SHADER: u32 = 8;
    pub const RAY_FLAG_CULL_BACK_FACING: u32 = 16;
    pub const RAY_FLAG_CULL_FRONT_FACING: u32 = 32;
    pub const RAY_FLAG_CULL_OPAQUE: u32 = 64;
    pub const RAY_FLAG_CULL_NO_OPAQUE: u32 = 128;
    pub const RAY_FLAG_SKIP_TRIANGLES: u32 = 256;
    pub const RAY_FLAG_SKIP_AABBS: u32 = 512;

    pub const RAY_QUERY_INTERSECTION_NONE: u32 = 0;
    pub const RAY_QUERY_INTERSECTION_TRIANGLE: u32 = 1;
    pub const RAY_QUERY_INTERSECTION_GENERATED: u32 = 2;
    pub const RAY_QUERY_INTERSECTION_AABB: u32 = 3;
}

pub use ray::*;
