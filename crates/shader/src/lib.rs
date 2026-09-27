//! Rust types for shaders written in the [synaga] dialect, so that a shader
//! module is an ordinary Rust module: `rustc` checks it, `cargo fmt` formats
//! it, and rust-analyzer understands it.
//!
//! ```ignore
//! use synaga_shader::*;
//!
//! #[derive(Clone, Copy, Io)]
//! struct VsOut {
//!     #[builtin(position)] clip: vec4,
//!     #[location(0)] uv: vec2,
//! }
//!
//! static camera: Uniform<mat4> = binding();
//!
//! #[entry_point(vertex)]
//! fn vs(#[location(0)] pos: vec3, #[location(1)] uv: vec2) -> VsOut {
//!     VsOut { clip: *camera * pos.extend(1.0), uv }
//! }
//! ```
//!
//! # What runs on the CPU
//!
//! Very little, yet. The atomics are real atomics; everything else panics.
//! The types are here to be *checked*, and the shader runs on a GPU.
//! Implementations can be filled in later without a signature changing.
//!
//! # Where this differs from WGSL
//!
//! What Rust cannot express the way WGSL does:
//!
//! - **Swizzles are methods.** `v.x` is a field, but `v.xyz` would need a
//!   hundred overlapping names for one piece of memory, so it is `v.xyz()`.
//!   `.r`/`.g`/`.b`/`.a` are methods for the same reason.
//! - **Constructors are fixed-arity.** `vec3(x, y, z)` is a function, so the
//!   other WGSL forms get their own names: [`vec3::splat`], [`vec2::extend`],
//!   and `From` for joining two vectors.
//! - **Vector comparisons are methods.** `a < b` yields one `bool` in Rust and
//!   one per lane in a shader, so the lane-wise forms are `cmplt`, `cmple` and
//!   the rest, as glam spells them.
//!
//! And what it expresses better: texture and ray-query operations are methods
//! on the types they apply to, atomics are the standard ones without an
//! `Ordering`, and a write to shared memory says `unsafe`.
//!
//! [synaga]: https://github.com/kvark/synaga

// Bodies never run, so every parameter is unused by construction.
#![allow(unused_variables)]
// WGSL names its builtins in camelCase; keeping the spelling is the point.
#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments, clippy::needless_lifetimes)]

pub mod atomic;
pub mod builtins;
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

/// The body of everything in this crate.
///
/// A shader runs on a GPU; these types are here so `rustc` can check the
/// source that describes it. Reaching one of these at runtime means something
/// called a shader function on the CPU.
#[inline]
#[track_caller]
pub fn unimplemented_on_cpu<T>() -> T {
    panic!("shader functions describe GPU work and cannot run on the CPU")
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
