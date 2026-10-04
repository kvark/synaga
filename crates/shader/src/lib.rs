//! Rust types for shaders written in the [synaga] dialect, so that a shader
//! module is an ordinary Rust module: `rustc` checks it, `cargo fmt` formats
//! it, and rust-analyzer understands it.
//!
//! ```ignore
//! use synaga_shader::*;
//!
//! #[derive(Io)]
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
//! The [`subgroup`] operations run as a subgroup of one, which the invocation
//! calling them is.
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
//! #[derive(Shared)]
//! pub struct Globals {
//!     pub view_proj: Mat4,
//!     pub sprite_size: Vec2,
//!     pub _pad: Vec2,
//! }
//! ```
//!
//! [`Shared`](derive@Shared) derives what uploading one takes: `Clone`,
//! `Copy`, a `Default` of all zeroes, as the GPU's default is, and
//! `bytemuck`'s `Zeroable` and `NoUninit`. The `bytemuck` feature makes the
//! vectors and matrices `Pod`, which is both. They convert from arrays, a
//! matrix column by column, and with the `mint` feature from mint's types,
//! which most math crates convert to. `vec3(x, y, z)` and the other
//! constructors work on the CPU too.
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
//! atomics, [`Atomic<u32>`] and [`Atomic<i32>`], have the standard methods
//! without an `Ordering`.
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
pub mod subgroup;
pub mod texture;
pub mod vector;

pub use atomic::*;
pub use builtins::*;
pub use matrix::*;
pub use resource::*;
pub use subgroup::*;
pub use synaga_macros::{entry_point, Io, Shared};
pub use texture::*;
pub use vector::*;

/// A struct the host shares with a shader, which `#[derive(Shared)]`
/// implements, with `Clone`, `Copy`, a zeroed `Default`, and `bytemuck`'s
/// `Zeroable` and `NoUninit`.
///
/// ```
/// use synaga_shader::*;
/// #[repr(C)]
/// #[derive(Shared)]
/// struct Params {
///     tint: Vec4,
///     size: Vec2<u32>,
///     _pad: Vec2<u32>,
/// }
/// assert_eq!(Params::default().tint, Vec4::ZERO);
/// ```
///
/// A struct with padding in Rust cannot be, since padding is not data. The
/// shader's own types leave none, and the GPU's gaps are the build's layout
/// check to find, but a `u8` can:
///
/// ```compile_fail
/// use synaga_shader::*;
/// #[repr(C)]
/// #[derive(Shared)]
/// struct Padded {
///     flag: u8,
///     size: u32,
/// }
/// ```
///
/// and neither can one without `#[repr(C)]`, whose layout is `rustc`'s to
/// choose:
///
/// ```compile_fail
/// use synaga_shader::*;
/// #[derive(Shared)]
/// struct Loose {
///     size: u32,
/// }
/// ```
///
/// A struct can hold an enum, which is a `u32` on the GPU. Its `Default` is all
/// zeroes too, so the enum needs a variant that is zero:
///
/// ```
/// use synaga_shader::*;
/// #[repr(u32)]
/// #[derive(Clone, Copy, Debug, PartialEq, bytemuck::NoUninit, bytemuck::Zeroable)]
/// enum Mode {
///     Final,
///     Depth,
/// }
/// #[repr(C)]
/// #[derive(Shared)]
/// struct Params {
///     mode: Mode,
///     _pad: u32,
/// }
/// assert_eq!(Params::default().mode, Mode::Final);
/// ```
///
/// and one without such a variant cannot be zeroed:
///
/// ```compile_fail,E0277
/// use synaga_shader::*;
/// #[repr(u32)]
/// #[derive(Clone, Copy, bytemuck::NoUninit)]
/// enum Mode {
///     Depth = 1,
///     Normal,
/// }
/// #[repr(C)]
/// #[derive(Shared)]
/// struct Params {
///     mode: Mode,
///     _pad: u32,
/// }
/// ```
///
/// `NoUninit` is all an upload takes. A host that reads one back from bytes
/// derives `bytemuck`'s `CheckedBitPattern` beside `Shared`, on the struct and
/// on its enums, so that a `u32` that is none of the variants is refused rather
/// than read. `bytemuck` checks no array that way, so the struct's arrays are
/// of `Pod` elements:
///
/// ```
/// use synaga_shader::*;
/// #[repr(u32)]
/// #[derive(Clone, Copy, Debug, PartialEq)]
/// #[derive(bytemuck::NoUninit, bytemuck::Zeroable, bytemuck::CheckedBitPattern)]
/// enum Mode {
///     Final,
///     Depth,
/// }
/// #[repr(C)]
/// #[derive(Debug, Shared, bytemuck::CheckedBitPattern)]
/// struct Params {
///     mode: Mode,
///     count: u32,
/// }
/// let read: &Params = bytemuck::checked::from_bytes(bytemuck::cast_slice(&[1u32, 3]));
/// assert_eq!((read.mode, read.count), (Mode::Depth, 3));
/// let unknown = bytemuck::checked::try_from_bytes::<Params>(bytemuck::cast_slice(&[7u32, 3]));
/// assert!(unknown.is_err());
/// ```
#[cfg(feature = "bytemuck")]
pub trait Shared: bytemuck::NoUninit + bytemuck::Zeroable + Default {}

/// What `#[derive(Shared)]` reaches for, from a crate that need not depend on
/// `bytemuck` itself.
#[cfg(feature = "bytemuck")]
#[doc(hidden)]
pub mod __private {
    pub use bytemuck;
}

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
