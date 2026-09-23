//! The math and relational builtins, over scalars and vectors alike.
//!
//! Each is generic over the operand type: WGSL's `abs` works on `f32` and on
//! `vec3<f32>`, and so does this one.

use crate::unimplemented_on_cpu;
use crate::vector::*;

/// A type a component-wise builtin accepts: a scalar or a vector of them.
pub trait Numeric: Copy {}
/// A float scalar or vector.
pub trait Floating: Numeric {}
/// An integer scalar or vector.
pub trait Integral: Numeric {}

macro_rules! mark {
    ($trait:ident: $($ty:ty),* $(,)?) => {
        $(impl $trait for $ty {})*
    };
}

mark!(Numeric: f32, i32, u32,
      vec2, vec3, vec4, vec2i, vec3i, vec4i, vec2u, vec3u, vec4u);
mark!(Floating: f32, vec2, vec3, vec4);
mark!(Integral: i32, u32, vec2i, vec3i, vec4i, vec2u, vec3u, vec4u);

macro_rules! builtins {
    ($($(#[$doc:meta])* $bound:ident $name:ident ($($arg:ident),*);)*) => {
        $(
            $(#[$doc])*
            #[inline]
            pub fn $name<T: $bound>($($arg: T),*) -> T { unimplemented_on_cpu() }
        )*
    };
}

builtins! {
    Numeric abs(x);
    Numeric sign(x);
    Numeric min(x, y);
    Numeric max(x, y);
    Numeric clamp(x, low, high);
    Floating saturate(x);
    Floating sin(x);
    Floating cos(x);
    Floating tan(x);
    Floating asin(x);
    Floating acos(x);
    Floating atan(x);
    Floating sinh(x);
    Floating cosh(x);
    Floating tanh(x);
    Floating asinh(x);
    Floating acosh(x);
    Floating atanh(x);
    Floating atan2(y, x);
    Floating floor(x);
    Floating ceil(x);
    Floating round(x);
    Floating trunc(x);
    Floating fract(x);
    Floating sqrt(x);
    Floating exp(x);
    Floating exp2(x);
    Floating log(x);
    Floating log2(x);
    Floating pow(x, y);
    Floating degrees(x);
    Floating radians(x);
    Floating step(edge, x);
    Floating fma(a, b, c);
    Floating normalize(x);
    Floating reflect(i, n);
    Floating refract(i, n, eta);
    Floating faceforward(n, i, nref);
    Integral countOneBits(x);
    Integral count_one_bits(x);
    Integral reverseBits(x);
    Integral reverse_bits(x);
    Integral firstLeadingBit(x);
    Integral firstTrailingBit(x);
    Integral countLeadingZeros(x);
    Integral countTrailingZeros(x);
}

/// `mix(a, b, t)`. The factor may be a scalar while `a` and `b` are vectors.
#[inline]
pub fn mix<T: Floating, A>(_x: T, _y: T, _a: A) -> T {
    unimplemented_on_cpu()
}

/// `smoothstep`, which takes its edges and the value.
#[inline]
pub fn smoothstep<T: Floating>(_low: T, _high: T, _x: T) -> T {
    unimplemented_on_cpu()
}

/// The inverse square root, under both spellings WGSL and Rust would use.
#[inline]
#[allow(non_snake_case)]
pub fn inverseSqrt<T: Floating>(_x: T) -> T {
    unimplemented_on_cpu()
}

#[inline]
pub fn inverse_sqrt<T: Floating>(_x: T) -> T {
    unimplemented_on_cpu()
}

/// Pick `accept` where `condition` holds, `reject` where it does not.
///
/// WGSL's argument order: the value taken when the condition holds is second.
#[inline]
pub fn select<T, C>(_reject: T, _accept: T, _condition: C) -> T {
    unimplemented_on_cpu()
}

/// A vector whose lanes reduce to one `bool`.
pub trait BoolVector: Copy {}
mark!(BoolVector: bool, vec2b, vec3b, vec4b);

/// True when every lane is.
#[inline]
pub fn all<T: BoolVector>(_v: T) -> bool {
    unimplemented_on_cpu()
}

/// True when any lane is.
#[inline]
pub fn any<T: BoolVector>(_v: T) -> bool {
    unimplemented_on_cpu()
}

/// Things `dot`, `length` and friends reduce to a scalar.
pub trait Reducible: Copy {
    /// What one of them produces.
    type Scalar;
}

macro_rules! reducible {
    ($($ty:ty => $scalar:ty),* $(,)?) => {
        $(impl Reducible for $ty { type Scalar = $scalar; })*
    };
}
reducible!(
    f32 => f32, vec2 => f32, vec3 => f32, vec4 => f32,
    i32 => i32, vec2i => i32, vec3i => i32, vec4i => i32,
    u32 => u32, vec2u => u32, vec3u => u32, vec4u => u32,
);

/// Dot product.
#[inline]
pub fn dot<T: Reducible>(_a: T, _b: T) -> T::Scalar {
    unimplemented_on_cpu()
}

/// Length of a vector.
#[inline]
pub fn length<T: Reducible>(_v: T) -> T::Scalar {
    unimplemented_on_cpu()
}

/// Distance between two points.
#[inline]
pub fn distance<T: Reducible>(_a: T, _b: T) -> T::Scalar {
    unimplemented_on_cpu()
}

/// Cross product, which only three-lane vectors have.
#[inline]
pub fn cross(_a: vec3, _b: vec3) -> vec3 {
    unimplemented_on_cpu()
}

/// Determinant of a square matrix.
#[inline]
pub fn determinant<M>(_m: M) -> f32 {
    unimplemented_on_cpu()
}

/// Transpose, which swaps a matrix's columns and rows.
pub trait Transposable {
    /// The matrix with columns and rows exchanged.
    type Output;
}

macro_rules! transposable {
    ($($from:ty => $to:ty),* $(,)?) => {
        $(impl Transposable for $from { type Output = $to; })*
    };
}
use crate::matrix::*;
transposable!(
    mat2x2 => mat2x2, mat3x3 => mat3x3, mat4x4 => mat4x4,
    mat2x3 => mat3x2, mat3x2 => mat2x3,
    mat2x4 => mat4x2, mat4x2 => mat2x4,
    mat3x4 => mat4x3, mat4x3 => mat3x4,
);

#[inline]
pub fn transpose<M: Transposable>(_m: M) -> M::Output {
    unimplemented_on_cpu()
}

/// The packing builtins, which move between a `u32` and four or two lanes.
macro_rules! packing {
    ($($name:ident($arg:ty) -> $ret:ty;)*) => {
        $(
            #[inline]
            #[allow(non_snake_case)]
            pub fn $name(_v: $arg) -> $ret { unimplemented_on_cpu() }
        )*
    };
}

packing! {
    pack4x8snorm(vec4) -> u32;
    pack4x8unorm(vec4) -> u32;
    pack2x16snorm(vec2) -> u32;
    pack2x16unorm(vec2) -> u32;
    pack2x16float(vec2) -> u32;
    unpack4x8snorm(u32) -> vec4;
    unpack4x8unorm(u32) -> vec4;
    unpack2x16snorm(u32) -> vec2;
    unpack2x16unorm(u32) -> vec2;
    unpack2x16float(u32) -> vec2;
}

/// Reinterpret the bits of `x` as `T`. `bitcast::<u32>(1.0)` is WGSL's
/// `bitcast<u32>(1.0)`. The widths have to match: `f32` and `u32` do.
#[inline]
pub fn bitcast<T>(_x: impl Copy) -> T {
    unimplemented_on_cpu()
}

/// Stop this invocation without writing anything.
#[inline]
pub fn discard() {
    unimplemented_on_cpu()
}

/// Wait for every invocation in the workgroup.
#[inline]
#[allow(non_snake_case)]
pub fn workgroupBarrier() {
    unimplemented_on_cpu()
}

#[inline]
pub fn workgroup_barrier() {
    unimplemented_on_cpu()
}

#[inline]
#[allow(non_snake_case)]
pub fn storageBarrier() {
    unimplemented_on_cpu()
}
