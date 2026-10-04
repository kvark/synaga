//! The math and relational builtins, over scalars and vectors alike.
//!
//! Each is generic over the operand type: WGSL's `abs` works on `f32` and on
//! `vec3<f32>`, and this one on `f32` and `Vec3`.
//!
//! These keep WGSL's names and WGSL's meanings, on the CPU as on the GPU:
//! `fract(x)` is `x - floor(x)`, `round(x)` takes a half to the even
//! neighbour, `sign(0.0)` is zero, and `abs(i32::MIN)` is `i32::MIN`. The
//! methods of the same names are Rust's.

use crate::matrix::{Square, Transposable};
use crate::vector::*;

use lanes::{Bits, Factor, Integer, Lanes, Number, Pick};

/// A type a component-wise builtin accepts: a scalar or a vector of them.
pub trait Numeric: Lanes<Lane: Number> {}
/// A float scalar or vector.
pub trait Floating: Numeric + Lanes<Lane = f32> {}
/// An integer scalar or vector.
pub trait Integral: Numeric + Lanes<Lane: Integer> {}
/// A `bool`, or a vector of them, which `all` and `any` fold to one.
pub trait BoolVector: Lanes<Lane = bool> {}

macro_rules! mark {
    ($trait:ident: $($ty:ty),* $(,)?) => {
        $(impl $trait for $ty {})*
    };
}

mark!(Numeric: f32, i32, u32,
      Vec2, Vec3, Vec4, Vec2<i32>, Vec3<i32>, Vec4<i32>, Vec2<u32>, Vec3<u32>, Vec4<u32>);
mark!(Floating: f32, Vec2, Vec3, Vec4);
mark!(Integral: i32, u32, Vec2<i32>, Vec3<i32>, Vec4<i32>, Vec2<u32>, Vec3<u32>, Vec4<u32>);
mark!(BoolVector: bool, Vec2<bool>, Vec3<bool>, Vec4<bool>);

/// The magnitude. The most negative `i32` has none that fits, and stays as
/// it is, as it does on the GPU.
#[inline]
pub fn abs<T: Numeric>(x: T) -> T {
    x.map(Number::abs)
}

/// `1`, `0` or `-1`, by the sign; `sign(0.0)` is zero, unlike Rust's
/// `0.0f32.signum()`.
#[inline]
pub fn sign<T: Numeric>(x: T) -> T {
    x.map(Number::sign)
}

#[inline]
pub fn min<T: Numeric>(x: T, y: T) -> T {
    x.zip(y, Number::min)
}

#[inline]
pub fn max<T: Numeric>(x: T, y: T) -> T {
    x.zip(y, Number::max)
}

/// `min(max(x, low), high)`, which does not mind `low` above `high`, as
/// Rust's `clamp` does.
#[inline]
pub fn clamp<T: Numeric>(x: T, low: T, high: T) -> T {
    x.zip3(low, high, |x, low, high| x.max(low).min(high))
}

/// Held between zero and one.
#[inline]
pub fn saturate<T: Floating>(x: T) -> T {
    x.map(|x| x.clamp(0.0, 1.0))
}

/// Builtins that take one float to one float, lane by lane.
macro_rules! float_lanes {
    ($($(#[$doc:meta])* $name:ident => $f:expr;)*) => {$(
        $(#[$doc])*
        #[inline]
        pub fn $name<T: Floating>(x: T) -> T { x.map($f) }
    )*};
}

float_lanes! {
    sin => f32::sin;
    cos => f32::cos;
    tan => f32::tan;
    asin => f32::asin;
    acos => f32::acos;
    atan => f32::atan;
    sinh => f32::sinh;
    cosh => f32::cosh;
    tanh => f32::tanh;
    asinh => f32::asinh;
    acosh => f32::acosh;
    atanh => f32::atanh;
    floor => f32::floor;
    ceil => f32::ceil;
    /// A half goes to the even neighbour: Rust's `round_ties_even`, not its
    /// `round`.
    round => f32::round_ties_even;
    trunc => f32::trunc;
    /// `x - floor(x)`, which for a negative `x` is not Rust's `x.fract()`.
    fract => |x| x - x.floor();
    sqrt => f32::sqrt;
    /// `1 / sqrt(x)`: WGSL's `inverseSqrt`.
    inverse_sqrt => |x| x.sqrt().recip();
    exp => f32::exp;
    exp2 => f32::exp2;
    /// The natural logarithm: Rust's `ln`.
    log => f32::ln;
    log2 => f32::log2;
    degrees => f32::to_degrees;
    radians => f32::to_radians;
}

/// The angle of `(x, y)`: Rust's `y.atan2(x)`.
#[inline]
pub fn atan2<T: Floating>(y: T, x: T) -> T {
    y.zip(x, f32::atan2)
}

/// `x` to the power `y`. The GPU's is only defined for `x >= 0`.
#[inline]
pub fn pow<T: Floating>(x: T, y: T) -> T {
    x.zip(y, f32::powf)
}

/// One where `edge <= x`, zero where not.
#[inline]
pub fn step<T: Floating>(edge: T, x: T) -> T {
    edge.zip(x, |edge, x| if edge <= x { 1.0 } else { 0.0 })
}

/// `a * b + c`, rounded once: Rust's `a.mul_add(b, c)`.
#[inline]
pub fn fma<T: Floating>(a: T, b: T, c: T) -> T {
    a.zip3(b, c, f32::mul_add)
}

/// `a` at `t == 0` and `b` at `t == 1`: `a * (1 - t) + b * t`. The factor
/// may be one `f32` for every lane.
#[inline]
pub fn mix<T: Floating, A: Factor<T>>(a: T, b: T, t: A) -> T {
    a.zip3(b, t.spread(), |a, b, t| a * (1.0 - t) + b * t)
}

/// Zero below `low`, one above `high`, and a smooth curve between.
#[inline]
pub fn smoothstep<T: Floating>(low: T, high: T, x: T) -> T {
    low.zip3(high, x, |low, high, x| {
        let t = ((x - low) / (high - low)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    })
}

/// The sum of the lane-wise products.
#[inline]
pub fn dot<T: Numeric>(a: T, b: T) -> T::Lane {
    a.zip(b, |a, b| a * b).reduce(|a, b| a + b)
}

/// The Euclidean length.
#[inline]
pub fn length<T: Floating>(v: T) -> f32 {
    dot(v, v).sqrt()
}

/// The distance between two points.
#[inline]
pub fn distance<T: Floating>(a: T, b: T) -> f32 {
    length(a.zip(b, |a, b| a - b))
}

/// The same direction at unit length.
#[inline]
pub fn normalize<T: Floating>(v: T) -> T {
    let length = length(v);
    v.map(|lane| lane / length)
}

/// `i` reflected off a surface facing `n`: `i - 2 * dot(n, i) * n`.
#[inline]
pub fn reflect<T: Floating>(i: T, n: T) -> T {
    let d = dot(n, i);
    i.zip(n, |i, n| i - 2.0 * d * n)
}

/// `i` refracted through a surface facing `n`, with `eta` the ratio of the
/// indices of refraction; zero where the light is reflected entirely.
#[inline]
pub fn refract<T: Floating>(i: T, n: T, eta: f32) -> T {
    let d = dot(n, i);
    let k = 1.0 - eta * eta * (1.0 - d * d);
    if k < 0.0 {
        return T::splat(0.0);
    }
    i.zip(n, |i, n| eta * i - (eta * d + k.sqrt()) * n)
}

/// `n` if `dot(nref, i)` is negative, `-n` if not.
#[inline]
pub fn faceforward<T: Floating>(n: T, i: T, nref: T) -> T {
    match dot(nref, i) < 0.0 {
        true => n,
        false => n.map(|lane| -lane),
    }
}

/// The cross product, which only three-lane vectors have.
#[inline]
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    a.cross(b)
}

/// The determinant of a square matrix.
#[inline]
pub fn determinant<M: Square>(m: M) -> f32 {
    m.determinant()
}

/// Columns and rows exchanged.
#[inline]
pub fn transpose<M: Transposable>(m: M) -> M::Output {
    m.transpose()
}

/// Builtins that take one integer to one integer, lane by lane.
macro_rules! integer_lanes {
    ($($(#[$doc:meta])* $name:ident;)*) => {$(
        $(#[$doc])*
        #[inline]
        pub fn $name<T: Integral>(x: T) -> T { x.map(Integer::$name) }
    )*};
}

integer_lanes! {
    count_one_bits;
    reverse_bits;
    /// Where the highest bit that differs from the sign bit is: the highest
    /// set one of a `u32`. All ones, or `-1`, when there is none.
    first_leading_bit;
    /// Where the lowest set bit is. All ones, or `-1`, when there is none.
    first_trailing_bit;
    count_leading_zeros;
    count_trailing_zeros;
}

/// Pick `accept` where `condition` holds, `reject` where it does not: lane by
/// lane under a vector of `bool`s, or all at once under one.
///
/// WGSL's argument order: the value taken when the condition holds is second.
#[inline]
pub fn select<T: Pick<C>, C>(reject: T, accept: T, condition: C) -> T {
    T::pick(reject, accept, condition)
}

/// True when every lane is.
#[inline]
pub fn all<T: BoolVector>(v: T) -> bool {
    v.reduce(|a, b| a & b)
}

/// True when any lane is.
#[inline]
pub fn any<T: BoolVector>(v: T) -> bool {
    v.reduce(|a, b| a | b)
}

/// Reinterpret the bits of `x` as `T`. `bitcast::<u32>(1.0)` is WGSL's
/// `bitcast<u32>(1.0)`. The widths have to match: `f32` and `u32` do, and
/// so do `Vec3` and `Vec3<i32>`.
///
/// ```
/// use synaga_shader::*;
/// assert_eq!(bitcast::<u32>(1.0), 0x3f80_0000);
/// assert_eq!(bitcast::<i32>(u32::MAX), -1);
/// ```
#[inline]
pub fn bitcast<T: Bits>(x: impl Bits<Words = T::Words>) -> T {
    T::from_words(x.to_words())
}

// The packing builtins, which move between a `u32` and four or two lanes, the
// first in the lowest bits. Each is WGSL's formula, which rounds to the
// nearest step and clamps what is out of range.

/// Four lanes in `[-1, 1]` as signed bytes: `round(127 * v)`.
#[inline]
pub fn pack4x8snorm(v: Vec4) -> u32 {
    let byte = |x: f32| (0.5 + 127.0 * x.clamp(-1.0, 1.0)).floor() as i8 as u8 as u32;
    byte(v.x) | byte(v.y) << 8 | byte(v.z) << 16 | byte(v.w) << 24
}

/// Four lanes in `[0, 1]` as unsigned bytes: `round(255 * v)`.
#[inline]
pub fn pack4x8unorm(v: Vec4) -> u32 {
    let byte = |x: f32| (0.5 + 255.0 * x.clamp(0.0, 1.0)).floor() as u32;
    byte(v.x) | byte(v.y) << 8 | byte(v.z) << 16 | byte(v.w) << 24
}

/// Two lanes in `[-1, 1]` as signed 16-bit integers: `round(32767 * v)`.
#[inline]
pub fn pack2x16snorm(v: Vec2) -> u32 {
    let half = |x: f32| (0.5 + 32767.0 * x.clamp(-1.0, 1.0)).floor() as i16 as u16 as u32;
    half(v.x) | half(v.y) << 16
}

/// Two lanes in `[0, 1]` as unsigned 16-bit integers: `round(65535 * v)`.
#[inline]
pub fn pack2x16unorm(v: Vec2) -> u32 {
    let half = |x: f32| (0.5 + 65535.0 * x.clamp(0.0, 1.0)).floor() as u32;
    half(v.x) | half(v.y) << 16
}

/// Two lanes as half-precision floats.
#[inline]
pub fn pack2x16float(v: Vec2) -> u32 {
    u32::from(f16_bits(v.x)) | u32::from(f16_bits(v.y)) << 16
}

/// Four signed bytes as lanes in `[-1, 1]`.
#[inline]
pub fn unpack4x8snorm(bits: u32) -> Vec4 {
    let lane = |shift: u32| ((bits >> shift) as u8 as i8 as f32 / 127.0).max(-1.0);
    vec4(lane(0), lane(8), lane(16), lane(24))
}

/// Four unsigned bytes as lanes in `[0, 1]`.
#[inline]
pub fn unpack4x8unorm(bits: u32) -> Vec4 {
    let lane = |shift: u32| ((bits >> shift) & 0xFF) as f32 / 255.0;
    vec4(lane(0), lane(8), lane(16), lane(24))
}

/// Two signed 16-bit integers as lanes in `[-1, 1]`.
#[inline]
pub fn unpack2x16snorm(bits: u32) -> Vec2 {
    let lane = |shift: u32| ((bits >> shift) as u16 as i16 as f32 / 32767.0).max(-1.0);
    vec2(lane(0), lane(16))
}

/// Two unsigned 16-bit integers as lanes in `[0, 1]`.
#[inline]
pub fn unpack2x16unorm(bits: u32) -> Vec2 {
    let lane = |shift: u32| ((bits >> shift) & 0xFFFF) as f32 / 65535.0;
    vec2(lane(0), lane(16))
}

/// Two half-precision floats as lanes.
#[inline]
pub fn unpack2x16float(bits: u32) -> Vec2 {
    vec2(f16_value(bits as u16), f16_value((bits >> 16) as u16))
}

// The packed 4x8 integer builtins, which read a `u32` as four bytes, the
// first in the lowest bits, as WGSL's `dot4U8Packed` and the rest do.

/// The byte of `bits` at lane `lane`, as it is: `0..=255`.
#[inline]
fn byte_u8(bits: u32, lane: u32) -> u32 {
    (bits >> (8 * lane)) & 0xFF
}

/// The byte of `bits` at lane `lane`, sign-extended: `-128..=127`.
#[inline]
fn byte_i8(bits: u32, lane: u32) -> i32 {
    i32::from((bits >> (8 * lane)) as u8 as i8)
}

/// The dot product of two `u32`s read as four unsigned bytes each.
///
/// ```
/// use synaga_shader::*;
/// assert_eq!(dot4_u8_packed(0x0403_0201, 0x0101_0101), 1 + 2 + 3 + 4);
/// ```
#[inline]
pub fn dot4_u8_packed(a: u32, b: u32) -> u32 {
    (0..4).map(|lane| byte_u8(a, lane) * byte_u8(b, lane)).sum()
}

/// The dot product of two `u32`s read as four signed bytes each.
///
/// ```
/// use synaga_shader::*;
/// assert_eq!(dot4_i8_packed(0x0000_00FF, 0x0000_0002), -2);
/// ```
#[inline]
pub fn dot4_i8_packed(a: u32, b: u32) -> i32 {
    (0..4).map(|lane| byte_i8(a, lane) * byte_i8(b, lane)).sum()
}

/// Four `i32`s as signed bytes, each its lowest eight bits.
#[inline]
pub fn pack4x_i8(v: Vec4<i32>) -> u32 {
    let byte = |x: i32| u32::from(x as u8);
    byte(v.x) | byte(v.y) << 8 | byte(v.z) << 16 | byte(v.w) << 24
}

/// Four `u32`s as unsigned bytes, each its lowest eight bits.
#[inline]
pub fn pack4x_u8(v: Vec4<u32>) -> u32 {
    let byte = |x: u32| x & 0xFF;
    byte(v.x) | byte(v.y) << 8 | byte(v.z) << 16 | byte(v.w) << 24
}

/// Four `i32`s as signed bytes, each clamped to `-128..=127` first.
#[inline]
pub fn pack4x_i8_clamp(v: Vec4<i32>) -> u32 {
    pack4x_i8(v.map(|x| Ord::clamp(x, -128, 127)))
}

/// Four `u32`s as unsigned bytes, each clamped to `0..=255` first.
#[inline]
pub fn pack4x_u8_clamp(v: Vec4<u32>) -> u32 {
    pack4x_u8(v.map(|x| Ord::min(x, 255)))
}

/// Four signed bytes as `i32`s.
#[inline]
pub fn unpack4x_i8(bits: u32) -> Vec4<i32> {
    vec4(
        byte_i8(bits, 0),
        byte_i8(bits, 1),
        byte_i8(bits, 2),
        byte_i8(bits, 3),
    )
}

/// Four unsigned bytes as `u32`s.
#[inline]
pub fn unpack4x_u8(bits: u32) -> Vec4<u32> {
    vec4(
        byte_u8(bits, 0),
        byte_u8(bits, 1),
        byte_u8(bits, 2),
        byte_u8(bits, 3),
    )
}

/// `value` as a half-precision float's bits, rounded to the nearest, ties to
/// even, as IEEE 754 rounds.
fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;
    if exponent == 0xFF {
        // Infinity stays infinite, and a NaN stays a NaN.
        let nan = if mantissa != 0 { 0x0200 } else { 0 };
        return sign | 0x7C00 | nan;
    }
    // The exponent re-biased from `f32`'s 127 to `f16`'s 15.
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1F {
        return sign | 0x7C00;
    }
    // What is left of the mantissa, and how much of it is shifted out.
    let (kept, dropped, shift) = if half_exponent <= 0 {
        // Too small for a normal `f16`: a subnormal, with the implicit one
        // made explicit, or zero.
        if half_exponent < -10 {
            return sign;
        }
        let shift = (14 - half_exponent) as u32;
        let full = mantissa | 0x0080_0000;
        (full >> shift, full & ((1 << shift) - 1), shift)
    } else {
        let kept = ((half_exponent as u32) << 10) | (mantissa >> 13);
        (kept, mantissa & 0x1FFF, 13)
    };
    let halfway = 1 << (shift - 1);
    // A carry out of the mantissa steps the exponent, which is right, and
    // past the largest finite value makes infinity, which is too.
    let rounded = if dropped > halfway || (dropped == halfway && kept & 1 == 1) {
        kept + 1
    } else {
        kept
    };
    sign | rounded as u16
}

/// A half-precision float's value.
fn f16_value(bits: u16) -> f32 {
    let negative = bits & 0x8000 != 0;
    let exponent = u32::from((bits >> 10) & 0x1F);
    let mantissa = u32::from(bits & 0x03FF);
    let magnitude = match exponent {
        // Zero, or a subnormal: the mantissa in units of 2^-24.
        0 => mantissa as f32 / 16_777_216.0,
        0x1F if mantissa == 0 => f32::INFINITY,
        0x1F => f32::NAN,
        _ => f32::from_bits(((exponent + 127 - 15) << 23) | (mantissa << 13)),
    };
    if negative {
        -magnitude
    } else {
        magnitude
    }
}

/// Stop this invocation without writing anything. It never returns, so
/// `discard()` can end a function whatever that function returns.
///
/// On the CPU there is no fragment to throw away, so it panics.
#[inline]
pub fn discard() -> ! {
    panic!("`discard()` ends a fragment shader's invocation, which only the GPU runs")
}

/// Wait for every invocation in the workgroup: WGSL's `workgroupBarrier`.
///
/// Running one needs every invocation of the workgroup at once, which the
/// CPU does not have, so it panics there.
#[inline]
pub fn workgroup_barrier() {
    crate::unimplemented_on_cpu()
}

/// Make storage writes visible across the workgroup: WGSL's `storageBarrier`.
#[inline]
pub fn storage_barrier() {
    crate::unimplemented_on_cpu()
}

/// How the builtins run on the CPU: lane by lane. Out of reach outside this
/// crate, so a shader sees none of it.
mod lanes {
    use core::ops::{Add, Div, Mul, Sub};

    use crate::vector::*;

    /// A scalar, or a vector of them.
    pub trait Lanes: Copy {
        /// What one lane holds.
        type Lane: Scalar;
        fn map(self, f: impl Fn(Self::Lane) -> Self::Lane) -> Self;
        fn zip(self, b: Self, f: impl Fn(Self::Lane, Self::Lane) -> Self::Lane) -> Self;
        fn zip3(
            self,
            b: Self,
            c: Self,
            f: impl Fn(Self::Lane, Self::Lane, Self::Lane) -> Self::Lane,
        ) -> Self;
        /// The lanes, combined first to last.
        fn reduce(self, f: impl Fn(Self::Lane, Self::Lane) -> Self::Lane) -> Self::Lane;
        /// Every lane `lane`.
        fn splat(lane: Self::Lane) -> Self;
    }

    macro_rules! scalar_lanes {
        ($($ty:ty),*) => {$(
            impl Lanes for $ty {
                type Lane = $ty;
                #[inline]
                fn map(self, f: impl Fn($ty) -> $ty) -> $ty { f(self) }
                #[inline]
                fn zip(self, b: $ty, f: impl Fn($ty, $ty) -> $ty) -> $ty { f(self, b) }
                #[inline]
                fn zip3(self, b: $ty, c: $ty, f: impl Fn($ty, $ty, $ty) -> $ty) -> $ty {
                    f(self, b, c)
                }
                #[inline]
                fn reduce(self, _: impl Fn($ty, $ty) -> $ty) -> $ty { self }
                #[inline]
                fn splat(lane: $ty) -> $ty { lane }
            }
        )*};
    }
    scalar_lanes!(f32, i32, u32, bool);

    macro_rules! vector_lanes {
        ($($vec:ident),*) => {$(
            impl<T: Scalar> Lanes for $vec<T> {
                type Lane = T;
                #[inline]
                fn map(self, f: impl Fn(T) -> T) -> Self { self.map_lanes(f) }
                #[inline]
                fn zip(self, b: Self, f: impl Fn(T, T) -> T) -> Self { self.zip_lanes(b, f) }
                #[inline]
                fn zip3(self, b: Self, c: Self, f: impl Fn(T, T, T) -> T) -> Self {
                    self.zip3_lanes(b, c, f)
                }
                #[inline]
                fn reduce(self, f: impl Fn(T, T) -> T) -> T { self.reduce_lanes(f) }
                #[inline]
                fn splat(lane: T) -> Self { $vec::splat(lane) }
            }
        )*};
    }
    vector_lanes!(Vec2, Vec3, Vec4);

    /// A lane that is a number: what `abs`, `min` and `dot` need.
    pub trait Number:
        Scalar + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
    {
        /// WGSL's `abs`, which leaves the most negative `i32` as it is.
        fn abs(self) -> Self;
        /// WGSL's `sign`: `1`, `0` or `-1`.
        fn sign(self) -> Self;
        fn min(self, other: Self) -> Self;
        fn max(self, other: Self) -> Self;
    }

    impl Number for f32 {
        #[inline]
        fn abs(self) -> f32 {
            f32::abs(self)
        }
        #[inline]
        fn sign(self) -> f32 {
            if self > 0.0 {
                1.0
            } else if self < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        // Which of the two a NaN yields is the GPU's choice; this is Rust's.
        #[inline]
        fn min(self, other: f32) -> f32 {
            f32::min(self, other)
        }
        #[inline]
        fn max(self, other: f32) -> f32 {
            f32::max(self, other)
        }
    }

    impl Number for i32 {
        #[inline]
        fn abs(self) -> i32 {
            self.wrapping_abs()
        }
        #[inline]
        fn sign(self) -> i32 {
            self.signum()
        }
        #[inline]
        fn min(self, other: i32) -> i32 {
            Ord::min(self, other)
        }
        #[inline]
        fn max(self, other: i32) -> i32 {
            Ord::max(self, other)
        }
    }

    impl Number for u32 {
        #[inline]
        fn abs(self) -> u32 {
            self
        }
        #[inline]
        fn sign(self) -> u32 {
            u32::from(self != 0)
        }
        #[inline]
        fn min(self, other: u32) -> u32 {
            Ord::min(self, other)
        }
        #[inline]
        fn max(self, other: u32) -> u32 {
            Ord::max(self, other)
        }
    }

    /// An integer lane: what the bit builtins need.
    pub trait Integer: Number {
        fn count_one_bits(self) -> Self;
        fn reverse_bits(self) -> Self;
        fn first_leading_bit(self) -> Self;
        fn first_trailing_bit(self) -> Self;
        fn count_leading_zeros(self) -> Self;
        fn count_trailing_zeros(self) -> Self;
    }

    impl Integer for u32 {
        #[inline]
        fn count_one_bits(self) -> u32 {
            self.count_ones()
        }
        #[inline]
        fn reverse_bits(self) -> u32 {
            u32::reverse_bits(self)
        }
        #[inline]
        fn first_leading_bit(self) -> u32 {
            match self {
                0 => u32::MAX,
                n => 31 - n.leading_zeros(),
            }
        }
        #[inline]
        fn first_trailing_bit(self) -> u32 {
            match self {
                0 => u32::MAX,
                n => n.trailing_zeros(),
            }
        }
        #[inline]
        fn count_leading_zeros(self) -> u32 {
            self.leading_zeros()
        }
        #[inline]
        fn count_trailing_zeros(self) -> u32 {
            self.trailing_zeros()
        }
    }

    impl Integer for i32 {
        #[inline]
        fn count_one_bits(self) -> i32 {
            self.count_ones() as i32
        }
        #[inline]
        fn reverse_bits(self) -> i32 {
            i32::reverse_bits(self)
        }
        // A negative number's sign bit is set, so what differs from it is
        // the highest bit that is clear.
        #[inline]
        fn first_leading_bit(self) -> i32 {
            let differs = if self < 0 { !self } else { self } as u32;
            match differs {
                0 => -1,
                n => (31 - n.leading_zeros()) as i32,
            }
        }
        #[inline]
        fn first_trailing_bit(self) -> i32 {
            match self {
                0 => -1,
                n => n.trailing_zeros() as i32,
            }
        }
        #[inline]
        fn count_leading_zeros(self) -> i32 {
            self.leading_zeros() as i32
        }
        #[inline]
        fn count_trailing_zeros(self) -> i32 {
            self.trailing_zeros() as i32
        }
    }

    /// What `mix` blends by: a value of the blended type, or one `f32` for
    /// every lane.
    pub trait Factor<T> {
        fn spread(self) -> T;
    }
    impl<T: Lanes<Lane = f32>> Factor<T> for f32 {
        #[inline]
        fn spread(self) -> T {
            T::splat(self)
        }
    }
    macro_rules! own_factor {
        ($($vec:ty),*) => {$(
            impl Factor<$vec> for $vec {
                #[inline]
                fn spread(self) -> $vec { self }
            }
        )*};
    }
    own_factor!(Vec2, Vec3, Vec4);

    /// What `select` picks between, under a condition of type `C`: anything
    /// under one `bool`, and a vector's lanes under a vector of them.
    pub trait Pick<C>: Copy {
        fn pick(reject: Self, accept: Self, condition: C) -> Self;
    }
    impl<T: Copy> Pick<bool> for T {
        #[inline]
        fn pick(reject: T, accept: T, condition: bool) -> T {
            if condition {
                accept
            } else {
                reject
            }
        }
    }
    macro_rules! pick_lanes {
        ($($vec:ident: $count:literal),*) => {$(
            impl<S: Scalar> Pick<$vec<bool>> for $vec<S> {
                #[inline]
                fn pick(reject: Self, accept: Self, condition: $vec<bool>) -> Self {
                    let mut picked = reject;
                    for lane in 0..$count {
                        if condition[lane] {
                            picked[lane] = accept[lane];
                        }
                    }
                    picked
                }
            }
        )*};
    }
    pick_lanes!(Vec2: 2, Vec3: 3, Vec4: 4);

    /// A value made of 32-bit words, which `bitcast` reads as another.
    pub trait Bits: Copy {
        /// The same bits as `u32` lanes.
        type Words;
        fn to_words(self) -> Self::Words;
        fn from_words(words: Self::Words) -> Self;
    }
    impl Bits for f32 {
        type Words = u32;
        #[inline]
        fn to_words(self) -> u32 {
            self.to_bits()
        }
        #[inline]
        fn from_words(words: u32) -> f32 {
            f32::from_bits(words)
        }
    }
    impl Bits for i32 {
        type Words = u32;
        #[inline]
        fn to_words(self) -> u32 {
            self as u32
        }
        #[inline]
        fn from_words(words: u32) -> i32 {
            words as i32
        }
    }
    impl Bits for u32 {
        type Words = u32;
        #[inline]
        fn to_words(self) -> u32 {
            self
        }
        #[inline]
        fn from_words(words: u32) -> u32 {
            words
        }
    }
    macro_rules! vector_bits {
        ($($vec:ident),*) => {$(
            impl<S: Scalar + Bits<Words = u32>> Bits for $vec<S> {
                type Words = $vec<u32>;
                #[inline]
                fn to_words(self) -> $vec<u32> { self.map_lanes(S::to_words) }
                #[inline]
                fn from_words(words: $vec<u32>) -> Self { words.map_lanes(S::from_words) }
            }
        )*};
    }
    vector_bits!(Vec2, Vec3, Vec4);
}
