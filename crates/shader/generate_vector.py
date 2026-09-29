"""Generates crates/shader/src/vector.rs — the vector types and everything on them."""
from itertools import permutations

SCALARS = ["f32", "i32", "u32", "bool"]
SIZES = [2, 3, 4]
# Every vector gets the same operators, differing only in which ones apply:
# `bool` has no arithmetic, only integers shift, only signed types negate. One
# macro says that once; the alternative is 2,500 lines of identical impls.
MACRO = r"""/// The operators every vector has, and the ones only some do.
///
/// `macro_rules!` cannot paste `Add` and `Assign` into one identifier, so each
/// operator names its assigning form too. `$shift` is the `VecN<u32>` a
/// lane-wise shift takes, since WGSL wants an unsigned shift amount whatever
/// is shifted.
macro_rules! vector_ops {
    ($name:ty, $scalar:ty, $shift:ty $(, $group:ident)*) => {
        $(vector_ops!(@group $group, $name, $scalar, $shift);)*
    };

    (@group arith, $name:ty, $scalar:ty, $shift:ty) => {
        vector_ops!(@scalar_too Add, add, AddAssign, add_assign, $name, $scalar);
        vector_ops!(@scalar_too Sub, sub, SubAssign, sub_assign, $name, $scalar);
        vector_ops!(@scalar_too Mul, mul, MulAssign, mul_assign, $name, $scalar);
        vector_ops!(@scalar_too Div, div, DivAssign, div_assign, $name, $scalar);
        vector_ops!(@scalar_too Rem, rem, RemAssign, rem_assign, $name, $scalar);
    };
    (@group bitwise, $name:ty, $scalar:ty, $shift:ty) => {
        vector_ops!(@scalar_too BitAnd, bitand, BitAndAssign, bitand_assign, $name, $scalar);
        vector_ops!(@scalar_too BitOr, bitor, BitOrAssign, bitor_assign, $name, $scalar);
        vector_ops!(@scalar_too BitXor, bitxor, BitXorAssign, bitxor_assign, $name, $scalar);
    };
    (@group shift, $name:ty, $scalar:ty, $shift:ty) => {
        vector_ops!(@shift Shl, shl, ShlAssign, shl_assign, $name, $shift);
        vector_ops!(@shift Shr, shr, ShrAssign, shr_assign, $name, $shift);
    };
    (@group neg, $name:ty, $scalar:ty, $shift:ty) => {
        impl Neg for $name {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self { self.map_lanes(|a| -a) }
        }
    };
    (@group not, $name:ty, $scalar:ty, $shift:ty) => {
        impl Not for $name {
            type Output = Self;
            #[inline]
            fn not(self) -> Self { self.map_lanes(|a| !a) }
        }
    };

    // Arithmetic and bitwise operators also work against a scalar, from
    // either side, as glam's do: a shader writes both `v * 2.0` and
    // `2.0 * v`, and `bits & 0xFF`.
    (@scalar_too $trait:ident, $method:ident, $assign:ident, $assign_fn:ident,
     $name:ty, $scalar:ty) => {
        vector_ops!(@lanewise $trait, $method, $assign, $assign_fn, $name);
        impl $trait<$scalar> for $name {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: $scalar) -> Self { self.map_lanes(|a| $trait::$method(a, rhs)) }
        }
        impl $trait<$name> for $scalar {
            type Output = $name;
            #[inline]
            fn $method(self, rhs: $name) -> $name { rhs.map_lanes(|b| $trait::$method(self, b)) }
        }
        impl $assign<$scalar> for $name {
            #[inline]
            fn $assign_fn(&mut self, rhs: $scalar) { *self = $trait::$method(*self, rhs) }
        }
    };
    (@lanewise $trait:ident, $method:ident, $assign:ident, $assign_fn:ident, $name:ty) => {
        impl $trait for $name {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: Self) -> Self { self.zip_lanes(rhs, $trait::$method) }
        }
        impl $assign for $name {
            #[inline]
            fn $assign_fn(&mut self, rhs: Self) { *self = $trait::$method(*self, rhs) }
        }
    };
    (@shift $trait:ident, $method:ident, $assign:ident, $assign_fn:ident,
     $name:ty, $shift:ty) => {
        impl $trait<$shift> for $name {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: $shift) -> Self { self.zip_lanes(rhs, $trait::$method) }
        }
        impl $trait<u32> for $name {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: u32) -> Self { self.map_lanes(|a| $trait::$method(a, rhs)) }
        }
        impl $assign<$shift> for $name {
            #[inline]
            fn $assign_fn(&mut self, rhs: $shift) { *self = $trait::$method(*self, rhs) }
        }
        impl $assign<u32> for $name {
            #[inline]
            fn $assign_fn(&mut self, rhs: u32) { *self = $trait::$method(*self, rhs) }
        }
    };
}

/// The math on a vector, named as Rust names it: `f32`'s own methods
/// (`v.sqrt()`, `v.max(w)`, `v.mul_add(a, b)`), and glam's for what only a
/// vector has (`v.dot(w)`, `v.normalize()`, `a.lerp(b, t)`).
///
/// Each means what the Rust method means, on the CPU as on the GPU. Two of
/// the GPU's builtins round differently from the Rust methods of the same
/// name, so `fract` here is `self - self.trunc()` as `f32::fract` is, and the
/// GPU's `round`, which takes a half to the even neighbour, is
/// `round_ties_even`. The free functions keep WGSL's names and meanings:
/// `fract(v)` is `v - floor(v)`.
macro_rules! vector_math {
    ($name:ty, $scalar:ty $(, $group:ident)*) => {
        $(vector_math!(@group $group, $name, $scalar);)*
    };

    (@group ord, $name:ty, $scalar:ty) => {
        impl $name {
            /// The lesser of each pair of lanes.
            #[inline]
            pub fn min(self, rhs: Self) -> Self { self.zip_lanes(rhs, |a, b| a.min(b)) }
            /// The greater of each pair of lanes.
            #[inline]
            pub fn max(self, rhs: Self) -> Self { self.zip_lanes(rhs, |a, b| a.max(b)) }
            /// Each lane held between the lanes of `min` and `max`, which
            /// have to be in order.
            #[inline]
            pub fn clamp(self, min: Self, max: Self) -> Self {
                debug_assert!(min.cmple(max).all(), "clamp: expected min <= max");
                self.max(min).min(max)
            }
            /// The sum of the lane-wise products.
            #[inline]
            pub fn dot(self, rhs: Self) -> $scalar { (self * rhs).element_sum() }
            /// The sum of the lanes.
            #[inline]
            pub fn element_sum(self) -> $scalar { self.reduce_lanes(|a, b| a + b) }
        }
    };
    (@group signed, $name:ty, $scalar:ty) => {
        impl $name {
            /// The magnitude of each lane.
            #[inline]
            pub fn abs(self) -> Self { self.map_lanes(<$scalar>::abs) }
            /// `-1`, `0` or `1` per lane, by its sign.
            #[inline]
            pub fn signum(self) -> Self { self.map_lanes(<$scalar>::signum) }
        }
    };
    (@group float, $name:ty, $scalar:ty) => {
        vector_math!(@lanewise $name, $scalar, abs floor ceil trunc round_ties_even fract sqrt
            recip exp exp2 ln log2 sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh
            to_degrees to_radians);
        impl $name {
            /// This direction at unit length: `self / self.length()`.
            #[inline]
            pub fn normalize(self) -> Self { self * self.length().recip() }
            /// Each lane raised to the power `n`.
            #[inline]
            pub fn powf(self, n: $scalar) -> Self { self.map_lanes(|a| a.powf(n)) }
            /// The angle of each pair of lanes, `self` being `y`, as `f32::atan2`.
            #[inline]
            pub fn atan2(self, x: Self) -> Self { self.zip_lanes(x, <$scalar>::atan2) }
            /// `self * a + b`: the GPU's `fma`.
            #[inline]
            pub fn mul_add(self, a: Self, b: Self) -> Self {
                self.zip3_lanes(a, b, <$scalar>::mul_add)
            }
            /// The Euclidean length.
            #[inline]
            pub fn length(self) -> $scalar { self.dot(self).sqrt() }
            /// The squared length, which saves the square root.
            #[inline]
            pub fn length_squared(self) -> $scalar { self.dot(self) }
            /// The distance to `rhs`.
            #[inline]
            pub fn distance(self, rhs: Self) -> $scalar { (self - rhs).length() }
            /// This direction reflected off a surface facing `normal`, which
            /// has to be normalized.
            #[inline]
            pub fn reflect(self, normal: Self) -> Self { self - 2.0 * self.dot(normal) * normal }
            /// This direction refracted through a surface facing `normal`,
            /// with `eta` the ratio of the indices of refraction. Zero where
            /// the light is reflected entirely.
            #[inline]
            pub fn refract(self, normal: Self, eta: $scalar) -> Self {
                let n_dot_i = normal.dot(self);
                let k = 1.0 - eta * eta * (1.0 - n_dot_i * n_dot_i);
                if k >= 0.0 {
                    eta * self - (eta * n_dot_i + k.sqrt()) * normal
                } else {
                    Self::ZERO
                }
            }
            /// `self` at `s == 0` and `rhs` at `s == 1`: the GPU's `mix`.
            #[inline]
            pub fn lerp(self, rhs: Self, s: $scalar) -> Self { self * (1.0 - s) + rhs * s }
        }
    };
    (@group wrapping, $name:ty, $scalar:ty) => {
        impl $name {
            /// Lane-wise `+`, wrapping around on overflow, as the GPU's `+`
            /// does, rather than panicking as Rust's does under overflow checks.
            #[inline]
            pub fn wrapping_add(self, rhs: Self) -> Self {
                self.zip_lanes(rhs, <$scalar>::wrapping_add)
            }
            /// Lane-wise `-`, wrapping around on overflow.
            #[inline]
            pub fn wrapping_sub(self, rhs: Self) -> Self {
                self.zip_lanes(rhs, <$scalar>::wrapping_sub)
            }
            /// Lane-wise `*`, wrapping around on overflow.
            #[inline]
            pub fn wrapping_mul(self, rhs: Self) -> Self {
                self.zip_lanes(rhs, <$scalar>::wrapping_mul)
            }
            /// Each lane negated, wrapping around on overflow: `0 - self`.
            #[inline]
            pub fn wrapping_neg(self) -> Self { self.map_lanes(<$scalar>::wrapping_neg) }
        }
    };
    (@group bool, $name:ty, $scalar:ty) => {
        impl $name {
            /// Whether every lane is `true`.
            #[inline]
            pub fn all(self) -> bool { self.reduce_lanes(|a, b| a & b) }
            /// Whether any lane is `true`.
            #[inline]
            pub fn any(self) -> bool { self.reduce_lanes(|a, b| a | b) }
        }
    };

    // One-lane-in, one-lane-out methods that `f32` has under the same name,
    // with the same meaning, which is all their documentation needs to say.
    (@lanewise $name:ty, $scalar:ty, $($method:ident)*) => {
        impl $name {
            $(
                #[doc = concat!("`f32::", stringify!($method), "` on each lane.")]
                #[inline]
                pub fn $method(self) -> Self { self.map_lanes(<$scalar>::$method) }
            )*
        }
    };
}"""

XYZW = "xyzw"
RGBA = "rgba"

# Under `rgba`, only the aliases anyone writes: the single components, the
# prefix runs, and the two tails that come up in colour code.
RGBA_ALIASES = [(0,), (1,), (2,), (3,), (0, 1), (1, 0), (2, 3), (0, 1, 2), (0, 1, 2, 3)]


def swizzles(size):
    """Which swizzle methods a vector of `size` lanes gets, as (lanes, letters).

    Every reordering and subset of the lanes it has, and nothing that repeats
    one: `v.xyz()`, `v.zyx()`, `v.yx()`, but not `v.xxyy()`. The exhaustive
    product is 996 methods for the three sizes and almost none of them are
    ever called -- it is not free, since every one is compiled by everybody who
    depends on this crate.

    A shader that does want a repeating swizzle can still write `v.xxyy` in the
    field spelling, which the transpiler accepts; `rustc` will not check that
    one, which is the trade.

    Single components are fields under `xyzw`, so those start at two lanes.
    """
    for n in range(2, size + 1):
        for combo in permutations(range(size), n):
            yield combo, XYZW
    for combo in RGBA_ALIASES:
        if max(combo) < size:
            yield combo, RGBA


def vname(size, scalar="T"):
    """The type as the generated code writes it: `Vec3<T>`, `Vec3<i32>`."""
    return f"Vec{size}<{scalar}>"


def vctor(size):
    """The function that builds one, whatever its scalar: `vec3`."""
    return f"vec{size}"


ZERO_ONE = {"f32": ("0.0", "1.0"), "i32": ("0", "1"), "u32": ("0", "1"), "bool": ("false", "true")}

out = []
w = out.append

w('''//! Vector types.
//!
//! Generated by `generate_vector.py`, then `cargo fmt`; edit that, not this.
//!
//! `Vec3<T>` is WGSL's `vec3<T>`, and a bare `Vec3` is `Vec3<f32>`, as `vec3f`
//! is. `vec3(x, y, z)` builds one of whatever its components are:
//! `vec3(0.0, 1.0, 0.0)` is a `Vec3<f32>` and `vec3(1, 2, 3)` a `Vec3<i32>`, as
//! WGSL's literals go, and `vec3::<u32>(1, 2, 3)` says which outright:
//!
//! ```
//! use synaga_shader::*;
//!
//! let up = vec3(0.0, 1.0, 0.0);
//! let cell = vec2(3, 4);
//! let size = vec2::<u32>(640, 480);
//! # use core::any::{type_name, type_name_of_val as of};
//! # assert_eq!(of(&up), type_name::<Vec3<f32>>());
//! # assert_eq!(of(&cell), type_name::<Vec2<i32>>());
//! # assert_eq!(of(&size), type_name::<Vec2<u32>>());
//! ```
//!
//! Component access splits two ways: a single `x`/`y`/`z`/`w` is a field, so
//! `v.x` reads and `v.x = 1.0` writes, while every other swizzle is a method.
//! Rust has no way to give one piece of memory a hundred overlapping names, so
//! `v.xyz` has to be `v.xyz()`; `.r`/`.g`/`.b`/`.a` are methods for the same
//! reason, since they would alias the `x`/`y`/`z`/`w` fields.
//!
//! Comparisons are methods too. `a < b` in a shader yields one bool per lane,
//! and Rust's `PartialOrd` yields a single `bool`, so the lane-wise forms are
//! spelled `cmplt`, `cmple`, and so on, as glam spells them.
//!
//! So is the math, named as `f32` names it, and as glam does for what only a
//! vector has. `cast` converts the lanes, as `as` converts a scalar:
//!
//! ```
//! use synaga_shader::*;
//!
//! fn lambert(normal: Vec3, light: Vec3) -> f32 {
//!     normal.normalize().dot(light).max(0.0)
//! }
//! fn inside(p: Vec2<i32>, extent: Vec2<i32>) -> bool {
//!     p.cmpge(Vec2::ZERO).all() && p.cmplt(extent).all()
//! }
//! fn texel(uv: Vec2, size: Vec2<u32>) -> Vec2<i32> {
//!     (uv * size.cast::<f32>()).cast()
//! }
//! # assert_eq!(lambert(vec3(0.0, 0.0, 2.0), vec3(0.0, 0.6, 0.8)), 0.8);
//! # assert!(inside(vec2(1, 2), vec2(4, 4)) && !inside(vec2(-1, 2), vec2(4, 4)));
//! # assert_eq!(texel(vec2(0.5, 0.25), vec2(640, 480)), vec2(320, 120));
//! ```
//!
//! Each means what the Rust method of that name means, which for two of them
//! is not what the GPU builtin of that name does: `v.fract()` is
//! `v - v.trunc()`, and the GPU's rounding is `v.round_ties_even()`. The free
//! functions, `fract(v)` and `round(v)`, are the GPU's.
//!
//! All of it runs on the CPU too, lane by lane, as Rust runs the scalar
//! code: `+` on a `Vec3<u32>` panics on overflow where `+` on a `u32` would,
//! and `wrapping_add` wraps, as the GPU's `+` does.''')
w("")
w("use core::ops::*;")
w("")
w('''/// What a vector's lanes hold: `f32`, `i32`, `u32` or `bool`.
///
/// The bound is also what makes `vec3(0.0, 1.0, 0.0)` a `Vec3<f32>`: `f32` is
/// the one float type that is a `Scalar`, so Rust gives the literals that type
/// rather than its usual `f64`.
pub trait Scalar: Copy + PartialEq + PartialOrd + sealed::Sealed {
    /// Zero, or `false`.
    const ZERO: Self;
    /// One, or `true`.
    const ONE: Self;
}

mod sealed {
    /// How a lane converts to another, as `as` converts a primitive. Out of
    /// reach outside this crate, so it adds no methods to `f32` and the rest.
    pub trait Sealed: Copy {
        fn to_f32(self) -> f32;
        fn to_i32(self) -> i32;
        fn to_u32(self) -> u32;
        /// `true` unless it is zero, as the GPU's `bool(x)`.
        fn to_bool(self) -> bool;
        /// `lane as Self`.
        fn convert<S: Sealed>(lane: S) -> Self;
    }
''')
# Rust's `as` between the numeric lanes: a float saturates into an integer,
# NaN becoming zero, and `i32` and `u32` reinterpret each other's bits.
CONVERT = {
    "f32": {"f32": "self", "i32": "self as i32", "u32": "self as u32", "bool": "self != 0.0"},
    "i32": {"f32": "self as f32", "i32": "self", "u32": "self as u32", "bool": "self != 0"},
    "u32": {"f32": "self as f32", "i32": "self as i32", "u32": "self", "bool": "self != 0"},
    "bool": {"f32": "self as u32 as f32", "i32": "self as i32", "u32": "self as u32", "bool": "self"},
}
for scalar in SCALARS:
    w(f"    impl Sealed for {scalar} {{")
    for target in SCALARS:
        w(f"        #[inline] fn to_{target}(self) -> {target} {{ {CONVERT[scalar][target]} }}")
    w(f"        #[inline] fn convert<S: Sealed>(lane: S) -> Self {{ lane.to_{scalar}() }}")
    w("    }")
w("}")
w("")
for scalar in SCALARS:
    zero, one = ZERO_ONE[scalar]
    w(f"impl Scalar for {scalar} {{ const ZERO: Self = {zero}; const ONE: Self = {one}; }}")
w("")
w(MACRO)
w("")

for size in SIZES:
    name = f"Vec{size}"
    ctor = vctor(size)
    comps = XYZW[:size]
    fields = ", ".join(f"pub {c}: T" for c in comps)
    args = ", ".join(f"{c}: T" for c in comps)
    init = ", ".join(comps)

    w(f"/// `vec{size}<T>` in WGSL. A bare `{name}` is `{name}<f32>`, WGSL's `vec{size}f`.")
    w("#[derive(Clone, Copy, Debug, Default, PartialEq)]")
    w("#[repr(C)]")
    w(f"pub struct {name}<T = f32> {{ {fields} }}")
    w("")
    w(f"/// Build a [`{name}`] from its components.")
    w("#[inline]")
    w(f"pub const fn {ctor}<T: Scalar>({args}) -> {vname(size)} {{ {name} {{ {init} }} }}")
    w("")
    w(f"impl<T: Scalar> {vname(size)} {{")
    w(f"    pub const ZERO: Self = {ctor}({', '.join(['T::ZERO'] * size)});")
    w(f"    pub const ONE: Self = {ctor}({', '.join(['T::ONE'] * size)});")
    w("")
    w("    /// Every lane set to `v`.")
    w("    #[inline]")
    w(f"    pub const fn splat(v: T) -> Self {{ {ctor}({', '.join(['v'] * size)}) }}")
    w("")
    w("    /// Each lane converted to `U`, as `as` converts a scalar: `v.cast::<i32>()`")
    w(f"    /// is WGSL's `vec{size}<i32>(v)`. Into `bool`, a lane is `true` unless it is zero.")
    w("    #[inline]")
    w(f"    pub fn cast<U: Scalar>(self) -> {vname(size, 'U')} {{ self.map_lanes(U::convert) }}")
    lanes_of = lambda *vs: ", ".join(f"f({', '.join(v + '.' + c for v in vs)})" for c in comps)
    w("")
    w("    /// `f` of each lane.")
    w("    #[inline]")
    w(f"    pub(crate) fn map_lanes<U: Scalar>(self, f: impl Fn(T) -> U) -> {vname(size, 'U')} {{")
    w(f"        {ctor}({lanes_of('self')})")
    w("    }")
    w("")
    w("    /// `f` of each pair of lanes.")
    w("    #[inline]")
    w(f"    pub(crate) fn zip_lanes<U: Scalar, V: Scalar>(self, rhs: {vname(size, 'U')}, f: impl Fn(T, U) -> V) -> {vname(size, 'V')} {{")
    w(f"        {ctor}({lanes_of('self', 'rhs')})")
    w("    }")
    w("")
    w("    /// `f` of each three lanes.")
    w("    #[inline]")
    w(f"    pub(crate) fn zip3_lanes(self, b: Self, c: Self, f: impl Fn(T, T, T) -> T) -> Self {{")
    w(f"        {ctor}({lanes_of('self', 'b', 'c')})")
    w("    }")
    w("")
    w("    /// The lanes, combined first to last.")
    w("    #[inline]")
    w("    pub(crate) fn reduce_lanes(self, f: impl Fn(T, T) -> T) -> T {")
    folded = "self.x"
    for c in comps[1:]:
        folded = f"f({folded}, self.{c})"
    w(f"        {folded}")
    w("    }")
    if size < 4:
        nc = XYZW[size]
        w("")
        w(f"    /// One lane wider, with `{nc}` appended. This is how a shader's")
        w(f"    /// `vec{size + 1}(v, {nc})` is spelled.")
        w("    #[inline]")
        w(f"    pub const fn extend(self, {nc}: T) -> {vname(size + 1)} {{")
        w(f"        {vctor(size + 1)}({', '.join('self.' + c for c in comps)}, {nc})")
        w("    }")
    if size > 2:
        w("")
        w("    /// One lane narrower, dropping the last.")
        w("    #[inline]")
        w(f"    pub const fn truncate(self) -> {vname(size - 1)} {{")
        w(f"        {vctor(size - 1)}({', '.join('self.' + c for c in comps[:-1])})")
        w("    }")
    for op, doc in [("cmpeq", "=="), ("cmpne", "!=")]:
        w("")
        w(f"    /// Lane-wise `{doc}`.")
        w("    #[inline]")
        w(f"    pub fn {op}(self, rhs: Self) -> {vname(size, 'bool')} {{ self.zip_lanes(rhs, |a, b| a {doc} b) }}")
    for combo, letters in swizzles(size):
        n = len(combo)
        sw = "".join(letters[i] for i in combo)
        ret = "T" if n == 1 else vname(n)
        body = (f"self.{XYZW[combo[0]]}" if n == 1
                else f"{vctor(n)}({', '.join('self.' + XYZW[i] for i in combo)})")
        w("")
        w("    #[inline]")
        w(f"    pub const fn {sw}(self) -> {ret} {{ {body} }}")
    w("}")
    w("")
    # Real on the CPU, where a host reads a shared struct's lanes.
    lanes = ", ".join(f"{i} => &self.{c}" for i, c in enumerate(comps))
    lanes_mut = ", ".join(f"{i} => &mut self.{c}" for i, c in enumerate(comps))
    oob = f'_ => panic!("lane {{index}} of a {name}")'
    w(f"impl<T: Scalar> Index<usize> for {vname(size)} {{")
    w("    type Output = T;")
    w("    #[inline]")
    w(f"    fn index(&self, index: usize) -> &T {{ match index {{ {lanes}, {oob} }} }}")
    w("}")
    w(f"impl<T: Scalar> IndexMut<usize> for {vname(size)} {{")
    w("    #[inline]")
    w(f"    fn index_mut(&mut self, index: usize) -> &mut T {{ match index {{ {lanes_mut}, {oob} }} }}")
    w("}")
    # two-vector concatenation, for a shader's vec4(vec2, vec2)
    if size == 4:
        two = vname(2)
        w(f"impl<T: Scalar> From<({two}, {two})> for {vname(4)} {{")
        w("    #[inline]")
        w(f"    fn from((a, b): ({two}, {two})) -> Self {{ vec4(a.x, a.y, b.x, b.y) }}")
        w("}")
    w("")

    for scalar in SCALARS:
        ty = vname(size, scalar)
        # Lane-wise ordering, for everything but `bool`.
        if scalar != "bool":
            w(f"impl {ty} {{")
            for i, (op, doc) in enumerate([("cmplt", "<"), ("cmple", "<="),
                                           ("cmpgt", ">"), ("cmpge", ">=")]):
                if i:
                    w("")
                w(f"    /// Lane-wise `{doc}`.")
                w("    #[inline]")
                w(f"    pub fn {op}(self, rhs: Self) -> {vname(size, 'bool')} {{ self.zip_lanes(rhs, |a, b| a {doc} b) }}")
            w("}")
        # Operators are uniform per type, so they go through a macro rather
        # than 2,500 lines of impls that differ only in a name.
        traits = []
        if scalar != "bool":
            traits.append("arith")
        if scalar in ("i32", "u32", "bool"):
            traits.append("bitwise")
        if scalar in ("i32", "u32"):
            traits.append("shift")
        if scalar in ("f32", "i32"):
            traits.append("neg")
        if scalar in ("i32", "u32", "bool"):
            traits.append("not")
        w(f"vector_ops!({ty}, {scalar}, {vname(size, 'u32')}{''.join(', ' + t for t in traits)});")
        groups = {"f32": ["ord", "float"], "i32": ["ord", "signed", "wrapping"],
                  "u32": ["ord", "wrapping"], "bool": ["bool"]}
        w(f"vector_math!({ty}, {scalar}{''.join(', ' + g for g in groups[scalar])});")
        if size == 3 and scalar == "f32":
            w(f"impl {ty} {{")
            w("    /// The cross product, perpendicular to both.")
            w("    #[inline]")
            w("    pub fn cross(self, rhs: Self) -> Self {")
            w("        vec3(")
            w("            self.y * rhs.z - rhs.y * self.z,")
            w("            self.z * rhs.x - rhs.z * self.x,")
            w("            self.x * rhs.y - rhs.x * self.y,")
            w("        )")
            w("    }")
            w("}")
        w("")

# What the host needs to fill a struct it shares with a shader: arrays and
# mint's vectors, whichever it has, and bytemuck to upload the result. These
# run on the CPU.
for size in SIZES:
    name = f"Vec{size}"
    comps = XYZW[:size]
    lanes = ", ".join(comps)
    of = ", ".join(f"v.{c}" for c in comps)
    w(f"impl<T: Scalar> From<[T; {size}]> for {vname(size)} {{")
    w("    #[inline]")
    w(f"    fn from([{lanes}]: [T; {size}]) -> Self {{ {name} {{ {lanes} }} }}")
    w("}")
    w(f"impl<T: Scalar> From<{vname(size)}> for [T; {size}] {{")
    w("    #[inline]")
    w(f"    fn from(v: {vname(size)}) -> Self {{ [{of}] }}")
    w("}")
    w('#[cfg(feature = "mint")]')
    w(f"impl<T: Scalar> From<mint::Vector{size}<T>> for {vname(size)} {{")
    w("    #[inline]")
    w(f"    fn from(v: mint::Vector{size}<T>) -> Self {{ {name} {{ {', '.join(f'{c}: v.{c}' for c in comps)} }} }}")
    w("}")
    if size == 4:
        w('#[cfg(feature = "mint")]')
        w(f"impl<T: Scalar> From<mint::Quaternion<T>> for {vname(size)} {{")
        w("    /// `xyz` is the vector part and `w` the scalar, as a shader keeps one.")
        w("    #[inline]")
        w(f"    fn from(q: mint::Quaternion<T>) -> Self {{ {name} {{ x: q.v.x, y: q.v.y, z: q.v.z, w: q.s }} }}")
        w("}")
    w('#[cfg(feature = "mint")]')
    w(f"impl<T: Scalar> From<{vname(size)}> for mint::Vector{size}<T> {{")
    w("    #[inline]")
    w(f"    fn from(v: {vname(size)}) -> Self {{ mint::Vector{size} {{ {', '.join(f'{c}: v.{c}' for c in comps)} }} }}")
    w("}")
    w(f"// SAFETY: `#[repr(C)]` lanes of one type, so there is no padding, and")
    w(f"// all zeroes, like any bytes of a `Pod` lane type, are valid lanes.")
    w('#[cfg(feature = "bytemuck")]')
    w(f"unsafe impl<T: Scalar + bytemuck::Zeroable> bytemuck::Zeroable for {vname(size)} {{}}")
    w('#[cfg(feature = "bytemuck")]')
    w(f"unsafe impl<T: Scalar + bytemuck::Pod> bytemuck::Pod for {vname(size)} {{}}")
w("")

# component-type conversions, for a shader's `vec3<f32>(v)`
for size in SIZES:
    for a in SCALARS:
        for b in SCALARS:
            if a == b:
                continue
            w(f"impl From<{vname(size, a)}> for {vname(size, b)} {{")
            w("    #[inline]")
            w(f"    fn from(v: {vname(size, a)}) -> Self {{ v.cast() }}")
            w("}")

import pathlib
target = pathlib.Path(__file__).resolve().parent / "src" / "vector.rs"
target.write_text("\n".join(out) + "\n")
print(f"{len(out)} lines -> {target}; now run `cargo fmt`")
