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
        vector_ops!(@lanewise BitAnd, bitand, BitAndAssign, bitand_assign, $name);
        vector_ops!(@lanewise BitOr, bitor, BitOrAssign, bitor_assign, $name);
        vector_ops!(@lanewise BitXor, bitxor, BitXorAssign, bitxor_assign, $name);
    };
    (@group shift, $name:ty, $scalar:ty, $shift:ty) => {
        vector_ops!(@shift Shl, shl, $name, $shift);
        vector_ops!(@shift Shr, shr, $name, $shift);
    };
    (@group neg, $name:ty, $scalar:ty, $shift:ty) => {
        impl Neg for $name {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self { unimplemented_on_cpu() }
        }
    };
    (@group not, $name:ty, $scalar:ty, $shift:ty) => {
        impl Not for $name {
            type Output = Self;
            #[inline]
            fn not(self) -> Self { unimplemented_on_cpu() }
        }
    };

    // Arithmetic also works against a scalar, from either side: a shader
    // writes both `v * 2.0` and `2.0 * v`.
    (@scalar_too $trait:ident, $method:ident, $assign:ident, $assign_fn:ident,
     $name:ty, $scalar:ty) => {
        vector_ops!(@lanewise $trait, $method, $assign, $assign_fn, $name);
        impl $trait<$scalar> for $name {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: $scalar) -> Self { unimplemented_on_cpu() }
        }
        impl $trait<$name> for $scalar {
            type Output = $name;
            #[inline]
            fn $method(self, rhs: $name) -> $name { unimplemented_on_cpu() }
        }
        impl $assign<$scalar> for $name {
            #[inline]
            fn $assign_fn(&mut self, rhs: $scalar) { unimplemented_on_cpu() }
        }
    };
    (@lanewise $trait:ident, $method:ident, $assign:ident, $assign_fn:ident, $name:ty) => {
        impl $trait for $name {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: Self) -> Self { unimplemented_on_cpu() }
        }
        impl $assign for $name {
            #[inline]
            fn $assign_fn(&mut self, rhs: Self) { unimplemented_on_cpu() }
        }
    };
    (@shift $trait:ident, $method:ident, $name:ty, $shift:ty) => {
        impl $trait<$shift> for $name {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: $shift) -> Self { unimplemented_on_cpu() }
        }
        impl $trait<u32> for $name {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: u32) -> Self { unimplemented_on_cpu() }
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
//! spelled `cmplt`, `cmple`, and so on, as glam spells them.''')
w("")
w("use core::ops::*;")
w("")
w("use crate::unimplemented_on_cpu;")
w("")
w('''/// What a vector's lanes hold: `f32`, `i32`, `u32` or `bool`.
///
/// The bound is also what makes `vec3(0.0, 1.0, 0.0)` a `Vec3<f32>`: `f32` is
/// the one float type that is a `Scalar`, so Rust gives the literals that type
/// rather than its usual `f64`.
pub trait Scalar: Copy + sealed::Sealed {
    /// Zero, or `false`.
    const ZERO: Self;
    /// One, or `true`.
    const ONE: Self;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for i32 {}
    impl Sealed for u32 {}
    impl Sealed for bool {}
}
''')
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
        w(f"    pub fn {op}(self, rhs: Self) -> {vname(size, 'bool')} {{ unimplemented_on_cpu() }}")
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
    w(f"impl<T: Scalar> Index<usize> for {vname(size)} {{")
    w("    type Output = T;")
    w("    #[inline]")
    w("    fn index(&self, index: usize) -> &T { unimplemented_on_cpu() }")
    w("}")
    w(f"impl<T: Scalar> IndexMut<usize> for {vname(size)} {{")
    w("    #[inline]")
    w("    fn index_mut(&mut self, index: usize) -> &mut T { unimplemented_on_cpu() }")
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
                w(f"    pub fn {op}(self, rhs: Self) -> {vname(size, 'bool')} {{ unimplemented_on_cpu() }}")
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
        w("")

# component-type conversions, for a shader's `vec3<f32>(v)`
for size in SIZES:
    for a in SCALARS:
        for b in SCALARS:
            if a == b:
                continue
            w(f"impl From<{vname(size, a)}> for {vname(size, b)} {{")
            w("    #[inline]")
            w(f"    fn from(v: {vname(size, a)}) -> Self {{ unimplemented_on_cpu() }}")
            w("}")

import pathlib
target = pathlib.Path(__file__).resolve().parent / "src" / "vector.rs"
target.write_text("\n".join(out) + "\n")
print(f"{len(out)} lines -> {target}; now run `cargo fmt`")
