//! Matrix types: column-major, as WGSL and Naga have them.
//!
//! `matN(c0, c1, ...)` takes column vectors. A shader's column-major scalar
//! form has the same name at a different arity, which Rust cannot express, so
//! that one is [`from_cols_array`](mat2::from_cols_array).

use core::ops::*;

use crate::unimplemented_on_cpu;
use crate::vector::{vec2, vec3, vec4};

macro_rules! matrix {
    ($name:ident, $cols:literal, $col:ident, $row:ident, $($c:ident),+) => {
        /// Column-major matrix.
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        #[repr(C)]
        #[allow(non_camel_case_types)]
        pub struct $name { $(pub $c: $col),+ }

        /// Build from column vectors.
        #[allow(non_snake_case)]
        #[inline]
        pub const fn $name($($c: $col),+) -> $name { $name { $($c),+ } }

        impl $name {
            pub const ZERO: Self = $name { $($c: $col::ZERO),+ };

            /// Column-major scalars, which a shader spells as another arity of
            /// the same constructor.
            #[inline]
            pub fn from_cols_array(cols: &[f32]) -> Self { unimplemented_on_cpu() }
        }

        impl Index<usize> for $name {
            type Output = $col;
            #[inline]
            fn index(&self, index: usize) -> &$col { unimplemented_on_cpu() }
        }
        impl IndexMut<usize> for $name {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut $col { unimplemented_on_cpu() }
        }

        impl Add for $name {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self { unimplemented_on_cpu() }
        }
        impl Sub for $name {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self { unimplemented_on_cpu() }
        }
        impl Mul<f32> for $name {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: f32) -> Self { unimplemented_on_cpu() }
        }
        impl Mul<$name> for f32 {
            type Output = $name;
            #[inline]
            fn mul(self, rhs: $name) -> $name { unimplemented_on_cpu() }
        }
        /// `m * v`: the matrix transforms the column vector.
        impl Mul<$col> for $name {
            type Output = $row;
            #[inline]
            fn mul(self, rhs: $col) -> $row { unimplemented_on_cpu() }
        }
    };
}

matrix!(mat2x2, 2, vec2, vec2, x_axis, y_axis);
matrix!(mat3x2, 3, vec2, vec3, x_axis, y_axis, z_axis);
matrix!(mat4x2, 4, vec2, vec4, x_axis, y_axis, z_axis, w_axis);
matrix!(mat2x3, 2, vec3, vec2, x_axis, y_axis);
matrix!(mat3x3, 3, vec3, vec3, x_axis, y_axis, z_axis);
matrix!(mat4x3, 4, vec3, vec4, x_axis, y_axis, z_axis, w_axis);
matrix!(mat2x4, 2, vec4, vec2, x_axis, y_axis);
matrix!(mat3x4, 3, vec4, vec3, x_axis, y_axis, z_axis);
matrix!(mat4x4, 4, vec4, vec4, x_axis, y_axis, z_axis, w_axis);

/// Square matrices, as a shader spells them.
#[allow(non_camel_case_types)]
pub type mat2 = mat2x2;
#[allow(non_camel_case_types)]
pub type mat3 = mat3x3;
#[allow(non_camel_case_types)]
pub type mat4 = mat4x4;

/// `mat2(c0, c1)` and friends, under the square names.
#[allow(non_snake_case)]
#[inline]
pub const fn mat2(x_axis: vec2, y_axis: vec2) -> mat2 {
    mat2x2(x_axis, y_axis)
}
#[allow(non_snake_case)]
#[inline]
pub const fn mat3(x_axis: vec3, y_axis: vec3, z_axis: vec3) -> mat3 {
    mat3x3(x_axis, y_axis, z_axis)
}
#[allow(non_snake_case)]
#[inline]
pub const fn mat4(x_axis: vec4, y_axis: vec4, z_axis: vec4, w_axis: vec4) -> mat4 {
    mat4x4(x_axis, y_axis, z_axis, w_axis)
}

/// Square matrix products, where the result is the same type.
macro_rules! square_mul {
    ($name:ident) => {
        impl Mul for $name {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                unimplemented_on_cpu()
            }
        }
    };
}
square_mul!(mat2x2);
square_mul!(mat3x3);
square_mul!(mat4x4);

/// `vec4 * mat3x4 -> vec3`, the row-vector product an affine skinning matrix uses.
impl Mul<mat3x4> for vec4 {
    type Output = vec3;
    #[inline]
    fn mul(self, _rhs: mat3x4) -> vec3 {
        crate::unimplemented_on_cpu()
    }
}

/// `mat3x2 * vec3 -> vec2`, a 2-row matrix times a column.
impl Mul<vec3> for mat3x2 {
    type Output = vec2;
    #[inline]
    fn mul(self, _rhs: vec3) -> vec2 {
        crate::unimplemented_on_cpu()
    }
}

/// `mat4x3 * vec4 -> vec3`, an affine matrix times a homogeneous point.
impl Mul<vec4> for mat4x3 {
    type Output = vec3;
    #[inline]
    fn mul(self, _rhs: vec4) -> vec3 {
        crate::unimplemented_on_cpu()
    }
}

/// `mat4x3 * mat3x4 -> mat3x3`, the object-to-world linear part of a hit.
impl Mul<mat3x4> for mat4x3 {
    type Output = mat3x3;
    #[inline]
    fn mul(self, _rhs: mat3x4) -> mat3x3 {
        crate::unimplemented_on_cpu()
    }
}
