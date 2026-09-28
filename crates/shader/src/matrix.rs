//! Matrix types: column-major, as WGSL and Naga have them.
//!
//! `matN(c0, c1, ...)` takes column vectors. A shader's column-major scalar
//! form has the same name at a different arity, which Rust cannot express, so
//! that one is [`from_cols_array`](Mat2::from_cols_array).

use core::ops::*;

use crate::unimplemented_on_cpu;
use crate::vector::{Vec2, Vec3, Vec4};

macro_rules! matrix {
    ($name:ident, $ctor:ident, $cols:literal, $col:ident, $row:ident, $($c:ident),+) => {
        /// Column-major matrix.
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        #[repr(C)]
        pub struct $name { $(pub $c: $col),+ }

        /// Build from column vectors.
        #[inline]
        pub const fn $ctor($($c: $col),+) -> $name { $name { $($c),+ } }

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

matrix!(Mat2, mat2, 2, Vec2, Vec2, x_axis, y_axis);
matrix!(Mat3x2, mat3x2, 3, Vec2, Vec3, x_axis, y_axis, z_axis);
matrix!(Mat4x2, mat4x2, 4, Vec2, Vec4, x_axis, y_axis, z_axis, w_axis);
matrix!(Mat2x3, mat2x3, 2, Vec3, Vec2, x_axis, y_axis);
matrix!(Mat3, mat3, 3, Vec3, Vec3, x_axis, y_axis, z_axis);
matrix!(Mat4x3, mat4x3, 4, Vec3, Vec4, x_axis, y_axis, z_axis, w_axis);
matrix!(Mat2x4, mat2x4, 2, Vec4, Vec2, x_axis, y_axis);
matrix!(Mat3x4, mat3x4, 3, Vec4, Vec3, x_axis, y_axis, z_axis);
matrix!(Mat4, mat4, 4, Vec4, Vec4, x_axis, y_axis, z_axis, w_axis);

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
square_mul!(Mat2);
square_mul!(Mat3);
square_mul!(Mat4);

/// `Vec4 * Mat3x4 -> Vec3`, the row-vector product an affine skinning matrix uses.
impl Mul<Mat3x4> for Vec4 {
    type Output = Vec3;
    #[inline]
    fn mul(self, _rhs: Mat3x4) -> Vec3 {
        crate::unimplemented_on_cpu()
    }
}

/// `Mat3x2 * Vec3 -> Vec2`, a 2-row matrix times a column.
impl Mul<Vec3> for Mat3x2 {
    type Output = Vec2;
    #[inline]
    fn mul(self, _rhs: Vec3) -> Vec2 {
        crate::unimplemented_on_cpu()
    }
}

/// `Mat4x3 * Vec4 -> Vec3`, an affine matrix times a homogeneous point.
impl Mul<Vec4> for Mat4x3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, _rhs: Vec4) -> Vec3 {
        crate::unimplemented_on_cpu()
    }
}

/// `Mat4x3 * Mat3x4 -> Mat3`, the object-to-world linear part of a hit.
impl Mul<Mat3x4> for Mat4x3 {
    type Output = Mat3;
    #[inline]
    fn mul(self, _rhs: Mat3x4) -> Mat3 {
        crate::unimplemented_on_cpu()
    }
}
