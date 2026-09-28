//! Matrix types: column-major, as WGSL and Naga have them.
//!
//! `matN(c0, c1, ...)` takes column vectors. A shader's column-major scalar
//! form has the same name at a different arity, which Rust cannot express, so
//! that one is [`from_cols_array`](Mat2::from_cols_array).

use core::ops::*;

use crate::unimplemented_on_cpu;
use crate::vector::{Vec2, Vec3, Vec4};

/// A matrix of `$cols` columns, each a `$col` of `$rows` lanes. `$row` has one
/// lane per column: what a row is, and what the matrix multiplies.
///
/// A column is laid out as the GPU lays it out, which for three lanes is four:
/// a `$pad` lane follows each `Vec3` column, so that a struct the host shares
/// can hold any of these.
macro_rules! matrix {
    ($name:ident, $ctor:ident, $cols:literal, $rows:literal, $col:ident, $row:ident,
     $mint:ident, $($c:ident: $m:ident $(+ $pad:ident)?),+) => {
        /// Column-major matrix.
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        #[repr(C)]
        pub struct $name { $(pub $c: $col, $($pad: f32,)?)+ }

        /// Build from column vectors.
        #[inline]
        pub const fn $ctor($($c: $col),+) -> $name { $name { $($c, $($pad: 0.0,)?)+ } }

        impl $name {
            pub const ZERO: Self = $name { $($c: $col::ZERO, $($pad: 0.0,)?)+ };

            /// Column-major scalars, which a shader spells as another arity of
            /// the same constructor.
            #[inline]
            pub fn from_cols_array(cols: &[f32]) -> Self { unimplemented_on_cpu() }
        }

        // A column, on the CPU as in a shader.
        impl Index<usize> for $name {
            type Output = $col;
            #[inline]
            fn index(&self, index: usize) -> &$col { [$(&self.$c),+][index] }
        }
        impl IndexMut<usize> for $name {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut $col {
                let columns = [$(&mut self.$c),+];
                let count = columns.len();
                columns.into_iter().nth(index).unwrap_or_else(|| {
                    panic!("column {index} of a {count}-column matrix")
                })
            }
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
        /// `m * v`: the matrix transforms a column vector, one lane per column,
        /// into one lane per row.
        impl Mul<$row> for $name {
            type Output = $col;
            #[inline]
            fn mul(self, rhs: $row) -> $col { unimplemented_on_cpu() }
        }
        /// `v * m`: a row vector, one lane per row, times the matrix.
        impl Mul<$name> for $col {
            type Output = $row;
            #[inline]
            fn mul(self, rhs: $name) -> $row { unimplemented_on_cpu() }
        }

        // What the host needs to fill a struct it shares with a shader. These
        // run on the CPU.
        impl From<[[f32; $rows]; $cols]> for $name {
            /// Column after column.
            #[inline]
            fn from([$($c),+]: [[f32; $rows]; $cols]) -> Self { $ctor($($c.into()),+) }
        }
        impl From<$name> for [[f32; $rows]; $cols] {
            #[inline]
            fn from(m: $name) -> Self { [$(m.$c.into()),+] }
        }
        #[cfg(feature = "mint")]
        impl From<mint::$mint<f32>> for $name {
            #[inline]
            fn from(m: mint::$mint<f32>) -> Self { $ctor($(m.$m.into()),+) }
        }
        #[cfg(feature = "mint")]
        impl From<$name> for mint::$mint<f32> {
            #[inline]
            fn from(m: $name) -> Self { mint::$mint { $($m: m.$c.into()),+ } }
        }
        // SAFETY: `#[repr(C)]` columns of one `Pod` type, and `f32` lanes
        // where the GPU pads them, so no padding.
        #[cfg(feature = "bytemuck")]
        unsafe impl bytemuck::Zeroable for $name {}
        #[cfg(feature = "bytemuck")]
        unsafe impl bytemuck::Pod for $name {}
    };
}

// mint names a matrix by rows, then columns; WGSL and these by columns first.
matrix!(Mat2, mat2, 2, 2, Vec2, Vec2, ColumnMatrix2, x_axis: x, y_axis: y);
matrix!(Mat3x2, mat3x2, 3, 2, Vec2, Vec3, ColumnMatrix2x3, x_axis: x, y_axis: y, z_axis: z);
matrix!(Mat4x2, mat4x2, 4, 2, Vec2, Vec4, ColumnMatrix2x4, x_axis: x, y_axis: y, z_axis: z, w_axis: w);
matrix!(Mat2x3, mat2x3, 2, 3, Vec3, Vec2, ColumnMatrix3x2,
    x_axis: x + x_pad, y_axis: y + y_pad);
matrix!(Mat3, mat3, 3, 3, Vec3, Vec3, ColumnMatrix3,
    x_axis: x + x_pad, y_axis: y + y_pad, z_axis: z + z_pad);
matrix!(Mat4x3, mat4x3, 4, 3, Vec3, Vec4, ColumnMatrix3x4,
    x_axis: x + x_pad, y_axis: y + y_pad, z_axis: z + z_pad, w_axis: w + w_pad);
matrix!(Mat2x4, mat2x4, 2, 4, Vec4, Vec2, ColumnMatrix4x2, x_axis: x, y_axis: y);
matrix!(Mat3x4, mat3x4, 3, 4, Vec4, Vec3, ColumnMatrix4x3, x_axis: x, y_axis: y, z_axis: z);
matrix!(Mat4, mat4, 4, 4, Vec4, Vec4, ColumnMatrix4, x_axis: x, y_axis: y, z_axis: z, w_axis: w);

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

/// `Mat4x3 * Mat3x4 -> Mat3`, the object-to-world linear part of a hit.
impl Mul<Mat3x4> for Mat4x3 {
    type Output = Mat3;
    #[inline]
    fn mul(self, _rhs: Mat3x4) -> Mat3 {
        crate::unimplemented_on_cpu()
    }
}
