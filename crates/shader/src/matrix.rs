//! Matrix types: column-major, as WGSL and Naga have them.
//!
//! `matN(c0, c1, ...)` takes column vectors. A shader's column-major scalar
//! form has the same name at a different arity, which Rust cannot express, so
//! that one is [`from_cols_array`](Mat2::from_cols_array).

use core::ops::*;

use crate::vector::{Vec2, Vec3, Vec4};

/// A matrix of `$cols` columns, each a `$col` of `$rows` lanes. `$row` has one
/// lane per column: what a row is, and what the matrix multiplies. `$m` names
/// both mint's column and the lane of a `$row` that column meets. `$trans` is
/// the matrix with columns and rows swapped.
///
/// A column is laid out as the GPU lays it out, which for three lanes is four:
/// a `$pad` lane follows each `Vec3` column, so that a struct the host shares
/// can hold any of these.
macro_rules! matrix {
    ($name:ident, $ctor:ident, $cols:literal, $rows:literal, $col:ident, $row:ident,
     $mint:ident, $trans:ident, $($c:ident: $m:ident $(+ $pad:ident)?),+) => {
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
            pub fn from_cols_array(cols: &[f32]) -> Self {
                assert_eq!(cols.len(), $cols * $rows, "a {}x{} matrix", $cols, $rows);
                let mut m = Self::ZERO;
                for (c, column) in cols.chunks_exact($rows).enumerate() {
                    for (r, &value) in column.iter().enumerate() {
                        m[c][r] = value;
                    }
                }
                m
            }
        }

        impl Transposable for $name {
            type Output = $trans;
            #[inline]
            fn transpose(self) -> $trans {
                let mut t = $trans::ZERO;
                for c in 0..$cols {
                    for r in 0..$rows {
                        t[r][c] = self[c][r];
                    }
                }
                t
            }
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
            fn add(self, rhs: Self) -> Self { $ctor($(self.$c + rhs.$c),+) }
        }
        impl Sub for $name {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self { $ctor($(self.$c - rhs.$c),+) }
        }
        impl Mul<f32> for $name {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: f32) -> Self { $ctor($(self.$c * rhs),+) }
        }
        impl Mul<$name> for f32 {
            type Output = $name;
            #[inline]
            fn mul(self, rhs: $name) -> $name { $ctor($(self * rhs.$c),+) }
        }
        /// `m * v`: the matrix transforms a column vector, one lane per column,
        /// into one lane per row.
        impl Mul<$row> for $name {
            type Output = $col;
            #[inline]
            fn mul(self, rhs: $row) -> $col {
                let mut sum = $col::<f32>::ZERO;
                $(sum += self.$c * rhs.$m;)+
                sum
            }
        }
        /// `v * m`: a row vector, one lane per row, times the matrix.
        impl Mul<$name> for $col {
            type Output = $row;
            #[inline]
            fn mul(self, rhs: $name) -> $row { $row { $($m: self.dot(rhs.$c)),+ } }
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
matrix!(Mat2, mat2, 2, 2, Vec2, Vec2, ColumnMatrix2, Mat2, x_axis: x, y_axis: y);
matrix!(Mat3x2, mat3x2, 3, 2, Vec2, Vec3, ColumnMatrix2x3, Mat2x3,
    x_axis: x, y_axis: y, z_axis: z);
matrix!(Mat4x2, mat4x2, 4, 2, Vec2, Vec4, ColumnMatrix2x4, Mat2x4,
    x_axis: x, y_axis: y, z_axis: z, w_axis: w);
matrix!(Mat2x3, mat2x3, 2, 3, Vec3, Vec2, ColumnMatrix3x2, Mat3x2,
    x_axis: x + x_pad, y_axis: y + y_pad);
matrix!(Mat3, mat3, 3, 3, Vec3, Vec3, ColumnMatrix3, Mat3,
    x_axis: x + x_pad, y_axis: y + y_pad, z_axis: z + z_pad);
matrix!(Mat4x3, mat4x3, 4, 3, Vec3, Vec4, ColumnMatrix3x4, Mat3x4,
    x_axis: x + x_pad, y_axis: y + y_pad, z_axis: z + z_pad, w_axis: w + w_pad);
matrix!(Mat2x4, mat2x4, 2, 4, Vec4, Vec2, ColumnMatrix4x2, Mat4x2, x_axis: x, y_axis: y);
matrix!(Mat3x4, mat3x4, 3, 4, Vec4, Vec3, ColumnMatrix4x3, Mat4x3,
    x_axis: x, y_axis: y, z_axis: z);
matrix!(Mat4, mat4, 4, 4, Vec4, Vec4, ColumnMatrix4, Mat4,
    x_axis: x, y_axis: y, z_axis: z, w_axis: w);

/// Square matrix products, where the result is the same type: each column of
/// the product is the left matrix times that column of the right.
macro_rules! square_mul {
    ($name:ident, $ctor:ident, $($c:ident),+) => {
        impl Mul for $name {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                $ctor($(self * rhs.$c),+)
            }
        }
    };
}
square_mul!(Mat2, mat2, x_axis, y_axis);
square_mul!(Mat3, mat3, x_axis, y_axis, z_axis);
square_mul!(Mat4, mat4, x_axis, y_axis, z_axis, w_axis);

/// `Mat4x3 * Mat3x4 -> Mat3`, the object-to-world linear part of a hit.
impl Mul<Mat3x4> for Mat4x3 {
    type Output = Mat3;
    #[inline]
    fn mul(self, rhs: Mat3x4) -> Mat3 {
        mat3(self * rhs.x_axis, self * rhs.y_axis, self * rhs.z_axis)
    }
}

/// A matrix `transpose` applies to: any of them.
pub trait Transposable {
    /// The matrix with columns and rows exchanged.
    type Output;
    /// Columns and rows exchanged: WGSL's `transpose(m)`.
    fn transpose(self) -> Self::Output;
}

/// A matrix `determinant` applies to: a square one.
pub trait Square {
    /// WGSL's `determinant(m)`.
    fn determinant(self) -> f32;
}

impl Square for Mat2 {
    #[inline]
    fn determinant(self) -> f32 {
        self.x_axis.x * self.y_axis.y - self.y_axis.x * self.x_axis.y
    }
}

impl Square for Mat3 {
    #[inline]
    fn determinant(self) -> f32 {
        self.z_axis.dot(self.x_axis.cross(self.y_axis))
    }
}

impl Square for Mat4 {
    fn determinant(self) -> f32 {
        let [m00, m01, m02, m03]: [f32; 4] = self.x_axis.into();
        let [m10, m11, m12, m13]: [f32; 4] = self.y_axis.into();
        let [m20, m21, m22, m23]: [f32; 4] = self.z_axis.into();
        let [m30, m31, m32, m33]: [f32; 4] = self.w_axis.into();
        // Along the first column, with the 2x2 minors of the last two.
        let a2323 = m22 * m33 - m23 * m32;
        let a1323 = m21 * m33 - m23 * m31;
        let a1223 = m21 * m32 - m22 * m31;
        let a0323 = m20 * m33 - m23 * m30;
        let a0223 = m20 * m32 - m22 * m30;
        let a0123 = m20 * m31 - m21 * m30;
        m00 * (m11 * a2323 - m12 * a1323 + m13 * a1223)
            - m01 * (m10 * a2323 - m12 * a0323 + m13 * a0223)
            + m02 * (m10 * a1323 - m11 * a0323 + m13 * a0123)
            - m03 * (m10 * a1223 - m11 * a0223 + m12 * a0123)
    }
}
