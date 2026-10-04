//! Cooperative matrices: a matrix the invocations of a subgroup hold between
//! them, and multiply as one operation. They are WGSL's `coop_mat8x8` and
//! `coop_mat16x16`, from wgpu's `wgpu_cooperative_matrix` extension.
//!
//! `a.mul_add(b, c)` is `a * b + c`, with `a` an [`A`] matrix, `b` a [`B`]
//! and `c` a [`C`], the accumulator. A matrix is loaded from a slice of its
//! scalars and stored back into one. [`load`](CoopMat8x8::load) and
//! [`store`](CoopMat8x8::store) go column by column, as WGSL's `coopLoad`
//! and `coopStore` do. The `_row_major` forms go row by row, as `coopLoadT`
//! and `coopStoreT` do. `stride` is how many scalars apart two columns, or
//! two rows, start. The scalar is `f32`, or `f16` with that feature.
//!
//! A module that has one needs Naga's `COOPERATIVE_MATRIX` capability, and
//! only a compute shader can use one.
//!
//! On the GPU each invocation of the subgroup holds a part of the matrix,
//! in a layout the GPU does not say, so all of them have to reach each
//! operation, as they reach a barrier. On the CPU a shader runs as one
//! invocation, which is a subgroup of one, so the one value holds all of it,
//! and each operation is the arithmetic it stands for. The order in which
//! `mul_add` adds its products, and how precisely, is the GPU's own; the CPU
//! adds each product to `c` in turn.
//!
//! ```
//! use synaga_shader::*;
//! let identity: Vec<f32> = (0..64).map(|i| if i % 9 == 0 { 1.0 } else { 0.0 }).collect();
//! let counting: Vec<f32> = (0..64).map(|i| i as f32).collect();
//! let a = CoopMat8x8::<f32, A>::load(&identity, 8);
//! let b = CoopMat8x8::<f32, B>::load(&counting, 8);
//! let mut out = vec![0.0; 64];
//! a.mul_add(b, CoopMat8x8::default()).store(&mut out, 8);
//! assert_eq!(out, counting);
//! ```

use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use crate::vector::Scalar;

mod sealed {
    pub trait Sealed {}
}

/// What a cooperative matrix holds: `f32`, or `f16` with that feature.
pub trait CoopScalar:
    Scalar + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + sealed::Sealed
{
}

impl sealed::Sealed for f32 {}
impl CoopScalar for f32 {}
#[cfg(feature = "f16")]
impl sealed::Sealed for half::f16 {}
#[cfg(feature = "f16")]
impl CoopScalar for half::f16 {}

/// Which operand of `a * b + c` a cooperative matrix is: [`A`], [`B`] or
/// [`C`].
pub trait Role: sealed::Sealed {}

/// The left operand of `a * b + c`.
pub enum A {}
/// The right operand of `a * b + c`.
pub enum B {}
/// The accumulator of `a * b + c`, and what it comes to.
pub enum C {}

impl sealed::Sealed for A {}
impl sealed::Sealed for B {}
impl sealed::Sealed for C {}
impl Role for A {}
impl Role for B {}
impl Role for C {}

macro_rules! cooperative_matrix {
    ($(#[$doc:meta])* $name:ident, $n:literal) => {
        $(#[$doc])*
        pub struct $name<T: CoopScalar, R: Role> {
            /// Column by column.
            columns: [[T; $n]; $n],
            role: PhantomData<R>,
        }

        impl<T: CoopScalar, R: Role> Clone for $name<T, R> {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<T: CoopScalar, R: Role> Copy for $name<T, R> {}

        /// All zeroes, as WGSL's zero value is.
        impl<T: CoopScalar, R: Role> Default for $name<T, R> {
            #[inline]
            fn default() -> Self {
                Self::from_fn(|_, _| T::ZERO)
            }
        }

        impl<T: CoopScalar, R: Role> $name<T, R> {
            /// The matrix whose column `c` starts at `data[c * stride]`:
            /// WGSL's `coopLoad`.
            pub fn load(data: &[T], stride: u32) -> Self {
                Self::from_fn(|row, column| data[column * stride as usize + row])
            }

            /// The matrix whose row `r` starts at `data[r * stride]`: WGSL's
            /// `coopLoadT`.
            pub fn load_row_major(data: &[T], stride: u32) -> Self {
                Self::from_fn(|row, column| data[row * stride as usize + column])
            }

            /// Column `c` written from `data[c * stride]` on: WGSL's
            /// `coopStore`.
            pub fn store(self, data: &mut [T], stride: u32) {
                for (column, values) in self.columns.iter().enumerate() {
                    for (row, &value) in values.iter().enumerate() {
                        data[column * stride as usize + row] = value;
                    }
                }
            }

            /// Row `r` written from `data[r * stride]` on: WGSL's
            /// `coopStoreT`.
            pub fn store_row_major(self, data: &mut [T], stride: u32) {
                for (column, values) in self.columns.iter().enumerate() {
                    for (row, &value) in values.iter().enumerate() {
                        data[row * stride as usize + column] = value;
                    }
                }
            }

            fn from_fn(element: impl Fn(usize, usize) -> T) -> Self {
                let mut columns = [[T::ZERO; $n]; $n];
                for (column, values) in columns.iter_mut().enumerate() {
                    for (row, value) in values.iter_mut().enumerate() {
                        *value = element(row, column);
                    }
                }
                Self {
                    columns,
                    role: PhantomData,
                }
            }

            fn map(self, f: impl Fn(T) -> T) -> Self {
                Self::from_fn(|row, column| f(self.columns[column][row]))
            }

            fn zip(self, other: Self, f: impl Fn(T, T) -> T) -> Self {
                Self::from_fn(|row, column| {
                    f(self.columns[column][row], other.columns[column][row])
                })
            }
        }

        impl<T: CoopScalar> $name<T, A> {
            /// `self * b + c`: WGSL's `coopMultiplyAdd(a, b, c)`, named as
            /// `f32::mul_add` is.
            pub fn mul_add(self, b: $name<T, B>, c: $name<T, C>) -> $name<T, C> {
                $name::from_fn(|row, column| {
                    let mut sum = c.columns[column][row];
                    for k in 0..$n {
                        sum = sum + self.columns[k][row] * b.columns[column][k];
                    }
                    sum
                })
            }
        }

        impl<T: CoopScalar, R: Role> Add for $name<T, R> {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self {
                self.zip(rhs, |a, b| a + b)
            }
        }

        impl<T: CoopScalar, R: Role> Sub for $name<T, R> {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                self.zip(rhs, |a, b| a - b)
            }
        }

        impl<T: CoopScalar, R: Role> Mul<T> for $name<T, R> {
            type Output = Self;
            #[inline]
            fn mul(self, scale: T) -> Self {
                self.map(|a| a * scale)
            }
        }

        impl<T: CoopScalar, R: Role> AddAssign for $name<T, R> {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl<T: CoopScalar, R: Role> SubAssign for $name<T, R> {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl<T: CoopScalar, R: Role> MulAssign<T> for $name<T, R> {
            #[inline]
            fn mul_assign(&mut self, scale: T) {
                *self = *self * scale;
            }
        }

        impl<R: Role> Mul<$name<f32, R>> for f32 {
            type Output = $name<f32, R>;
            #[inline]
            fn mul(self, matrix: $name<f32, R>) -> $name<f32, R> {
                matrix.map(|a| self * a)
            }
        }

        #[cfg(feature = "f16")]
        impl<R: Role> Mul<$name<half::f16, R>> for half::f16 {
            type Output = $name<half::f16, R>;
            #[inline]
            fn mul(self, matrix: $name<half::f16, R>) -> $name<half::f16, R> {
                matrix.map(|a| self * a)
            }
        }
    };
}

cooperative_matrix!(
    /// WGSL's `coop_mat8x8<T, R>`: an 8×8 matrix of `T`, which a subgroup
    /// holds as the `R` of `a * b + c`.
    CoopMat8x8,
    8
);
cooperative_matrix!(
    /// WGSL's `coop_mat16x16<T, R>`: a 16×16 matrix of `T`, which a subgroup
    /// holds as the `R` of `a * b + c`.
    CoopMat16x16,
    16
);
