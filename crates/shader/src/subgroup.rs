//! Subgroup operations: what the invocations of a subgroup compute together.
//!
//! Each is WGSL's builtin of the same name, in snake case: `subgroup_add` is
//! `subgroupAdd`. A module that uses one needs Naga's `SUBGROUP` capability,
//! and `subgroup_barrier` needs `SUBGROUP_BARRIER` as well.
//!
//! On the GPU an operation reads every active invocation of the subgroup. On
//! the CPU a shader runs as one invocation, which is a subgroup of one, and
//! each operation means what it means for one: a reduction or an inclusive
//! scan of `x` is `x`, an exclusive scan is the operation's identity, a ballot
//! has the one invocation's bit, and a broadcast, a shuffle or a quad swap
//! reads `x` back, since there is no other lane. That is exact for a subgroup
//! of one, which the `subgroup_size` the test passes says it is; what depends
//! on more than one lane takes a GPU to test.
//!
//! ```
//! use synaga_shader::*;
//! assert_eq!(subgroup_add(vec2(1, 2)), vec2(1, 2));
//! assert_eq!(subgroup_exclusive_mul(3.0), 1.0);
//! assert_eq!(subgroup_ballot(true), vec4(1, 0, 0, 0));
//! ```

use crate::builtins::{Integral, Numeric};
use crate::vector::{vec4, Scalar, Vec4};

/// `x` in every lane: the sum of the subgroup's `x`.
#[inline]
pub fn subgroup_add<T: Numeric>(x: T) -> T {
    x
}

/// The product of the subgroup's `x`.
#[inline]
pub fn subgroup_mul<T: Numeric>(x: T) -> T {
    x
}

/// The least of the subgroup's `x`, lane by lane.
#[inline]
pub fn subgroup_min<T: Numeric>(x: T) -> T {
    x
}

/// The greatest of the subgroup's `x`, lane by lane.
#[inline]
pub fn subgroup_max<T: Numeric>(x: T) -> T {
    x
}

/// The subgroup's `x`, bitwise and-ed.
#[inline]
pub fn subgroup_and<T: Integral>(x: T) -> T {
    x
}

/// The subgroup's `x`, bitwise or-ed.
#[inline]
pub fn subgroup_or<T: Integral>(x: T) -> T {
    x
}

/// The subgroup's `x`, bitwise xor-ed.
#[inline]
pub fn subgroup_xor<T: Integral>(x: T) -> T {
    x
}

/// Whether `predicate` holds for every invocation of the subgroup.
#[inline]
pub fn subgroup_all(predicate: bool) -> bool {
    predicate
}

/// Whether `predicate` holds for any invocation of the subgroup.
#[inline]
pub fn subgroup_any(predicate: bool) -> bool {
    predicate
}

/// The sum of `x` over the invocations before this one: zero for the first.
#[inline]
pub fn subgroup_exclusive_add<T: Numeric>(x: T) -> T {
    T::splat(<T::Lane as Scalar>::ZERO)
}

/// The product of `x` over the invocations before this one: one for the
/// first.
#[inline]
pub fn subgroup_exclusive_mul<T: Numeric>(x: T) -> T {
    T::splat(<T::Lane as Scalar>::ONE)
}

/// The sum of `x` over this invocation and those before it.
#[inline]
pub fn subgroup_inclusive_add<T: Numeric>(x: T) -> T {
    x
}

/// The product of `x` over this invocation and those before it.
#[inline]
pub fn subgroup_inclusive_mul<T: Numeric>(x: T) -> T {
    x
}

/// A bit for each invocation of the subgroup, set where `predicate` holds:
/// invocation `i` is bit `i % 32` of lane `i / 32`.
#[inline]
pub fn subgroup_ballot(predicate: bool) -> Vec4<u32> {
    vec4(u32::from(predicate), 0, 0, 0)
}

/// `x` as the active invocation with the lowest id has it.
#[inline]
pub fn subgroup_broadcast_first<T: Numeric>(x: T) -> T {
    x
}

/// `x` as invocation `id` has it. `id` is the same for every invocation.
#[inline]
pub fn subgroup_broadcast<T: Numeric>(x: T, id: u32) -> T {
    x
}

/// `x` as invocation `id` has it, where `id` may differ between invocations.
#[inline]
pub fn subgroup_shuffle<T: Numeric>(x: T, id: u32) -> T {
    x
}

/// `x` as the invocation `mask` away, by xor, has it.
#[inline]
pub fn subgroup_shuffle_xor<T: Numeric>(x: T, mask: u32) -> T {
    x
}

/// `x` as the invocation `delta` before this one has it.
#[inline]
pub fn subgroup_shuffle_up<T: Numeric>(x: T, delta: u32) -> T {
    x
}

/// `x` as the invocation `delta` after this one has it.
#[inline]
pub fn subgroup_shuffle_down<T: Numeric>(x: T, delta: u32) -> T {
    x
}

/// `x` as invocation `id` of this one's quad has it.
#[inline]
pub fn quad_broadcast<T: Numeric>(x: T, id: u32) -> T {
    x
}

/// `x` as the other invocation of this one's quad row has it.
#[inline]
pub fn quad_swap_x<T: Numeric>(x: T) -> T {
    x
}

/// `x` as the other invocation of this one's quad column has it.
#[inline]
pub fn quad_swap_y<T: Numeric>(x: T) -> T {
    x
}

/// `x` as the diagonally opposite invocation of this one's quad has it.
#[inline]
pub fn quad_swap_diagonal<T: Numeric>(x: T) -> T {
    x
}

/// Wait until every invocation of the subgroup reaches this point: WGSL's
/// `subgroupBarrier`. A subgroup of one, as the CPU has, is always there.
#[inline]
pub fn subgroup_barrier() {}
