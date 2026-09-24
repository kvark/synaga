#![cfg(feature = "wgsl")]
//! Operator typing: what the frontend accepts, and what it turns away before
//! Naga's validator would have to.

mod common;

use common::*;

#[test]
fn shift_by_untyped_literal() {
    // Naga insists the shift amount be `u32`; Rust would have inferred that
    // from the context, and so does the frontend.
    let wgsl = roundtrip("fn f(a: i32) -> i32 { a << 1 }");
    assert!(wgsl.contains("1u"), "{wgsl}");
    validate_only("fn f(a: u32) -> u32 { a >> 3 }");
}

#[test]
fn shift_vector_by_scalar() {
    let wgsl = roundtrip("fn f(a: vec3<u32>) -> vec3<u32> { a << 2 }");
    assert!(wgsl.contains("vec3(2u)"), "{wgsl}");
}

#[test]
fn shift_amount_must_be_unsigned() {
    let msg = reject("fn f(a: i32, b: i32) -> i32 { a << b }");
    assert!(msg.contains("shift"), "{msg}");
}

#[test]
fn rejects_negating_unsigned() {
    // `Negate` is defined on signed and float operands only.
    let msg = reject("fn f(a: u32) -> u32 { -a }");
    assert!(msg.contains('-'), "{msg}");
}

#[test]
fn rejects_negating_matrix() {
    let msg = reject("fn f(m: mat2) -> mat2 { -m }");
    assert!(msg.contains('-'), "{msg}");
}

#[test]
fn not_is_bitwise_on_integers() {
    let wgsl = roundtrip("fn f(a: u32) -> u32 { !a }");
    assert!(wgsl.contains('~'), "{wgsl}");
}

#[test]
fn not_is_logical_on_bool() {
    let wgsl = roundtrip("fn f(a: bool) -> bool { !a }");
    assert!(wgsl.contains('!'), "{wgsl}");
    assert!(!wgsl.contains('~'), "{wgsl}");
}

#[test]
fn rejects_not_on_float() {
    let msg = reject("fn f(a: f32) -> f32 { !a }");
    assert!(msg.contains('!'), "{msg}");
}

#[test]
fn rejects_arithmetic_on_bool() {
    let msg = reject("fn f(a: bool) -> bool { a + a }");
    assert!(msg.contains('+'), "{msg}");
}

#[test]
fn rejects_bitwise_on_float() {
    let msg = reject("fn f(a: f32) -> f32 { a & a }");
    assert!(msg.contains('&'), "{msg}");
}

#[test]
fn rejects_dividing_matrices() {
    let msg = reject("fn f(m: mat2) -> mat2 { m / m }");
    assert!(msg.contains('/'), "{msg}");
}

#[test]
fn rejects_ordering_bools() {
    let msg = reject("fn f(a: bool, b: bool) -> bool { a < b }");
    assert!(msg.contains('<'), "{msg}");
}

#[test]
fn matrices_add_and_subtract() {
    validate_only("fn f(a: mat3, b: mat3) -> mat3 { a + b - b }");
}

#[test]
fn untyped_literal_follows_the_other_operand() {
    let wgsl = roundtrip("fn f(a: u32) -> u32 { a * 2 }");
    assert!(wgsl.contains("2u"), "{wgsl}");
    // …from either side.
    let wgsl = roundtrip("fn f(a: u32) -> u32 { 2 * a }");
    assert!(wgsl.contains("2u"), "{wgsl}");
}

#[test]
fn untyped_literal_does_not_become_a_float() {
    // Rust would not accept `1.0 + 1` either.
    let msg = reject("fn f(a: f32) -> f32 { a + 1 }");
    assert!(msg.contains("mismatch") || msg.contains('+'), "{msg}");
}

#[test]
fn compound_assign_scales_a_vector() {
    // `v * 2.0` splats the scalar, and so should `v *= 2.0`.
    let wgsl = roundtrip("fn f(v: vec3) -> vec3 { let x = v; x *= 2.0; x }");
    assert!(wgsl.contains("vec3(2f)"), "{wgsl}");
}

#[test]
fn compound_assign_shift() {
    validate_only("fn f(a: u32) -> u32 { let x = a; x <<= 1; x }");
}

#[test]
fn compound_assign_keeps_the_target_type() {
    // `v` is a vec3, `v += 1.0` would be a vec3, but `s += v` would not be an f32.
    let msg = reject("fn f(v: vec3) -> f32 { let s = 0.0; s += v; s }");
    assert!(msg.contains("mismatch"), "{msg}");
}

#[test]
fn casts_between_scalars() {
    let wgsl = roundtrip("fn f(a: i32) -> f32 { a as f32 }");
    assert!(wgsl.contains("f32("), "{wgsl}");
    validate_only("fn f(a: f32) -> u32 { a as u32 }");
    validate_only("fn f(a: u32) -> i32 { a as i32 }");
}

#[test]
fn casts_between_vectors() {
    let wgsl = roundtrip("fn f(v: vec3<i32>) -> vec3 { v as vec3<f32> }");
    assert!(wgsl.contains("vec3<f32>(v)"), "{wgsl}");
}

#[test]
fn cast_to_the_same_type_is_a_no_op() {
    validate_only("fn f(a: f32) -> f32 { a as f32 }");
}

#[test]
fn rejects_cast_that_changes_shape() {
    let msg = reject("fn f(v: vec3) -> f32 { v as f32 }");
    assert!(msg.contains("cast"), "{msg}");
}
