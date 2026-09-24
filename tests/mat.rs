#![cfg(feature = "wgsl")]
mod common;

use common::*;

#[test]
fn mat4_from_columns() {
    let wgsl = roundtrip(
        r#"
        fn f(a: vec4, b: vec4, c: vec4, d: vec4) -> mat4 {
            mat4(a, b, c, d)
        }
        "#,
    );
    assert!(wgsl.contains("mat4") || wgsl.contains("mat4x4"), "{wgsl}");
}

#[test]
fn mat2_from_scalars() {
    validate_only("fn f(a: f32, b: f32, c: f32, d: f32) -> mat2 { mat2(a, b, c, d) }");
}

#[test]
fn mat3_typed() {
    validate_only(
        r#"
        fn f(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> mat3<f32> {
            mat3(a, b, c)
        }
        "#,
    );
}

#[test]
fn mat2x3_rect() {
    validate_only(
        r#"
        fn f(a: vec3, b: vec3) -> mat2x3 {
            mat2x3(a, b)
        }
        "#,
    );
}

#[test]
fn mat_vec_mul() {
    validate_only("fn f(m: mat4, v: vec4) -> vec4 { m * v }");
}

#[test]
fn vec_mat_mul() {
    validate_only("fn f(v: vec4, m: mat4) -> vec4 { v * m }");
}

#[test]
fn mat_mat_mul() {
    validate_only("fn f(a: mat4, b: mat4) -> mat4 { a * b }");
}

#[test]
fn mat_scale() {
    validate_only("fn f(m: mat3, s: f32) -> mat3 { m * s }");
}

#[test]
fn mat_add() {
    validate_only("fn f(a: mat3, b: mat3) -> mat3 { a + b }");
}

#[test]
fn column_index() {
    validate_only("fn f(m: mat4, i: i32) -> vec4 { m[0] + m[i] }");
}

#[test]
fn transpose_det() {
    validate_only("fn f(m: mat3) -> f32 { determinant(transpose(m)) }");
}

#[test]
fn transform_point() {
    validate_only(
        r#"
        fn vs(pos: vec3, mvp: mat4) -> vec4 {
            mvp * vec4(pos, 1.0)
        }
        "#,
    );
}

#[test]
fn rejects_ctor_arity() {
    let msg = reject("fn f(a: vec4) -> mat4 { mat4(a, a) }");
    assert!(
        msg.contains("component") || msg.contains("constructor"),
        "{msg}"
    );
}
