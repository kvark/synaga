#![cfg(feature = "wgsl")]
mod common;

use common::*;

#[test]
fn vec3_compose() {
    let wgsl = roundtrip(
        r#"
        fn f(a: f32, b: f32, c: f32) -> vec3<f32> {
            vec3(a, b, c)
        }
        "#,
    );
    assert!(wgsl.contains("vec3"), "{wgsl}");
}

#[test]
fn vec2_default_f32() {
    validate_only("fn f(a: f32, b: f32) -> vec2 { vec2(a, b) }");
}

#[test]
fn vec3_splat() {
    validate_only("fn f(a: f32) -> vec3<f32> { vec3(a) }");
}

#[test]
fn component_x() {
    validate_only("fn f(v: vec3<f32>) -> f32 { v.x }");
}

#[test]
fn swizzle_xy() {
    validate_only("fn f(v: vec4<f32>) -> vec2<f32> { v.xy }");
}

#[test]
fn swizzle_wzyx() {
    validate_only("fn f(v: vec4<f32>) -> vec4<f32> { v.wzyx }");
}

#[test]
fn vec_add() {
    validate_only("fn f(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> { a + b }");
}

#[test]
fn vec_scale() {
    validate_only("fn f(v: vec3<f32>, s: f32) -> vec3<f32> { v * s }");
}

#[test]
fn vec_let() {
    validate_only("fn f(a: f32, b: f32) -> vec2<f32> { let v = vec2(a, b); v }");
}

#[test]
fn vec_i32() {
    validate_only("fn f(a: i32, b: i32) -> vec2<i32> { vec2(a, b) }");
}

#[test]
fn compose_from_vec2_and_scalar() {
    validate_only("fn f(xy: vec2, z: f32) -> vec3 { vec3(xy, z) }");
}

#[test]
fn index_literal_and_dynamic() {
    validate_only("fn f(v: vec3, i: i32) -> f32 { v[0] + v[i] }");
}

#[test]
fn splat_scalar_add() {
    validate_only("fn f(v: vec3, s: f32) -> vec3 { v + s }");
}

#[test]
fn comparing_two_vectors_is_one_bool() {
    // As Rust's `PartialOrd` and `PartialEq` on them: every lane, or for `!=`
    // any lane. The lane-wise forms are methods.
    let wgsl = roundtrip(
        r#"
        fn inside(p: Vec2<i32>, extent: Vec2<i32>) -> bool { p >= Vec2::ZERO && p < extent }
        fn outside(p: Vec2<i32>, extent: Vec2<i32>) -> bool { !(p < extent) }
        fn same(a: Vec3, b: Vec3) -> bool { a == b }
        fn differ(a: Vec3, b: Vec3) -> bool { a != b }
        fn lanes(a: Vec3, b: Vec3) -> Vec3<bool> { a.cmplt(b) }
        "#,
    );
    assert!(wgsl.contains("all((p >= vec2<i32>()))"), "{wgsl}");
    assert!(wgsl.contains("all((p < extent))"), "{wgsl}");
    assert!(wgsl.contains("all((a == b))"), "{wgsl}");
    // Naga renames the later functions' parameters apart.
    assert!(wgsl.contains("any((a_1 != b_1))"), "{wgsl}");
    assert!(wgsl.contains("return (a_2 < b_2);"), "{wgsl}");
    // A comparison that is one `bool` can be a condition.
    validate_only("fn f(a: Vec3, b: Vec3) -> f32 { if a <= b { 1.0 } else { 0.0 } }");
}

#[test]
fn vec_assign() {
    validate_only(
        r#"
        fn f(a: vec3, b: vec3) -> vec3 {
            let x = a;
            x = x + b;
            x
        }
        "#,
    );
}

#[test]
fn rejects_swizzle_out_of_range() {
    let msg = reject("fn f(v: vec2) -> f32 { v.z }");
    assert!(
        msg.contains("swizzle") || msg.contains("unsupported"),
        "{msg}"
    );
}

#[test]
fn rejects_ctor_arity() {
    let msg = reject("fn f(a: f32) -> vec3 { vec3(a, a) }");
    assert!(
        msg.contains("component") || msg.contains("constructor"),
        "{msg}"
    );
}

#[test]
fn typed_constructor_types_its_literals() {
    let wgsl = roundtrip("fn f() -> vec3<u32> { vec3u(1, 2, 3) }");
    assert!(wgsl.contains("1u"), "{wgsl}");
}

#[test]
fn first_component_types_the_rest() {
    let wgsl = roundtrip("fn f(a: u32) -> vec3<u32> { vec3(a, 2, 3) }");
    assert!(wgsl.contains("2u"), "{wgsl}");
}
