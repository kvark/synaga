#![cfg(feature = "wgsl")]
mod common;

use common::*;

#[test]
fn user_fn_call() {
    let wgsl = roundtrip(
        r#"
        fn add(a: f32, b: f32) -> f32 { a + b }
        fn mad(a: f32, b: f32, c: f32) -> f32 { add(a, b) + c }
        "#,
    );
    assert!(wgsl.contains("fn add"), "{wgsl}");
    assert!(wgsl.contains("fn mad"), "{wgsl}");
}

#[test]
fn call_in_entry() {
    validate_only(
        r#"
        fn scale(v: vec3, s: f32) -> vec3 { v * s }
        #[vertex]
        #[output(builtin(position))]
        fn vs_main(#[location(0)] pos: vec3) -> vec4 {
            let p = scale(pos, 2.0);
            vec4(p.x, p.y, p.z, 1.0)
        }
        "#,
    );
}

#[test]
fn dot_normalize() {
    validate_only(
        r#"
        fn f(a: vec3, b: vec3) -> f32 {
            dot(normalize(a), b)
        }
        "#,
    );
}

#[test]
fn cross_length() {
    validate_only("fn f(a: vec3, b: vec3) -> f32 { length(cross(a, b)) }");
}

#[test]
fn clamp_mix() {
    validate_only("fn f(x: f32, a: f32, b: f32) -> f32 { mix(a, b, clamp(x, 0.0, 1.0)) }");
}

#[test]
fn rejects_unknown_fn() {
    let msg = reject("fn f(a: f32) -> f32 { foo(a) }");
    assert!(
        msg.contains("foo") || msg.contains("unknown") || msg.contains("constructor"),
        "{msg}"
    );
}

#[test]
fn rejects_forward_ref() {
    let msg = reject(
        r#"
        fn a(x: f32) -> f32 { b(x) }
        fn b(x: f32) -> f32 { x }
        "#,
    );
    assert!(msg.contains("b") || msg.contains("unknown"), "{msg}");
}

#[test]
fn user_function_shadows_a_builtin() {
    // Resolving `length` to the builtin here would silently call something
    // other than what the source says.
    let wgsl = roundtrip("fn length(a: f32) -> f32 { a + 1.0 } fn g(x: f32) -> f32 { length(x) }");
    assert!(wgsl.contains("length_(x)"), "{wgsl}");
}

#[test]
fn rejects_duplicate_function_names() {
    let msg = reject("fn f(a: f32) -> f32 { a } fn f(a: f32) -> f32 { a }");
    assert!(msg.contains("duplicate"), "{msg}");
}

#[test]
fn rejects_entry_point_clashing_with_a_function() {
    let msg = reject(
        r#"
        fn fs(a: f32) -> f32 { a }
        #[fragment] #[output(location(0))] fn fs() -> vec4 { vec4(1.0) }
        "#,
    );
    assert!(msg.contains("duplicate"), "{msg}");
}

#[test]
fn untyped_literal_takes_the_parameter_type() {
    let wgsl = roundtrip("fn g(a: u32) -> u32 { a } fn f() -> u32 { g(1) }");
    assert!(wgsl.contains("g(1u)"), "{wgsl}");
}

#[test]
fn untyped_literal_follows_the_first_math_argument() {
    let wgsl = roundtrip("fn f(a: u32) -> u32 { clamp(a, 0, 10) }");
    assert!(wgsl.contains("0u") && wgsl.contains("10u"), "{wgsl}");
}

#[test]
fn sign_is_not_abs() {
    let wgsl = roundtrip("fn f(a: f32) -> f32 { sign(a) }");
    assert!(wgsl.contains("sign("), "{wgsl}");
}

#[test]
fn names_an_unsupported_construct_in_the_error() {
    let msg = reject("fn f(a: f32) -> f32 { match a { _ => a } }");
    assert!(msg.contains("match"), "{msg}");
}
