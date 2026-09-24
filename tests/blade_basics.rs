#![cfg(feature = "wgsl")]
//! The parts of the dialect Blade-style shaders lean on: host-assigned
//! bindings, `select`, deferred `let`, `for` over a range, and `const`.

mod common;

use common::*;
use synaga::{parse_str, validate, validate_unbound};

#[test]
fn globals_can_leave_bindings_to_the_host() {
    // Blade omits @group/@binding from its shaders and matches globals up by
    // name at pipeline creation, asserting the module has none.
    let src = r#"
        struct Globals { mvp: mat4, sprite_size: vec2 }
        static globals: Globals = ();
        #[storage(read_write)] static out_color: vec4 = ();
        fn f() -> vec2 { globals.sprite_size }
    "#;
    let module = parse_str(src).expect("parse");
    assert!(
        module
            .global_variables
            .iter()
            .all(|(_, v)| v.binding.is_none()),
        "bindings should be left unset"
    );
    validate_unbound(&module).expect("validate");

    // The default flags still hold unbound globals against the module.
    assert!(validate(&module).is_err());
}

#[test]
fn bindings_still_work_when_given() {
    let wgsl = roundtrip(
        r#"
        #[group(1)] #[binding(2)] static scale: f32 = ();
        fn f() -> f32 { scale }
        "#,
    );
    assert!(wgsl.contains("@group(1) @binding(2)"), "{wgsl}");
}

#[test]
fn rejects_half_a_binding() {
    let msg = reject("#[group(0)] static x: f32 = (); fn f() -> f32 { x }");
    assert!(msg.contains("group") && msg.contains("binding"), "{msg}");
}

#[test]
fn select_picks_the_second_value_when_true() {
    let wgsl = roundtrip("fn f(a: f32, b: f32, c: bool) -> f32 { select(a, b, c) }");
    assert!(wgsl.contains("select(a, b, c)"), "{wgsl}");
}

#[test]
fn select_with_a_vector_condition() {
    // Blade's encode_srgb, verbatim apart from the type spellings.
    let wgsl = roundtrip(
        r#"
        fn encode_srgb(linear: vec3) -> vec3 {
            let low = 12.92 * linear;
            let high = 1.055 * pow(max(linear, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
            select(high, low, linear <= vec3(0.0031308))
        }
        "#,
    );
    assert!(wgsl.contains("select("), "{wgsl}");
}

#[test]
fn rejects_select_with_mismatched_condition() {
    let msg = reject("fn f(a: vec3, b: vec3, c: vec2<bool>) -> vec3 { select(a, b, c) }");
    assert!(msg.contains("mismatch"), "{msg}");
}

#[test]
fn deferred_let_is_assigned_in_branches() {
    // WGSL's bare `var q: vec4<f32>;`, which Blade's make_quat is built on.
    let wgsl = roundtrip(
        r#"
        fn f(c: bool) -> vec4 {
            let q: vec4;
            if c { q = vec4(1.0); } else { q = vec4(0.0); }
            normalize(q)
        }
        "#,
    );
    assert!(wgsl.contains("var q: vec4<f32>"), "{wgsl}");
}

#[test]
fn deferred_let_needs_a_type() {
    let msg = reject("fn f() -> f32 { let x; x = 1.0; x }");
    assert!(msg.contains("initializer"), "{msg}");
}

#[test]
fn for_over_a_range() {
    let wgsl = roundtrip("fn f(n: i32) -> i32 { let t = 0; for i in 0..n { t += i; } t }");
    assert!(wgsl.contains("loop {"), "{wgsl}");
    assert!(wgsl.contains("continuing {"), "{wgsl}");
}

#[test]
fn for_bound_takes_the_counter_type() {
    // `0` follows `n`, as Rust's inference would make it.
    let wgsl = roundtrip("fn f(n: u32) -> u32 { let t = 0u32; for i in 0..n { t += i; } t }");
    assert!(wgsl.contains("var i: u32"), "{wgsl}");
}

#[test]
fn for_inclusive_range() {
    let wgsl = roundtrip("fn f() -> i32 { let t = 0; for x in -1..=1 { t += x; } t }");
    assert!(wgsl.contains("<= 1i"), "{wgsl}");
}

#[test]
fn continue_inside_for_still_steps() {
    // The step lives in `continuing`, so `continue` cannot skip it.
    validate_only(
        "fn f(n: u32) -> u32 { let t = 0u32; for i in 0..n { if i == 2 { continue; } t += i; } t }",
    );
}

#[test]
fn for_over_a_computed_bound() {
    validate_only(
        "fn f(n: u32, m: u32) -> u32 { let t = 0u32; for i in 0..min(n, m) { t += i; } t }",
    );
}

#[test]
fn rejects_for_over_a_non_range() {
    let msg = reject("fn f(v: vec3) -> f32 { for x in v { } 1.0 }");
    assert!(msg.contains("range"), "{msg}");
}

#[test]
fn const_scalars_and_vectors() {
    let wgsl = roundtrip(
        r#"
        const PI: f32 = 3.1415926;
        const MAX_BOUNCES: u32 = 4;
        const LUMA: vec3 = vec3(0.2126, 0.7152, 0.0722);
        const BUMP: f32 = -0.025;
        fn f(c: vec3) -> f32 { dot(c, LUMA) * PI + (MAX_BOUNCES as f32) + BUMP }
        "#,
    );
    assert!(wgsl.contains("const PI: f32"), "{wgsl}");
    assert!(wgsl.contains("const LUMA: vec3<f32>"), "{wgsl}");
    assert!(wgsl.contains("-0.025"), "{wgsl}");
}

#[test]
fn const_can_name_another_const() {
    validate_only("const A: u32 = 4; const B: u32 = A; fn f() -> u32 { B }");
}

#[test]
fn rejects_arithmetic_in_a_const() {
    // Naga wants constants folded; saying so beats a validation failure.
    let msg = reject("const X: f32 = 1.0 + 2.0; fn f() -> f32 { X }");
    assert!(msg.contains("constant"), "{msg}");
}

#[test]
fn rejects_duplicate_const() {
    let msg = reject("const A: u32 = 1; const A: u32 = 2; fn f() -> u32 { A }");
    assert!(msg.contains("duplicate"), "{msg}");
}
