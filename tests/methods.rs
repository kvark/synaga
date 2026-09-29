//! Math as methods, named as Rust names it.
//!
//! `f32` has `x.sqrt()` and `y.atan2(x)`, and the vectors have the same, plus
//! glam's names for what only a vector has: `v.dot(w)`, `v.normalize()`,
//! `a.lerp(b, t)`. Each lowers to the builtin the free function does, except
//! where Rust means something else by the name.

mod common;

use common::*;

/// Lower both sources and require the same module.
fn same_module(methods: &str, functions: &str) {
    let lower = |src: &str| {
        let module = synaga::parse_str(src).unwrap_or_else(|e| panic!("parse: {e}\n{src}"));
        synaga::validate(&module).unwrap_or_else(|e| panic!("validate: {e}\n{src}"));
        format!("{module:#?}")
    };
    assert_eq!(lower(methods), lower(functions), "\n{methods}\n{functions}");
}

#[test]
fn a_method_is_the_free_function_with_the_receiver_first() {
    for (method, function) in [
        ("v.sqrt()", "sqrt(v)"),
        (
            "v.abs().floor().ceil().trunc()",
            "trunc(ceil(floor(abs(v))))",
        ),
        ("v.normalize()", "normalize(v)"),
        ("v.max(w).min(w)", "min(max(v, w), w)"),
        ("v.clamp(w, v)", "clamp(v, w, v)"),
        ("v.reflect(w)", "reflect(v, w)"),
        ("v.refract(w, 1.5)", "refract(v, w, 1.5)"),
        ("v.cross(w)", "cross(v, w)"),
        ("v.atan2(w)", "atan2(v, w)"),
        (
            "v.sin().cos().exp().exp2().log2()",
            "log2(exp2(exp(cos(sin(v)))))",
        ),
        (
            "Vec3::splat(v.dot(w) + v.length() + v.distance(w))",
            "Vec3::splat(dot(v, w) + length(v) + distance(v, w))",
        ),
    ] {
        same_module(
            &format!("fn f(v: Vec3, w: Vec3) -> Vec3 {{ {method} }}"),
            &format!("fn f(v: Vec3, w: Vec3) -> Vec3 {{ {function} }}"),
        );
    }
}

#[test]
fn where_rust_names_a_builtin_differently() {
    for (method, function) in [
        ("x.ln()", "log(x)"),
        ("x.powf(y)", "pow(x, y)"),
        ("x.mul_add(y, x)", "fma(x, y, x)"),
        ("x.to_degrees() + y.to_radians()", "degrees(x) + radians(y)"),
        // The GPU's `round` takes a half to the even neighbour.
        ("x.round_ties_even()", "round(x)"),
    ] {
        same_module(
            &format!("fn f(x: f32, y: f32) -> f32 {{ {method} }}"),
            &format!("fn f(x: f32, y: f32) -> f32 {{ {function} }}"),
        );
    }
    same_module(
        "fn f(a: Vec3, b: Vec3, t: f32) -> Vec3 { a.lerp(b, t) }",
        "fn f(a: Vec3, b: Vec3, t: f32) -> Vec3 { mix(a, b, t) }",
    );
    // glam raises every lane to one power; the GPU's `pow` wants a vector.
    same_module(
        "fn f(v: Vec3) -> Vec3 { v.powf(2.0) }",
        "fn f(v: Vec3) -> Vec3 { pow(v, Vec3::splat(2.0)) }",
    );
}

#[test]
fn integer_vectors_have_the_ordering_ones() {
    same_module(
        "fn f(a: Vec2<i32>, b: Vec2<i32>) -> i32 { a.abs().max(b).clamp(a, b).signum().dot(b) }",
        "fn f(a: Vec2<i32>, b: Vec2<i32>) -> i32 { dot(sign(clamp(max(abs(a), b), a, b)), b) }",
    );
    // The literal takes the receiver's scalar, as `rustc` gives it.
    validate_only("fn f(a: Vec2<u32>) -> Vec2<u32> { a.min(Vec2::splat(7)) }");
    validate_only("fn f(n: u32) -> u32 { n.max(1) }");
}

#[test]
fn a_bool_vector_folds_with_all_and_any() {
    same_module(
        "fn f(p: Vec2<i32>, e: Vec2<i32>) -> bool { p.cmpge(Vec2::ZERO).all() && p.cmplt(e).any() }",
        "fn f(p: Vec2<i32>, e: Vec2<i32>) -> bool { all(p.cmpge(Vec2::ZERO)) && any(p.cmplt(e)) }",
    );
    let msg = reject("fn f(v: Vec3) -> bool { v.all() }");
    assert!(msg.contains("all"), "{msg}");
}

#[test]
fn cast_converts_every_lane() {
    same_module(
        "fn f(v: Vec3<u32>) -> Vec3<i32> { v.cast::<i32>() }",
        "fn f(v: Vec3<u32>) -> Vec3<i32> { Vec3::<i32>::from(v) }",
    );
    // Without a turbofish, where the value goes says, as it does for `rustc`.
    same_module(
        "fn f(v: Vec3) -> Vec3<u32> { let c: Vec3<u32> = v.cast(); c }",
        "fn f(v: Vec3) -> Vec3<u32> { let c: Vec3<u32> = Vec3::<u32>::from(v); c }",
    );
    same_module(
        "fn f(v: Vec2<u32>) -> Vec2 { v.cast() }",
        "fn f(v: Vec2<u32>) -> Vec2 { Vec2::<f32>::from(v) }",
    );
    validate_only("fn g(p: Vec2<i32>) -> i32 { p.x } fn f(v: Vec2) -> i32 { g(v.cast()) }");
}

#[test]
fn fract_is_rusts() {
    // `f32::fract` rounds toward zero: `-1.25.fract()` is `-0.25`, where the
    // GPU's `fract(-1.25)` is `0.75`. The free function stays the GPU's.
    same_module(
        "fn f(x: f32) -> f32 { x.fract() }",
        "fn f(x: f32) -> f32 { x - trunc(x) }",
    );
    same_module(
        "fn f(v: Vec2) -> Vec2 { v.fract() }",
        "fn f(v: Vec2) -> Vec2 { v - trunc(v) }",
    );
    validate_only("fn f(v: Vec3) -> Vec3 { v.recip() + Vec3::splat(v.length_squared()) }");
}

#[test]
fn a_method_that_rounds_differently_on_the_gpu_is_refused() {
    for (src, hint) in [
        ("fn f(x: f32) -> f32 { x.round() }", "round_ties_even()"),
        ("fn f(x: f32) -> f32 { x.signum() }", "sign(x)"),
    ] {
        let msg = reject(src);
        assert!(msg.contains("means something else on the GPU"), "{msg}");
        assert!(msg.contains(hint), "{msg}");
    }
    // An integer's sign is the same thing on both.
    validate_only("fn f(n: i32) -> i32 { n.signum() }");
}

#[test]
fn bit_casts_are_rusts() {
    same_module(
        "fn f(x: f32) -> f32 { f32::from_bits(x.to_bits() | 1) }",
        "fn f(x: f32) -> f32 { bitcast::<f32>(bitcast::<u32>(x) | 1) }",
    );
    same_module(
        "fn f() -> u32 { 1.0f32.to_bits() }",
        "fn f() -> u32 { bitcast::<u32>(1.0) }",
    );
}

#[test]
fn integer_methods_are_rusts_too() {
    // Rust takes the amount modulo 32, and so does the module, whatever the
    // backend does with an amount of 32 or more.
    same_module(
        "fn f(x: u32, n: u32) -> u32 { x.rotate_left(n) }",
        "fn f(x: u32, n: u32) -> u32 { (x << (n & 31)) | (x >> ((32 - n) & 31)) }",
    );
    same_module(
        "fn f(x: u32, n: u32) -> u32 { x.rotate_right(n) }",
        "fn f(x: u32, n: u32) -> u32 { (x >> (n & 31)) | (x << ((32 - n) & 31)) }",
    );
    // An `i32` rotates as its bits do, not as `>>` would with its sign.
    same_module(
        "fn f(x: i32, n: u32) -> i32 { x.rotate_left(n) }",
        "fn f(x: i32, n: u32) -> i32 { let b = bitcast::<u32>(x); bitcast::<i32>((b << (n & 31)) | (b >> ((32 - n) & 31))) }",
    );
    same_module(
        "fn f(n: i32) -> u32 { n.unsigned_abs() }",
        "fn f(n: i32) -> u32 { abs(n) as u32 }",
    );
    let msg = reject("fn f(x: f32, n: u32) -> f32 { x.rotate_left(n) }");
    assert!(msg.contains("type mismatch"), "{msg}");
}

#[test]
fn element_sum_is_a_dot_with_ones() {
    same_module(
        "fn f(v: Vec4) -> f32 { v.element_sum() }",
        "fn f(v: Vec4) -> f32 { dot(v, Vec4::splat(1.0)) }",
    );
    same_module(
        "fn f(v: Vec3<i32>) -> i32 { v.element_sum() }",
        "fn f(v: Vec3<i32>) -> i32 { dot(v, Vec3::<i32>::splat(1)) }",
    );
}
