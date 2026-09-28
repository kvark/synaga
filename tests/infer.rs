//! A vector's scalar, and a literal's type, from where the value goes.
//!
//! `rustc` infers `vec3(1, 2, 3)` in `let c: Vec3<u32> = vec3(1, 2, 3)` to be
//! `u32` lanes, and the transpiler has to arrive at the same type: nothing in
//! the call says so, only the annotation.

mod common;

use common::*;

/// The type of the single function's result, as Naga has it.
fn result_type(src: &str) -> naga::TypeInner {
    let module = synaga::parse_str(src).unwrap_or_else(|e| panic!("parse: {e}\n{src}"));
    synaga::validate(&module).unwrap_or_else(|e| panic!("validate: {e}\n{src}"));
    let (_, function) = module.functions.iter().last().expect("a function");
    let result = function.result.as_ref().expect("a result");
    module.types[result.ty].inner.clone()
}

fn uvec3() -> naga::TypeInner {
    naga::TypeInner::Vector {
        size: naga::VectorSize::Tri,
        scalar: naga::Scalar::U32,
    }
}

#[test]
fn a_constructor_follows_its_annotation() {
    let src = "fn f() -> Vec3<u32> { let c: Vec3<u32> = vec3(1, 2, 3); c }";
    assert_eq!(result_type(src), uvec3());
}

#[test]
fn a_constructor_follows_what_it_is_passed_to() {
    validate_only("fn g(v: Vec2<u32>) -> u32 { v.x } fn f() -> u32 { g(vec2(1, 2)) }");
}

#[test]
fn what_a_function_returns_follows_its_result() {
    assert_eq!(
        result_type("fn f() -> Vec3<u32> { vec3(1, 2, 3) }"),
        uvec3()
    );
    assert_eq!(result_type("fn f() -> Vec3<u32> { Vec3::ZERO }"), uvec3());
    assert_eq!(
        result_type("fn f() -> Vec3<u32> { Vec3::ONE * 2 }"),
        uvec3()
    );
    assert_eq!(
        result_type("fn f() -> Vec3<u32> { Vec3::splat(7) }"),
        uvec3()
    );
    assert_eq!(
        result_type("fn f(v: Vec3) -> Vec3<u32> { Vec3::from(v) }"),
        uvec3()
    );
    // Scalars too, which is where this was missing before vectors needed it.
    validate_only("fn f() -> u32 { 1 }");
    validate_only("fn f(c: bool) -> u32 { if c { return 1; } 2 }");
    validate_only("fn f(c: bool) -> u32 { if c { 1 } else { 2 } }");
}

#[test]
fn an_operator_passes_its_type_to_its_operands() {
    // The shape `examples/sprites` unpacks a colour with.
    let src = "fn f(raw: u32) -> Vec4<u32> { (Vec4::splat(raw) >> vec4(0, 8, 16, 24)) & Vec4::splat(0xFF) }";
    assert_eq!(
        result_type(src),
        naga::TypeInner::Vector {
            size: naga::VectorSize::Quad,
            scalar: naga::Scalar::U32,
        }
    );
    validate_only("fn f() -> Vec3<u32> { vec3(1, 2, 3) + Vec3::ONE * 2 }");
    validate_only("fn f() -> u32 { let x: u32 = max(1, 2) + 3; x }");
}

#[test]
fn a_comparison_says_nothing_about_its_operands() {
    // The `bool` the comparison makes is no hint for what it compares.
    validate_only("fn f(a: Vec3<u32>) -> Vec3<bool> { a.cmplt(Vec3::splat(1)) }");
    validate_only("fn f(n: u32) -> bool { let b: bool = n < 3; b }");
}

#[test]
fn a_returned_value_has_to_be_the_declared_type() {
    // Naga notices too, but as "the `return` expression [0] does not match".
    for src in [
        "fn f() -> u32 { 1.0 }",
        "fn f() -> Vec3 { vec3(1, 2, 3) }",
        "fn f(c: bool) -> u32 { if c { return 1.0; } 2 }",
    ] {
        let msg = reject(src);
        assert!(
            msg.contains("returns a value that is not the type it declares"),
            "{msg}\n{src}"
        );
    }
}
