#![cfg(feature = "wgsl")]
mod common;

use common::*;

#[test]
fn let_inferred() {
    let wgsl = roundtrip("fn add(a: f32, b: f32) -> f32 { let x = a + b; x }");
    assert!(wgsl.contains("fn add"), "{wgsl}");
    // `x` is never stored to, so it stays a value and the sum is inlined.
    assert!(wgsl.contains("a + b"), "{wgsl}");
}

#[test]
fn let_annotated() {
    let wgsl = roundtrip("fn id(a: f32) -> f32 { let x: f32 = a; x }");
    assert!(wgsl.contains("fn id"), "{wgsl}");
    validate_only("fn id(a: f32) -> f32 { let x: f32 = a; x }");
}

#[test]
fn let_shadows() {
    validate_only("fn f(a: f32, b: f32) -> f32 { let x = a; let x = x + b; x }");
}

#[test]
fn let_shadows_arg() {
    validate_only("fn f(a: f32) -> f32 { let a = a + a; a }");
}

#[test]
fn if_expr_tail() {
    let wgsl = roundtrip("fn pick(c: bool, a: f32, b: f32) -> f32 { if c { a } else { b } }");
    assert!(wgsl.contains("if"), "{wgsl}");
}

#[test]
fn if_expr_in_let() {
    validate_only("fn pick(c: bool, a: f32, b: f32) -> f32 { let x = if c { a } else { b }; x }");
}

#[test]
fn if_stmt_then_return() {
    validate_only(
        r#"
        fn pick(c: bool, a: f32, b: f32) -> f32 {
            if c {
                return a;
            }
            b
        }
    "#,
    );
}

#[test]
fn else_if_expr() {
    validate_only(
        r#"
        fn pick(c: bool, d: bool, a: f32, b: f32, e: f32) -> f32 {
            if c { a } else if d { b } else { e }
        }
    "#,
    );
}

#[test]
fn nested_block_tail() {
    validate_only("fn f(a: f32) -> f32 { { let x = a; x } }");
}

#[test]
fn rejects_bare_let() {
    let msg = reject("fn f(a: f32) -> f32 { let x; x }");
    assert!(msg.contains("let") || msg.contains("initializer"), "{msg}");
}

#[test]
fn rejects_if_expr_without_else() {
    let msg = reject("fn f(c: bool, a: f32) -> f32 { let x = if c { a }; x }");
    assert!(msg.contains("else"), "{msg}");
}

#[test]
fn rejects_body_that_can_fall_through() {
    let msg = reject("fn f(c: bool, a: f32) -> f32 { if c { a } }");
    assert!(msg.contains("without returning"), "{msg}");
}

#[test]
fn return_in_both_branches() {
    validate_only("fn f(a: f32) -> f32 { if a > 0.0 { return a; } else { return -a; } }");
}

#[test]
fn tail_if_that_returns_from_both_branches() {
    let wgsl = roundtrip("fn f(a: f32) -> f32 { if a > 0.0 { return a; } else { return -a; } }");
    assert_eq!(wgsl.matches("return").count(), 2, "{wgsl}");
}

#[test]
fn tail_if_else_if_chain_that_returns() {
    validate_only(
        r#"
        fn sign_of(a: f32) -> f32 {
            if a > 0.0 {
                return 1.0;
            } else if a < 0.0 {
                return -1.0;
            } else {
                return 0.0;
            }
        }
        "#,
    );
}

#[test]
fn loop_without_break_counts_as_returning() {
    validate_only("fn f() -> f32 { loop { } }");
}

#[test]
fn rejects_return_missing_from_one_branch() {
    let msg = reject("fn f(a: f32) -> f32 { if a > 0.0 { return a; } else { let x = a; } }");
    assert!(msg.contains("without returning"), "{msg}");
}

#[test]
fn rejects_loop_with_break_as_the_only_exit() {
    let msg = reject("fn f(a: f32) -> f32 { loop { break; } }");
    assert!(msg.contains("without returning"), "{msg}");
}

#[test]
fn let_annotation_types_an_untyped_literal() {
    let wgsl = roundtrip("fn f() -> u32 { let x: u32 = 1; x }");
    assert!(wgsl.contains("1u"), "{wgsl}");
}
