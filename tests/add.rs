#![cfg(feature = "wgsl")]
mod common;

use common::*;
use synaga::parse_str;

#[test]
fn add_lowers_and_validates() {
    let wgsl = roundtrip("fn add(a: f32, b: f32) -> f32 { a + b }");
    assert!(wgsl.contains("fn add(a: f32, b: f32) -> f32"), "{wgsl}");
    assert!(wgsl.contains("a + b") || wgsl.contains("(a + b)"), "{wgsl}");
}

#[test]
fn explicit_return() {
    validate_only("fn add(a: f32, b: f32) -> f32 { return a + b; }");
}

#[test]
fn arithmetic_and_compare() {
    let src = r#"
        fn mad(a: f32, b: f32, c: f32) -> f32 { a * b + c }
        fn lt(a: i32, b: i32) -> bool { a < b }
        fn bits(x: u32, y: u32) -> u32 { x & y | y }
    "#;
    let module = parse_str(src).unwrap();
    assert_eq!(module.functions.len(), 3);
    validate_only(src);
}

#[test]
fn unary_and_bool() {
    validate_only("fn neg(x: f32) -> f32 { -x } fn not_b(x: bool) -> bool { !x }");
}
