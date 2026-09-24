#![cfg(feature = "wgsl")]
mod common;

use common::*;

#[test]
fn while_counts() {
    let wgsl = roundtrip(
        r#"
        fn f(n: i32) -> i32 {
            let i = 0;
            while i < n {
                i += 1;
            }
            i
        }
        "#,
    );
    assert!(wgsl.contains("loop") || wgsl.contains("while"), "{wgsl}");
}

#[test]
fn loop_with_break() {
    validate_only(
        r#"
        fn f(n: i32) -> i32 {
            let x = 0;
            loop {
                if x >= n {
                    break;
                }
                x += 1;
            }
            x
        }
        "#,
    );
}

#[test]
fn while_continue() {
    validate_only(
        r#"
        fn f(n: i32) -> i32 {
            let i = 0;
            let s = 0;
            while i < n {
                i += 1;
                if i < 0 {
                    continue;
                }
                s += i;
            }
            s
        }
        "#,
    );
}

#[test]
fn return_inside_loop() {
    validate_only(
        r#"
        fn f(n: i32) -> i32 {
            let i = 0;
            loop {
                if i >= n {
                    return i;
                }
                i += 1;
            }
        }
        "#,
    );
}

#[test]
fn rejects_labeled_loop() {
    let msg = reject("fn f(a: i32) -> i32 { 'x: loop { break; } a }");
    assert!(
        msg.contains("label") || msg.contains("unsupported"),
        "{msg}"
    );
}

#[test]
fn rejects_break_value() {
    let msg = reject("fn f(a: i32) -> i32 { loop { break a; } }");
    assert!(msg.contains("break") || msg.contains("value"), "{msg}");
}
