#![cfg(feature = "wgsl")]
//! `match`, which is a `switch`, or an `if` where a `switch` would mean
//! something else.

mod common;

use common::*;

const MODE: &str = r#"
    #[repr(u32)]
    #[derive(Clone, Copy, PartialEq)]
    pub enum Mode { Final, Depth, Variance, Normal }
"#;

#[test]
fn an_enum_is_a_switch() {
    let wgsl = roundtrip(&format!(
        "{MODE}
        fn weight(mode: Mode) -> f32 {{
            match mode {{
                Mode::Final => 1.0,
                Mode::Depth | Mode::Normal => 0.5,
                _ => 0.0,
            }}
        }}"
    ));
    assert!(wgsl.contains("switch mode {"), "{wgsl}");
    assert!(wgsl.contains("case 1u, 3u: {"), "{wgsl}");
    assert!(wgsl.contains("default: {"), "{wgsl}");
    // Each arm returns from where it is, as each branch of an `if` does.
    assert!(wgsl.contains("return 0.5f;"), "{wgsl}");
}

#[test]
fn every_variant_needs_no_wildcard() {
    // `rustc` has checked that the arms cover the enum, so the last one also
    // takes the default Naga wants.
    let wgsl = roundtrip(&format!(
        "{MODE}
        fn code(mode: Mode) -> u32 {{
            let n: u32 = match mode {{
                Mode::Final => 10,
                Mode::Depth => 20,
                Mode::Variance => 30,
                Mode::Normal => 40,
            }};
            n + 1
        }}"
    ));
    assert!(wgsl.contains("case 3u, default: {"), "{wgsl}");
}

#[test]
fn a_variant_brought_in_by_name_is_the_variant() {
    // With `use Mode::*`, `Depth` is the variant, not a binding.
    let wgsl = roundtrip(&format!(
        "{MODE}
        use Mode::*;
        fn f(mode: Mode) -> u32 {{
            match mode {{
                Depth => 1,
                _ => 0,
            }}
        }}"
    ));
    assert!(wgsl.contains("case 1u: {"), "{wgsl}");
}

#[test]
fn integers_consts_and_bindings() {
    let wgsl = roundtrip(
        r#"
        const LAST: i32 = 7;
        fn f(n: i32) -> i32 {
            match n {
                -1 => 0,
                0 | 1 => 1,
                LAST => 70,
                other => other * 2,
            }
        }
        "#,
    );
    assert!(wgsl.contains("case -1: {"), "{wgsl}");
    assert!(wgsl.contains("case 0, 1: {"), "{wgsl}");
    assert!(wgsl.contains("case 7: {"), "{wgsl}");
    assert!(wgsl.contains("(n * 2i)"), "{wgsl}");
}

#[test]
fn the_first_arm_that_matches_wins() {
    // A value an earlier arm takes, and every arm after `_`, is never reached.
    let wgsl = roundtrip(
        r#"
        fn f(n: u32) -> u32 {
            match n {
                1 => 10,
                1 | 2 => 20,
                _ => 0,
                3 => 30,
            }
        }
        "#,
    );
    assert!(wgsl.contains("case 1u: {"), "{wgsl}");
    assert!(wgsl.contains("case 2u: {"), "{wgsl}");
    assert!(!wgsl.contains("case 3u"), "{wgsl}");
}

#[test]
fn a_statement_and_an_early_return() {
    let wgsl = roundtrip(&format!(
        "{MODE}
        fn f(mode: Mode, x: f32) -> f32 {{
            let mut y = x;
            match mode {{
                Mode::Final => {{
                    if x < 0.0 {{
                        return 0.0;
                    }}
                    y *= 2.0;
                }}
                Mode::Depth => y = 1.0,
                _ => {{}}
            }}
            y
        }}"
    ));
    assert!(wgsl.contains("switch mode {"), "{wgsl}");
}

#[test]
fn a_bool_is_an_if() {
    let wgsl = roundtrip(
        r#"
        fn f(b: bool) -> u32 {
            match b {
                true => 1,
                false => 2,
            }
        }
        "#,
    );
    assert!(!wgsl.contains("switch"), "{wgsl}");
    assert!(wgsl.contains("if b {"), "{wgsl}");
}

#[test]
fn a_break_leaves_the_loop_not_the_match() {
    // In a `switch`, `break` would leave the `switch`.
    let wgsl = roundtrip(
        r#"
        fn f(limit: u32) -> u32 {
            let mut i = 0u32;
            loop {
                match i {
                    5 => break,
                    _ => {}
                }
                if i == limit { break; }
                i += 1;
            }
            i
        }
        "#,
    );
    assert!(!wgsl.contains("switch"), "{wgsl}");
    assert!(
        wgsl.contains("(i_1 == 5u)") || wgsl.contains("== 5u)"),
        "{wgsl}"
    );
}

#[test]
fn a_flag_or_a_kind_is_a_pattern() {
    validate_only(
        r#"
        bitflags::bitflags! {
            #[repr(transparent)]
            #[derive(Clone, Copy, PartialEq, Eq)]
            pub struct Draw: u32 { const SPACE = 1; const LINES = 2; }
        }
        fn f(d: Draw) -> u32 {
            match d {
                Draw::SPACE => 1,
                Draw::LINES => 2,
                _ => 0,
            }
        }
        "#,
    );
}

#[test]
fn what_a_match_cannot_do() {
    let msg = reject("fn f(n: u32) -> u32 { match n { x if x > 3 => 1, _ => 0 } }");
    assert!(msg.contains("guard"), "{msg}");
    let msg = reject("fn f(n: u32) -> u32 { match n { 1..=3 => 1, _ => 0 } }");
    assert!(msg.contains("pattern"), "{msg}");
    let msg = reject("fn f(v: f32) -> u32 { match v { _ => 0 } }");
    assert!(msg.contains("`match` is on"), "{msg}");
    let msg = reject("fn f(n: u32) -> u32 { let x = match n { 1 => 1u32, _ => 2.0 }; x }");
    assert!(msg.contains("mismatch"), "{msg}");
}
