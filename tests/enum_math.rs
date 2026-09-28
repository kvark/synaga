//! Enum discriminants and the `f32` math methods.

use synaga::{parse_str, validate_unbound};

#[test]
fn an_enum_variant_is_its_discriminant() {
    let module = parse_str(
        r#"
        enum Mode { Final = 0, Variance = 15 }
        fn selected(mode: u32) -> bool { mode == Mode::Variance as u32 }
        "#,
    )
    .expect("parse");
    validate_unbound(&module).expect("validate");
    let text = format!("{module:?}");
    assert!(text.contains("15"), "{text}");
}

#[test]
fn scalar_math_methods_match_the_free_functions() {
    let module = parse_str(
        r#"
        fn f(y: f32, x: f32) -> f32 { y.asin() + y.atan2(x) + y.sin() + x.cos() }
        "#,
    )
    .expect("parse");
    validate_unbound(&module).expect("validate");
}
