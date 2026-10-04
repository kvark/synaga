//! Constant arithmetic: folded to the literal `rustc` gets, and refused where
//! `rustc` refuses it.

mod common;
use common::reject;
use synaga::naga::{self, Literal};

/// The literal the `const` called `name` came to.
fn folded(module: &naga::Module, name: &str) -> Literal {
    let (_, constant) = module
        .constants
        .iter()
        .find(|(_, c)| c.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no `const {name}`"));
    let mut init = constant.init;
    loop {
        match module.global_expressions[init] {
            naga::Expression::Literal(literal) => return literal,
            naga::Expression::Constant(other) => init = module.constants[other].init,
            ref other => panic!("`{name}` is {other:?}, not a literal"),
        }
    }
}

trait IntoLiteral {
    fn literal(self) -> Literal;
}
impl IntoLiteral for u32 {
    fn literal(self) -> Literal {
        Literal::U32(self)
    }
}
impl IntoLiteral for i32 {
    fn literal(self) -> Literal {
        Literal::I32(self)
    }
}
impl IntoLiteral for f32 {
    fn literal(self) -> Literal {
        Literal::F32(self)
    }
}
impl IntoLiteral for bool {
    fn literal(self) -> Literal {
        Literal::Bool(self)
    }
}

/// Each `const` is evaluated twice: by `rustc`, as this test's own constant,
/// and by the transpiler, from the same tokens. The two have to agree, to the
/// bit for a float.
macro_rules! folds_as_rustc_does {
    ($($name:ident: $ty:ident = $value:expr;)*) => {{
        $(
            #[allow(clippy::all, clippy::pedantic, non_upper_case_globals, invalid_nan_comparisons)]
            const $name: $ty = $value;
        )*
        let source = concat!(
            $("const ", stringify!($name), ": ", stringify!($ty), " = ", stringify!($value), ";\n",)*
        );
        let module = synaga::parse_str(source).unwrap_or_else(|e| panic!("{e}\n{source}"));
        synaga::validate(&module).unwrap_or_else(|e| panic!("{e}\n{source}"));
        $(
            let (got, want) = (folded(&module, stringify!($name)), $name.literal());
            let same = match (got, want) {
                (Literal::F32(a), Literal::F32(b)) => a.to_bits() == b.to_bits(),
                _ => got == want,
            };
            assert!(same, "`{}`: rustc has {want:?}, the transpiler {got:?}", stringify!($value));
        )*
    }};
}

#[test]
fn integer_arithmetic_folds_as_rustc_does() {
    folds_as_rustc_does! {
        SUM: u32 = 4 * 16 + 1;
        BITS: u32 = (1 << 4) | 3;
        MASK: u32 = !0xF0u32 & 0xFF;
        // Division truncates, and the remainder takes the dividend's sign.
        QUOTIENT: i32 = -7 / 2;
        REMAINDER: i32 = -7 % 3;
        // `>>` on a signed integer keeps the sign.
        HALVED: i32 = -8 >> 1;
        TOP: u32 = u32::MAX >> 31;
        // The amount can be any integer type.
        SHIFTED: u32 = 1u32 << 3i32;
        LOWEST: i32 = -2147483648;
        NAMED: i32 = i32::MIN + 1;
        CHAINED: u32 = SUM * 2;
    };
}

#[test]
fn casts_convert_as_rustc_does() {
    folds_as_rustc_does! {
        WRAPPED: u32 = -1i32 as u32;
        BACK: i32 = 0xFFFF_FFFFu32 as i32;
        TRUNCATED: i32 = 3.99 as i32;
        // A float saturates into an integer.
        BELOW: u32 = -1.0f32 as u32;
        ABOVE: i32 = 1e10 as i32;
        // `1.5 * 2.0` is an `f64` here: `as` gives its operand no type.
        DOUBLE: u32 = (1.5 * 2.0) as u32;
        NARROWED: f32 = (1.0 / 3.0) as f32;
        WIDENED: f32 = 7 as f32 / 2.0;
        TRUE: u32 = true as u32;
    };
}

#[test]
fn float_arithmetic_folds_as_rustc_does() {
    folds_as_rustc_does! {
        THIRD: f32 = 1.0 / 3.0;
        INVERSE: f32 = 1.0 / 255.0;
        REMAINDER: f32 = 5.5 % 2.0;
        NEGATIVE: f32 = -(0.25 * 3.0);
        TURN: f32 = core::f32::consts::PI * 2.0;
        TINY: f32 = f32::EPSILON * 0.5;
    };
}

#[test]
fn primitive_methods_fold_as_rustc_does() {
    folds_as_rustc_does! {
        WRAPS: u32 = u32::MAX.wrapping_add(2);
        WRAPS_DOWN: i32 = i32::MIN.wrapping_sub(1);
        SATURATES: u32 = u32::MAX.saturating_add(1);
        POWER: u32 = 2u32.pow(10);
        ONES: u32 = 0xF0u32.count_ones();
        LEADING: u32 = 40u32.leading_zeros();
        ROTATED: u32 = 0x8000_0001u32.rotate_left(1);
        CEILING: u32 = 10u32.div_ceil(4);
        LOG: u32 = 1000u32.ilog2();
        ABSOLUTE: i32 = (-5i32).abs();
        UNSIGNED: u32 = (-5i32).unsigned_abs();
        EUCLID: i32 = (-7i32).rem_euclid(3);
        POWER_OF_TWO: bool = 64u32.is_power_of_two();
        NEXT: u32 = 65u32.next_power_of_two();
        BITS: u32 = 1.5f32.to_bits();
        FROM_BITS: f32 = f32::from_bits(0x3f80_0000);
        LARGER: f32 = 2.5f32.max(1.0);
        CLAMPED: f32 = 7.0f32.clamp(0.0, 1.0);
        SIGN: f32 = (-3.0f32).signum();
    };
}

#[test]
fn conditions_fold_as_rustc_does() {
    folds_as_rustc_does! {
        BIG: bool = 3 > 2 && !false;
        EITHER: bool = 1.5f32 < 1.0 || 2u32 == 2;
        PICKED: u32 = if 3 > 2 { 10 } else { 20 };
        NAN: bool = f32::NAN == f32::NAN;
    };
}

#[test]
fn a_condition_short_circuits() {
    // The right side is never reached, as at run time, so what it would do
    // there does not matter.
    let module = synaga::parse_str("const X: bool = false && 1 / 0 == 0;").unwrap();
    assert_eq!(folded(&module, "X"), Literal::Bool(false));
}

#[test]
fn an_array_is_as_long_as_arithmetic_says() {
    let module = synaga::parse_str(
        r#"
        const SIDE: u32 = 4;
        static tiles: Workgroup<[u32; SIDE * SIDE]> = binding();
        fn first(row: [f32; SIDE * 2]) -> f32 { row[0] }
        #[entry_point(compute, threads(1))]
        fn main() { let _ = tiles[15] as f32; }
        "#,
    )
    .unwrap();
    synaga::validate(&module).unwrap();
    let lengths: Vec<_> = module
        .types
        .iter()
        .filter_map(|(_, ty)| match ty.inner {
            naga::TypeInner::Array {
                size: naga::ArraySize::Constant(len),
                ..
            } => Some(len.get()),
            _ => None,
        })
        .collect();
    assert!(lengths.contains(&16) && lengths.contains(&8), "{lengths:?}");
}

#[test]
fn a_workgroup_size_is_any_constant_expression() {
    let module = synaga::parse_str(
        r#"
        const LANES: u32 = 32;
        const ROWS: u32 = LANES / 16;
        #[entry_point(compute, threads(LANES * 2, ROWS))]
        fn main() {}
        "#,
    )
    .unwrap();
    synaga::validate(&module).unwrap();
    assert_eq!(module.entry_points[0].workgroup_size, [64, 2, 1]);
}

#[test]
fn a_zero_workgroup_size_is_refused() {
    let err = reject(
        r#"
        const NONE: u32 = 4 - 4;
        #[entry_point(compute, threads(NONE))]
        fn main() {}
        "#,
    );
    assert!(err.contains("threads(0)"), "{err}");
}

#[test]
fn arithmetic_rustc_refuses_is_refused() {
    for (source, says) in [
        ("const X: u32 = 0 - 1;", "overflows `u32`"),
        ("const X: i32 = i32::MAX + 1;", "overflows `i32`"),
        ("const X: u32 = 1 / (2 - 2);", "divides by zero"),
        ("const X: u32 = 1 << 32;", "its width or more"),
        ("const X: u32 = 2u32.pow(40);", "overflows `u32`"),
    ] {
        let err = reject(source);
        assert!(err.contains(says), "`{source}`: {err}");
        assert!(err.contains("constant arithmetic"), "`{source}`: {err}");
    }
}

#[test]
fn an_infinite_constant_is_refused() {
    // `rustc` is content with an infinity; WGSL has no way to write one.
    let err = reject("const X: f32 = f32::MAX * 2.0;");
    assert!(err.contains("an infinity or a NaN"), "{err}");
}

#[test]
fn a_constant_cannot_call_what_rustc_cannot() {
    let err = reject("const X: f32 = 2.0f32.sqrt();");
    assert!(
        err.contains("`sqrt()` is not allowed in a constant"),
        "{err}"
    );
    let err = reject("fn half(x: f32) -> f32 { x * 0.5 }\nconst X: f32 = half(2.0);");
    assert!(
        err.contains("`half()` is not allowed in a constant"),
        "{err}"
    );
}

#[test]
fn a_binding_number_is_any_constant_expression() {
    let module = synaga::parse_str(
        r#"
        const FIRST: u32 = 2;
        static a: Uniform<f32> = group(FIRST - 1).binding(FIRST * 2);
        fn f() -> f32 { *a }
        "#,
    )
    .unwrap();
    let (_, var) = module
        .global_variables
        .iter()
        .find(|(_, var)| var.name.as_deref() == Some("a"))
        .expect("the uniform");
    let at = var.binding.as_ref().expect("a binding");
    assert_eq!((at.group, at.binding), (1, 4));
}
