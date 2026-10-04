#![cfg(feature = "wgsl")]
//! `f16`, which is `half`'s: Rust's own is not stable. A shader names it `f16`
//! either way, and a module that has one needs Naga's `SHADER_FLOAT16`.

mod common;
use common::reject;
use half::f16;
use synaga::naga::valid::{Capabilities, ValidationFlags};
use synaga::naga::{self, Literal};

/// Lower, validate with `SHADER_FLOAT16`, and print.
fn roundtrip_half(src: &str) -> String {
    let module = synaga::parse_str(src).unwrap_or_else(|e| panic!("parse: {e}\n{src}"));
    let info = synaga::validate_with(
        &module,
        ValidationFlags::all(),
        Capabilities::SHADER_FLOAT16,
    )
    .unwrap_or_else(|e| panic!("validate: {e}\n{src}"));
    synaga::to_wgsl(&module, &info).unwrap_or_else(|e| panic!("wgsl: {e}\n{src}"))
}

/// A shader checked twice from the same tokens: by `rustc`, as a module here
/// against `synaga-shader`, and by the transpiler. The WGSL it lowers to.
macro_rules! shader {
    ($($source:tt)*) => {{
        #[allow(dead_code, non_upper_case_globals)]
        mod rust {
            use synaga_shader::*;
            $($source)*
        }
        roundtrip_half(stringify!($($source)*))
    }};
}

#[test]
fn a_shader_computes_in_half_precision() {
    let wgsl = shader! {
        static halves: StorageMut<[f16]> = group(0).binding(0);
        static tint: Uniform<Vec4<f16>> = group(0).binding(1);

        const HALF: f16 = f16::from_f32_const(0.5);
        const ACROSS: Vec2<f16> = vec2(f16::ONE, f16::NEG_ONE);

        fn brighten(color: Vec3<f16>, by: f16) -> Vec3<f16> {
            (color * by + Vec3::<f16>::ONE * HALF).max(Vec3::splat(f16::ZERO))
        }

        #[entry_point(compute, threads(64))]
        fn shade(global_invocation_id: Vec3<u32>) {
            let i = global_invocation_id.x as usize;
            let h = halves.get_mut()[i];
            let color = brighten(vec3(h, tint.y, f16::from_f32(1.5)), tint.x);
            let wide = color.cast::<f32>();
            let narrow = f16::from_f32(wide.x * 2.0) - f16::from_f32(-0.25);
            let sum = color.dot(color) + narrow * ACROSS.y + f16::PI;
            if sum.to_f32() > f32::from(h) {
                halves.get_mut()[i] = sum.max(-sum) / f16::MAX;
            }
        }
    };
    for wgsl_spelling in [
        "enable f16;",
        "var<storage, read_write> halves: array<f16>;",
        "var<uniform> tint: vec4<f16>;",
        "const ACROSS: vec2<f16> = vec2<f16>(1h, -1h);",
        // A literal is converted here, rounding as `half` does on the CPU.
        "const HALF: f16 = 0.5h;",
        "1.5h",
        "-0.25h",
        "3.140625h",
        "65504h",
        // Anything else is the GPU's conversion, either way.
        "f16((vec3<f32>(",
        "if (f32(",
    ] {
        assert!(wgsl.contains(wgsl_spelling), "{wgsl_spelling}\n{wgsl}");
    }
}

#[test]
fn a_conversion_the_build_can_fold_is_half_s() {
    let source = r#"
        const SCALE: f32 = 0.1;
        fn scaled() -> f16 { f16::from_f32(SCALE * 2.0) }
        #[allow(non_snake_case)]
        fn shadowed(x: f32) -> f16 { let SCALE = x; f16::from_f32(SCALE * 2.0) }
    "#;
    let wgsl = roundtrip_half(source);
    let folded = format!("return {}h;", f16::from_f32(0.1f32 * 2.0));
    assert!(wgsl.contains(&folded), "{folded}\n{wgsl}");
    // A local is read, as in Rust, though a constant has its name.
    assert!(wgsl.contains("return f16((x * 2f));"), "{wgsl}");

    let err = reject("fn f() -> f16 { f16::from_f32(1e5) }");
    assert!(err.contains("an infinity or a NaN"), "{err}");
}

#[test]
fn the_builtins_for_any_number_take_a_half() {
    let wgsl = shader! {
        fn shape(a: Vec3<f16>, b: Vec3<f16>, h: f16) -> f16 {
            let bounded = clamp(h, f16::ZERO, f16::ONE);
            let steps = abs(a - b) + sign(a) * min(a, b);
            dot(max(steps, b), a) * bounded + f16::default()
        }
    };
    for builtin in [
        "clamp(h, 0h, 1h)",
        "abs(",
        "sign(a)",
        "min(a, b)",
        "max(",
        "dot(",
    ] {
        assert!(wgsl.contains(builtin), "{builtin}\n{wgsl}");
    }
}

#[test]
fn a_subgroup_adds_halves() {
    let module = synaga::parse_str(
        "fn total(v: Vec2<f16>) -> Vec2<f16> { subgroup_add(v) + subgroup_shuffle(v, 1) }",
    )
    .unwrap();
    let caps = Capabilities::SHADER_FLOAT16 | Capabilities::SUBGROUP;
    synaga::validate_with(&module, ValidationFlags::all(), caps).unwrap();
}

#[test]
fn a_half_needs_the_capability() {
    let module = synaga::parse_str("fn f(h: f16) -> f16 { h }").unwrap();
    let err = synaga::validate(&module).expect_err("validates without SHADER_FLOAT16");
    assert!(err.to_string().contains("FLOAT16"), "{err}");
}

/// The literal the `const` called `name` came to.
fn folded(module: &naga::Module, name: &str) -> Literal {
    let (_, constant) = module
        .constants
        .iter()
        .find(|(_, c)| c.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no `const {name}`"));
    match module.global_expressions[constant.init] {
        naga::Expression::Literal(literal) => literal,
        ref other => panic!("`{name}` is {other:?}, not a literal"),
    }
}

trait IntoLiteral {
    fn literal(self) -> Literal;
}
impl IntoLiteral for f16 {
    fn literal(self) -> Literal {
        Literal::F16(self)
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

/// Each `const` is evaluated by `half`, as this test's own, and by the
/// transpiler from the same tokens, which have to agree to the bit.
macro_rules! folds_as_half_does {
    ($($name:ident: $ty:ident = $value:expr;)*) => {{
        $(
            const $name: $ty = $value;
        )*
        let source = concat!(
            $("const ", stringify!($name), ": ", stringify!($ty), " = ", stringify!($value), ";\n",)*
        );
        let module = synaga::parse_str(source).unwrap_or_else(|e| panic!("{e}\n{source}"));
        synaga::validate_with(&module, ValidationFlags::all(), Capabilities::SHADER_FLOAT16)
            .unwrap_or_else(|e| panic!("{e}\n{source}"));
        $(
            let (got, want) = (folded(&module, stringify!($name)), $name.literal());
            let same = match (got, want) {
                (Literal::F16(a), Literal::F16(b)) => a.to_bits() == b.to_bits(),
                (Literal::F32(a), Literal::F32(b)) => a.to_bits() == b.to_bits(),
                _ => got == want,
            };
            assert!(same, "`{}`: half has {want:?}, the transpiler {got:?}", stringify!($value));
        )*
    }};
}

#[test]
fn a_constant_half_rounds_as_half_does() {
    folds_as_half_does! {
        HALF: f16 = f16::from_f32_const(0.5);
        // 2049 is halfway between 2048 and 2050, so it goes to the even one.
        TIE: f16 = f16::from_f32_const(2049.0);
        // An `f64` rounds once, straight to half precision.
        THIRD: f16 = f16::from_f64_const(1.0 / 3.0);
        SMALLEST: f16 = f16::from_bits(1);
        NEGATED: f16 = f16::from_f32_const(-(0.25 * 3.0));
        LARGEST: f16 = f16::MAX;
        TURN: f16 = f16::PI;
        BACK: f32 = f16::from_f32_const(0.1).to_f32_const();
        FLIPPED: f16 = f16::ONE.copysign(f16::NEG_ONE);
        SIGN: f16 = f16::from_f32_const(-3.0).signum();
        UNORDERED: bool = f16::NAN.is_nan();
        CHAINED: f32 = HALF.to_f32_const() * 3.0;
    };
}

#[test]
fn a_constant_half_has_only_half_s_const_fns() {
    for (source, says) in [
        // `half`'s operators are trait methods, which a `const` cannot call.
        ("const X: f16 = f16::ONE + f16::ONE;", "`+` on an `f16`"),
        ("const X: f16 = -f16::ONE;", "`-` on an `f16`"),
        ("const X: bool = f16::ONE < f16::MAX;", "`<` on an `f16`"),
        // Rust has no literal for one, and `as` is for primitives.
        ("const X: f16 = 1.5;", "type mismatch"),
        ("const X: f32 = f16::ONE as f32;", "cast"),
        (
            "const X: f16 = f16::from_bits(0x10000);",
            "out of range for `u16`",
        ),
        (
            "const X: f16 = f16::from_f32_const(1e5);",
            "an infinity or a NaN",
        ),
        ("const X: f16 = f16::INFINITY;", "an infinity or a NaN"),
    ] {
        let err = reject(source);
        assert!(err.contains(says), "`{source}`: {err}");
    }
}
