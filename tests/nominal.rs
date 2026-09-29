#![cfg(feature = "wgsl")]
//! Enums and `bitflags!` sets as types: a `u32` on the GPU, with the Rust
//! type's meaning.

mod common;

use common::*;

const MODE: &str = r#"
    #[repr(u32)]
    #[derive(Clone, Copy, Default, PartialEq, Eq)]
    enum Mode {
        #[default]
        Final,
        Depth,
        Variance = 15,
    }
"#;

#[test]
fn an_enum_is_a_field_and_compares_as_itself() {
    let wgsl = roundtrip_unbound(&format!(
        r#"{MODE}
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Params {{ mode: Mode, count: u32 }}
        static params: Uniform<Params> = binding();
        fn shows_depth() -> bool {{ params.mode == Mode::Depth }}
        fn shows_variance(mode: Mode) -> bool {{ mode == Mode::Variance }}
        "#
    ));
    assert!(wgsl.contains("mode: u32"), "{wgsl}");
    assert!(wgsl.contains("== 1u)"), "{wgsl}");
    assert!(wgsl.contains("(mode == 15u)"), "{wgsl}");
}

#[test]
fn an_enum_cast_to_u32_converts_nothing() {
    let wgsl = roundtrip(&format!(
        r#"{MODE}
        fn index(mode: Mode) -> u32 {{ mode as u32 + 1 }}
        "#
    ));
    assert!(wgsl.contains("(mode + 1u)"), "{wgsl}");
}

#[test]
fn an_enum_default_is_its_default_variant() {
    let wgsl = roundtrip(
        r#"
        #[repr(u32)]
        #[derive(Clone, Copy, Default, PartialEq)]
        enum Mode { Final, #[default] Depth }
        fn f() -> Mode { Mode::default() }
        "#,
    );
    assert!(wgsl.contains("return 1u"), "{wgsl}");
}

#[test]
fn a_zeroed_struct_needs_an_enum_whose_default_is_zero() {
    let message = reject(
        r#"
        #[repr(u32)]
        #[derive(Clone, Copy, Default, PartialEq)]
        enum Mode { Final, #[default] Depth }
        #[derive(Clone, Copy, Default)]
        struct Params { mode: Mode }
        fn f() -> Params { Params::default() }
        "#,
    );
    assert!(message.contains("`Mode`"), "{message}");
}

#[test]
fn an_enum_type_needs_repr_u32() {
    let message = reject(
        r#"
        #[derive(Clone, Copy, PartialEq)]
        enum Mode { Final, Depth }
        fn f(mode: Mode) -> bool { mode == Mode::Depth }
        "#,
    );
    assert!(message.contains("#[repr(u32)]"), "{message}");
    // Its discriminants still serve as `u32`s, as they always have.
    validate_only(
        r#"
        enum Mode { Final, Depth }
        fn f(mode: u32) -> bool { mode == Mode::Depth as u32 }
        "#,
    );
}

const DRAW: &str = r#"
    #[repr(transparent)]
    #[derive(Clone, Copy, Default, PartialEq, Eq)]
    pub struct Draw(u32);

    bitflags::bitflags! {
        impl Draw: u32 {
            /// Doc comments are fine.
            const SPACE = 1;
            const GEOMETRY = 1 << 1;
            const RESTIR = 1 << 2;
            const BOTH = Self::SPACE.bits() | Self::GEOMETRY.bits();
        }
    }
"#;

#[test]
fn a_set_is_a_field_and_answers_as_itself() {
    let wgsl = roundtrip_unbound(&format!(
        r#"{DRAW}
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Params {{ mode: u32, flags: Draw }}
        static params: Uniform<Params> = binding();
        fn space() -> bool {{ params.flags.contains(Draw::SPACE) }}
        fn restir() -> bool {{ params.flags.intersects(Draw::RESTIR | Draw::BOTH) }}
        fn none() -> bool {{ params.flags.is_empty() }}
        fn bits() -> u32 {{ params.flags.bits() + 1 }}
        "#
    ));
    assert!(wgsl.contains("flags: u32"), "{wgsl}");
    assert!(wgsl.contains(" & 1u) == 1u)"), "{wgsl}");
    assert!(wgsl.contains(" & (4u | 3u)) != 0u)"), "{wgsl}");
    assert!(wgsl.contains(" == 0u)"), "{wgsl}");
}

#[test]
fn a_complement_stays_within_the_declared_flags() {
    // Rust's `!` on a set is `complement`, which drops the bits no flag
    // declares; a `u32`'s `!` would keep them.
    let wgsl = roundtrip(&format!(
        r#"{DRAW}
        fn others(flags: Draw) -> Draw {{ !flags }}
        fn all_but(flags: Draw) -> bool {{ flags.complement().is_all() }}
        "#
    ));
    assert!(wgsl.contains("(~(flags) & 7u)"), "{wgsl}");
}

#[test]
fn a_difference_is_not_a_subtraction() {
    let wgsl = roundtrip(&format!(
        r#"{DRAW}
        fn without(flags: Draw, other: Draw) -> Draw {{ flags - other }}
        fn drop_space(flags: Draw) -> Draw {{
            let mut left = flags;
            left -= Draw::SPACE;
            left |= Draw::RESTIR;
            left
        }}
        "#
    ));
    assert!(wgsl.contains("(flags & ~(other))"), "{wgsl}");
    assert!(!wgsl.contains("flags - other"), "{wgsl}");
    assert!(wgsl.contains("& ~(1u)"), "{wgsl}");
}

#[test]
fn a_set_is_built_from_its_bits() {
    let wgsl = roundtrip(&format!(
        r#"{DRAW}
        fn none() -> Draw {{ Draw::empty() }}
        fn every() -> Draw {{ Draw::all() }}
        fn known(bits: u32) -> Draw {{ Draw::from_bits_truncate(bits) }}
        fn kept(bits: u32) -> Draw {{ Draw::from_bits_retain(bits) }}
        fn zero() -> Draw {{ Draw::default() }}
        "#
    ));
    assert!(wgsl.contains("return 7u"), "{wgsl}");
    assert!(wgsl.contains("(bits & 7u)"), "{wgsl}");
}

#[test]
fn a_whole_declaration_is_a_set_too() {
    let wgsl = roundtrip(
        r#"
        bitflags::bitflags! {
            #[repr(transparent)]
            #[derive(Clone, Copy, PartialEq, Eq)]
            pub struct Textures: u32 {
                const ALBEDO = 1;
                const NORMAL = 2;
                const _ = 1 << 7;
            }
        }
        fn ignored(textures: Textures) -> bool { textures.contains(Textures::NORMAL) }
        fn all() -> Textures { Textures::all() }
        "#,
    );
    // The unnamed flag is a declared bit too.
    assert!(wgsl.contains("return 131u"), "{wgsl}");
}

#[test]
fn a_set_needs_to_be_a_u32() {
    let message = reject(
        r#"
        bitflags::bitflags! {
            #[repr(transparent)]
            pub struct Small: u8 { const A = 1; }
        }
        "#,
    );
    assert!(message.contains("not a `u32`"), "{message}");

    let message = reject(
        r#"
        bitflags::bitflags! {
            pub struct Loose: u32 { const A = 1; }
        }
        fn f(loose: Loose) -> bool { loose.is_empty() }
        "#,
    );
    assert!(message.contains("#[repr(transparent)]"), "{message}");

    let message = reject(
        r#"
        #[repr(transparent)]
        pub struct Wide(u64);
        bitflags::bitflags! { impl Wide: u32 { const A = 1; } }
        "#,
    );
    assert!(message.contains("`Wide`"), "{message}");

    let message = reject(r#"bitflags::bitflags! { impl Nowhere: u32 { const A = 1; } }"#);
    assert!(message.contains("struct Nowhere(u32)"), "{message}");
}

#[test]
fn a_shared_struct_holds_enums_and_sets() {
    // The layout check sees each as the `u32` it is on both sides.
    validate_only_unbound(&format!(
        r#"{MODE}{DRAW}
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Debug {{ mode: Mode, draw: Draw, textures: u32, pad: u32, mouse: Vec2<u32> }}
        static debug: Uniform<Debug> = binding();
        #[entry_point(compute, threads(1))]
        fn main() {{
            let _ = debug.mode == Mode::Final && debug.draw.contains(Draw::SPACE);
        }}
        "#
    ));
}
