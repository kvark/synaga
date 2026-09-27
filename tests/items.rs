#![cfg(feature = "wgsl")]
//! Items the way Rust has them: in any order, across modules, reached by
//! `use` and by path, and kept or dropped by `#[cfg]`.

mod common;

use common::*;
use synaga::{parse, Cfg, Source};

fn named<'a>(sources: &[(&'a str, &'a str)]) -> Vec<Source<'a>> {
    sources
        .iter()
        .map(|&(name, text)| Source {
            name: Some(name),
            text,
        })
        .collect()
}

#[test]
fn a_function_may_be_called_before_it_is_defined() {
    validate_only(
        r#"
        #[compute] #[workgroup_size(1)]
        fn main() { let x = twice(1.0); }
        fn twice(a: f32) -> f32 { half(a) * 4.0 }
        fn half(a: f32) -> f32 { a * 0.5 }
        "#,
    );
}

#[test]
fn a_struct_may_be_named_before_it_is_defined() {
    validate_only(
        r#"
        fn f(p: Later) -> f32 { p.inner.a }
        struct Later { inner: Inner }
        struct Inner { a: f32 }
        #[compute] #[workgroup_size(1)]
        fn main() { let x = f(Later { inner: Inner { a: 1.0 } }); }
        "#,
    );
}

#[test]
fn a_const_may_use_one_defined_after_it() {
    validate_only(
        r#"
        const LEN: u32 = TWO;
        const TWO: u32 = 2u32;
        #[compute] #[workgroup_size(1)]
        fn main() { let x: [u32; LEN] = [1u32, 2u32]; }
        "#,
    );
}

#[test]
fn callees_come_before_callers_in_the_module() {
    // Naga validates a call only against a function it has already seen.
    let module = synaga::parse_str(
        r#"
        fn a() -> f32 { b() + 1.0 }
        fn b() -> f32 { c() + 1.0 }
        fn c() -> f32 { 1.0 }
        "#,
    )
    .unwrap();
    let order: Vec<_> = module
        .functions
        .iter()
        .map(|(_, f)| f.name.clone().unwrap())
        .collect();
    assert_eq!(order, ["c", "b", "a"]);
    synaga::validate(&module).unwrap();
}

#[test]
fn recursion_is_refused_by_name() {
    let msg = reject(
        r#"
        fn a(x: f32) -> f32 { b(x) }
        fn b(x: f32) -> f32 { a(x) }
        "#,
    );
    assert!(msg.contains("depends on itself"), "{msg}");
}

#[test]
fn an_entry_point_cannot_be_called() {
    let msg = reject(
        r#"
        #[compute] #[workgroup_size(1)]
        fn main() {}
        fn helper() { main(); }
        "#,
    );
    assert!(msg.contains("only the GPU calls"), "{msg}");
}

#[test]
fn modules_reach_each_other_by_use_and_by_path() {
    let module = parse(
        &named(&[
            (
                "shade",
                r#"
                use super::math::*;
                use super::light::{Sun as Light, strength};
                #[fragment]
                #[output(location(0))]
                fn fs() -> vec4 {
                    let l = Light { dir: vec3(0.0, 1.0, 0.0) };
                    let k = square(strength(l)) + super::light::strength(l) + crate::shaders::math::square(2.0);
                    vec4::splat(k)
                }
                "#,
            ),
            ("math", "pub fn square(x: f32) -> f32 { x * x }"),
            (
                "light",
                r#"
                pub struct Sun { pub dir: vec3 }
                pub fn strength(s: Sun) -> f32 { s.dir.y }
                "#,
            ),
        ]),
        &Cfg::new(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    synaga::validate(&module).unwrap();
}

#[test]
fn each_module_keeps_its_own_helpers() {
    // Two modules may each have a `helper`; each call reaches its own.
    let module = parse(
        &named(&[
            (
                "main",
                r#"
                use super::a;
                use super::b;
                #[compute] #[workgroup_size(1)]
                fn main() { let x = a::run() + b::run(); }
                "#,
            ),
            (
                "a",
                "fn helper() -> f32 { 1.0 } pub fn run() -> f32 { helper() }",
            ),
            (
                "b",
                "fn helper() -> f32 { 2.0 } pub fn run() -> f32 { helper() }",
            ),
        ]),
        &Cfg::new(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    synaga::validate(&module).unwrap();
    let helpers = module
        .functions
        .iter()
        .filter(|(_, f)| f.name.as_deref() == Some("helper"))
        .count();
    assert_eq!(helpers, 2);
}

#[test]
fn a_name_two_globs_bring_in_is_ambiguous() {
    let err = parse(
        &named(&[
            (
                "main",
                r#"
                use super::a::*;
                use super::b::*;
                #[compute] #[workgroup_size(1)]
                fn main() { let x = helper(); }
                "#,
            ),
            ("a", "pub fn helper() -> f32 { 1.0 }"),
            ("b", "pub fn helper() -> f32 { 2.0 }"),
        ]),
        &Cfg::new(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("more than one module"), "{err}");
    assert_eq!(err.index, 0);
}

#[test]
fn an_error_is_blamed_on_the_source_it_is_in() {
    let err = parse(
        &named(&[
            (
                "main",
                "use super::broken::f;\n#[compute] #[workgroup_size(1)]\nfn main() { let x = f(); }",
            ),
            ("broken", "pub fn f() -> f32 {\n    nope\n}"),
        ]),
        &Cfg::new(),
    )
    .unwrap_err();
    assert_eq!(err.index, 1, "{err}");
    assert!(err.to_string().contains("nope"), "{err}");
}

#[test]
fn a_type_alias_names_its_type() {
    validate_only(
        r#"
        type Color = vec4;
        fn tint(c: Color) -> Color { c * 0.5 }
        "#,
    );
}

#[test]
fn cfg_drops_items_and_answers_cfg_macros() {
    let src = r#"
        #[cfg(debug_assertions)]
        const LEVEL: u32 = 2u32;
        #[cfg(not(debug_assertions))]
        const LEVEL: u32 = 0u32;
        const CHECKED: bool = cfg!(all(debug_assertions, feature = "checks"));
        #[compute] #[workgroup_size(1)]
        fn main() {
            let checked = cfg!(any(feature = "checks", feature = "more"));
            let x = LEVEL;
        }
    "#;
    let level = |cfg: &Cfg| {
        let module = parse(
            &[Source {
                name: None,
                text: src,
            }],
            cfg,
        )
        .unwrap();
        synaga::validate(&module).unwrap();
        let (_, level) = module
            .constants
            .iter()
            .find(|(_, c)| c.name.as_deref() == Some("LEVEL"))
            .unwrap();
        module.global_expressions[level.init].clone()
    };
    assert!(matches!(
        level(&Cfg::new()),
        naga::Expression::Literal(naga::Literal::U32(0))
    ));
    assert!(matches!(
        level(&Cfg::new().with("debug_assertions")),
        naga::Expression::Literal(naga::Literal::U32(2))
    ));
    let checks = Cfg::new()
        .with("debug_assertions")
        .with_value("feature", "checks");
    let module = parse(
        &[Source {
            name: None,
            text: src,
        }],
        &checks,
    )
    .unwrap();
    let (_, checked) = module
        .constants
        .iter()
        .find(|(_, c)| c.name.as_deref() == Some("CHECKED"))
        .unwrap();
    assert!(matches!(
        module.global_expressions[checked.init],
        naga::Expression::Literal(naga::Literal::Bool(true))
    ));
}

#[test]
fn a_glob_brings_in_only_what_its_module_has() {
    // `c` has a `helper` too, but nothing here imports `c`.
    let module = parse(
        &named(&[
            (
                "main",
                r#"
                use super::a::*;
                use super::b::*;
                #[compute] #[workgroup_size(1)]
                fn main() { let x = helper() + other(); }
                "#,
            ),
            ("a", "pub fn helper() -> f32 { 1.0 }"),
            ("b", "pub fn other() -> f32 { 2.0 }"),
            ("c", "pub fn helper() -> f32 { 3.0 }"),
        ]),
        &Cfg::new(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    synaga::validate(&module).unwrap();
}
