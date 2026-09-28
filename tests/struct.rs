#![cfg(feature = "wgsl")]
mod common;

use common::*;

#[test]
fn struct_field_access() {
    let wgsl = roundtrip(
        r#"
        struct Pair { a: f32, b: f32 }
        fn f(p: Pair) -> f32 { p.a + p.b }
        "#,
    );
    assert!(wgsl.contains("struct") || wgsl.contains("Pair"), "{wgsl}");
}

#[test]
fn struct_literal() {
    validate_only(
        r#"
        struct Pair { a: f32, b: f32 }
        fn f(x: f32, y: f32) -> Pair { Pair { a: x, b: y } }
        "#,
    );
}

#[test]
fn struct_literal_shorthand_and_reorder() {
    validate_only(
        r#"
        struct Pair { a: f32, b: f32 }
        fn f(b: f32, a: f32) -> Pair { Pair { b, a } }
        "#,
    );
}

#[test]
fn nested_struct() {
    validate_only(
        r#"
        struct Inner { x: f32 }
        struct Outer { inner: Inner, s: f32 }
        fn f(o: Outer) -> f32 { o.inner.x + o.s }
        "#,
    );
}

#[test]
fn struct_as_uniform() {
    validate_only(
        r#"
        struct Camera { mvp: mat4, pos: vec3 }
        #[group(0)]
        #[binding(0)]
        static camera: Camera = ();

        #[entry_point(vertex)]
        #[output(builtin(position))]
        fn vs_main(#[location(0)] pos: vec3) -> vec4 {
            camera.mvp * vec4(pos, 1.0)
        }
        "#,
    );
}

#[test]
fn struct_let_and_return() {
    validate_only(
        r#"
        struct Pair { a: vec3, b: f32 }
        fn f(a: vec3, b: f32) -> vec3 {
            let p = Pair { a, b };
            p.a * p.b
        }
        "#,
    );
}

#[test]
fn rejects_unknown_field() {
    let msg = reject(
        r#"
        struct Pair { a: f32, b: f32 }
        fn f(p: Pair) -> f32 { p.c }
        "#,
    );
    assert!(msg.contains("field") || msg.contains("c"), "{msg}");
}

#[test]
fn rejects_missing_field() {
    let msg = reject(
        r#"
        struct Pair { a: f32, b: f32 }
        fn f(a: f32) -> Pair { Pair { a } }
        "#,
    );
    assert!(msg.contains("field") || msg.contains("b"), "{msg}");
}

#[test]
fn rejects_unknown_struct() {
    let msg = reject("fn f() -> Ghost { Ghost { a: 1.0 } }");
    assert!(
        msg.contains("Ghost") || msg.contains("struct") || msg.contains("type"),
        "{msg}"
    );
}

#[test]
fn struct_update_takes_the_rest_from_the_base() {
    let lower = |src: &str| {
        let module = synaga::parse_str(src).unwrap_or_else(|e| panic!("parse: {e}\n{src}"));
        synaga::validate(&module).unwrap_or_else(|e| panic!("validate: {e}\n{src}"));
        format!("{module:#?}")
    };
    let decl = "#[derive(Clone, Copy, Default)] struct S { a: f32, b: u32, c: Vec2 }";
    // `..Default::default()` is how Rust avoids assigning fields one by one
    // after `S::default()`, which clippy asks for.
    for rest in ["Default::default()", "S::default()"] {
        assert_eq!(
            lower(&format!(
                "{decl} fn f(x: f32) -> S {{ S {{ a: x, ..{rest} }} }}"
            )),
            lower(&format!(
                "{decl} fn f(x: f32) -> S {{ S {{ a: x, b: u32::default(), c: Vec2::default() }} }}"
            )),
        );
    }
    assert_eq!(
        lower(&format!(
            "{decl} fn f(s: S, x: f32) -> S {{ S {{ a: x, ..s }} }}"
        )),
        lower(&format!(
            "{decl} fn f(s: S, x: f32) -> S {{ S {{ a: x, b: s.b, c: s.c }} }}"
        )),
    );
    let msg = reject(&format!("{decl} fn f(x: f32) -> S {{ S {{ a: x }} }}"));
    assert!(msg.contains("wrong number of fields"), "{msg}");
}
