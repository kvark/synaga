#![cfg(feature = "wgsl")]
//! Structs carrying their own `#[location]` / `#[builtin]` bindings: the
//! vertex-output / fragment-input shape every real shader is written in.

mod common;

use common::*;

const VS_FS: &str = r#"
    struct VsOut {
        #[builtin(position)] pos: vec4,
        #[location(0)] uv: vec2,
    }

    #[vertex]
    fn vs(#[location(0)] p: vec4, #[location(1)] uv: vec2) -> VsOut {
        VsOut { pos: p, uv }
    }

    #[fragment]
    #[output(location(0))]
    fn fs(varying: VsOut) -> vec4 {
        vec4(varying.uv, 0.0, 1.0)
    }
"#;

#[test]
fn vertex_out_fragment_in() {
    let wgsl = roundtrip(VS_FS);
    assert!(wgsl.contains("@builtin(position) pos"), "{wgsl}");
    assert!(wgsl.contains("@location(0) uv"), "{wgsl}");
    assert!(wgsl.contains("-> VsOut"), "{wgsl}");
    // A float varying interpolates perspective-correct by default, which WGSL
    // spells by saying nothing at all.
    assert!(!wgsl.contains("@interpolate"), "{wgsl}");
}

#[test]
fn multiple_render_targets() {
    let wgsl = roundtrip(
        r#"
        struct FsOut {
            #[location(0)] color: vec4,
            #[location(1)] normal: vec4,
        }

        #[fragment]
        fn fs() -> FsOut {
            FsOut { color: vec4(1.0), normal: vec4(0.0) }
        }
        "#,
    );
    assert!(wgsl.contains("@location(1) normal"), "{wgsl}");
}

#[test]
fn integer_varying_needs_flat() {
    let msg = reject(
        r#"
        struct VsOut {
            #[builtin(position)] pos: vec4,
            #[location(0)] id: u32,
        }
        #[vertex]
        fn vs(#[location(0)] p: vec4) -> VsOut { VsOut { pos: p, id: 0u32 } }
        "#,
    );
    assert!(msg.contains("flat"), "{msg}");
}

#[test]
fn integer_varying_with_flat() {
    let wgsl = roundtrip(
        r#"
        struct VsOut {
            #[builtin(position)] pos: vec4,
            #[location(0)] #[flat] id: u32,
        }
        #[vertex]
        fn vs(#[location(0)] p: vec4) -> VsOut { VsOut { pos: p, id: 0u32 } }
        "#,
    );
    assert!(wgsl.contains("@interpolate(flat)"), "{wgsl}");
}

#[test]
fn plain_struct_keeps_no_bindings() {
    // Data structs are unaffected; they still work as uniform buffers.
    let wgsl = roundtrip(
        r#"
        struct Camera { view: mat4, near: f32 }
        #[group(0)] #[binding(0)] static cam: Camera = ();
        fn f() -> f32 { cam.near }
        "#,
    );
    assert!(!wgsl.contains("@location"), "{wgsl}");
}

#[test]
fn rejects_half_bound_struct() {
    let msg = reject("struct S { #[location(0)] a: f32, b: f32 } fn f(s: S) -> f32 { s.b }");
    assert!(msg.contains("mixes"), "{msg}");
}

#[test]
fn rejects_redundant_output_attribute() {
    let msg = reject(
        r#"
        struct VsOut { #[builtin(position)] pos: vec4 }
        #[vertex]
        #[output(location(0))]
        fn vs() -> VsOut { VsOut { pos: vec4(0.0) } }
        "#,
    );
    assert!(msg.contains("output"), "{msg}");
}

#[test]
fn rejects_flat_without_location() {
    let msg = reject("struct S { #[flat] a: f32 } fn f(s: S) -> f32 { s.a }");
    assert!(msg.contains("flat"), "{msg}");
}
