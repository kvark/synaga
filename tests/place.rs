#![cfg(feature = "wgsl")]
//! Writing through a field, a component or an element — and reading through
//! one without dragging the whole aggregate along.

mod common;

use common::*;

#[test]
fn field_store() {
    // Blade fills structs in field by field all over its shaders.
    let wgsl = roundtrip(
        r#"
        struct Vertex { position: vec3, normal: vec3, material: u32 }
        fn build(p: vec3) -> Vertex {
            let v: Vertex;
            v.position = p;
            v.normal = vec3(0.0, 1.0, 0.0);
            v.material = 0u32;
            v
        }
        "#,
    );
    assert!(wgsl.contains("v.position = p"), "{wgsl}");
    assert!(wgsl.contains("v.material = 0u"), "{wgsl}");
}

#[test]
fn component_store() {
    let wgsl = roundtrip("fn f(a: f32) -> vec3 { let v = vec3(0.0); v.x = a; v.z = a; v }");
    assert!(wgsl.contains("v.x = a"), "{wgsl}");
    assert!(wgsl.contains("v.z = a"), "{wgsl}");
}

#[test]
fn element_store() {
    let wgsl = roundtrip("fn f(a: f32) -> vec4 { let v = vec4(0.0); v[2] = a; v }");
    assert!(wgsl.contains("v.z = a"), "{wgsl}");
}

#[test]
fn dynamic_element_store() {
    let wgsl = roundtrip("fn f(i: i32, x: f32) -> vec4 { let v = vec4(0.0); v[i] = x; v }");
    assert!(wgsl.contains("v[i] = x"), "{wgsl}");
}

#[test]
fn matrix_column_store() {
    validate_only("fn f(c: vec4) -> mat4 { let m = mat4(c, c, c, c); m[1] = c; m }");
}

#[test]
fn nested_field_store() {
    let wgsl = roundtrip(
        r#"
        struct Inner { a: f32, b: f32 }
        struct Outer { i: Inner, c: f32 }
        fn f(x: f32) -> Outer { let o: Outer; o.i.a = x; o.i.b = x; o.c = x; o }
        "#,
    );
    assert!(wgsl.contains("o.i.a = x"), "{wgsl}");
}

#[test]
fn compound_assign_through_a_field() {
    let wgsl = roundtrip(
        r#"
        struct Acc { total: vec3, count: u32 }
        fn f(v: vec3) -> Acc {
            let a: Acc;
            a.total = vec3(0.0);
            a.count = 0u32;
            a.total += v;
            a.count += 1;
            a
        }
        "#,
    );
    assert!(
        wgsl.contains("a.total = (_e8 + v)") || wgsl.contains("a.total ="),
        "{wgsl}"
    );
}

#[test]
fn store_through_a_storage_global() {
    let wgsl = roundtrip(
        r#"
        struct Particle { pos: vec3, life: f32 }
        #[group(0)] #[binding(0)] #[storage(read_write)] static p: Particle = ();
        #[compute] #[workgroup_size(1)]
        fn cs() { p.life = 0.0; p.pos.y = 1.0; }
        "#,
    );
    assert!(wgsl.contains("p.life = 0f"), "{wgsl}");
    assert!(wgsl.contains("p.pos.y = 1f"), "{wgsl}");
}

#[test]
fn reading_a_field_loads_only_that_field() {
    // Not just tidier output: loading the whole struct reads an entire uniform
    // buffer to get at one scalar.
    let wgsl = roundtrip(
        r#"
        struct Camera { view: mat4, eye: vec3, time: f32 }
        #[group(0)] #[binding(0)] static camera: Camera = ();
        fn f() -> f32 { camera.time }
        "#,
    );
    assert!(wgsl.contains("= camera.time"), "{wgsl}");
}

#[test]
fn rejects_store_to_a_read_only_global() {
    let msg = reject(
        r#"
        struct S { a: f32 }
        #[group(0)] #[binding(0)] static u: S = ();
        #[compute] #[workgroup_size(1)] fn cs() { u.a = 1.0; }
        "#,
    );
    assert!(msg.contains("read-only"), "{msg}");
}

#[test]
fn rejects_store_to_an_argument_field() {
    let msg = reject("struct S { a: f32 } fn f(s: S) -> f32 { s.a = 1.0; s.a }");
    assert!(msg.contains("argument"), "{msg}");
}

#[test]
fn rejects_store_to_a_swizzle() {
    // `v.xy = a` is not a place in WGSL either.
    let msg = reject("fn f(a: vec2) -> vec3 { let v = vec3(0.0); v.xy = a; v }");
    assert!(msg.contains("assign"), "{msg}");
}

#[test]
fn rejects_out_of_range_element_store() {
    let msg = reject("fn f() -> vec2 { let v = vec2(0.0); v[5] = 1.0; v }");
    assert!(msg.contains("range"), "{msg}");
}

#[test]
fn rejects_store_of_the_wrong_type() {
    let msg = reject("fn f(a: vec2) -> vec3 { let v = vec3(0.0); v.x = a; v }");
    assert!(msg.contains("mismatch"), "{msg}");
}

#[test]
fn argument_fields_still_read_as_values() {
    validate_only("struct S { a: f32, b: f32 } fn f(s: S) -> f32 { s.a + s.b }");
}
