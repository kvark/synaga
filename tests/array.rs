#![cfg(feature = "wgsl")]
//! `[T; N]` and `[T]` — the latter being what a storage buffer holds.

mod common;

use common::*;

#[test]
fn runtime_sized_storage_buffer() {
    let wgsl = roundtrip(
        r#"
        struct Particle { pos: vec3, life: f32 }
        #[group(0)] #[binding(0)] #[storage(read_write)] static particles: [Particle] = ();
        #[compute] #[workgroup_size(64)]
        fn cs(#[builtin(global_invocation_id)] id: vec3<u32>) {
            particles[id.x].life = 0.0;
        }
        "#,
    );
    assert!(wgsl.contains("array<Particle>"), "{wgsl}");
    assert!(wgsl.contains("particles[id.x].life = 0f"), "{wgsl}");
}

#[test]
fn runtime_sized_scalars() {
    let wgsl = roundtrip(
        r#"
        #[group(0)] #[binding(0)] #[storage] static indices: [u32] = ();
        fn f(i: u32) -> u32 { indices[i] }
        "#,
    );
    assert!(wgsl.contains("array<u32>"), "{wgsl}");
}

#[test]
fn runtime_sized_tail_of_a_struct() {
    validate_only(
        r#"
        struct Buf { count: u32, items: [f32] }
        #[group(0)] #[binding(0)] #[storage] static b: Buf = ();
        fn f(i: u32) -> f32 { b.items[i] }
        "#,
    );
}

#[test]
fn extern_block_declares_one_too() {
    validate_only(
        r#"
        extern { #[group(0)] #[binding(0)] #[storage] static data: [u32]; }
        fn f(i: u32) -> u32 { data[i] }
        "#,
    );
}

#[test]
fn rejects_runtime_array_outside_storage() {
    // WGSL keeps runtime-sized arrays in storage; Naga only notices as an
    // alignment complaint about a stride nobody wrote.
    let msg = reject("static data: [u32] = (); fn f(i: u32) -> u32 { data[i] }");
    assert!(msg.contains("storage"), "{msg}");
}

#[test]
fn fixed_array_literal() {
    let wgsl = roundtrip("fn f() -> f32 { let a = [1.0, 2.0, 3.0]; a[1] }");
    assert!(wgsl.contains("array<f32, 3>"), "{wgsl}");
}

#[test]
fn fixed_array_declared_then_filled() {
    let wgsl = roundtrip("fn f(i: i32) -> f32 { let a: [f32; 4]; a[0] = 1.0; a[i] }");
    assert!(wgsl.contains("var a: array<f32, 4>"), "{wgsl}");
}

#[test]
fn array_literal_types_its_elements() {
    let wgsl = roundtrip("fn f(n: u32) -> u32 { let a = [n, 1, 2]; a[0] }");
    assert!(wgsl.contains("1u"), "{wgsl}");
}

#[test]
fn array_length_from_a_const() {
    let wgsl = roundtrip("const N: u32 = 4; fn f() -> f32 { let a: [f32; N]; a[0] = 1.0; a[0] }");
    assert!(wgsl.contains("array<f32, 4>"), "{wgsl}");
}

#[test]
fn array_of_vectors_in_a_uniform() {
    // vec4 strides at 16, so this one satisfies the uniform layout rules.
    validate_only(
        r#"
        struct Frame { taps: [vec4; 4] }
        #[group(0)] #[binding(0)] static frame: Frame = ();
        fn f() -> vec4 { frame.taps[2] }
        "#,
    );
}

#[test]
fn rejects_out_of_range_literal_index() {
    let msg = reject("fn f() -> f32 { let a = [1.0, 2.0]; a[5] }");
    assert!(msg.contains("range"), "{msg}");
}

#[test]
fn rejects_mixed_element_types() {
    let msg = reject("fn f() -> f32 { let a = [1.0, 2u32]; a[0] }");
    assert!(msg.contains("mismatch"), "{msg}");
}

#[test]
fn validation_errors_say_why() {
    // `[f32; 4]` strides at 4, which uniform layout will not have. The reason
    // lives in Naga's source chain; losing it leaves only "is invalid".
    let src = r#"
        struct Frame { weights: [f32; 4] }
        #[group(0)] #[binding(0)] static frame: Frame = ();
        fn f() -> f32 { frame.weights[2] }
    "#;
    let module = synaga::parse_str(src).expect("parse");
    let err = synaga::validate(&module).expect_err("uniform layout");
    let msg = err.to_string();
    assert!(msg.contains("stride"), "{msg}");
    assert!(msg.contains("alignment"), "{msg}");
}
