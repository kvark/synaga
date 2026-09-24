#![cfg(feature = "wgsl")]
//! The rest of WGSL's builtin surface: data packing, relational folds,
//! barriers, `discard`, atomics, and the memory spaces that are not resources.

mod common;

use common::*;

#[test]
fn rgba_component_names() {
    // WGSL names components twice over; `.a` is `.w`.
    let wgsl = roundtrip("fn f(c: vec4) -> f32 { c.a + c.r + c.gb.x }");
    assert!(wgsl.contains("c.w"), "{wgsl}");
    assert!(wgsl.contains("c.yz"), "{wgsl}");
}

#[test]
fn rejects_a_swizzle_mixing_both_sets() {
    // `.xg` names the same components twice over and is rejected in WGSL too.
    let msg = reject("fn f(c: vec4) -> vec2 { c.xg }");
    assert!(msg.contains("swizzle"), "{msg}");
}

#[test]
fn packing_functions() {
    let wgsl = roundtrip("fn f(v: vec4) -> vec4 { unpack4x8unorm(pack4x8snorm(v)) }");
    assert!(wgsl.contains("pack4x8snorm"), "{wgsl}");
    assert!(wgsl.contains("unpack4x8unorm"), "{wgsl}");
}

#[test]
fn relational_folds() {
    let wgsl = roundtrip("fn f(v: vec3) -> bool { all(v > vec3(0.0)) || any(v < vec3(1.0)) }");
    assert!(wgsl.contains("all("), "{wgsl}");
    assert!(wgsl.contains("any("), "{wgsl}");
}

#[test]
fn rejects_all_on_a_float_vector() {
    let msg = reject("fn f(v: vec3) -> bool { all(v) }");
    assert!(msg.contains("mismatch"), "{msg}");
}

#[test]
fn more_math() {
    validate_only(
        "fn f(a: f32, b: f32) -> f32 { step(a, b) + trunc(a) + degrees(b) + radians(a) + tanh(b) }",
    );
    validate_only("fn f(a: u32) -> u32 { countOneBits(a) + reverseBits(a) + firstLeadingBit(a) }");
}

#[test]
fn workgroup_memory_and_barrier() {
    let wgsl = roundtrip_unbound(
        r#"
        #[workgroup] static scratch: [f32; 64] = ();
        #[compute] #[workgroup_size(64)]
        fn cs(#[builtin(local_invocation_index)] i: u32) {
            scratch[i] = 1.0;
            workgroupBarrier();
        }
        "#,
    );
    assert!(wgsl.contains("var<workgroup> scratch"), "{wgsl}");
    assert!(wgsl.contains("workgroupBarrier()"), "{wgsl}");
}

#[test]
fn private_memory() {
    let wgsl = roundtrip_unbound(
        "#[private] static counter: u32 = (); fn f() -> u32 { counter = 1u32; counter }",
    );
    assert!(wgsl.contains("var<private> counter"), "{wgsl}");
}

#[test]
fn rejects_a_binding_on_workgroup_memory() {
    let msg =
        reject("#[group(0)] #[binding(0)] #[workgroup] static s: f32 = (); fn f() -> f32 { s }");
    assert!(msg.contains("binding"), "{msg}");
}

#[test]
fn discard_in_a_fragment_shader() {
    let wgsl = roundtrip_unbound(
        r#"
        #[fragment] #[output(location(0))]
        fn fs(#[location(0)] c: vec4) -> vec4 { if c.a < 0.5 { discard(); } c }
        "#,
    );
    assert!(wgsl.contains("discard;"), "{wgsl}");
}

#[test]
fn rejects_discard_as_a_value() {
    let msg = reject("fn f() -> f32 { discard() }");
    assert!(msg.contains("no value"), "{msg}");
}

#[test]
fn atomics() {
    // The variable is passed directly; there is nothing else the argument to an
    // atomic could mean, so the `&` WGSL requires is left out.
    let wgsl = roundtrip_unbound(
        r#"
        #[storage(read_write)] static counter: atomic<u32> = ();
        #[compute] #[workgroup_size(1)]
        fn cs() {
            atomicAdd(counter, 1);
            atomicStore(counter, 0u32);
        }
        fn read_back() -> u32 { atomicLoad(counter) }
        "#,
    );
    assert!(wgsl.contains("atomic<u32>"), "{wgsl}");
    assert!(wgsl.contains("atomicAdd((&counter), 1u)"), "{wgsl}");
    assert!(wgsl.contains("atomicStore"), "{wgsl}");
    assert!(wgsl.contains("atomicLoad"), "{wgsl}");
}

#[test]
fn atomic_read_modify_write_yields_the_old_value() {
    let wgsl = roundtrip_unbound(
        r#"
        #[storage(read_write)] static counter: atomic<u32> = ();
        fn bump() -> u32 { atomicAdd(counter, 1) }
        "#,
    );
    assert!(wgsl.contains("= atomicAdd("), "{wgsl}");
}

#[test]
fn rejects_an_atomic_op_on_a_plain_variable() {
    let msg = reject(
        r#"
        #[storage(read_write)] static counter: u32 = ();
        #[compute] #[workgroup_size(1)] fn cs() { atomicAdd(counter, 1); }
        "#,
    );
    assert!(msg.contains("atomic"), "{msg}");
}

#[test]
fn array_length() {
    let wgsl = roundtrip_unbound(
        r#"
        #[storage] static data: [f32] = ();
        fn n() -> u32 { arrayLength(data) }
        "#,
    );
    assert!(wgsl.contains("arrayLength("), "{wgsl}");
}

#[test]
fn rejects_array_length_of_a_fixed_array() {
    let msg = reject("fn f() -> u32 { let a = [1.0, 2.0]; arrayLength(a) }");
    assert!(msg.contains("mismatch"), "{msg}");
}

#[test]
fn bitcast_reinterprets_same_width_scalars() {
    let wgsl = roundtrip("fn f(x: f32) -> f32 { bitcast::<f32>(bitcast::<u32>(x)) }");
    assert!(wgsl.contains("bitcast<u32>"), "{wgsl}");
    assert!(wgsl.contains("bitcast<f32>"), "{wgsl}");
}
