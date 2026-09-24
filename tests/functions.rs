#![cfg(feature = "wgsl")]
//! Functions that hand nothing back, and functions that write through their
//! arguments.

mod common;

use common::*;

#[test]
fn void_function() {
    let wgsl = roundtrip_unbound(
        r#"
        #[storage(read_write)] static out: [vec4] = ();
        fn write_at(i: u32, v: vec4) {
            out[i] = v;
        }
        #[compute] #[workgroup_size(1)]
        fn cs(#[builtin(global_invocation_id)] id: vec3<u32>) {
            write_at(id.x, vec4(1.0));
        }
        "#,
    );
    assert!(
        wgsl.contains("fn write_at(i: u32, v: vec4<f32>) {"),
        "{wgsl}"
    );
    assert!(wgsl.contains("write_at(id.x,"), "{wgsl}");
}

#[test]
fn unit_return_type_is_the_same_thing() {
    validate_only("fn f() -> () { } fn g() { f(); }");
}

#[test]
fn rejects_a_void_call_as_a_value() {
    let msg = reject("fn nothing() { } fn f() -> f32 { nothing() }");
    assert!(msg.contains("no value"), "{msg}");
}

#[test]
fn a_void_function_need_not_return() {
    validate_only("fn f(a: f32) { let x = a; }");
}

/// Blade's random number generator, which threads its state through a
/// `ptr<function, RandomState>`.
#[test]
fn out_parameter() {
    let wgsl = roundtrip_unbound(
        r#"
        struct RandomState { seed: u32, index: u32 }

        fn random_u32(rng: &mut RandomState) -> u32 {
            rng.index += 1;
            rng.seed = rng.seed * 1664525u32 + 1013904223u32;
            rng.seed
        }

        fn random_f32(rng: &mut RandomState) -> f32 {
            (random_u32(rng) >> 8) as f32 / 16777216.0
        }

        fn use_it() -> f32 {
            let rng: RandomState;
            rng.seed = 1u32;
            rng.index = 0u32;
            random_f32(&mut rng)
        }
        "#,
    );
    assert!(wgsl.contains("ptr<function, RandomState>"), "{wgsl}");
    // Writes land through the pointer…
    assert!(wgsl.contains("(*rng).seed = "), "{wgsl}");
    // …a name that is already a pointer passes straight through…
    assert!(wgsl.contains("random_u32(rng_1)"), "{wgsl}");
    // …and a fresh borrow takes the address. `random_f32` ends in a digit;
    // the optional WGSL printer puts that source name back.
    assert!(wgsl.contains("random_f32((&rng_2))"), "{wgsl}");
}

#[test]
fn shared_reference_is_read_only() {
    let wgsl = roundtrip_unbound(
        r#"
        struct S { a: f32 }
        fn read_it(s: &S) -> f32 { s.a }
        fn f() -> f32 { let x: S; x.a = 1.0; read_it(&x) }
        "#,
    );
    assert!(wgsl.contains("(*s).a"), "{wgsl}");
}

#[test]
fn rejects_a_write_through_a_shared_reference() {
    // WGSL has no read-only pointer, so this is the frontend's rule to keep.
    let msg = reject("struct S { a: f32 } fn bad(s: &S) -> f32 { s.a = 1.0; s.a }");
    assert!(msg.contains("read-only"), "{msg}");
}

#[test]
fn rejects_a_mutable_borrow_of_a_read_only_global() {
    let msg = reject(
        r#"
        struct S { a: f32 }
        static u: S = ();
        fn take(s: &mut S) -> f32 { s.a }
        fn f() -> f32 { take(&mut u) }
        "#,
    );
    assert!(msg.contains("read-only"), "{msg}");
}

#[test]
fn rejects_a_reference_to_something_that_is_not_storage() {
    let msg = reject(
        r#"
        struct S { a: f32 }
        fn take(s: &mut S) -> f32 { s.a }
        fn make() -> S { S { a: 1.0 } }
        fn f() -> f32 { take(&mut make()) }
        "#,
    );
    assert!(msg.contains("assign") || msg.contains("storage"), "{msg}");
}

#[test]
fn rejects_a_reference_of_the_wrong_type() {
    let msg = reject(
        r#"
        struct S { a: f32 }
        struct T { b: f32 }
        fn take(s: &mut S) -> f32 { s.a }
        fn f() -> f32 { let t: T; t.b = 1.0; take(&mut t) }
        "#,
    );
    assert!(msg.contains("mismatch"), "{msg}");
}
