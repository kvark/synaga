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
        #[entry_point(compute, threads(64))]
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
        #[entry_point(fragment)] #[output(location(0))]
        fn fs(#[location(0)] c: vec4) -> vec4 { if c.a < 0.5 { discard(); } c }
        "#,
    );
    assert!(wgsl.contains("discard;"), "{wgsl}");
}

#[test]
fn rejects_discard_as_a_value() {
    let msg = reject("fn f() -> f32 { discard() + 1.0 }");
    assert!(msg.contains("no value"), "{msg}");
}

#[test]
fn discard_can_end_a_function_that_returns() {
    // It never returns, so nothing falls off the end.
    validate_only(
        r#"
        #[entry_point(fragment)]
        #[output(location(0))]
        fn fs(#[location(0)] a: f32) -> vec4 {
            if a < 0.5 { discard() }
            vec4::splat(a)
        }
        fn f() -> f32 { discard() }
        "#,
    );
}

#[test]
fn atomics() {
    // `synaga_shader::AtomicU32`: the standard methods, without an `Ordering`.
    // They take `&self`, so a buffer changed only through its atomics needs
    // no `static mut`.
    let wgsl = roundtrip_unbound(
        r#"
        struct Counters { hits: AtomicU32, low: AtomicI32 }
        static counters: StorageMut<Counters> = binding();
        static scratch: Workgroup<AtomicU32> = binding();
        #[entry_point(compute, threads(64))]
        fn cs() {
            counters.hits.fetch_add(1);
            counters.hits.store(0);
            counters.low.fetch_min(-4);
            scratch.fetch_or(2);
            let old = counters.hits.swap(7);
        }
        fn read_back() -> u32 { counters.hits.load() + scratch.load() }
        "#,
    );
    assert!(wgsl.contains("hits: atomic<u32>"), "{wgsl}");
    assert!(wgsl.contains("low: atomic<i32>"), "{wgsl}");
    assert!(wgsl.contains("atomicAdd((&counters.hits), 1u)"), "{wgsl}");
    assert!(wgsl.contains("atomicStore((&counters.hits), 0u)"), "{wgsl}");
    assert!(wgsl.contains("atomicMin((&counters.low), -(4i))"), "{wgsl}");
    assert!(wgsl.contains("atomicOr((&scratch), 2u)"), "{wgsl}");
    assert!(
        wgsl.contains("atomicExchange((&counters.hits), 7u)"),
        "{wgsl}"
    );
    assert!(wgsl.contains("atomicLoad((&counters.hits))"), "{wgsl}");
}

#[test]
fn atomic_read_modify_write_yields_the_old_value() {
    let wgsl = roundtrip_unbound(
        r#"
        static counter: StorageMut<AtomicU32> = binding();
        fn bump() -> u32 { counter.fetch_add(1) }
        "#,
    );
    assert!(wgsl.contains("= atomicAdd("), "{wgsl}");
}

#[test]
fn compare_exchange_hands_back_a_plain_pair() {
    // WGSL's `__atomic_compare_exchange_result`, not a `Result`.
    let wgsl = roundtrip_unbound(
        r#"
        static lock: StorageMut<AtomicU32> = binding();
        fn try_lock() -> bool {
            let r: CompareExchange<u32> = lock.compare_exchange_weak(0, 1);
            r.exchanged && r.old_value == 0
        }
        "#,
    );
    assert!(
        wgsl.contains("atomicCompareExchangeWeak((&lock), 0u, 1u)"),
        "{wgsl}"
    );
    assert!(wgsl.contains(".exchanged"), "{wgsl}");
    assert!(wgsl.contains(".old_value"), "{wgsl}");
}

#[test]
fn an_atomic_method_on_a_plain_variable_is_not_one() {
    let msg = reject(
        r#"
        static counter: StorageMut<u32> = binding();
        #[entry_point(compute, threads(1))] fn cs() { counter.fetch_add(1); }
        "#,
    );
    assert!(msg.contains("fetch_add"), "{msg}");
}

#[test]
fn a_runtime_sized_array_asks_for_its_length() {
    // Rust's `len()`: `arrayLength` in WGSL, and `usize` is `u32` here.
    let wgsl = roundtrip_unbound(
        r#"
        #[storage] static data: [f32] = ();
        fn n() -> u32 { data.len() }
        "#,
    );
    assert!(wgsl.contains("arrayLength("), "{wgsl}");
}

#[test]
fn a_fixed_array_knows_its_length() {
    // `.len()` as Rust has it: part of the type, so a constant.
    let wgsl = roundtrip("fn f() -> u32 { let a = [1.0, 2.0]; a.len() }");
    assert!(wgsl.contains("return 2u;"), "{wgsl}");
}

#[test]
fn bitcast_reinterprets_same_width_scalars() {
    let wgsl = roundtrip("fn f(x: f32) -> f32 { bitcast::<f32>(bitcast::<u32>(x)) }");
    assert!(wgsl.contains("bitcast<u32>"), "{wgsl}");
    assert!(wgsl.contains("bitcast<f32>"), "{wgsl}");
}
