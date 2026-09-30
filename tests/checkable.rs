#![cfg(feature = "wgsl")]
//! The spelling that is also valid Rust.
//!
//! A shader written this way is checked twice: `rustc` type-checks the module
//! as ordinary Rust against `synaga-shader`, and the transpiler reads the
//! same text. These cover the transpiler's half — `examples/sprites` is the
//! arrangement working end to end, including the rustc half.

mod common;

use common::*;

#[test]
fn swizzles_as_methods() {
    // `v.xyz` cannot be a field: one piece of memory cannot carry a hundred
    // overlapping names.
    let wgsl = roundtrip("fn f(v: Vec4) -> Vec3 { v.xyz() }");
    assert!(wgsl.contains("v.xyz"), "{wgsl}");
    let wgsl = roundtrip("fn f(v: Vec4) -> f32 { v.a() + v.x }");
    assert!(wgsl.contains("v.w"), "{wgsl}");
    let wgsl = roundtrip("fn f(v: Vec4) -> Vec3 { v.rgb() }");
    assert!(wgsl.contains("v.xyz"), "{wgsl}");
}

#[test]
fn splat_and_extend_and_truncate() {
    let wgsl = roundtrip("fn f(x: f32) -> Vec3 { Vec3::splat(x) }");
    assert!(wgsl.contains("vec3(x)"), "{wgsl}");

    let wgsl = roundtrip("fn f(v: Vec3, w: f32) -> Vec4 { v.extend(w) }");
    assert!(wgsl.contains("vec4<f32>(v, w)"), "{wgsl}");

    let wgsl = roundtrip("fn f(v: Vec4) -> Vec3 { v.truncate() }");
    assert!(wgsl.contains("v.xyz"), "{wgsl}");
}

#[test]
fn typed_splat_keeps_its_component_type() {
    let wgsl = roundtrip("fn f(x: u32) -> Vec4<u32> { Vec4::<u32>::splat(x) }");
    assert!(wgsl.contains("vec4(x)"), "{wgsl}");
}

#[test]
fn extend_chains_to_four() {
    validate_only("fn f(v: Vec2) -> Vec4 { v.extend(0.0).extend(1.0) }");
}

#[test]
fn from_converts_components() {
    // Rust's `as` only works on primitives, so a vector conversion is `From`.
    let wgsl = roundtrip("fn f(v: Vec3<u32>) -> Vec3 { Vec3::from(v) }");
    assert!(wgsl.contains("vec3<f32>(v)"), "{wgsl}");
}

#[test]
fn lanewise_comparisons_are_methods() {
    // `a < b` yields one bool in Rust and one per lane in a shader.
    let wgsl = roundtrip("fn f(a: Vec3, b: Vec3) -> Vec3<bool> { a.cmple(b) }");
    assert!(wgsl.contains("a <= b"), "{wgsl}");
    validate_only("fn f(a: Vec3, b: Vec3) -> Vec3<bool> { a.cmpgt(b) }");
    validate_only("fn f(a: Vec3<u32>, b: Vec3<u32>) -> Vec3<bool> { a.cmpeq(b) }");
}

#[test]
fn zero_value_as_an_associated_constant() {
    let wgsl = roundtrip("fn f() -> Vec3 { Vec3::ZERO }");
    assert!(wgsl.contains("vec3<f32>()"), "{wgsl}");
}

#[test]
fn address_space_in_the_type() {
    let wgsl = roundtrip_unbound(
        r#"
        struct Camera { view: Mat4 }
        static camera: Uniform<Camera> = binding();
        static indices: Storage<[u32]> = binding();
        static counters: StorageMut<[u32]> = binding();
        fn f(i: u32) -> u32 { counters[i] = indices[i]; counters[i] }
        "#,
    );
    assert!(wgsl.contains("var<uniform> camera"), "{wgsl}");
    assert!(wgsl.contains("var<storage> indices"), "{wgsl}");
    assert!(wgsl.contains("var<storage, read_write> counters"), "{wgsl}");
}

#[test]
fn rejects_saying_the_address_space_twice() {
    let msg = reject("#[storage] static a: Uniform<f32> = binding(); fn f() -> f32 { a }");
    assert!(msg.contains("address space"), "{msg}");
}

#[test]
fn use_and_mod_are_skipped() {
    // A shader module is also a Rust module, so it carries the `use` that
    // brings the prelude into scope.
    validate_only(
        r#"
        use synaga_shader::*;
        use super::common::{globals, unpack_color};
        fn f(a: f32) -> f32 { a }
        "#,
    );
}

#[test]
fn a_reference_to_a_handle_is_the_handle() {
    // `&tex` has no address to take: the resource lives on the GPU.
    let wgsl = roundtrip_unbound(
        r#"
        static tex: Texture2D<f32> = binding();
        static samp: Sampler = binding();
        fn f(uv: Vec2) -> Vec4 { tex.sample_level(&samp, uv, 0.0) }
        "#,
    );
    assert!(wgsl.contains("textureSampleLevel(tex, samp"), "{wgsl}");
}

#[test]
fn storage_load_has_its_own_spelling() {
    // WGSL calls it `textureLoad` at one argument fewer, which Rust cannot do.
    validate_only(
        r#"
        #[group(0)] #[binding(0)] static acc: TextureStorage2D<Rgba32Float, ReadWrite> = binding();
        fn f(c: Vec2<i32>) -> Vec4 { acc.load(c) }
        "#,
    );
}

#[test]
fn let_underscore_evaluates_and_binds_nothing() {
    validate_only("fn f(a: f32) -> f32 { let _ = a + 1.0; a }");
}

#[test]
fn the_wgsl_shaped_spelling_still_works() {
    // Both spellings lower the same way; only one of them type-checks as Rust.
    let method = roundtrip("fn f(v: Vec4) -> Vec3 { v.xyz() }");
    let field = roundtrip("fn f(v: Vec4) -> Vec3 { v.xyz }");
    assert_eq!(method, field);
}

#[test]
fn default_is_the_zero_value() {
    // WGSL spells a zero value `T()`, and Rust spells it `T::default()`.
    let decl = "#[derive(Default)] struct S { a: f32, b: Vec3 }";
    let named = roundtrip(&format!("{decl} fn f() -> S {{ S::default() }}"));
    let called = roundtrip(&format!("{decl} fn f() -> S {{ S() }}"));
    assert_eq!(named, called);

    // Not just structs: anything with a zero value has one.
    assert!(roundtrip("fn f() -> Mat3 { Mat3::default() }").contains("mat3x3<f32>()"));
    assert!(roundtrip("fn f() -> u32 { u32::default() }").contains("u32()"));
}

#[test]
fn default_is_zero_only_where_it_is_derived() {
    // A `Default` written by hand is where the transpiler cannot see it, in
    // the host, and need not be zero.
    let msg = reject("struct S { a: f32 } fn f() -> S { S::default() }");
    assert!(
        msg.contains(
            "`S::default()` is zero on the GPU, and in Rust too only if `S` derives `Default`, \
             which it does not"
        ),
        "{msg}"
    );
    // A derived one calls each field's.
    let nested = "struct Inner { a: f32 } #[derive(Default)] struct S { i: Inner, b: f32 }";
    let msg = reject(&format!("{nested} fn f() -> S {{ S::default() }}"));
    assert!(msg.contains("only if `Inner` derives `Default`"), "{msg}");

    // `..Default::default()` is the struct's own `default()`, for the fields
    // left out.
    let msg =
        reject("struct S { a: f32, b: f32 } fn f() -> S { S { a: 1.0, ..Default::default() } }");
    assert!(msg.contains("`S::default()` is zero"), "{msg}");
    let msg = reject(&format!(
        "{nested} fn f() -> S {{ S {{ b: 1.0, ..Default::default() }} }}"
    ));
    assert!(msg.contains("only if `Inner` derives `Default`"), "{msg}");
    validate_only(&format!(
        "{nested} fn f(i: Inner) -> S {{ S {{ i, ..Default::default() }} }}"
    ));
    // `Shared` derives a `Default` that is all zeroes.
    let shared = "#[repr(C)] #[derive(Shared)] struct P { a: f32, b: Vec3 }";
    let wgsl = roundtrip(&format!("{shared} fn f() -> P {{ P::default() }}"));
    assert!(wgsl.contains("return P();"), "{wgsl}");
}

#[test]
fn usize_indexes_what_rust_insists_on_indexing_by_usize() {
    // `[T; N]`, `[T]` and the prelude's vectors index by `usize` because that
    // is the only thing `Index` offers, so a checkable shader has to be able to
    // write it. On a GPU an index is 32-bit.
    let by_usize = roundtrip("fn f(a: &mut [u32; 4], i: u32) -> u32 { a[i as usize] }");
    let by_u32 = roundtrip("fn f(a: &mut [u32; 4], i: u32) -> u32 { a[i] }");
    assert_eq!(by_usize, by_u32);

    let aliased = roundtrip("fn f(i: usize) -> usize { i }");
    assert!(
        aliased.contains("i: u32") && aliased.contains("-> u32"),
        "{aliased}"
    );
}

#[test]
fn set_writes_a_whole_resource() {
    // `*emit_end = n` would borrow a `static` mutably, which Rust refuses, so
    // a resource has `Cell`'s `set` for it.
    let wgsl = roundtrip_unbound(
        r#"
        static emit_end: Workgroup<i32> = binding();
        static totals: StorageMut<Vec2<u32>> = binding();
        #[entry_point(compute, threads(64))]
        fn emit(local_invocation_index: u32) {
            if local_invocation_index == 0 {
                emit_end.set(3);
                totals.set(vec2(1, 2));
            }
            workgroup_barrier();
            let read = *emit_end;
        }
        "#,
    );
    assert!(wgsl.contains("emit_end = 3i;"), "{wgsl}");
    assert!(wgsl.contains("totals = vec2<u32>(1u, 2u);"), "{wgsl}");
    let msg = reject("static tint: Uniform<Vec4> = binding(); fn f() { tint.set(Vec4::ONE); }");
    assert!(msg.contains("read-only"), "{msg}");
    // A part of a resource is written by assigning to it.
    let msg =
        reject("struct S { a: u32 } static s: StorageMut<S> = binding(); fn f() { s.a.set(1); }");
    assert!(msg.contains("set"), "{msg}");
}

#[test]
fn a_write_goes_through_get_mut() {
    // A writable resource is a plain `static`: a read derefs it, and a write
    // goes through `get_mut`. The accessor is for `rustc`; the storage is the
    // same. `unsafe`, which an older `get_mut` wanted, is still read through.
    let wgsl = roundtrip_unbound(
        r#"
        static counts: StorageMut<[u32]> = binding();
        #[entry_point(compute, threads(64))]
        fn tally(#[builtin(global_invocation_id)] id: Vec3<u32>) {
            counts.get_mut()[id.x as usize] += 1;
            let doubled = counts[0] * 2;
            unsafe { counts.get_mut()[1] = doubled };
        }
        "#,
    );
    assert!(wgsl.contains("counts[id.x] = ("), "{wgsl}");
    assert!(wgsl.contains("counts[1] = ("), "{wgsl}");
    assert!(wgsl.contains(" * 2u)"), "{wgsl}");
}
