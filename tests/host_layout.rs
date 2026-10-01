//! A `#[repr(C)]` struct is one the host shares: it uploads the bytes `rustc`
//! lays out, and the shader reads them where the GPU's layout rules put each
//! field. The transpiler lays out every such struct a buffer holds both ways,
//! and says where they part, since nothing else would.

mod common;

use common::*;

/// A shader that reads a `P` from a uniform buffer, after `items`.
fn uniform(items: &str) -> String {
    format!("{items} static p: Uniform<P> = binding(); fn f() -> P {{ *p }}")
}

#[test]
fn a_struct_laid_out_the_same_both_ways_is_accepted() {
    validate_only_unbound(&uniform(
        "#[repr(C)] #[derive(Clone, Copy)]
         struct P { a: Vec3, b: f32, c: Vec2<u32>, d: Vec2, m: Mat4, e: [Vec4; 2], n: Inner }
         #[repr(C)] #[derive(Clone, Copy)]
         struct Inner { x: u32, y: u32, z: f32, w: f32 }",
    ));
}

#[test]
fn a_field_the_gpu_aligns_further_is_reported_with_the_padding_it_needs() {
    let msg = reject(&uniform("#[repr(C)] struct P { a: f32, b: Vec2 }"));
    assert!(
        msg.contains(
            "`P` is `#[repr(C)]`, which says the host shares it, but `b` is at byte 4 in Rust \
             and 8 on the GPU; 4 bytes of padding before it line them up"
        ),
        "{msg}"
    );
    validate_only_unbound(&uniform(
        "#[repr(C)] struct P { a: f32, _pad: f32, b: Vec2 }",
    ));
}

#[test]
fn a_size_the_gpu_rounds_up_is_reported_too() {
    let msg = reject(&uniform("#[repr(C)] struct P { a: Vec3 }"));
    assert!(
        msg.contains("it is 12 bytes in Rust and 16 on the GPU; 4 bytes of padding at the end"),
        "{msg}"
    );
    validate_only_unbound(&uniform("#[repr(C)] struct P { a: Vec3, _pad: f32 }"));
    // `align(16)` rounds Rust's size up the same way.
    validate_only_unbound(&uniform("#[repr(C, align(16))] struct P { a: Vec3 }"));
}

#[test]
fn a_matrix_is_laid_out_as_the_gpu_lays_it_out() {
    // Each 3-lane column takes four, in Rust as on the GPU.
    validate_only_unbound(&uniform(
        "#[repr(C)] struct P { m: Mat3, t: Mat4x3, s: Mat2x3 }",
    ));
}

#[test]
fn a_field_laid_out_differently_inside_cannot_be_padded_into_place() {
    let msg = reject(
        "#[repr(C)] struct P { v: [Vec3; 2] }
         static p: Storage<P> = binding(); fn f() -> Vec3 { p.v[0] }",
    );
    assert!(
        msg.contains("`v` is an array whose elements are 12 bytes apart in Rust and 16 on the GPU"),
        "{msg}"
    );
    let msg = reject(&uniform("#[repr(C)] struct P { b: bool }"));
    assert!(msg.contains("`b` is a `bool`"), "{msg}");
}

#[test]
fn a_shared_struct_holds_only_shared_structs() {
    let msg = reject(&uniform(
        "#[derive(Clone, Copy)] struct Inner { x: f32 } #[repr(C)] struct P { i: Inner }",
    ));
    assert!(
        msg.contains("`i` is `Inner`, which is not `#[repr(C)]`"),
        "{msg}"
    );
    // Checked before the struct that holds it, and named as the culprit.
    let msg = reject(&uniform(
        "#[repr(C)] struct Inner { a: f32, b: Vec2 } #[repr(C)] struct P { i: Inner }",
    ));
    assert!(msg.contains("`Inner` is `#[repr(C)]`"), "{msg}");
}

#[test]
fn a_runtime_sized_array_is_placed_but_has_no_size() {
    validate_only_unbound(
        "#[repr(C)] struct Buf { count: u32, _pad: u32, data: [Vec2] }
         static buf: Storage<Buf> = binding();
         fn f() -> f32 { buf.data[0].x }",
    );
    let msg = reject(
        "#[repr(C)] struct Buf { count: u32, data: [Vec2] }
         static buf: Storage<Buf> = binding();
         fn f() -> f32 { buf.data[0].x }",
    );
    assert!(
        msg.contains("`data` is at byte 4 in Rust and 8 on the GPU"),
        "{msg}"
    );
    // The elements of a buffer that is only an array are checked too.
    let msg = reject(
        "#[repr(C)] #[derive(Clone, Copy)] struct E { a: f32, b: Vec2 }
         static items: Storage<[E]> = binding();
         fn f() -> f32 { items[0].a }",
    );
    assert!(msg.contains("`E` is `#[repr(C)]`"), "{msg}");
}

#[test]
fn only_what_a_buffer_holds_has_to_match() {
    // Laid out the GPU's way only, which is all the shader itself needs.
    validate_only_unbound(
        "struct P { a: f32, b: Vec2 } static p: Uniform<P> = binding(); fn f() -> f32 { p.a }",
    );
    // A vertex's attributes are wherever the vertex buffer's format says.
    validate_only_unbound(
        "#[repr(C)] #[derive(Clone, Copy)] struct V { pos: Vec2, uv: Vec2, color: u32 }
         #[entry_point(vertex)] fn vs(v: V) -> Vec4 { v.pos.extend(v.uv.x).extend(1.0) }",
    );
    // So are an interface struct's fields, which are bindings.
    validate_only(
        "#[repr(C)] #[derive(Io)] struct V { #[location(0)] a: f32, #[location(1)] b: Vec2 }
         #[entry_point(vertex)] fn vs(v: V) -> Vec4 { vec4(v.a, v.b.x, v.b.y, 1.0) }",
    );
    let msg = reject("#[repr(C, packed)] struct P { a: f32 } fn f(p: P) -> f32 { p.a }");
    assert!(msg.contains("the GPU has no packed layout"), "{msg}");
}

#[test]
fn a_pointer_sized_field_cannot_be_shared() {
    // `usize` is a `u32` on the GPU, which is also what the layout check sees,
    // so without this the struct would look like 4 bytes on both sides.
    let msg = reject(&uniform("#[repr(C)] struct P { count: usize }"));
    assert!(
        msg.contains(
            "`P` is `#[repr(C)]`, which says the host shares it, but its field `count` is a \
             `usize`, which is as wide as a pointer on the host and 32 bits on the GPU: use `u32`"
        ),
        "{msg}"
    );
    let msg = reject(&uniform("#[repr(C)] struct P { offset: isize }"));
    assert!(
        msg.contains("`isize`") && msg.contains("use `i32`"),
        "{msg}"
    );
    // Through an array, an alias, or a struct that holds one.
    let msg = reject(&uniform("#[repr(C)] struct P { counts: [usize; 4] }"));
    assert!(msg.contains("`counts` is a `usize`"), "{msg}");
    let msg = reject(&uniform(
        "type Count = usize; type Counts = [Count; 2]; #[repr(C)] struct P { counts: Counts }",
    ));
    assert!(msg.contains("`counts` is a `usize`"), "{msg}");
    let msg = reject(&uniform(
        "#[repr(C)] struct Inner { n: usize } #[repr(C)] struct P { inner: Inner }",
    ));
    assert!(msg.contains("`Inner` is `#[repr(C)]`"), "{msg}");
}

#[test]
fn a_buffer_of_pointer_sized_integers_cannot_be_shared() {
    let msg = reject("static counts: Storage<[usize]> = binding(); fn f() -> usize { counts[0] }");
    assert!(
        msg.contains(
            "`counts` is shared with the host, but it holds a `usize`, which is as wide as a \
             pointer there and 32 bits on the GPU: use `u32`"
        ),
        "{msg}"
    );
    let msg = reject("static n: Uniform<isize> = binding(); fn f() -> isize { *n }");
    assert!(msg.contains("use `i32`"), "{msg}");
}

#[test]
fn usize_still_indexes() {
    // Indexing takes `usize` in Rust, so a shader has to be able to say it.
    // Only sharing one with the host is refused.
    validate_only_unbound(
        "static items: Storage<[u32]> = binding();
         #[derive(Clone, Copy)] struct Local { i: usize }
         fn f(i: u32) -> u32 { let local = Local { i: i as usize }; items[local.i] }",
    );
}

#[test]
fn a_transparent_newtype_is_checked_as_shared() {
    // `Shared` accepts `#[repr(transparent)]`, so the transpiler has to read it
    // as saying the host shares the struct too — otherwise the derive would make
    // it a shared struct and the layout check would skip it.
    // A `Vec3<f32>` is 12 bytes in Rust and takes 16 on the GPU, which shows as
    // a size difference whether anything follows it or not.
    let msg = reject(&uniform(
        "#[repr(transparent)] #[derive(Clone, Copy)]
         struct P { value: Vec3<f32> }",
    ));
    assert!(
        msg.contains("is 12 bytes in Rust and 16 on the GPU"),
        "{msg}"
    );
}

#[test]
fn a_transparent_newtype_that_lines_up_is_accepted() {
    // The same `repr`, where the field is where the GPU puts it.
    validate_only_unbound(&uniform(
        "#[repr(transparent)] #[derive(Clone, Copy)]
         struct P { value: u32 }",
    ));
}
