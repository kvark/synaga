//! The Rust name of every type and builtin, against the WGSL one it stands for.
//!
//! A checkable shader writes `Vec3<i32>`, `Texture2D<f32>` and
//! `workgroup_barrier()`, as Rust names things. The transpiler still reads
//! WGSL's own names, for a source `rustc` never sees, and the two spellings
//! have to build the same module.

mod common;

use common::*;

/// Lower both sources and require the same module. Nothing of a type's
/// spelling survives into Naga, so any difference is a difference in meaning.
fn same_module(rust: &str, wgsl: &str) {
    let lower = |src: &str| {
        let module = synaga::parse_str(src).unwrap_or_else(|e| panic!("parse: {e}\n{src}"));
        format!("{module:#?}")
    };
    assert_eq!(lower(rust), lower(wgsl), "\n{rust}\n{wgsl}");
}

#[test]
fn handle_types_are_wgsl_words_capitalized() {
    for (rust, wgsl) in [
        ("Texture1D<f32>", "texture_1d<f32>"),
        ("Texture2D<u32>", "texture_2d<u32>"),
        ("Texture2DArray<i32>", "texture_2d_array<i32>"),
        ("Texture3D<f32>", "texture_3d<f32>"),
        ("TextureCube<f32>", "texture_cube<f32>"),
        ("TextureCubeArray<f32>", "texture_cube_array<f32>"),
        ("TextureMultisampled2D<f32>", "texture_multisampled_2d<f32>"),
        ("TextureDepth2D", "texture_depth_2d"),
        ("TextureDepth2DArray", "texture_depth_2d_array"),
        ("TextureDepthCube", "texture_depth_cube"),
        ("TextureDepthCubeArray", "texture_depth_cube_array"),
        (
            "TextureDepthMultisampled2D",
            "texture_depth_multisampled_2d",
        ),
        (
            "TextureStorage1D<R8Unorm, Write>",
            "texture_storage_1d<R8Unorm, Write>",
        ),
        (
            "TextureStorage2D<Rgba8Unorm, ReadWrite>",
            "texture_storage_2d<Rgba8Unorm, ReadWrite>",
        ),
        (
            "TextureStorage2DArray<R16Uint, Read>",
            "texture_storage_2d_array<R16Uint, Read>",
        ),
        (
            "TextureStorage3D<Rg8Snorm, Write>",
            "texture_storage_3d<Rg8Snorm, Write>",
        ),
        ("Sampler", "sampler"),
        ("SamplerComparison", "sampler_comparison"),
        ("AccelerationStructure", "acceleration_structure"),
        (
            "BindingArray<Texture2D<f32>, 4>",
            "binding_array<texture_2d<f32>, 4>",
        ),
        ("BindingArray<Sampler>", "binding_array<sampler>"),
    ] {
        same_module(
            &format!("static t: {rust} = binding();"),
            &format!("static t: {wgsl} = binding();"),
        );
    }
}

#[test]
fn a_struct_of_the_shaders_own_is_not_taken_for_a_texture() {
    validate_only(
        r#"
        struct TextureParams { scale: f32 }
        static params: Uniform<TextureParams> = group(0).binding(0);
        fn f() -> f32 { params.scale }
        "#,
    );
}

#[test]
fn vectors_take_their_scalar_as_a_type_argument() {
    // `Vec3<u32>` is `vec3u` as a type, a constructor, a path and a constant.
    same_module(
        r#"
        fn f(a: Vec3<u32>, b: Vec2<i32>, c: Vec4<bool>, d: Vec3<f32>) -> Vec3<u32> {
            let _ = b;
            let _ = c;
            a + vec3::<u32>(1, 2, 3) + Vec3::<u32>::splat(4) + Vec3::<u32>::ONE
                + Vec3::<u32>::ZERO + Vec3::<u32>::default() + Vec3::<u32>::from(d)
        }
        "#,
        r#"
        fn f(a: vec3u, b: vec2i, c: vec4<bool>, d: vec3f) -> vec3u {
            let _ = b;
            let _ = c;
            a + vec3u(1, 2, 3) + vec3u::splat(4) + vec3u::ONE
                + vec3u::ZERO + vec3u::default() + vec3u::from(d)
        }
        "#,
    );
}

#[test]
fn a_bare_vector_is_f32() {
    same_module(
        "fn f(v: Vec3) -> Vec4 { v.extend(1.0) + Vec4::ONE + vec4(0.0, 1.0, 0.0, 1.0) }",
        "fn f(v: vec3<f32>) -> vec4<f32> { v.extend(1.0) + vec4f::ONE + vec4(0.0, 1.0, 0.0, 1.0) }",
    );
}

#[test]
fn a_vectors_scalar_is_said_once() {
    let msg = reject("fn f() -> Vec3<u32> { vec3u::<u32>(1, 2, 3) }");
    assert!(msg.contains("vec3u"), "{msg}");
    let msg = reject("fn f() -> Vec3 { Vec3::<f64>::ZERO }");
    assert!(msg.contains("f64"), "{msg}");
}

#[test]
fn matrices_are_square_by_name_and_columns_first_otherwise() {
    same_module(
        "fn f(m: Mat4, n: Mat4x3, v: Vec4) -> Vec3 { n * (m * v) }",
        "fn f(m: mat4x4<f32>, n: mat4x3f, v: vec4f) -> vec3f { n * (m * v) }",
    );
}

#[test]
fn builtins_are_snake_case() {
    same_module(
        r#"
        #[entry_point(compute, threads(1))]
        fn main() {
            workgroup_barrier();
            storage_barrier();
            let _ = inverse_sqrt(2.0);
            let _ = count_one_bits(7u32) + reverse_bits(1u32);
            let _ = first_leading_bit(8u32) + first_trailing_bit(8u32);
            let _ = count_leading_zeros(1u32) + count_trailing_zeros(2u32);
        }
        "#,
        r#"
        #[entry_point(compute, threads(1))]
        fn main() {
            workgroupBarrier();
            storageBarrier();
            let _ = inverseSqrt(2.0);
            let _ = countOneBits(7u32) + reverseBits(1u32);
            let _ = firstLeadingBit(8u32) + firstTrailingBit(8u32);
            let _ = countLeadingZeros(1u32) + countTrailingZeros(2u32);
        }
        "#,
    );
}

#[test]
fn a_ray_query_is_named_as_a_type_is() {
    same_module(
        r#"
        static acc: AccelerationStructure = binding();
        fn f(o: Vec3, d: Vec3) -> u32 {
            let mut rq = RayQuery::default();
            rq.initialize(&acc, RayDesc {
                flags: 0u32, cull_mask: 0xFF, tmin: 0.0, tmax: 1.0, origin: o, dir: d,
            });
            rq.proceed();
            rq.committed_intersection().kind
        }
        "#,
        r#"
        static acc: acceleration_structure = binding();
        fn f(o: vec3, d: vec3) -> u32 {
            let mut rq = ray_query::default();
            rq.initialize(&acc, RayDesc {
                flags: 0u32, cull_mask: 0xFF, tmin: 0.0, tmax: 1.0, origin: o, dir: d,
            });
            rq.proceed();
            rq.committed_intersection().kind
        }
        "#,
    );
}
