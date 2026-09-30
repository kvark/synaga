#![cfg(feature = "wgsl")]
//! Textures and samplers, spelled as WGSL spells them.

mod common;

use common::*;

#[test]
fn sample_a_texture() {
    let wgsl = roundtrip_unbound(
        r#"
        static sprite_texture: texture_2d<f32> = ();
        static sprite_sampler: sampler = ();
        #[entry_point(fragment)]
        fn fs(#[location(0)] uv: vec2) -> vec4 {
            sprite_texture.sample_level(&sprite_sampler, uv, 0.0)
        }
        "#,
    );
    assert!(
        wgsl.contains("var sprite_texture: texture_2d<f32>"),
        "{wgsl}"
    );
    assert!(wgsl.contains("var sprite_sampler: sampler"), "{wgsl}");
    assert!(wgsl.contains("textureSampleLevel("), "{wgsl}");
}

#[test]
fn plain_sample_picks_its_own_level() {
    let _ = roundtrip_unbound(
        r#"
        static t: texture_2d<f32> = ();
        static s: sampler = ();
        #[entry_point(fragment)]
        fn fs(#[location(0)] uv: vec2) -> vec4 { t.sample(&s, uv) }
        "#,
    );
}

#[test]
fn load_and_store() {
    let wgsl = roundtrip_unbound(
        r#"
        static input: texture_2d<f32> = ();
        static output: texture_storage_2d<Rgba8Unorm, Write> = ();
        #[entry_point(compute, threads(8, 8))]
        fn cs(#[builtin(global_invocation_id)] id: vec3<u32>) {
            let c = input.load(id.xy as vec2<i32>, 0);
            output.store(id.xy as vec2<i32>, c);
        }
        "#,
    );
    assert!(
        wgsl.contains("texture_storage_2d<rgba8unorm,write>"),
        "{wgsl}"
    );
    assert!(wgsl.contains("textureLoad(input,"), "{wgsl}");
    assert!(wgsl.contains("textureStore(output,"), "{wgsl}");
}

#[test]
fn storage_load_takes_no_level() {
    // A storage texture has no mips, so `textureLoad` stops at the coordinate.
    let _ = roundtrip_unbound(
        r#"
        static acc: texture_storage_2d<Rgba32Float, ReadWrite> = ();
        #[entry_point(compute, threads(1))]
        fn cs(#[builtin(global_invocation_id)] id: vec3<u32>) {
            let prev = acc.load(id.xy as vec2<i32>);
            acc.store(id.xy as vec2<i32>, prev + vec4(1.0));
        }
        "#,
    );
}

#[test]
fn formats_take_either_spelling() {
    // WGSL writes `rgba16float`; Rust would write `Rgba16Float`.
    let _ = roundtrip_unbound(
        "static o: texture_storage_2d<rgba16float, write> = (); fn f() -> f32 { 1.0 }",
    );
    let _ = roundtrip_unbound(
        "static o: texture_storage_2d<Rgba16Float, Write> = (); fn f() -> f32 { 1.0 }",
    );
}

#[test]
fn queries() {
    let wgsl = roundtrip_unbound(
        r#"
        static t: texture_2d<f32> = ();
        fn size() -> vec2<u32> { t.dimensions() }
        fn levels() -> u32 { t.num_levels() }
        "#,
    );
    assert!(wgsl.contains("textureDimensions(t)"), "{wgsl}");
    assert!(wgsl.contains("textureNumLevels(t)"), "{wgsl}");
}

#[test]
fn depth_comparison() {
    let wgsl = roundtrip_unbound(
        r#"
        static shadow_tex: texture_depth_2d = ();
        static shadow_samp: sampler_comparison = ();
        fn f(uv: vec2, z: f32) -> f32 { shadow_tex.sample_compare(&shadow_samp, uv, z) }
        "#,
    );
    assert!(wgsl.contains("texture_depth_2d"), "{wgsl}");
    assert!(wgsl.contains("sampler_comparison"), "{wgsl}");
}

#[test]
fn rejects_texture_store_as_a_value() {
    let msg = reject(
        r#"
        static output: texture_storage_2d<Rgba8Unorm, Write> = ();
        fn f(c: vec4) -> vec4 { let done = output.store(vec2(0, 0), c); c }
        "#,
    );
    assert!(msg.contains("no value"), "{msg}");
}

#[test]
fn rejects_an_address_space_on_a_handle() {
    let msg = reject("#[storage] static s: sampler = (); fn f() -> f32 { 1.0 }");
    assert!(msg.contains("handle"), "{msg}");
}

#[test]
fn rejects_a_non_texture_argument() {
    let msg = reject("fn f(v: vec3) -> vec2<u32> { v.dimensions() }");
    assert!(msg.contains("`dimensions`"), "{msg}");
}

#[test]
fn rejects_the_wrong_argument_count() {
    let msg = reject(
        r#"
        static t: texture_2d<f32> = ();
        static s: sampler = ();
        fn f(uv: vec2) -> vec4 { t.sample(&s, uv, 0.0) }
        "#,
    );
    assert!(msg.contains("arguments"), "{msg}");
}

#[test]
fn rejects_an_unknown_format() {
    let msg = reject("static o: texture_storage_2d<Bgra4Unorm, Write> = (); fn f() -> f32 { 1.0 }");
    assert!(msg.contains("texture_storage_2d"), "{msg}");
}

#[test]
fn names_that_end_in_a_digit_keep_it() {
    // Naga's writer appends `_` so a later suffix stays distinct. The host
    // matches `t_specular_f0` by that spelling, so the underscore comes back off.
    let wgsl = roundtrip_unbound(
        r#"
        static t_specular_f0: texture_2d<f32> = ();
        static samp: sampler = ();
        fn w4(w: f32) -> vec4 { vec4(w, w, w, w) }
        #[entry_point(fragment)]
        fn fs(#[location(0)] uv: vec2) -> vec4 {
            t_specular_f0.sample_level(&samp, uv, 0.0) + w4(uv.x)
        }
        "#,
    );
    assert!(wgsl.contains("t_specular_f0"), "{wgsl}");
    assert!(!wgsl.contains("t_specular_f0_"), "{wgsl}");
    assert!(wgsl.contains("fn w4("), "{wgsl}");
    assert!(!wgsl.contains("fn w4_("), "{wgsl}");
}

#[test]
fn every_sampling_method() {
    let wgsl = roundtrip_unbound(
        r#"
        static t: texture_2d<f32> = binding();
        static layers: texture_2d_array<f32> = binding();
        static cube: texture_cube<f32> = binding();
        static s: sampler = binding();
        #[entry_point(fragment)]
        fn fs(#[location(0)] uv: vec2) -> vec4 {
            t.sample(&s, uv)
                + t.sample_bias(&s, uv, 0.5)
                + t.sample_grad(&s, uv, vec2::splat(0.1), vec2::splat(0.2))
                + layers.sample_level(&s, uv, 3, 1.0)
                + cube.sample_level(&s, vec3(uv, 1.0), 0.0)
        }
        "#,
    );
    assert!(wgsl.contains("textureSample(t, s, uv)"), "{wgsl}");
    assert!(wgsl.contains("textureSampleBias(t, s, uv, 0.5f)"), "{wgsl}");
    assert!(wgsl.contains("textureSampleGrad(t, s, uv,"), "{wgsl}");
    assert!(
        wgsl.contains("textureSampleLevel(layers, s, uv, 3i, 1f)"),
        "{wgsl}"
    );
}

#[test]
fn level_dimensions_and_layers() {
    let wgsl = roundtrip_unbound(
        r#"
        static layers: texture_2d_array<u32> = binding();
        fn size() -> vec2u { layers.level_dimensions(2) }
        fn count() -> u32 { layers.num_layers() }
        fn texel(c: vec2i) -> vec4u { layers.load(c, 1, 0) }
        "#,
    );
    assert!(wgsl.contains("textureDimensions(layers, 2i)"), "{wgsl}");
    assert!(wgsl.contains("textureNumLayers(layers)"), "{wgsl}");
    assert!(wgsl.contains("textureLoad(layers, c, 1i, 0i)"), "{wgsl}");
}

#[test]
fn storage_textures_load_and_store_by_access() {
    let wgsl = roundtrip_unbound(
        r#"
        static image: texture_storage_2d<R32Float, ReadWrite> = binding();
        #[entry_point(compute, threads(8, 8))]
        fn cs(#[builtin(global_invocation_id)] id: vec3u) {
            let c = vec2i::from(id.xy());
            image.store(c, image.load(c) * 2.0)
        }
        "#,
    );
    assert!(wgsl.contains("textureLoad(image, "), "{wgsl}");
    assert!(wgsl.contains("textureStore(image, "), "{wgsl}");
}

#[test]
fn a_texture_from_a_binding_array() {
    let module = synaga::parse_str(
        r#"
        static textures: binding_array<texture_2d<f32>, 8> = binding();
        static s: sampler = binding();
        fn f(i: u32, uv: vec2) -> vec4 { textures[i as usize].sample_level(&s, uv, 0.0) }
        "#,
    )
    .unwrap();
    synaga::validate_with(
        &module,
        synaga::naga::valid::ValidationFlags::all() ^ synaga::naga::valid::ValidationFlags::BINDINGS,
        synaga::naga::valid::Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY
            | synaga::naga::valid::Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY_NON_UNIFORM_INDEXING,
    )
    .unwrap();
}
