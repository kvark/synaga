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
        #[fragment]
        #[output(location(0))]
        fn fs(#[location(0)] uv: vec2) -> vec4 {
            textureSampleLevel(sprite_texture, sprite_sampler, uv, 0.0)
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
        #[fragment] #[output(location(0))]
        fn fs(#[location(0)] uv: vec2) -> vec4 { textureSample(t, s, uv) }
        "#,
    );
}

#[test]
fn load_and_store() {
    let wgsl = roundtrip_unbound(
        r#"
        static input: texture_2d<f32> = ();
        static output: texture_storage_2d<Rgba8Unorm, Write> = ();
        #[compute] #[workgroup_size(8, 8)]
        fn cs(#[builtin(global_invocation_id)] id: vec3<u32>) {
            let c = textureLoad(input, id.xy as vec2<i32>, 0);
            textureStore(output, id.xy as vec2<i32>, c);
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
        #[compute] #[workgroup_size(1)]
        fn cs(#[builtin(global_invocation_id)] id: vec3<u32>) {
            let prev = textureLoad(acc, id.xy as vec2<i32>);
            textureStore(acc, id.xy as vec2<i32>, prev + vec4(1.0));
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
        fn size() -> vec2<u32> { textureDimensions(t) }
        fn levels() -> u32 { textureNumLevels(t) }
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
        fn f(uv: vec2, z: f32) -> f32 { textureSampleCompare(shadow_tex, shadow_samp, uv, z) }
        "#,
    );
    assert!(wgsl.contains("texture_depth_2d"), "{wgsl}");
    assert!(wgsl.contains("sampler_comparison"), "{wgsl}");
}

#[test]
fn snake_case_names_work_too() {
    let _ = roundtrip_unbound(
        r#"
        static t: texture_2d<f32> = ();
        static s: sampler = ();
        fn f(uv: vec2) -> vec4 { texture_sample_level(t, s, uv, 0.0) }
        "#,
    );
}

#[test]
fn rejects_texture_store_as_a_value() {
    let msg = reject(
        r#"
        static output: texture_storage_2d<Rgba8Unorm, Write> = ();
        fn f(c: vec4) -> vec4 { textureStore(output, vec2(0, 0), c) }
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
    let msg = reject("fn f(v: vec3) -> vec2<u32> { textureDimensions(v) }");
    assert!(msg.contains("texture"), "{msg}");
}

#[test]
fn rejects_the_wrong_argument_count() {
    let msg = reject(
        r#"
        static t: texture_2d<f32> = ();
        static s: sampler = ();
        fn f(uv: vec2) -> vec4 { textureSample(t, s, uv, 0.0) }
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
        #[fragment]
        #[output(location(0))]
        fn fs(#[location(0)] uv: vec2) -> vec4 {
            textureSampleLevel(t_specular_f0, samp, uv, 0.0) + w4(uv.x)
        }
        "#,
    );
    assert!(wgsl.contains("t_specular_f0"), "{wgsl}");
    assert!(!wgsl.contains("t_specular_f0_"), "{wgsl}");
    assert!(wgsl.contains("fn w4("), "{wgsl}");
    assert!(!wgsl.contains("fn w4_("), "{wgsl}");
}
