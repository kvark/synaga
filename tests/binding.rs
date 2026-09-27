//! Where a resource binds, and who says.
//!
//! A checkable shader says it in the initialiser: `group(G).binding(B)` is
//! WGSL's `@group(G) @binding(B)`, and `binding()` leaves it to the host. The
//! older spelling says it with `#[group]` and `#[binding]`, which `rustc`
//! does not accept, and which still work for a shader it never sees.

mod common;

use common::*;

/// Every global, and where it binds.
fn bindings(src: &str) -> Vec<(String, Option<(u32, u32)>)> {
    let module = synaga::parse_str(src).unwrap_or_else(|e| panic!("parse: {e}\n{src}"));
    module
        .global_variables
        .iter()
        .map(|(_, var)| {
            let at = var.binding.as_ref().map(|at| (at.group, at.binding));
            (var.name.clone().unwrap(), at)
        })
        .collect()
}

fn named(name: &str, at: Option<(u32, u32)>) -> (String, Option<(u32, u32)>) {
    (name.to_string(), at)
}

#[test]
fn the_initialiser_says_where() {
    let buffers_and_textures = r#"
        static camera: Uniform<mat4> = group(0).binding(0);
        static counts: StorageMut<[u32]> = group(0).binding(1);
        static albedo: texture_2d<f32> = group(1).binding(0);
        static linear: sampler = group(1).binding(1);
        fn f(uv: vec2) -> vec4 { camera[0] + albedo.sample(&linear, uv) }
    "#;
    // Every resource says, so the default validation has nothing to miss.
    validate_only(buffers_and_textures);

    // These two want capabilities the default validation leaves off.
    let src = format!(
        "{buffers_and_textures}
        static scene: acceleration_structure = group(2).binding(0);
        static textures: binding_array<texture_2d<f32>, 4> = group(3).binding(0);"
    );
    assert_eq!(
        bindings(&src),
        [
            named("camera", Some((0, 0))),
            named("counts", Some((0, 1))),
            named("albedo", Some((1, 0))),
            named("linear", Some((1, 1))),
            named("scene", Some((2, 0))),
            named("textures", Some((3, 0))),
        ]
    );
}

#[test]
fn a_const_can_number_it() {
    // A host sharing the module can use the same `const` for its layouts.
    let src = r#"
        pub const FRAME: u32 = 2;
        pub const CAMERA: u32 = FRAME;
        static camera: Uniform<mat4> = group(CAMERA).binding(SLOT);
        const SLOT: u32 = 5;
        fn f() -> mat4 { *camera }
    "#;
    validate_only(src);
    assert_eq!(bindings(src), [named("camera", Some((2, 5)))]);
}

#[test]
fn a_qualified_path_reads_the_same() {
    let src = r#"
        static a: Uniform<f32> = synaga_shader::group(0).binding(1);
        static b: Uniform<f32> = (group(0)).binding(2);
        static c: Uniform<f32> = synaga_shader::binding();
        fn f() -> f32 { *a + *b + *c }
    "#;
    assert_eq!(
        bindings(src),
        [
            named("a", Some((0, 1))),
            named("b", Some((0, 2))),
            named("c", None)
        ]
    );
}

#[test]
fn binding_leaves_it_to_the_host() {
    let src = r#"
        static camera: Uniform<mat4> = binding();
        static tile: Workgroup<[f32; 64]> = binding();
        fn f() -> mat4 { *camera }
    "#;
    validate_only_unbound(src);
    assert_eq!(bindings(src), [named("camera", None), named("tile", None)]);
}

#[test]
fn the_older_spelling_still_binds() {
    let src = r#"
        #[group(1)] #[binding(2)] static scale: f32 = ();
        #[group(1)] #[binding(3)] static tex: texture_2d<f32> = binding();
        fn f() -> f32 { scale }
    "#;
    assert_eq!(
        bindings(src),
        [named("scale", Some((1, 2))), named("tex", Some((1, 3)))]
    );
}

#[test]
fn workgroup_memory_takes_no_binding() {
    for src in [
        "static tile: Workgroup<f32> = group(0).binding(0); fn f() -> f32 { *tile }",
        "static seed: Private<f32> = group(0).binding(0); fn f() -> f32 { *seed }",
        // The attributes say the same thing, and are refused the same way.
        "#[group(0)] #[binding(0)] static tile: Workgroup<f32> = binding(); fn f() -> f32 { *tile }",
    ] {
        let msg = reject(src);
        assert!(msg.contains("takes no binding"), "{msg}\n{src}");
    }
}

#[test]
fn a_binding_is_said_once() {
    let msg = reject(
        "#[group(0)] #[binding(0)] static a: Uniform<f32> = group(0).binding(0); fn f() -> f32 { *a }",
    );
    assert!(msg.contains("binding twice"), "{msg}");
}

#[test]
fn half_a_binding_is_a_typo() {
    let msg = reject("#[binding(0)] static a: Uniform<f32> = binding(); fn f() -> f32 { *a }");
    assert!(
        msg.contains("`#[group]` and `#[binding]` together"),
        "{msg}"
    );
}

#[test]
fn a_binding_number_is_known_before_the_shader_runs() {
    for (src, what) in [
        (
            "static a: Uniform<f32> = group(1 + 1).binding(0);",
            "`1 + 1`",
        ),
        (
            "const G: f32 = 1.0; static a: Uniform<f32> = group(0).binding(G);",
            "`G`",
        ),
        (
            "static a: Uniform<f32> = group(self::G).binding(0); const G: i32 = -1;",
            "`self::G`",
        ),
    ] {
        let msg = reject(&format!("{src} fn f() -> f32 {{ *a }}"));
        assert!(msg.contains("a binding number is"), "{msg}\n{src}");
        assert!(msg.contains(what), "{msg}\n{src}");
    }
    let msg = reject("static a: Uniform<f32> = group(NOWHERE).binding(0); fn f() -> f32 { *a }");
    assert!(msg.contains("NOWHERE"), "{msg}");
}

#[test]
fn a_resource_is_initialised_one_of_two_ways() {
    for src in [
        "static a: Uniform<f32> = 0.0;",
        "static a: Uniform<f32> = group(0);",
        "static a: Uniform<f32> = group(0).at(1);",
        "static a: Uniform<f32> = binding(0);",
        "static a: Uniform<f32> = bind(0).binding(1);",
    ] {
        let msg = reject(&format!("{src} fn f() -> f32 {{ *a }}"));
        assert!(
            msg.contains("is initialised with `binding()`"),
            "{msg}\n{src}"
        );
    }
}
