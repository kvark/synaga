//! The build-script path, exercised without a build script.
//!
//! `examples/sprites` proves the Cargo wiring end to end; these cover the
//! behaviour around it — what gets generated, what gets pruned, and what a
//! failure says.

use std::path::{Path, PathBuf};

use synaga::build::{Bindings, BuildErrorKind, Shaders};

/// A scratch directory unique to one test. `CARGO_TARGET_TMPDIR` is cleaned by
/// `cargo clean` and needs no dependency.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("shaders")).expect("create scratch");
    std::fs::create_dir_all(dir.join("out")).expect("create out");
    dir
}

fn write(dir: &Path, name: &str, source: &str) {
    std::fs::write(dir.join("shaders").join(name), source).expect("write shader");
}

const TRIANGLE: &str = r#"
    #[vertex]
    #[output(builtin(position))]
    fn vs(#[location(0)] pos: vec3) -> vec4 {
        vec4(pos, 1.0)
    }
"#;

const SOLID: &str = r#"
    #[fragment]
    #[output(location(0))]
    fn fs() -> vec4 { vec4(1.0, 0.0, 0.0, 1.0) }
"#;

#[test]
fn generates_a_constant_per_module() {
    let dir = scratch("generates");
    write(&dir, "triangle.rs", TRIANGLE);
    write(&dir, "solid.rs", SOLID);

    let shaders = Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect("emit");

    // Directory order is arbitrary, so the output is sorted.
    let names: Vec<&str> = shaders.iter().map(|s| s.constant.as_str()).collect();
    assert_eq!(names, ["SOLID", "TRIANGLE"]);

    let generated = std::fs::read_to_string(dir.join("out/shaders.rs")).expect("read generated");
    assert!(
        generated.contains("pub const TRIANGLE: &[u8]"),
        "{generated}"
    );
    assert!(generated.contains("pub const SOLID: &[u8]"), "{generated}");

    let ir = std::fs::read_to_string(dir.join("out/triangle.json")).expect("read ir");
    assert!(ir.contains("\"vs\""), "{ir}");
}

#[test]
fn a_prelude_is_shared_and_not_compiled_alone() {
    let dir = scratch("prelude");
    write(
        &dir,
        "common.rs",
        "fn luminance(c: vec3) -> f32 { dot(c, vec3(0.2126, 0.7152, 0.0722)) }",
    );
    write(
        &dir,
        "grey.rs",
        r#"
        #[fragment]
        #[output(location(0))]
        fn fs(#[location(0)] c: vec4) -> vec4 { vec4(vec3(luminance(c.xyz)), 1.0) }
        "#,
    );

    let shaders = Shaders::new()
        .dir(dir.join("shaders"))
        .prelude("common.rs")
        .emit_to(&dir.join("out"))
        .expect("emit");

    // The prelude is a source of declarations, not a shader of its own.
    let names: Vec<&str> = shaders.iter().map(|s| s.constant.as_str()).collect();
    assert_eq!(names, ["GREY"]);

    let ir = std::fs::read_to_string(dir.join("out/grey.json")).expect("read ir");
    assert!(ir.contains("luminance"), "{ir}");
}

#[test]
fn what_a_module_does_not_use_is_pruned() {
    let dir = scratch("prune");
    write(
        &dir,
        "common.rs",
        r#"
        struct Camera { view: mat4 }
        static camera: Camera = ();
        fn unused_helper(x: f32) -> f32 { x * 2.0 }
        "#,
    );
    write(&dir, "solid.rs", SOLID);

    let out = dir.join("out");
    Shaders::new()
        .dir(dir.join("shaders"))
        .prelude("common.rs")
        .bindings(Bindings::Host)
        .emit_to(&out)
        .expect("emit");

    // Not merely untidy: a host that binds by name would have to find
    // something to bind an unused `camera` to.
    let pruned = std::fs::read_to_string(out.join("solid.json")).expect("read ir");
    assert!(!pruned.contains("camera"), "{pruned}");
    assert!(!pruned.contains("unused_helper"), "{pruned}");

    let kept_dir = dir.join("kept");
    std::fs::create_dir_all(&kept_dir).unwrap();
    Shaders::new()
        .dir(dir.join("shaders"))
        .prelude("common.rs")
        .bindings(Bindings::Host)
        .prune(false)
        .emit_to(&kept_dir)
        .expect("emit");
    let kept = std::fs::read_to_string(kept_dir.join("solid.json")).expect("read ir");
    assert!(kept.contains("camera"), "{kept}");
}

#[test]
fn host_bindings_accept_globals_with_none() {
    let dir = scratch("host_bindings");
    write(
        &dir,
        "tint.rs",
        r#"
        static tint: vec4 = ();
        #[fragment]
        #[output(location(0))]
        fn fs() -> vec4 { tint }
        "#,
    );

    Shaders::new()
        .dir(dir.join("shaders"))
        .bindings(Bindings::Host)
        .emit_to(&dir.join("out"))
        .expect("host bindings");

    // The default expects the shader to have said where things bind.
    let err = Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect_err("explicit bindings");
    assert!(matches!(err.kind, BuildErrorKind::Validate(_)), "{err}");
}

#[test]
fn a_failure_names_the_file_and_the_line() {
    let dir = scratch("failure");
    write(
        &dir,
        "broken.rs",
        "\n\n\nfn bad(a: u32) -> u32 {\n    -a\n}\n",
    );

    let err = Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect_err("broken shader");
    let msg = err.to_string();
    // `path:line:column: ...` is what Cargo and editors turn into a jump.
    assert!(msg.contains("broken.rs:4:1"), "{msg}");
    assert!(msg.contains("`fn bad`"), "{msg}");
    assert!(msg.contains("operator `-`"), "{msg}");
}

#[test]
fn a_failure_in_the_prelude_names_the_prelude() {
    let dir = scratch("prelude_failure");
    write(&dir, "common.rs", "fn bad(a: u32) -> u32 { -a }");
    write(&dir, "solid.rs", SOLID);

    let err = Shaders::new()
        .dir(dir.join("shaders"))
        .prelude("common.rs")
        .emit_to(&dir.join("out"))
        .expect_err("broken prelude");
    let msg = err.to_string();
    assert!(msg.contains("common.rs"), "{msg}");
    assert!(!msg.contains("solid.rs"), "{msg}");
}

#[test]
fn a_missing_directory_says_so() {
    let dir = scratch("missing");
    let err = Shaders::new()
        .dir(dir.join("nowhere"))
        .emit_to(&dir.join("out"))
        .expect_err("missing directory");
    assert!(matches!(err.kind, BuildErrorKind::Io(_)), "{err}");
    assert!(err.to_string().contains("nowhere"), "{err}");
}

#[test]
fn only_rust_files_are_compiled() {
    let dir = scratch("extensions");
    write(&dir, "solid.rs", SOLID);
    write(&dir, "notes.txt", "this is not a shader");
    write(&dir, "old.wgsl", "@fragment fn fs() {}");

    let shaders = Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect("emit");
    let names: Vec<&str> = shaders.iter().map(|s| s.constant.as_str()).collect();
    assert_eq!(names, ["SOLID"]);
}

#[test]
fn ray_queries_stay_in_the_module() {
    // The serialized module keeps the ray query. Nothing on this path prints WGSL.
    let dir = scratch("ray");
    write(
        &dir,
        "trace.rs",
        r#"
        static acc: acceleration_structure = ();
        static output: texture_storage_2d<Rgba8Unorm, Write> = ();
        #[compute]
        #[workgroup_size(8, 8)]
        fn cs(#[builtin(global_invocation_id)] gid: vec3<u32>) {
            let rq: ray_query;
            rayQueryInitialize(rq, acc, RayDesc {
                flags: RAY_FLAG_NONE, cull_mask: 0xFF,
                tmin: 0.0, tmax: 100.0, origin: vec3(0.0), dir: vec3(0.0, 0.0, 1.0),
            });
            rayQueryProceed(rq);
            let hit = rayQueryGetCommittedIntersection(rq);
            textureStore(output, gid.xy as vec2<i32>, vec4(hit.t));
        }
        "#,
    );

    Shaders::new()
        .dir(dir.join("shaders"))
        .bindings(Bindings::Host)
        .capabilities(synaga::naga::valid::Capabilities::RAY_QUERY)
        .emit_to(&dir.join("out"))
        .expect("ray query module");
    let ir = std::fs::read_to_string(dir.join("out").join("trace.json")).unwrap();
    assert!(ir.contains("RayQuery"), "{ir}");
    assert!(
        ir.contains("acceleration_structure") || ir.contains("AccelerationStructure"),
        "{ir}"
    );
}

#[test]
fn entry_point_names_are_the_source_names() {
    // The IR keeps the name from the source, including one that ends in a digit.
    let dir = scratch("names");
    write(
        &dir,
        "blur.rs",
        r#"
        #[compute]
        #[workgroup_size(8, 8)]
        fn blur3x3() {}

        #[compute]
        #[workgroup_size(8, 8)]
        fn blur() {}
        "#,
    );

    let shaders = Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect("emit");
    let reported: Vec<&str> = shaders[0]
        .entry_points
        .iter()
        .map(|e| e.name.as_str())
        .collect();
    assert_eq!(reported, ["blur3x3", "blur"]);

    let ir = std::fs::read_to_string(dir.join("out/blur.json")).expect("read ir");
    assert!(ir.contains("blur3x3"), "{ir}");
    assert!(ir.contains("\"blur\""), "{ir}");
}

#[test]
fn the_generated_module_lists_every_shader() {
    let dir = scratch("all");
    write(&dir, "triangle.rs", TRIANGLE);
    write(&dir, "solid.rs", SOLID);

    Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect("emit");

    let generated = std::fs::read_to_string(dir.join("out/shaders.rs")).expect("read generated");
    assert!(
        generated.contains(
            r#"pub const ALL: [(&str, &[u8]); 2] = [("solid", SOLID), ("triangle", TRIANGLE), ];"#
        ),
        "{generated}"
    );
}
