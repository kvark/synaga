//! The build-script path, exercised without a build script.
//!
//! `examples/sprites` proves the Cargo wiring end to end; these cover the
//! behaviour around it — what gets generated, what gets pruned, and what a
//! failure says.

use std::path::{Path, PathBuf};

use synaga::build::{Bindings, BuildErrorKind, Shaders};
use synaga::naga;

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

/// Read a module back the way a host does.
fn decode(path: impl AsRef<Path>) -> naga::Module {
    let bytes = std::fs::read(path).expect("read module");
    let ir = synaga_shader::ir::Ir::new(Box::leak(bytes.into_boxed_slice()));
    ir.decode().expect("decode module")
}

fn function_names(module: &naga::Module) -> Vec<String> {
    module
        .functions
        .iter()
        .filter_map(|(_, f)| f.name.clone())
        .collect()
}

const TRIANGLE: &str = r#"
    #[entry_point(vertex)]
    #[output(builtin(position))]
    fn vs(#[location(0)] pos: vec3) -> vec4 {
        vec4(pos, 1.0)
    }
"#;

const SOLID: &str = r#"
    #[entry_point(fragment)]
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
        generated.contains("pub const TRIANGLE: ::synaga_shader::ir::Ir"),
        "{generated}"
    );
    assert!(
        generated.contains("pub const SOLID: ::synaga_shader::ir::Ir"),
        "{generated}"
    );

    let module = decode(dir.join("out/triangle.naga"));
    assert_eq!(module.entry_points[0].name, "vs");
}

#[test]
fn the_header_says_who_wrote_it() {
    let dir = scratch("header");
    write(&dir, "solid.rs", SOLID);
    Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect("emit");
    let bytes = std::fs::read(dir.join("out/solid.naga")).unwrap();
    assert_eq!(&bytes[..6], b"SYNAGA");
    let ir = synaga_shader::ir::Ir::new(Box::leak(bytes.into_boxed_slice()));
    assert_eq!(ir.naga_major(), Some(synaga::build::NAGA_MAJOR));

    // Something else entirely is refused as such, not misread.
    let err = synaga_shader::ir::Ir::new(b"{\"json\": true}")
        .decode::<naga::Module>()
        .unwrap_err();
    assert_eq!(err, synaga_shader::ir::DecodeError::NotIr);
}

#[test]
fn a_helper_file_comes_along_through_use() {
    let dir = scratch("helpers");
    write(
        &dir,
        "common.rs",
        "pub fn luminance(c: vec3) -> f32 { dot(c, vec3(0.2126, 0.7152, 0.0722)) }",
    );
    write(
        &dir,
        "grey.rs",
        r#"
        use super::common::luminance;
        #[entry_point(fragment)]
        #[output(location(0))]
        fn fs(#[location(0)] c: vec4) -> vec4 { vec4(vec3(luminance(c.xyz)), 1.0) }
        "#,
    );
    write(
        &dir,
        "path.rs",
        r#"
        #[entry_point(fragment)]
        #[output(location(0))]
        fn fs(#[location(0)] c: vec4) -> vec4 { vec4(vec3(super::common::luminance(c.xyz)), 1.0) }
        "#,
    );

    let shaders = Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect("emit");

    // A file with no entry point is a source of declarations, not a shader.
    let names: Vec<&str> = shaders.iter().map(|s| s.constant.as_str()).collect();
    assert_eq!(names, ["GREY", "PATH"]);
    for shader in &shaders {
        let used: Vec<_> = shader
            .sources
            .iter()
            .map(|p| p.file_name().unwrap())
            .collect();
        assert_eq!(used.len(), 2, "{used:?}");
        assert_eq!(used[1], "common.rs");
        assert!(function_names(&decode(&shader.output_path)).contains(&"luminance".to_string()));
    }
}

#[test]
fn a_file_no_shader_uses_is_left_alone() {
    let dir = scratch("unused_file");
    // Not in the dialect at all, and nothing reaches it.
    write(
        &dir,
        "cpu_only.rs",
        "pub fn f() -> String { String::new() }",
    );
    write(&dir, "solid.rs", SOLID);
    let shaders = Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect("emit");
    assert_eq!(shaders.len(), 1);
}

#[test]
fn cfg_follows_the_build() {
    let dir = scratch("cfg");
    write(
        &dir,
        "debug.rs",
        r#"
        #[cfg(debug_assertions)]
        fn level() -> f32 { 1.0 }
        #[cfg(not(debug_assertions))]
        fn level() -> f32 { 0.0 }
        #[entry_point(fragment)]
        #[output(location(0))]
        fn fs() -> vec4 {
            if cfg!(feature = "red") { vec4(level(), 0.0, 0.0, 1.0) } else { vec4::splat(level()) }
        }
        "#,
    );
    let level = |cfg: synaga::Cfg| {
        Shaders::new()
            .dir(dir.join("shaders"))
            .cfg(cfg)
            .emit_to(&dir.join("out"))
            .expect("emit");
        let module = decode(dir.join("out/debug.naga"));
        let (_, level) = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("level"))
            .expect("level");
        let value = level
            .expressions
            .iter()
            .find_map(|(_, e)| match e {
                naga::Expression::Literal(naga::Literal::F32(v)) => Some(*v),
                _ => None,
            })
            .unwrap();
        value
    };
    assert_eq!(level(synaga::Cfg::new()), 0.0);
    assert_eq!(level(synaga::Cfg::new().with("debug_assertions")), 1.0);
}

#[test]
fn what_a_module_does_not_use_is_pruned() {
    let dir = scratch("prune");
    write(
        &dir,
        "common.rs",
        r#"
        pub struct Camera { view: mat4 }
        pub static camera: Camera = ();
        pub fn unused_helper(x: f32) -> f32 { x * 2.0 }
        "#,
    );
    write(&dir, "solid.rs", &format!("use super::common::*;\n{SOLID}"));

    let out = dir.join("out");
    Shaders::new()
        .dir(dir.join("shaders"))
        .bindings(Bindings::Host)
        .emit_to(&out)
        .expect("emit");

    // Not merely untidy: a host that binds by name would have to find
    // something to bind an unused `camera` to.
    let pruned = decode(out.join("solid.naga"));
    assert!(pruned.global_variables.is_empty());
    assert!(function_names(&pruned).is_empty());

    let kept_dir = dir.join("kept");
    std::fs::create_dir_all(&kept_dir).unwrap();
    Shaders::new()
        .dir(dir.join("shaders"))
        .bindings(Bindings::Host)
        .prune(false)
        .emit_to(&kept_dir)
        .expect("emit");
    let kept = decode(kept_dir.join("solid.naga"));
    assert_eq!(kept.global_variables.len(), 1);
    assert_eq!(function_names(&kept), ["unused_helper"]);
}

#[test]
fn host_bindings_accept_globals_with_none() {
    let dir = scratch("host_bindings");
    write(
        &dir,
        "tint.rs",
        r#"
        static tint: vec4 = ();
        #[entry_point(fragment)]
        #[output(location(0))]
        fn fs() -> vec4 { tint }
        "#,
    );

    Shaders::new()
        .dir(dir.join("shaders"))
        .bindings(Bindings::Host)
        .emit_to(&dir.join("out"))
        .expect("host bindings");

    // The default expects the shader to have said where things bind, and
    // says which global did not, where it is written.
    let err = Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect_err("explicit bindings");
    assert!(matches!(err.kind, BuildErrorKind::Transpile(_)), "{err}");
    let msg = err.to_string();
    assert!(msg.contains("tint.rs:2:"), "{msg}");
    assert!(msg.contains("`tint` has no binding"), "{msg}");
    assert!(msg.contains("group(G).binding(B)"), "{msg}");
}

#[test]
fn explicit_bindings_are_written_where_the_resource_is() {
    let dir = scratch("explicit_bindings");
    // A helper file says where its resource binds, with a `const` the host
    // can use too; the shader that brings it along adds its own.
    write(
        &dir,
        "common.rs",
        r#"
        use synaga_shader::*;
        pub const GLOBALS: u32 = 0;
        pub static globals: Uniform<vec4> = group(GLOBALS).binding(0);
        "#,
    );
    write(
        &dir,
        "tint.rs",
        r#"
        use synaga_shader::*;
        use super::common::*;
        const MATERIAL: u32 = 1;
        pub static albedo: texture_2d<f32> = group(MATERIAL).binding(0);
        pub static linear: sampler = synaga_shader::group(MATERIAL).binding(1);
        #[entry_point(fragment)]
        #[output(location(0))]
        pub fn fs(#[location(0)] uv: vec2) -> vec4 {
            *globals * albedo.sample(&linear, uv)
        }
        "#,
    );

    let shaders = Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect("explicit bindings");
    let module = decode(&shaders[0].output_path);
    let mut bound: Vec<(String, u32, u32)> = module
        .global_variables
        .iter()
        .map(|(_, var)| {
            let at = var.binding.as_ref().expect("every resource is bound");
            (var.name.clone().unwrap(), at.group, at.binding)
        })
        .collect();
    bound.sort();
    assert_eq!(
        bound,
        [
            ("albedo".to_string(), 1, 0),
            ("globals".to_string(), 0, 0),
            ("linear".to_string(), 1, 1),
        ]
    );
}

#[test]
fn a_vertex_struct_without_locations_needs_host_bindings() {
    let dir = scratch("unbound_vertex");
    // Blade fills a struct argument's fields in by name; any other host needs
    // them to say where they come from.
    write(
        &dir,
        "quad.rs",
        r#"
        use synaga_shader::*;
        pub struct Vertex { pub pos: vec2 }
        #[entry_point(vertex)]
        #[output(builtin(position))]
        pub fn vs(vertex: Vertex) -> vec4 { vertex.pos.extend(0.0).extend(1.0) }
        "#,
    );

    Shaders::new()
        .dir(dir.join("shaders"))
        .bindings(Bindings::Host)
        .emit_to(&dir.join("out"))
        .expect("host bindings");
    let err = Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect_err("explicit bindings");
    let msg = err.to_string();
    assert!(msg.contains("quad.rs:4:"), "{msg}");
    assert!(
        msg.contains("`vertex` is a struct with no `#[location]`s"),
        "{msg}"
    );
}

#[test]
fn host_bindings_refuse_a_resource_that_says_where() {
    let dir = scratch("host_refuses");
    write(
        &dir,
        "tint.rs",
        r#"
        use synaga_shader::*;
        pub static tint: Uniform<vec4> = group(0).binding(0);
        #[entry_point(fragment)]
        #[output(location(0))]
        pub fn fs() -> vec4 { *tint }
        "#,
    );

    // Blade asserts a module has no bindings, at pipeline creation; this says
    // so at build time, at the line that has one.
    let err = Shaders::new()
        .dir(dir.join("shaders"))
        .bindings(Bindings::Host)
        .emit_to(&dir.join("out"))
        .expect_err("a binding under Bindings::Host");
    let msg = err.to_string();
    assert!(msg.contains("tint.rs:3:"), "{msg}");
    assert!(msg.contains("the host assigns bindings"), "{msg}");

    // The same source is fine for a host that takes the shader's word.
    Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect("explicit bindings");
}

#[test]
fn a_failure_names_the_file_and_the_line() {
    let dir = scratch("failure");
    write(
        &dir,
        "broken.rs",
        "\n\n\nfn bad(a: u32) -> u32 {\n    -a\n}\n#[entry_point(compute, threads(1))]\nfn main() {}\n",
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
fn a_failure_in_a_helper_file_names_that_file() {
    let dir = scratch("helper_failure");
    write(&dir, "common.rs", "pub fn bad(a: u32) -> u32 { -a }");
    write(&dir, "solid.rs", &format!("use super::common::*;\n{SOLID}"));

    let err = Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect_err("broken helper");
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
        #[entry_point(compute, threads(8, 8))]
        fn cs(#[builtin(global_invocation_id)] gid: vec3<u32>) {
            let rq: ray_query;
            rq.initialize(&acc, RayDesc {
                flags: RAY_FLAG_NONE, cull_mask: 0xFF,
                tmin: 0.0, tmax: 100.0, origin: vec3(0.0), dir: vec3(0.0, 0.0, 1.0),
            });
            rq.proceed();
            let hit = rq.committed_intersection();
            output.store(gid.xy as vec2<i32>, vec4(hit.t));
        }
        "#,
    );

    Shaders::new()
        .dir(dir.join("shaders"))
        .bindings(Bindings::Host)
        .capabilities(synaga::naga::valid::Capabilities::RAY_QUERY)
        .emit_to(&dir.join("out"))
        .expect("ray query module");
    let module = decode(dir.join("out/trace.naga"));
    assert!(module
        .types
        .iter()
        .any(|(_, ty)| matches!(ty.inner, naga::TypeInner::RayQuery { .. })));
    assert!(module
        .global_variables
        .iter()
        .any(|(_, var)| var.name.as_deref() == Some("acc")));
}

#[test]
fn entry_point_names_are_the_source_names() {
    // The IR keeps the name from the source, including one that ends in a digit.
    let dir = scratch("names");
    write(
        &dir,
        "blur.rs",
        r#"
        #[entry_point(compute, threads(8, 8))]
        fn blur3x3() {}

        #[entry_point(compute, threads(8, 8))]
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

    let module = decode(dir.join("out/blur.naga"));
    let names: Vec<&str> = module
        .entry_points
        .iter()
        .map(|e| e.name.as_str())
        .collect();
    assert_eq!(names, ["blur3x3", "blur"]);
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
            r#"pub const ALL: [(&str, ::synaga_shader::ir::Ir); 2] = [("solid", SOLID), ("triangle", TRIANGLE), ];"#
        ),
        "{generated}"
    );
}
