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

/// A shader in a subdirectory of the shader directory, whose parent it creates.
fn write_at(dir: &Path, name: &str, source: &str) {
    let path = dir.join("shaders").join(name);
    std::fs::create_dir_all(path.parent().expect("a parent")).expect("create directory");
    std::fs::write(path, source).expect("write shader");
}

/// `err` as one message, whether it held one failure or several.
///
/// A build script reports all of them rather than stopping at the first, so
/// `err`'s own `Display` is the whole list and these tests read it whole.
fn reported(err: synaga::build::BuildError) -> String {
    err.to_string()
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
    fn vs(#[location(0)] pos: Vec3) -> Vec4 {
        vec4(pos, 1.0)
    }
"#;

const SOLID: &str = r#"
    #[entry_point(fragment)]
    fn fs() -> Vec4 { vec4(1.0, 0.0, 0.0, 1.0) }
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
        "pub fn luminance(c: Vec3) -> f32 { dot(c, vec3(0.2126, 0.7152, 0.0722)) }",
    );
    write(
        &dir,
        "grey.rs",
        r#"
        use super::common::luminance;
        #[entry_point(fragment)]
        fn fs(#[location(0)] c: Vec4) -> Vec4 { vec4(vec3(luminance(c.xyz)), 1.0) }
        "#,
    );
    write(
        &dir,
        "path.rs",
        r#"
        #[entry_point(fragment)]
        fn fs(#[location(0)] c: Vec4) -> Vec4 { vec4(vec3(super::common::luminance(c.xyz)), 1.0) }
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
fn a_file_no_shader_uses_is_reported() {
    let dir = scratch("unused_file");
    // Not in the dialect at all, and nothing reaches it.
    write(
        &dir,
        "cpu_only.rs",
        "pub fn f() -> String { String::new() }",
    );
    write(&dir, "solid.rs", SOLID);
    // `rustc` checks `cpu_only.rs` as a shader while nothing compiles it, which
    // is the gap this exists to close — so it is a build failure, not a shrug.
    let err = Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect_err("an unreachable file");
    let msg = reported(err);
    assert!(msg.contains("cpu_only.rs"), "{msg}");
    assert!(msg.contains("no shader reaches it"), "{msg}");
    assert!(msg.contains("use super::cpu_only::*"), "{msg}");
}

#[test]
fn a_helper_only_shader_tree_is_not_reported() {
    let dir = scratch("helper_only");
    // No entry point anywhere: nothing is a shader, so nothing is unreached.
    write(&dir, "common.rs", "pub fn f() -> f32 { 1.0 }");
    let shaders = Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect("emit");
    assert!(shaders.is_empty());
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
        fn fs() -> Vec4 {
            if cfg!(feature = "red") { vec4(level(), 0.0, 0.0, 1.0) } else { Vec4::splat(level()) }
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
fn cfg_test_needs_an_explicit_choice() {
    let dir = scratch("cfg_test");
    write(
        &dir,
        "both.rs",
        r#"
        #[cfg(test)]
        fn which() -> f32 { 1.0 }
        #[cfg(not(test))]
        fn which() -> f32 { 0.0 }
        #[entry_point(fragment)]
        fn fs() -> Vec4 { Vec4::splat(which()) }
        "#,
    );
    let which = |cfg: synaga::Cfg| {
        Shaders::new()
            .dir(dir.join("shaders"))
            .cfg(cfg)
            .emit_to(&dir.join("out"))
            .expect("emit");
        let module = decode(dir.join("out/both.naga"));
        let value = module
            .functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some("which"))
            .and_then(|(_, f)| {
                f.expressions.iter().find_map(|(_, e)| match e {
                    naga::Expression::Literal(naga::Literal::F32(v)) => Some(*v),
                    _ => None,
                })
            })
            .expect("which");
        value
    };
    // A build script's output is shared by the library and its test harness,
    // so it cannot infer which Rust branch that consumer is compiling.
    let err = Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect_err("Cargo cannot decide cfg(test)");
    let msg = reported(err);
    assert!(
        msg.contains("cfg(test)") && msg.contains("PROFILE"),
        "{msg}"
    );
    assert_eq!(which(synaga::Cfg::new()), 0.0);
    assert_eq!(which(synaga::Cfg::new().with("test")), 1.0);
    assert_eq!(which(synaga::Cfg::from_cargo_env().without("test")), 0.0);
    assert_eq!(which(synaga::Cfg::from_cargo_env().with("test")), 1.0);
}

#[test]
fn cfg_macros_cannot_guess_rustc_only_flags() {
    for predicate in ["test", "doctest", "miri"] {
        let dir = scratch(&format!("cfg_macro_{predicate}"));
        write(
            &dir,
            "branch.rs",
            &format!(
                "#[entry_point(fragment)] fn fs() -> Vec4 {{\n\
                      Vec4::splat(if cfg!({predicate}) {{ 1.0 }} else {{ 0.0 }}) }}"
            ),
        );
        let err = Shaders::new()
            .dir(dir.join("shaders"))
            .emit_to(&dir.join("out"))
            .expect_err("Cargo cannot decide this predicate");
        let msg = reported(err);
        assert!(msg.contains(&format!("cfg({predicate})")), "{msg}");
    }
}

#[test]
fn a_conditional_entry_point_cannot_disappear_from_the_build() {
    let dir = scratch("cfg_attr_entry_point");
    write(
        &dir,
        "conditional.rs",
        "#[cfg_attr(debug_assertions, entry_point(fragment))]\n\
         fn fs() -> Vec4 { Vec4::splat(1.0) }",
    );
    let err = Shaders::new()
        .dir(dir.join("shaders"))
        .cfg(synaga::Cfg::new().with("debug_assertions"))
        .emit_to(&dir.join("out"))
        .expect_err("a conditional entry point must not produce an empty shader tree");
    let msg = reported(err);
    assert!(
        msg.contains("conditional.rs:1:1") && msg.contains("cfg_attr"),
        "{msg}"
    );
}

#[test]
fn conditional_layout_and_binding_attributes_are_rejected_by_the_frontend() {
    for source in [
        "#[cfg_attr(debug_assertions, repr(C))] struct P { x: f32 }",
        "struct P { #[cfg_attr(debug_assertions, location(0))] x: f32 }",
    ] {
        let err = synaga::parse_str(source).expect_err("cannot ignore a conditional attribute");
        assert!(err.to_string().contains("cfg_attr"), "{err}");
    }
}

#[test]
fn what_a_module_does_not_use_is_pruned() {
    let dir = scratch("prune");
    write(
        &dir,
        "common.rs",
        r#"
        pub struct Camera { view: Mat4 }
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
        static tint: Vec4 = ();
        #[entry_point(fragment)]
        fn fs() -> Vec4 { tint }
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
    let msg = reported(err);
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
        pub static globals: Uniform<Vec4> = group(GLOBALS).binding(0);
        "#,
    );
    write(
        &dir,
        "tint.rs",
        r#"
        use synaga_shader::*;
        use super::common::*;
        const MATERIAL: u32 = 1;
        pub static albedo: Texture2D<f32> = group(MATERIAL).binding(0);
        pub static linear: Sampler = synaga_shader::group(MATERIAL).binding(1);
        #[entry_point(fragment)]
        pub fn fs(#[location(0)] uv: Vec2) -> Vec4 {
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
        pub struct Vertex { pub pos: Vec2 }
        #[entry_point(vertex)]
        pub fn vs(vertex: Vertex) -> Vec4 { vertex.pos.extend(0.0).extend(1.0) }
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
    let msg = reported(err);
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
        pub static tint: Uniform<Vec4> = group(0).binding(0);
        #[entry_point(fragment)]
        pub fn fs() -> Vec4 { *tint }
        "#,
    );

    // Blade asserts a module has no bindings, at pipeline creation; this says
    // so at build time, at the line that has one.
    let err = Shaders::new()
        .dir(dir.join("shaders"))
        .bindings(Bindings::Host)
        .emit_to(&dir.join("out"))
        .expect_err("a binding under Bindings::Host");
    let msg = reported(err);
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
    let msg = reported(err);
    // `path:line:column: ...` is what Cargo and editors turn into a jump, and
    // it names the offending expression rather than the item around it: `-a`
    // is on line 5, where `fn bad` opens on line 4.
    assert!(msg.contains("broken.rs:5:5"), "{msg}");
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
    let msg = reported(err);
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
fn a_subdirectory_is_searched_for_shaders() {
    let dir = scratch("subdir");
    write(&dir, "solid.rs", SOLID);
    // A shader tree past a screenful of files wants directories, and a file in
    // one is a module named after its stem — reached as `super::helper`, not as
    // a path through the directory it happens to sit in.
    std::fs::create_dir_all(dir.join("shaders/common")).expect("create subdir");
    write_at(
        &dir,
        "common/helper.rs",
        "pub fn tint() -> Vec4 { vec4(1.0, 0.0, 0.0, 1.0) }",
    );
    write_at(
        &dir,
        "grey.rs",
        r#"
        use super::helper::tint;
        #[entry_point(fragment)]
        fn fs() -> Vec4 { tint() }
        "#,
    );

    let shaders = Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect("emit");

    // Two shaders: `grey.rs` at the top and the `solid.rs` that was already
    // there. The helper is not a shader of its own; it is compiled into `grey`.
    let mut names: Vec<&str> = shaders.iter().map(|s| s.constant.as_str()).collect();
    names.sort();
    assert_eq!(names, ["GREY", "SOLID"]);

    let grey = shaders
        .iter()
        .find(|s| s.name == "grey")
        .expect("grey shader");
    let used: Vec<_> = grey
        .sources
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(used, ["grey.rs", "helper.rs"]);
    // The module has the helper in it, which is the point of reaching it.
    let module = decode(&grey.output_path);
    assert!(
        function_names(&module).contains(&"tint".to_string()),
        "{:?}",
        function_names(&module)
    );
}

#[test]
fn ray_queries_stay_in_the_module() {
    // The serialized module keeps the ray query. Nothing on this path prints WGSL.
    let dir = scratch("ray");
    write(
        &dir,
        "trace.rs",
        r#"
        static acc: AccelerationStructure = ();
        static output: TextureStorage2D<Rgba8Unorm, Write> = ();
        #[entry_point(compute, threads(8, 8))]
        fn cs(#[builtin(global_invocation_id)] gid: Vec3<u32>) {
            let rq: RayQuery;
            rq.initialize(&acc, RayDesc {
                flags: RAY_FLAG_NONE, cull_mask: 0xFF,
                tmin: 0.0, tmax: 100.0, origin: vec3(0.0), dir: vec3(0.0, 0.0, 1.0),
            });
            rq.proceed();
            let hit = rq.committed_intersection();
            output.store(gid.xy as Vec2<i32>, vec4(hit.t));
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

#[test]
fn the_generated_file_asserts_the_naga_version() {
    let dir = scratch("version");
    write(&dir, "solid.rs", SOLID);

    Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect("emit");

    // A module is written by one Naga and read by the host's own copy. Without
    // this the first `decode()` panics on bytes it cannot make sense of; with
    // it, the build fails and names both versions.
    let generated = std::fs::read_to_string(dir.join("out/shaders.rs")).expect("read generated");
    assert!(
        generated.contains("::synaga_shader::ir::NAGA_MAJOR"),
        "{generated}"
    );
    assert!(
        generated.contains(&format!("== {}", synaga::build::NAGA_MAJOR)),
        "{generated}"
    );
}

#[test]
fn the_written_module_carries_the_version_in_its_header() {
    let dir = scratch("header_version");
    write(&dir, "solid.rs", SOLID);

    Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect("emit");

    let bytes = std::fs::read(dir.join("out/solid.naga")).expect("read module");
    assert_eq!(&bytes[..6], synaga_shader::ir::MAGIC);
    assert_eq!(bytes[6], synaga_shader::ir::FORMAT);
    assert_eq!(bytes[7], synaga::build::NAGA_MAJOR);
    assert_eq!(
        synaga_shader::ir::naga_major(&bytes),
        Some(synaga::build::NAGA_MAJOR)
    );
}

#[test]
fn the_layout_file_asserts_what_rustc_has_to_agree_with() {
    let dir = scratch("layout_shape");
    write(
        &dir,
        "params.rs",
        r#"
        #[repr(C)]
        #[derive(Shared)]
        pub struct Globals {
            pub mvp_transform: Mat4,
            pub sprite_size: Vec2,
            pub _pad: Vec2,
        }
        pub static globals: Uniform<Globals> = group(0).binding(0);
        #[entry_point(fragment)]
        fn fs() -> Vec4 { let _ = globals.mvp_transform; Vec4::ZERO }
        "#,
    );

    Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect("emit");

    let layout = std::fs::read_to_string(dir.join("out/shaders_layout.rs")).expect("read layout");
    // The size, the alignment and each field's offset, spelled the way
    // `check_layout!` includes them: `size_of` for the struct, `align_of`
    // because a struct can have every offset right and the wrong alignment,
    // and `offset_of!` for each field the module above can name.
    assert!(
        layout.contains("assert!(::core::mem::size_of::<self::params::Globals>() == 80,"),
        "{layout}"
    );
    assert!(
        layout.contains("assert!(::core::mem::align_of::<self::params::Globals>() == 4,"),
        "{layout}"
    );
    assert!(
        layout
            .contains("assert!(::core::mem::offset_of!(self::params::Globals, sprite_size) == 64,"),
        "{layout}"
    );
    assert!(
        layout.contains("assert!(::core::mem::offset_of!(self::params::Globals, _pad) == 72,"),
        "{layout}"
    );
    // `const _: () = { .. }` is what makes the file safe to leave unincluded, and
    // what the warning above is about.
    assert!(layout.contains("const _: () = {"), "{layout}");
}

#[test]
fn the_layout_file_aligns_everything_it_checks() {
    let dir = scratch("layout_align");
    write(
        &dir,
        "params.rs",
        r#"
        #[repr(C)]
        #[derive(Shared)]
        pub struct Globals { pub a: f32, pub b: f32, pub c: f32 }
        pub static globals: Uniform<Globals> = group(0).binding(0);
        #[entry_point(fragment)]
        fn fs() -> Vec4 { let _ = globals.a; Vec4::ZERO }
        "#,
    );
    Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect("emit");
    let layout = std::fs::read_to_string(dir.join("out/shaders_layout.rs")).expect("read layout");
    // Three `f32` are 4-byte aligned and 12 bytes long, in Rust as on the GPU,
    // so both are asserted. Alignment is checked separately from the offsets
    // because a struct can have every offset right and the wrong alignment,
    // and that is what would place it wrongly inside an enclosing one.
    assert!(
        layout.contains("assert!(::core::mem::size_of::<self::params::Globals>() == 12,"),
        "{layout}"
    );
    assert!(
        layout.contains("assert!(::core::mem::align_of::<self::params::Globals>() == 4,"),
        "{layout}"
    );
}

#[test]
fn a_raised_alignment_is_checked_too() {
    let dir = scratch("layout_align_raised");
    // `repr(C, align(N))` raises the alignment above what any member asks for,
    // and the GPU has to agree about it: a struct nested in another is placed by
    // its alignment, not by its size.
    write(
        &dir,
        "params.rs",
        r#"
        #[repr(C, align(16))]
        #[derive(Shared)]
        pub struct Globals { pub a: f32, pub b: f32, pub c: f32, pub d: f32 }
        pub static globals: Uniform<Globals> = group(0).binding(0);
        #[entry_point(fragment)]
        fn fs() -> Vec4 { let _ = globals.a; Vec4::ZERO }
        "#,
    );
    Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect("emit");
    let layout = std::fs::read_to_string(dir.join("out/shaders_layout.rs")).expect("read layout");
    assert!(
        layout.contains("assert!(::core::mem::align_of::<self::params::Globals>() == 16,"),
        "{layout}"
    );
    assert!(
        layout.contains("assert!(::core::mem::size_of::<self::params::Globals>() == 16,"),
        "{layout}"
    );
}

#[test]
fn the_layout_of_each_shared_struct_is_left_for_rustc_to_check() {
    let dir = scratch("layout");
    write(
        &dir,
        "params.rs",
        r#"
        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct Params { pub scale: Vec2, pub count: u32, secret: u32 }
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Hidden { pub x: f32 }
        #[repr(C)]
        pub struct List { pub count: u32, pub items: [u32] }
        "#,
    );
    write(
        &dir,
        "fill.rs",
        r#"
        use super::params::{Hidden, List, Params};
        static params: Uniform<Params> = binding();
        static hidden: Uniform<Hidden> = binding();
        static list: StorageMut<List> = binding();
        #[entry_point(compute, threads(1))]
        fn fill() {
            list.get_mut().items[0] = params.count + hidden.x as u32 + params.secret;
        }
        "#,
    );
    Shaders::new()
        .dir(dir.join("shaders"))
        .bindings(Bindings::Host)
        .module_name("gpu.rs")
        .emit_to(&dir.join("out"))
        .expect("emit");
    let checks = std::fs::read_to_string(dir.join("out/gpu_layout.rs")).expect("read checks");
    // A crate may warn about qualified paths; these have to be qualified.
    assert!(
        checks.contains("#[allow(unused_qualifications)]\nconst _: () = {"),
        "{checks}"
    );
    assert!(
        checks.contains("::core::mem::size_of::<self::params::Params>() == 16"),
        "{checks}"
    );
    assert!(
        checks.contains("::core::mem::offset_of!(self::params::Params, count) == 8"),
        "{checks}"
    );
    // What the module listing the shaders cannot name, it cannot check.
    assert!(!checks.contains("secret"), "{checks}");
    assert!(!checks.contains("Hidden"), "{checks}");
    // A struct ending in a runtime-sized array has no size, in Rust either;
    // its sized fields still have offsets.
    assert!(
        !checks.contains("size_of::<self::params::List>"),
        "{checks}"
    );
    assert!(
        checks.contains("::core::mem::offset_of!(self::params::List, count) == 0"),
        "{checks}"
    );
    assert!(!checks.contains("List, items"), "{checks}");
}

#[test]
fn an_inline_module_is_refused() {
    let dir = scratch("inline_mod");
    write(
        &dir,
        "solid.rs",
        r#"
        mod helpers {
            #[entry_point(fragment)]
            pub fn fs() -> Vec4 { Vec4::ZERO }
        }
        "#,
    );
    // An entry point inside a `mod` is a shader `rustc` checks and the build
    // never compiles, which is the one outcome this design exists to prevent.
    let msg = reported(
        Shaders::new()
            .dir(dir.join("shaders"))
            .emit_to(&dir.join("out"))
            .expect_err("a mod holding an entry point"),
    );
    assert!(msg.contains("`#[entry_point]`"), "{msg}");
    assert!(msg.contains("mod {"), "{msg}");
    assert!(msg.contains("use super::"), "{msg}");
}

#[test]
fn a_mod_naming_a_file_is_refused() {
    let dir = scratch("file_mod");
    write(&dir, "solid.rs", &format!("mod helpers;\n{SOLID}"));
    let msg = reported(
        Shaders::new()
            .dir(dir.join("shaders"))
            .emit_to(&dir.join("out"))
            .expect_err("a mod naming a file"),
    );
    assert!(msg.contains("`mod helpers;`"), "{msg}");
    assert!(msg.contains("helpers.rs"), "{msg}");
}

#[test]
fn every_failure_in_the_tree_is_reported_at_once() {
    let dir = scratch("many_failures");
    write(
        &dir,
        "a.rs",
        "#[entry_point(fragment)] fn fs() -> Vec4 { let x: u32 = \"no\"; Vec4::ZERO }",
    );
    write(
        &dir,
        "b.rs",
        "#[entry_point(fragment)] fn fs() -> Vec4 { also_not_a_function() }",
    );
    write(
        &dir,
        "c.rs",
        "#[entry_point(fragment)] fn fs() -> Vec4 { -1u32 }",
    );
    let err = Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect_err("three broken shaders");
    let msg = reported(err);
    // Fixing a shader one rebuild at a time is a bad way to spend an afternoon,
    // so one build says everything that is wrong.
    assert!(msg.contains("a.rs"), "{msg}");
    assert!(msg.contains("b.rs"), "{msg}");
    assert!(msg.contains("c.rs"), "{msg}");
}

#[test]
fn a_validation_failure_names_where_naga_blames_it() {
    let dir = scratch("validate_blame");
    // A struct the GPU cannot lay out as a uniform, which is caught by Naga
    // rather than by the lowering — and Naga's complaint carries a span, which
    // is a byte range in one of the module's files.
    write(
        &dir,
        "bad.rs",
        r#"
        #[repr(C)]
        pub struct Odd { pub a: Vec3<f32>, pub b: Vec3<f32> }
        pub static odd: Uniform<Odd> = group(0).binding(0);
        #[entry_point(fragment)]
        fn fs() -> Vec4 { let _ = odd.a; Vec4::ZERO }
        "#,
    );
    // Whether Naga refuses this particular struct is its business; what is
    // synaga's is that when it does, the failure carries a file and a position.
    match Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
    {
        Ok(_) => {}
        Err(err) => {
            let msg = reported(err);
            assert!(msg.contains("bad.rs:"), "{msg}");
        }
    }
}

#[test]
fn the_module_is_left_for_a_caller_that_wants_it() {
    let dir = scratch("keep_modules");
    write(&dir, "solid.rs", SOLID);
    let shaders = Shaders::new()
        .dir(dir.join("shaders"))
        .keep_modules(true)
        .emit_to(&dir.join("out"))
        .expect("emit");
    // A build script that wants to report on what it built should not have to
    // decode the bytes it just wrote to find out.
    let module = shaders[0].module().expect("the module is kept");
    assert_eq!(module.entry_points[0].name, "fs");
}

#[test]
fn the_module_is_not_kept_unless_asked_for() {
    let dir = scratch("no_keep_modules");
    write(&dir, "solid.rs", SOLID);
    let shaders = Shaders::new()
        .dir(dir.join("shaders"))
        .emit_to(&dir.join("out"))
        .expect("emit");
    // Holding every module is a second copy of each, which a script that only
    // writes bytes has no use for.
    assert!(shaders[0].module().is_none());
}

#[cfg(feature = "wgsl")]
#[test]
fn the_module_is_written_as_text_too() {
    let dir = scratch("wgsl_dump");
    write(&dir, "solid.rs", SOLID);
    Shaders::new()
        .dir(dir.join("shaders"))
        .wgsl()
        .emit_to(&dir.join("out"))
        .expect("emit");
    // The IR is what a host runs and it is not readable. A shader whose module
    // is not what you expected is far easier to diagnose as text.
    let wgsl = std::fs::read_to_string(dir.join("out/wgsl/solid.wgsl")).expect("read wgsl");
    assert!(wgsl.contains("fn fs"), "{wgsl}");
    assert!(wgsl.contains("@fragment"), "{wgsl}");
}

#[test]
fn one_fault_is_reported_once_however_many_modules_hit_it() {
    let dir = scratch("dedup_failures");
    // A shader file is also a helper of the next one, so its fault is found
    // twice — once for each module that reaches it. Listing it twice would make
    // the aggregated report read as two problems.
    write(
        &dir,
        "a.rs",
        "pub fn helper(x: u32) -> u32 { -x }\npub fn uses() -> u32 { helper(1) }\n#[entry_point(fragment)]\nfn fs() -> Vec4 { Vec4::ZERO }\n",
    );
    write(
        &dir,
        "b.rs",
        "use super::a::*;\n#[entry_point(fragment)]\nfn fs() -> Vec4 { Vec4::splat(uses()) }\n",
    );
    let msg = reported(
        Shaders::new()
            .dir(dir.join("shaders"))
            .emit_to(&dir.join("out"))
            .expect_err("a helper that is also a shader"),
    );
    assert_eq!(
        msg.matches("operator `-` does not apply").count(),
        1,
        "{msg}"
    );
}
