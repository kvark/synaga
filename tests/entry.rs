#![cfg(feature = "wgsl")]
mod common;

use common::*;
use synaga::parse_str;

#[test]
fn compute_global_id() {
    let wgsl = roundtrip(
        r#"
        #[entry_point(compute, threads(8, 8, 1))]
        fn cs_main(#[builtin(global_invocation_id)] id: vec3<u32>) {
            let x = id.x;
        }
        "#,
    );
    assert!(
        wgsl.contains("@compute") || wgsl.contains("compute"),
        "{wgsl}"
    );
    assert!(
        wgsl.contains("workgroup_size") || wgsl.contains("8"),
        "{wgsl}"
    );
}

#[test]
fn vertex_position() {
    let wgsl = roundtrip(
        r#"
        #[entry_point(vertex)]
        #[output(builtin(position))]
        fn vs_main(#[location(0)] pos: vec3) -> vec4 {
            vec4(pos.x, pos.y, pos.z, 1.0)
        }
        "#,
    );
    assert!(
        wgsl.contains("@vertex") || wgsl.contains("vertex"),
        "{wgsl}"
    );
    assert!(wgsl.contains("position"), "{wgsl}");
}

#[test]
fn fragment_color() {
    validate_only(
        r#"
        #[entry_point(fragment)]
        #[output(location(0))]
        fn fs_main(#[location(0)] color: vec4) -> vec4 {
            color
        }
        "#,
    );
}

#[test]
fn vertex_with_index() {
    validate_only(
        r#"
        #[entry_point(vertex)]
        #[output(builtin(position))]
        fn vs_main(#[builtin(vertex_index)] vid: u32) -> vec4 {
            vec4(0.0, 0.0, 0.0, 1.0)
        }
        "#,
    );
}

#[test]
fn regular_fn_still_in_functions() {
    let module = parse_str("fn add(a: f32, b: f32) -> f32 { a + b }").unwrap();
    assert_eq!(module.functions.len(), 1);
    assert!(module.entry_points.is_empty());
}

#[test]
fn entry_not_in_functions() {
    let module = parse_str(
        r#"
        #[entry_point(compute, threads(1))]
        fn cs_main(#[builtin(local_invocation_index)] i: u32) {}
        "#,
    )
    .unwrap();
    assert_eq!(module.functions.len(), 0);
    assert_eq!(module.entry_points.len(), 1);
}

#[test]
fn rejects_compute_without_workgroup() {
    let msg = reject(
        r#"
        #[entry_point(compute)]
        fn cs_main(#[builtin(local_invocation_index)] i: u32) {}
        "#,
    );
    assert!(msg.contains("threads"), "{msg}");
}

#[test]
fn rejects_workgroup_on_vertex() {
    let msg = reject(
        r#"
        #[entry_point(vertex, threads(8))]
        #[output(builtin(position))]
        fn vs_main() -> vec4 { vec4(0.0, 0.0, 0.0, 1.0) }
        "#,
    );
    assert!(msg.contains("compute"), "{msg}");
}

#[test]
fn an_old_stage_attribute_is_named_not_ignored() {
    let msg = reject(
        r#"
        #[vertex]
        #[output(builtin(position))]
        fn vs_main() -> vec4 { vec4(0.0, 0.0, 0.0, 1.0) }
        "#,
    );
    assert!(msg.contains("#[entry_point(vertex)]"), "{msg}");
}

#[test]
fn fragment_may_return_nothing() {
    let wgsl = roundtrip(
        r#"
        #[entry_point(fragment)]
        fn fs_main() {}
        "#,
    );
    assert!(wgsl.contains("fn fs_main"), "{wgsl}");
    assert!(!wgsl.contains("->"), "{wgsl}");
}

#[test]
fn rejects_vertex_without_return() {
    let msg = reject(
        r#"
        #[entry_point(vertex)]
        fn vs_main() {}
        "#,
    );
    assert!(msg.contains("return"), "{msg}");
}

#[test]
fn rejects_missing_arg_binding() {
    let msg = reject(
        r#"
        #[entry_point(compute, threads(1))]
        fn cs_main(id: vec3<u32>) {}
        "#,
    );
    assert!(
        msg.contains("location") || msg.contains("builtin") || msg.contains("binding"),
        "{msg}"
    );
}

/// Lower both and require the same module: what the first leaves unsaid, the
/// second says.
fn same_module(implied: &str, explicit: &str) {
    let lower = |src: &str| {
        let module = parse_str(src).unwrap_or_else(|e| panic!("parse: {e}\n{src}"));
        synaga::validate(&module).unwrap_or_else(|e| panic!("validate: {e}\n{src}"));
        format!("{module:#?}")
    };
    assert_eq!(lower(implied), lower(explicit), "\n{implied}\n{explicit}");
}

#[test]
fn a_parameter_named_after_a_builtin_is_that_builtin() {
    same_module(
        r#"
        #[entry_point(compute, threads(64))]
        fn cs(global_invocation_id: Vec3<u32>, local_invocation_index: u32,
              workgroup_id: Vec3<u32>, num_workgroups: Vec3<u32>, local_invocation_id: Vec3<u32>) {}
        "#,
        r#"
        #[entry_point(compute, threads(64))]
        fn cs(#[builtin(global_invocation_id)] global_invocation_id: Vec3<u32>,
              #[builtin(local_invocation_index)] local_invocation_index: u32,
              #[builtin(workgroup_id)] workgroup_id: Vec3<u32>,
              #[builtin(num_workgroups)] num_workgroups: Vec3<u32>,
              #[builtin(local_invocation_id)] local_invocation_id: Vec3<u32>) {}
        "#,
    );
    same_module(
        r#"
        #[entry_point(vertex)]
        fn vs(vertex_index: u32, instance_index: u32) -> Vec4 { Vec4::ZERO }
        "#,
        r#"
        #[entry_point(vertex)]
        fn vs(#[builtin(vertex_index)] vertex_index: u32,
              #[builtin(instance_index)] instance_index: u32) -> Vec4 { Vec4::ZERO }
        "#,
    );
    same_module(
        r#"
        #[entry_point(fragment)]
        fn fs(position: Vec4, front_facing: bool) -> Vec4 { position }
        "#,
        r#"
        #[entry_point(fragment)]
        fn fs(#[builtin(position)] position: Vec4,
              #[builtin(front_facing)] front_facing: bool) -> Vec4 { position }
        "#,
    );
}

#[test]
fn only_a_builtin_the_stage_takes_goes_by_name() {
    // A vertex shader's `position` is one of its attributes, so it still has
    // to say which.
    let msg = reject("#[entry_point(vertex)] fn vs(position: Vec4) -> Vec4 { position }");
    assert!(
        msg.contains("the name of a builtin its stage takes"),
        "{msg}"
    );
    validate_only(
        "#[entry_point(vertex)] fn vs(#[location(0)] position: Vec4) -> Vec4 { position }",
    );
    // An attribute says otherwise wherever it is written.
    let module = parse_str(
        "#[entry_point(fragment)] fn fs(#[location(0)] position: Vec4) -> Vec4 { position }",
    )
    .unwrap();
    let binding = &module.entry_points[0].function.arguments[0].binding;
    assert!(
        matches!(binding, Some(naga::Binding::Location { location: 0, .. })),
        "{binding:?}"
    );
}

#[test]
fn a_bare_result_goes_where_its_stage_puts_one() {
    // A vertex shader has to produce its position, and a fragment shader's
    // first colour target is where a lone colour goes.
    same_module(
        "#[entry_point(vertex)] fn vs(vertex_index: u32) -> Vec4 { Vec4::ZERO }",
        "#[entry_point(vertex)] #[output(builtin(position))] fn vs(vertex_index: u32) -> Vec4 { Vec4::ZERO }",
    );
    same_module(
        "#[entry_point(fragment)] fn fs() -> Vec4 { Vec4::ONE }",
        "#[entry_point(fragment)] #[output(location(0))] fn fs() -> Vec4 { Vec4::ONE }",
    );
    // Anything else says so.
    let wgsl =
        roundtrip("#[entry_point(fragment)] #[output(builtin(frag_depth))] fn fs() -> f32 { 0.5 }");
    assert!(wgsl.contains("@builtin(frag_depth)"), "{wgsl}");
    let msg = reject("#[entry_point(compute, threads(1))] fn cs() -> u32 { 1 }");
    assert!(msg.contains("a compute entry point cannot"), "{msg}");
}
