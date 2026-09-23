//! Inline ray tracing: `ray_query`, `acceleration_structure`, and the
//! operations on them.
//!
//! These validate rather than round-trip. Naga's WGSL backend has no spelling
//! for a ray query, so the modules are checked as IR — which is the form Blade
//! and every other host actually consumes.

mod common;

use common::*;
use synaga::{naga, parse_str, to_wgsl, validate_with};

/// Validate the way a host with ray tracing would: bindings assigned elsewhere,
/// `RAY_QUERY` turned on.
fn trace(src: &str) -> naga::Module {
    let module = parse_str(src).unwrap_or_else(|e| panic!("parse: {e}\n{src}"));
    let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
    validate_with(&module, flags, naga::valid::Capabilities::RAY_QUERY)
        .unwrap_or_else(|e| panic!("validate: {e}\n{src}"));
    module
}

/// The tracing loop from `examples/ray-query/shader.wgsl`.
const RAY_QUERY_SHADER: &str = r#"
    const MAX_BOUNCES: i32 = 3;

    struct Parameters {
        cam_position: vec3,
        depth: f32,
        cam_orientation: vec4,
        fov: vec2,
        torus_radius: f32,
        rotation_angle: f32,
    }

    static parameters: Parameters = ();
    static acc_struct: acceleration_structure = ();
    static output: texture_storage_2d<Rgba8Unorm, Write> = ();

    fn qrot(q: vec4, v: vec3) -> vec3 {
        v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    }

    #[compute]
    #[workgroup_size(8, 8)]
    fn main(#[builtin(global_invocation_id)] global_id: vec3<u32>) {
        let target_size = textureDimensions(output);
        if any(global_id.xy > target_size) {
            return;
        }

        let ray_pos = parameters.cam_position;
        let ray_dir = qrot(parameters.cam_orientation, vec3(0.0, 0.0, 1.0));
        let num_bounces = 0;

        loop {
            let rq: ray_query;
            rayQueryInitialize(rq, acc_struct, RayDesc {
                flags: RAY_FLAG_NONE,
                cull_mask: 0xFF,
                tmin: 0.1,
                tmax: parameters.depth,
                origin: ray_pos,
                dir: ray_dir,
            });
            rayQueryProceed(rq);
            let intersection = rayQueryGetCommittedIntersection(rq);
            if intersection.kind == RAY_QUERY_INTERSECTION_NONE {
                break;
            }

            ray_pos += ray_dir * intersection.t;
            let normal = normalize(ray_pos);
            ray_dir -= 2.0 * dot(ray_dir, normal) * normal;

            num_bounces += 1;
            if num_bounces > MAX_BOUNCES {
                break;
            }
        }

        textureStore(output, global_id.xy as vec2<i32>, vec4(ray_dir, 1.0));
    }
"#;

#[test]
fn ray_query_loop() {
    let module = trace(RAY_QUERY_SHADER);
    assert!(module
        .global_variables
        .iter()
        .any(|(_, v)| v.space == naga::AddressSpace::Handle));
}

#[test]
fn ray_query_emits_wgsl() {
    let module = trace(RAY_QUERY_SHADER);
    let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
    let info = validate_with(&module, flags, naga::valid::Capabilities::RAY_QUERY).unwrap();
    let wgsl = to_wgsl(&module, &info).unwrap_or_else(|err| panic!("wgsl: {err}"));
    assert!(wgsl.contains("enable wgpu_ray_query;"), "{wgsl}");
    assert!(wgsl.contains("rayQueryInitialize("), "{wgsl}");
    assert!(wgsl.contains("rayQueryProceed("), "{wgsl}");
    assert!(wgsl.contains("rayQueryGetCommittedIntersection("), "{wgsl}");
    // The stand-in functions are gone, so a frontend binds the calls to the
    // builtins rather than to an empty definition.
    assert!(!wgsl.contains("fn rayQueryInitialize"), "{wgsl}");
    let parsed = naga::front::wgsl::parse_str(&wgsl)
        .unwrap_or_else(|err| panic!("reparse: {err:?}\n{wgsl}"));
    validate_with(&parsed, flags, naga::valid::Capabilities::RAY_QUERY)
        .unwrap_or_else(|err| panic!("revalidate: {err}\n{wgsl}"));
}

#[test]
fn proceed_drives_a_loop() {
    // `rayQueryProceed` produces a bool, so it can be a `while` condition.
    trace(
        r#"
        static acc: acceleration_structure = ();
        fn trace_one(o: vec3, d: vec3) -> f32 {
            let rq: ray_query;
            rayQueryInitialize(rq, acc, RayDesc {
                flags: RAY_FLAG_NONE, cull_mask: 0xFF,
                tmin: 0.0, tmax: 100.0, origin: o, dir: d,
            });
            while rayQueryProceed(rq) {
                rayQueryConfirmIntersection(rq);
            }
            rayQueryGetCommittedIntersection(rq).t
        }
        "#,
    );
}

#[test]
fn candidate_intersections_and_terminate() {
    trace(
        r#"
        static acc: acceleration_structure = ();
        fn any_hit(o: vec3, d: vec3) -> bool {
            let rq: ray_query;
            rayQueryInitialize(rq, acc, RayDesc {
                flags: RAY_FLAG_TERMINATE_ON_FIRST_HIT, cull_mask: 0xFF,
                tmin: 0.0, tmax: 100.0, origin: o, dir: d,
            });
            while rayQueryProceed(rq) {
                let candidate = rayQueryGetCandidateIntersection(rq);
                if candidate.t < 1.0 {
                    rayQueryTerminate(rq);
                }
            }
            rayQueryGetCommittedIntersection(rq).kind != RAY_QUERY_INTERSECTION_NONE
        }
        "#,
    );
}

#[test]
fn intersection_transforms() {
    // `RayIntersection` is Naga's own struct; its matrix fields come with it.
    trace(
        r#"
        static acc: acceleration_structure = ();
        fn local_point(o: vec3, d: vec3) -> vec3 {
            let rq: ray_query;
            rayQueryInitialize(rq, acc, RayDesc {
                flags: RAY_FLAG_NONE, cull_mask: 0xFF,
                tmin: 0.0, tmax: 100.0, origin: o, dir: d,
            });
            rayQueryProceed(rq);
            let hit = rayQueryGetCommittedIntersection(rq);
            (hit.world_to_object * vec4(o + d * hit.t, 1.0)).xyz
        }
        "#,
    );
}

#[test]
fn rejects_a_ray_op_on_something_else() {
    let msg = reject("fn f(x: f32) { rayQueryProceed(x); }");
    assert!(
        msg.contains("storage") || msg.contains("ray_query"),
        "{msg}"
    );
}

#[test]
fn rejects_initialize_without_an_acceleration_structure() {
    let msg = reject(
        r#"
        static acc: texture_2d<f32> = ();
        fn f(o: vec3, d: vec3) {
            let rq: ray_query;
            rayQueryInitialize(rq, acc, RayDesc {
                flags: 0u32, cull_mask: 0xFF, tmin: 0.0, tmax: 1.0, origin: o, dir: d,
            });
        }
        "#,
    );
    assert!(msg.contains("acceleration_structure"), "{msg}");
}

#[test]
fn rejects_initialize_as_a_value() {
    let msg = reject(
        r#"
        static acc: acceleration_structure = ();
        fn f(o: vec3, d: vec3) -> u32 {
            let rq: ray_query;
            rayQueryInitialize(rq, acc, RayDesc {
                flags: 0u32, cull_mask: 0xFF, tmin: 0.0, tmax: 1.0, origin: o, dir: d,
            })
        }
        "#,
    );
    assert!(msg.contains("no value"), "{msg}");
}

#[test]
fn ray_query_inside_a_function_emits_wgsl() {
    // The stand-in builtins have to be callable from a plain function, which
    // means they are declared before it. An entry point is allowed to call
    // anything, so it would not catch the ordering.
    let module = trace(
        r#"
        static acc: acceleration_structure = ();
        fn trace(o: vec3, d: vec3) -> bool {
            let rq: ray_query;
            rayQueryInitialize(rq, acc, RayDesc {
                flags: 0u32, cull_mask: 0xFF, tmin: 0.0, tmax: 1.0, origin: o, dir: d,
            });
            rayQueryProceed(rq)
        }
        #[compute]
        #[workgroup_size(1)]
        fn main(#[builtin(local_invocation_index)] i: u32) {
            let hit = trace(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0));
            let _ = hit;
            let _ = i;
        }
        "#,
    );
    let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
    let info = validate_with(&module, flags, naga::valid::Capabilities::RAY_QUERY).unwrap();
    let wgsl = to_wgsl(&module, &info).unwrap_or_else(|err| panic!("{err}"));
    assert!(wgsl.contains("fn trace("), "{wgsl}");
    assert!(wgsl.contains("rayQueryProceed("), "{wgsl}");
    assert!(!wgsl.contains("fn rayQueryProceed"), "{wgsl}");
}

#[test]
fn ray_query_default_is_a_local() {
    // Rust spells the query as `ray_query::default()`. It is not a value, so
    // the local is declared and left for `rayQueryInitialize` to start.
    trace(
        r#"
        static acc: acceleration_structure = ();
        fn f(o: vec3, d: vec3) {
            let mut rq = ray_query::default();
            rayQueryInitialize(rq, acc, RayDesc {
                flags: 0u32, cull_mask: 0xFF, tmin: 0.0, tmax: 1.0, origin: o, dir: d,
            });
            rayQueryProceed(rq);
        }
        "#,
    );
}

#[test]
fn wgsl_output_writes_a_ray_query_as_the_builtin() {
    // Naga's backend cannot print a ray query, so the calls come back as the
    // WGSL builtins and a frontend can read them again.
    let module = trace(RAY_QUERY_SHADER);
    let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
    let info = validate_with(&module, flags, naga::valid::Capabilities::RAY_QUERY).unwrap();
    let wgsl = to_wgsl(&module, &info).unwrap_or_else(|err| panic!("{err}"));
    assert!(wgsl.contains("rayQueryInitialize("), "{wgsl}");
}

#[test]
fn zero_values() {
    // `T()` is WGSL's zero value, which the reservoir and surface helpers lean on.
    let wgsl = roundtrip("struct S { a: f32, b: vec3 } fn f() -> S { let s = S(); s }");
    assert!(wgsl.contains("S()"), "{wgsl}");
    validate_only("fn f() -> vec3 { vec3() }");
    validate_only("fn f() -> mat4 { mat4() }");
}

/// Blade's ray-tracing passes index whole descriptor arrays of resources.
#[test]
fn binding_arrays() {
    let module = parse_str(
        r#"
        static textures: binding_array<texture_2d<f32>> = ();
        static samp: sampler = ();
        fn sample_one(i: u32, uv: vec2) -> vec4 {
            textureSampleLevel(textures[i], samp, uv, 0.0)
        }
        "#,
    )
    .expect("parse");
    let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
    let caps = naga::valid::Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY
        | naga::valid::Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY_NON_UNIFORM_INDEXING;
    validate_with(&module, flags, caps).expect("validate");
}

#[test]
fn sized_binding_array() {
    let module = parse_str(
        r#"
        static textures: binding_array<texture_2d<f32>, 8> = ();
        static samp: sampler = ();
        fn sample_first(uv: vec2) -> vec4 { textureSampleLevel(textures[0], samp, uv, 0.0) }
        "#,
    )
    .expect("parse");
    let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
    validate_with(
        &module,
        flags,
        naga::valid::Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY,
    )
    .expect("validate");
}

#[test]
fn binding_array_of_buffers_is_storage() {
    // A binding array of textures is a handle; one of buffers is not, and takes
    // an address space like any other buffer.
    let module = parse_str(
        r#"
        struct VertexBuffer { data: [u32] }
        #[storage] static vertex_buffers: binding_array<VertexBuffer> = ();
        fn fetch(i: u32, j: u32) -> u32 { vertex_buffers[i].data[j] }
        "#,
    )
    .expect("parse");
    let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
    let caps = naga::valid::Capabilities::STORAGE_BUFFER_BINDING_ARRAY
        | naga::valid::Capabilities::STORAGE_BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING;
    validate_with(&module, flags, caps).expect("validate");
}
