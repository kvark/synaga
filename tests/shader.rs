#![cfg(feature = "wgsl")]
//! One realistic multi-stage shader, exercised end to end.
//!
//! The per-feature tests each pin down one rule; this one checks they still
//! compose into something a real pipeline would use.

mod common;

use common::*;

const SHADER: &str = r#"
    struct Camera {
        view_proj: mat4,
        eye: vec3,
        time: f32,
    }

    struct VsOut {
        #[builtin(position)] clip: vec4,
        #[location(0)] world: vec3,
        #[location(1)] uv: vec2,
        #[location(2)] #[flat] material: u32,
    }

    #[group(0)] #[binding(0)] static camera: Camera = ();
    #[group(0)] #[binding(1)] #[storage(read_write)] static counter: u32 = ();

    fn luminance(color: vec3) -> f32 {
        dot(color, vec3(0.2126, 0.7152, 0.0722))
    }

    fn wave(p: vec3, t: f32) -> f32 {
        let phase = p.x * 4.0 + t;
        if phase > 0.0 { sin(phase) } else { -sin(-phase) }
    }

    #[vertex]
    fn vs(
        #[location(0)] pos: vec3,
        #[location(1)] uv: vec2,
        #[builtin(instance_index)] inst: u32,
    ) -> VsOut {
        let height = wave(pos, camera.time);
        let world = vec3(pos.x, pos.y + height, pos.z);
        VsOut {
            clip: camera.view_proj * vec4(world, 1.0),
            world,
            uv,
            material: inst & 3,
        }
    }

    #[fragment]
    #[output(location(0))]
    fn fs(v: VsOut) -> vec4 {
        let to_eye = normalize(camera.eye - v.world);
        let ndl = clamp(to_eye.y, 0.0, 1.0);
        let tint = vec3(v.uv, 1.0);
        let shade = luminance(tint) * ndl;
        let bias = (v.material as f32) * 0.25;
        vec4(tint * (shade + bias), 1.0)
    }

    #[compute]
    #[workgroup_size(64)]
    fn accumulate(#[builtin(global_invocation_id)] id: vec3<u32>) {
        let total = 0u32;
        let n = 0u32;
        while n < 4 {
            n += 1;
            if n == 2 {
                continue;
            }
            total += id.x << n;
        }
        counter = total;
    }
"#;

#[test]
fn multi_stage_shader() {
    let wgsl = roundtrip(SHADER);

    // The interface struct keeps its bindings, and only the integer varying
    // needs to spell its interpolation out.
    assert!(wgsl.contains("@builtin(position) clip"), "{wgsl}");
    assert!(
        wgsl.contains("@location(2) @interpolate(flat) material"),
        "{wgsl}"
    );
    assert_eq!(wgsl.matches("@interpolate").count(), 1, "{wgsl}");

    assert!(wgsl.contains("var<uniform> camera: Camera"), "{wgsl}");
    assert!(
        wgsl.contains("var<storage, read_write> counter: u32"),
        "{wgsl}"
    );

    assert!(wgsl.contains("@vertex"), "{wgsl}");
    assert!(wgsl.contains("@fragment"), "{wgsl}");
    assert!(
        wgsl.contains("@compute @workgroup_size(64, 1, 1)"),
        "{wgsl}"
    );

    // `inst & 3` and `id.x << n` took their literal types from context, and the
    // cast came through as a conversion.
    assert!(wgsl.contains("3u"), "{wgsl}");
    assert!(wgsl.contains("f32(v.material)"), "{wgsl}");
}
