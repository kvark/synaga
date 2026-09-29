#![cfg(feature = "wgsl")]
//! `let p = &mut place;`: a name for a place, stored to and loaded through.

mod common;

use common::*;

const PARTICLES: &str = r#"
    #[derive(Clone, Copy)]
    struct Particle { pos: Vec3, vel: Vec3, life: f32 }
    #[derive(Clone, Copy)]
    struct Params { time_delta: f32 }
    static particles: StorageMut<[Particle]> = binding();
    static params: Uniform<Params> = binding();
"#;

#[test]
fn a_mutable_borrow_writes_the_place() {
    let wgsl = roundtrip_unbound(&format!(
        r#"{PARTICLES}
        #[entry_point(compute, threads(64))]
        fn update(global_invocation_id: Vec3<u32>) {{
            let p = &mut particles.get_mut()[global_invocation_id.x as usize];
            p.pos += p.vel * params.time_delta;
            p.life -= params.time_delta;
            if p.life < 0.0 {{
                p.life = 0.0;
            }}
        }}
        "#
    ));
    // Every access goes through the element's pointer: nothing is copied out
    // into a local and written back.
    assert!(!wgsl.contains("var p"), "{wgsl}");
    assert!(
        wgsl.contains("particles[global_invocation_id.x].pos = "),
        "{wgsl}"
    );
    assert!(
        wgsl.contains("particles[global_invocation_id.x].life = 0f"),
        "{wgsl}"
    );
}

#[test]
fn a_shared_borrow_reads_the_place() {
    validate_only_unbound(&format!(
        r#"{PARTICLES}
        static ages: StorageMut<[f32]> = binding();
        #[entry_point(compute, threads(64))]
        fn age(global_invocation_id: Vec3<u32>) {{
            let index = global_invocation_id.x as usize;
            let p = &particles[index];
            ages.get_mut()[index] = p.life + p.pos.length();
        }}
        "#
    ));
}

#[test]
fn a_deref_stores_the_whole_place() {
    let wgsl = roundtrip_unbound(&format!(
        r#"{PARTICLES}
        #[entry_point(compute, threads(64))]
        fn reset(global_invocation_id: Vec3<u32>) {{
            let p = &mut particles.get_mut()[global_invocation_id.x as usize];
            let fresh = Particle {{ pos: Vec3::splat(0.0), vel: p.vel, life: 1.0 }};
            *p = fresh;
        }}
        "#
    ));
    assert!(
        wgsl.contains("particles[global_invocation_id.x] = "),
        "{wgsl}"
    );
}

#[test]
fn a_borrowed_local_is_changed_through_the_borrow() {
    let wgsl = roundtrip(
        r#"
        fn f(x: f32) -> Vec3 {
            let mut v = Vec3::splat(0.0);
            let r = &mut v;
            r.y = x;
            v
        }
        "#,
    );
    assert!(wgsl.contains("v.y = x"), "{wgsl}");
}

#[test]
fn a_mutable_borrow_of_a_uniform_is_refused() {
    let message = reject(&format!(
        r#"{PARTICLES}
        #[entry_point(compute, threads(64))]
        fn update() {{
            let dt = &mut params.time_delta;
            *dt = 0.0;
        }}
        "#
    ));
    assert!(message.contains("params"), "{message}");
}
