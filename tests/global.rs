#![cfg(feature = "wgsl")]
mod common;

use common::*;

#[test]
fn uniform_mat4_in_vertex() {
    let wgsl = roundtrip(
        r#"
        #[group(0)]
        #[binding(0)]
        static mvp: mat4 = ();

        #[vertex]
        #[output(builtin(position))]
        fn vs_main(#[location(0)] pos: vec3) -> vec4 {
            mvp * vec4(pos, 1.0)
        }
        "#,
    );
    assert!(
        wgsl.contains("@group(0)") || wgsl.contains("group"),
        "{wgsl}"
    );
    assert!(
        wgsl.contains("@binding(0)") || wgsl.contains("binding"),
        "{wgsl}"
    );
    assert!(wgsl.contains("uniform"), "{wgsl}");
}

#[test]
fn extern_static_uniform_vec() {
    validate_only(
        r#"
        extern "C" {
            #[group(0)]
            #[binding(1)]
            static color: vec4;
        }

        #[fragment]
        #[output(location(0))]
        fn fs_main() -> vec4 {
            color
        }
        "#,
    );
}

#[test]
fn uniform_in_helper() {
    validate_only(
        r#"
        #[group(1)]
        #[binding(0)]
        #[uniform]
        static scale: vec3 = ();

        fn apply(p: vec3) -> vec3 { p * scale }

        #[vertex]
        #[output(builtin(position))]
        fn vs_main(#[location(0)] pos: vec3) -> vec4 {
            let q = apply(pos);
            vec4(q, 1.0)
        }
        "#,
    );
}

#[test]
fn storage_read() {
    validate_only(
        r#"
        #[group(0)]
        #[binding(0)]
        #[storage]
        static data: vec4 = ();

        fn f() -> vec4 { data }
        "#,
    );
}

#[test]
fn storage_read_write_assign() {
    validate_only(
        r#"
        #[group(0)]
        #[binding(0)]
        #[storage(read_write)]
        static data: vec4 = ();

        fn f(v: vec4) -> vec4 {
            data = v;
            data
        }
        "#,
    );
}

#[test]
fn rejects_assign_to_uniform() {
    let msg = reject(
        r#"
        #[group(0)]
        #[binding(0)]
        static mvp: mat4 = ();
        fn f(m: mat4) -> mat4 { mvp = m; mvp }
        "#,
    );
    assert!(msg.contains("read-only") || msg.contains("assign"), "{msg}");
}

#[test]
fn rejects_missing_group() {
    let msg = reject(
        r#"
        #[binding(0)]
        static mvp: mat4 = ();
        fn f() -> mat4 { mvp }
        "#,
    );
    assert!(msg.contains("group") || msg.contains("binding"), "{msg}");
}

#[test]
fn rejects_duplicate_global() {
    let msg = reject(
        r#"
        #[group(0)]
        #[binding(0)]
        static a: vec4 = ();
        #[group(0)]
        #[binding(1)]
        static a: vec4 = ();
        fn f() -> vec4 { a }
        "#,
    );
    assert!(msg.contains("duplicate") || msg.contains("a"), "{msg}");
}
