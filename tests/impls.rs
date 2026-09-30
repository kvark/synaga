#![cfg(feature = "wgsl")]
//! `impl` blocks: associated functions and methods, on the shader's own types
//! and, through a trait, on a vector.

mod common;

use common::*;

const MATERIAL: &str = r#"
    #[derive(Clone, Copy, Default)]
    pub struct Material {
        pub albedo: Vec3,
        pub roughness: f32,
    }

    impl Material {
        pub fn from_metallic_roughness(base: Vec3, metalness: f32, roughness: f32) -> Self {
            Self { albedo: base * (1.0 - metalness), roughness }
        }
        pub fn alpha(self) -> f32 {
            let r = self.roughness.clamp(0.05, 1.0);
            r * r
        }
        pub fn is_rough(&self) -> bool {
            self.alpha() > 0.5
        }
        pub fn darken(&mut self, by: f32) {
            self.albedo *= by;
        }
    }
"#;

#[test]
fn an_impl_gives_a_type_functions_and_methods() {
    let wgsl = roundtrip(&format!(
        "{MATERIAL}
        fn shade(base: Vec3) -> f32 {{
            let mut mat = Material::from_metallic_roughness(base, 0.5, 0.8);
            mat.darken(0.5);
            if mat.is_rough() {{ mat.alpha() }} else {{ mat.albedo.x }}
        }}"
    ));
    // Each is a function, named after the type it is on.
    assert!(
        wgsl.contains("fn Material_from_metallic_roughness(base: vec3<f32>"),
        "{wgsl}"
    );
    // WGSL reserves `self`, so Naga names the receivers apart.
    assert!(wgsl.contains("fn Material_alpha(self_"), "{wgsl}");
    // `&self` takes a copy, and `&mut self` the storage.
    assert!(wgsl.contains(": Material) -> bool"), "{wgsl}");
    assert!(
        wgsl.contains(": ptr<function, Material>, by: f32)"),
        "{wgsl}"
    );
    assert!(wgsl.contains("Material_darken((&mat), 0.5f)"), "{wgsl}");
    assert!(wgsl.contains("= Material_alpha(self_"), "{wgsl}");
}

#[test]
fn a_method_is_lowered_when_something_calls_it() {
    // The host's `Debug` sits in the shader module without being lowered.
    let wgsl = roundtrip(&format!(
        "{MATERIAL}
        impl core::fmt::Debug for Material {{
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {{
                write!(f, \"{{}}\", self.roughness)
            }}
        }}
        fn alpha_of(mat: Material) -> f32 {{ mat.alpha() }}"
    ));
    assert!(wgsl.contains("fn Material_alpha"), "{wgsl}");
    assert!(!wgsl.contains("from_metallic_roughness"), "{wgsl}");
    assert!(!wgsl.contains("fmt"), "{wgsl}");
}

#[test]
fn a_trait_adds_methods_to_a_vector() {
    let wgsl = roundtrip(
        r#"
        pub trait Quaternion {
            fn inv(self) -> Self;
            fn rotate(self, v: Vec3) -> Vec3;
        }
        impl Quaternion for Vec4 {
            fn inv(self) -> Self {
                (-self.xyz()).extend(self.w)
            }
            fn rotate(self, v: Vec3) -> Vec3 {
                v + 2.0 * self.xyz().cross(self.xyz().cross(v) + self.w * v)
            }
        }
        fn to_local(q: Vec4, v: Vec3) -> Vec3 { q.inv().rotate(v) }
        fn through_the_type(q: Vec4) -> Vec4 { Vec4::inv(q) }
        "#,
    );
    assert!(
        wgsl.contains("fn Vec4_inv(self_: vec4<f32>) -> vec4<f32>"),
        "{wgsl}"
    );
    assert!(wgsl.contains("= Vec4_inv(q)"), "{wgsl}");
    assert!(wgsl.contains("Vec4_rotate("), "{wgsl}");
}

#[test]
fn a_vectors_own_method_comes_first() {
    // Rust calls the inherent `dot`, not the trait's, and so does the shader.
    let wgsl = roundtrip(
        r#"
        pub trait Odd { fn dot(self, other: Vec3) -> f32; }
        impl Odd for Vec3 { fn dot(self, other: Vec3) -> f32 { 7.0 } }
        fn f(a: Vec3, b: Vec3) -> f32 { a.dot(b) }
        "#,
    );
    assert!(wgsl.contains("dot(a, b)"), "{wgsl}");
    assert!(!wgsl.contains("Vec3_dot"), "{wgsl}");
}

#[test]
fn a_hand_written_default_is_called() {
    // A `Default` written in the shader is one it can see.
    let wgsl = roundtrip(
        r#"
        #[derive(Clone, Copy)]
        pub struct Weights { pub first: f32, pub rest: f32 }
        impl Default for Weights {
            fn default() -> Self { Self { first: 1.0, rest: 0.0 } }
        }
        fn f() -> f32 { Weights::default().first }
        "#,
    );
    assert!(wgsl.contains("fn Weights_default() -> Weights"), "{wgsl}");
}

#[test]
fn a_method_on_a_struct_a_buffer_holds() {
    let wgsl = roundtrip_unbound(&format!(
        "{MATERIAL}
        static material: Uniform<Material> = binding();
        static materials: StorageMut<[Material]> = binding();
        fn f(i: u32) -> f32 {{ material.alpha() + materials[i as usize].alpha() }}"
    ));
    assert!(wgsl.contains("Material_alpha(_e"), "{wgsl}");
    // `&mut self` needs a local: a buffer's element cannot be passed to it.
    let msg = reject(&format!(
        "{MATERIAL}
        static materials: StorageMut<[Material]> = binding();
        fn f(i: u32) {{ materials.get_mut()[i as usize].darken(0.5); }}"
    ));
    assert!(msg.contains("has to be a local"), "{msg}");
}

#[test]
fn a_method_through_its_type() {
    // `Material::alpha(mat)` is the same call as `mat.alpha()`.
    let wgsl = roundtrip(&format!(
        "{MATERIAL}
        fn f(mat: Material) -> f32 {{
            let mut copy = mat;
            Material::darken(&mut copy, 0.5);
            Material::alpha(copy)
        }}"
    ));
    assert!(wgsl.contains("Material_darken((&copy), 0.5f)"), "{wgsl}");
}

#[test]
fn what_an_impl_cannot_do() {
    let msg = reject(&format!(
        "{MATERIAL}
        fn f(mat: Material) -> Material {{ mat.from_metallic_roughness(Vec3::ONE, 0.0, 1.0) }}"
    ));
    assert!(msg.contains("takes no `self`"), "{msg}");
    let msg = reject(
        "struct S { a: f32 } impl S { fn f(mut self) -> f32 { self.a += 1.0; self.a } }
         fn g(s: S) -> f32 { s.f() }",
    );
    assert!(msg.contains("`mut self`"), "{msg}");
    // A generic `impl` is the host's, and sits in a shader module unread.
    validate_only("struct S { a: f32 } impl<T: Into<f32>> From<T> for S { fn from(t: T) -> Self { S { a: t.into() } } }");
    let msg = reject(
        "struct S { a: f32 } impl<T> From<T> for S { fn from(t: T) -> Self { S { a: 0.0 } } }
         fn f() -> S { S::from(1.0) }",
    );
    assert!(msg.contains("from"), "{msg}");
    // Two traits with the method are as ambiguous here as they are in Rust.
    let msg = reject(
        "struct S { a: f32 }
         trait A { fn f(self) -> f32; } impl A for S { fn f(self) -> f32 { 1.0 } }
         trait B { fn f(self) -> f32; } impl B for S { fn f(self) -> f32 { 2.0 } }
         fn g(s: S) -> f32 { s.f() }",
    );
    assert!(msg.contains("more than one"), "{msg}");
    // A method cannot call itself, any more than a function can.
    let msg = reject(
        "struct S { a: f32 } impl S { fn f(self) -> f32 { self.f() } } fn g(s: S) -> f32 { s.f() }",
    );
    assert!(msg.contains("`S::f` depends on itself"), "{msg}");
}
