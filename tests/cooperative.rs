#![cfg(feature = "wgsl")]
//! Cooperative matrices: `CoopMat8x8<f32, A>` and the rest, WGSL's
//! `coop_mat8x8` and `coop_mat16x16`. A module that has one needs Naga's
//! `COOPERATIVE_MATRIX`.

mod common;
use common::reject;
use synaga::naga::valid::{Capabilities, ValidationFlags};

fn caps() -> Capabilities {
    Capabilities::COOPERATIVE_MATRIX | Capabilities::SHADER_FLOAT16
}

/// Lower, validate with `COOPERATIVE_MATRIX`, and print.
fn roundtrip_cooperative(src: &str) -> String {
    let module = synaga::parse_str(src).unwrap_or_else(|e| panic!("parse: {e}\n{src}"));
    let info = synaga::validate_with(&module, ValidationFlags::all(), caps())
        .unwrap_or_else(|e| panic!("validate: {e}\n{src}"));
    synaga::to_wgsl(&module, &info).unwrap_or_else(|e| panic!("wgsl: {e}\n{src}"))
}

/// A shader checked twice from the same tokens: by `rustc`, as a module here
/// against `synaga-shader`, and by the transpiler. The WGSL it lowers to.
macro_rules! shader {
    ($($source:tt)*) => {{
        #[allow(dead_code, non_upper_case_globals)]
        mod rust {
            use synaga_shader::*;
            $($source)*
        }
        roundtrip_cooperative(stringify!($($source)*))
    }};
}

#[test]
fn a_workgroup_multiplies_a_tile() {
    let wgsl = shader! {
        static lhs: Storage<[f32]> = group(0).binding(0);
        static rhs: Storage<[f32]> = group(0).binding(1);
        static out: StorageMut<[f32]> = group(0).binding(2);

        const N: u32 = 64;
        const K: u32 = 64;
        const TILE: u32 = 8;

        type Lhs = CoopMat8x8<f32, A>;
        type Rhs = CoopMat8x8<f32, B>;
        type Acc = CoopMat8x8<f32, C>;

        /// `lhs` is M×K and `rhs` K×N, row by row, and each workgroup
        /// computes the tile of `out` its id says.
        #[entry_point(compute, threads(64))]
        fn multiply(workgroup_id: Vec3<u32>) {
            let row = workgroup_id.y * TILE;
            let column = workgroup_id.x * TILE;
            let mut acc = Acc::default();
            for step in 0..K / TILE {
                let k = step * TILE;
                let a = Lhs::load_row_major(&lhs[(row * K + k) as usize..], K);
                let b = Rhs::load_row_major(&rhs[(k * N + column) as usize..], N);
                acc = a.mul_add(b, acc);
            }
            acc.store_row_major(&mut out.get_mut()[(row * N + column) as usize..], N);
        }
    };
    for wgsl_spelling in [
        "enable wgpu_cooperative_matrix;",
        "coop_mat8x8<f32,C>()",
        "coopLoadT<coop_mat8x8<f32,A>>((&lhs[",
        "coopLoadT<coop_mat8x8<f32,B>>((&rhs[",
        "coopMultiplyAdd(",
        "coopStoreT(",
        "(&out[",
    ] {
        assert!(wgsl.contains(wgsl_spelling), "{wgsl_spelling}\n{wgsl}");
    }
}

#[test]
fn a_matrix_loads_from_workgroup_memory_column_by_column() {
    let wgsl = shader! {
        static tile: Workgroup<[f16; 256]> = binding();
        static out: StorageMut<[f16]> = group(0).binding(0);

        fn scaled(m: CoopMat16x16<f16, C>, by: f16) -> CoopMat16x16<f16, C> {
            let mut twice = m + m - m * by;
            twice += by * m;
            twice
        }

        #[entry_point(compute, threads(32))]
        fn square(local_invocation_index: u32) {
            tile.get_mut()[local_invocation_index as usize] = f16::ONE;
            workgroup_barrier();
            let a = CoopMat16x16::<f16, A>::load(&tile[..], 16);
            let b = CoopMat16x16::<f16, B>::load(&tile[16..], 16);
            let c = CoopMat16x16::<f16, C>::load(&*tile, 16);
            scaled(a.mul_add(b, c), f16::from_f32(0.5)).store(&mut out.get_mut()[..], 16);
        }
    };
    for wgsl_spelling in [
        "coopLoad<coop_mat16x16<f16,A>>((&tile[0]), 16u)",
        "coopLoad<coop_mat16x16<f16,B>>((&tile[16]), 16u)",
        "coopLoad<coop_mat16x16<f16,C>>((&tile[0]), 16u)",
        "fn scaled(m: coop_mat16x16<f16,C>, by: f16) -> coop_mat16x16<f16,C>",
        "twice = ((m + m) - (m * by));",
        "(by * m)",
        "coopStore(_e16, (&out[0]), 16u);",
    ] {
        assert!(wgsl.contains(wgsl_spelling), "{wgsl_spelling}\n{wgsl}");
    }
}

#[test]
fn a_cooperative_matrix_needs_the_capability() {
    let module = synaga::parse_str("fn f(m: CoopMat8x8<f32, C>) -> CoopMat8x8<f32, C> { m }")
        .expect("lowers");
    let err = synaga::validate_with(&module, ValidationFlags::all(), Capabilities::empty())
        .expect_err("validates without COOPERATIVE_MATRIX");
    assert!(err.to_string().contains("COOPERATIVE_MATRIX"), "{err}");
}

#[test]
fn a_matrix_says_what_it_holds() {
    for source in [
        "fn f(m: CoopMat8x8<f32>) {}",
        "fn f(m: CoopMat8x8<u32, A>) {}",
        "fn f(m: CoopMat8x8<f32, D>) {}",
        // Nothing else in a call says what it loads.
        "static data: Storage<[f32]> = group(0).binding(0);
         fn f() { let m = CoopMat8x8::load(&data[..], 8); }",
    ] {
        let err = reject(source);
        assert!(
            err.contains("says its scalar and its role"),
            "`{source}`: {err}"
        );
    }
}

#[test]
fn a_matrix_loads_its_own_scalars_and_stores_where_it_can() {
    for (source, says) in [
        (
            "static data: Storage<[Vec4]> = group(0).binding(0);
             fn f() { let m = CoopMat8x8::<f32, A>::load(&data[..], 8); }",
            "takes a slice of the matrix's own scalars",
        ),
        (
            "static data: Storage<[f32]> = group(0).binding(0);
             fn f() { let m = CoopMat8x8::<f32, A>::load(data[0], 8); }",
            "takes a slice of the matrix's own scalars",
        ),
        (
            "static data: Storage<[f32]> = group(0).binding(0);
             fn f(m: CoopMat8x8<f32, C>) { m.store(&mut data[..], 8); }",
            "cannot write through `data`",
        ),
        (
            "static data: StorageMut<[f32]> = group(0).binding(0);
             fn f(m: CoopMat8x8<f32, C>) { m.store(&data.get_mut()[..], 8); }",
            "takes a slice of the matrix's own scalars",
        ),
    ] {
        let err = reject(source);
        assert!(err.contains(says), "`{source}`: {err}");
    }
}

#[test]
fn a_multiply_add_takes_a_b_and_c() {
    for source in [
        "fn f(a: CoopMat8x8<f32, B>, b: CoopMat8x8<f32, B>, c: CoopMat8x8<f32, C>) -> CoopMat8x8<f32, C> { a.mul_add(b, c) }",
        "fn f(a: CoopMat8x8<f32, A>, b: CoopMat8x8<f32, B>, c: CoopMat16x16<f32, C>) -> CoopMat16x16<f32, C> { a.mul_add(b, c) }",
        "fn f(a: CoopMat8x8<f32, A>, b: CoopMat8x8<f16, B>, c: CoopMat8x8<f32, C>) -> CoopMat8x8<f32, C> { a.mul_add(b, c) }",
    ] {
        let err = reject(source);
        assert!(err.contains("`a.mul_add(b, c)` takes"), "`{source}`: {err}");
    }
}

#[test]
fn only_a_compute_shader_has_one() {
    let module = synaga::parse_str(
        r#"
        static data: Storage<[f32]> = group(0).binding(0);
        #[entry_point(fragment)]
        fn fs() -> Vec4 {
            let m = CoopMat8x8::<f32, C>::load(&data[..], 8);
            Vec4::ZERO
        }
        "#,
    )
    .expect("lowers");
    let err = synaga::validate_with(&module, ValidationFlags::all(), caps())
        .expect_err("a fragment shader cannot load one");
    assert!(
        format!("{err:?}").contains("ForbiddenStageOperations"),
        "{err:?}"
    );
}

#[test]
fn the_whole_subgroup_reaches_an_operation() {
    // Each invocation holds part of the matrix, so all of them take part, as
    // in a barrier. Naga refuses a load or a multiply-add under a branch they
    // may not all take.
    let source = |operation: &str| {
        format!(
            r#"
            static data: StorageMut<[f32]> = group(0).binding(0);
            #[entry_point(compute, threads(32))]
            fn cs(local_invocation_index: u32) {{
                let m = CoopMat8x8::<f32, C>::default();
                if local_invocation_index == 0 {{
                    {operation}
                }}
            }}
            "#
        )
    };
    let verdict = |operation: &str| {
        let module = synaga::parse_str(&source(operation)).expect("lowers");
        synaga::validate_with(&module, ValidationFlags::all(), caps()).map(|_| ())
    };
    let err = verdict(
        "let n = CoopMat8x8::<f32, C>::load(&data[..], 8); n.store(&mut data.get_mut()[..], 8);",
    )
    .expect_err("a load under a non-uniform branch");
    assert!(
        format!("{err:?}").contains("NonUniformControlFlow"),
        "{err:?}"
    );
    // A store is a statement, which Naga does not check, though the GPU needs
    // the same of it.
    verdict("m.store(&mut data.get_mut()[..], 8);").expect("Naga lets a store through");
}
