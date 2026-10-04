#![cfg(feature = "wgsl")]
//! Subgroup operations: WGSL's builtins in snake case, statements that write a
//! result, as Naga has them.

mod common;
use common::reject;
use synaga::naga::valid::{Capabilities, ValidationFlags};

fn subgroup_caps() -> Capabilities {
    Capabilities::SUBGROUP | Capabilities::SUBGROUP_BARRIER
}

/// Lower, validate with the subgroup capabilities, and print.
fn roundtrip_subgroup(src: &str) -> String {
    let module = synaga::parse_str(src).unwrap_or_else(|e| panic!("parse: {e}\n{src}"));
    let info = synaga::validate_with(&module, ValidationFlags::all(), subgroup_caps())
        .unwrap_or_else(|e| panic!("validate: {e}\n{src}"));
    synaga::to_wgsl(&module, &info).unwrap_or_else(|e| panic!("wgsl: {e}\n{src}"))
}

const REDUCE: &str = r#"
    static totals: StorageMut<[u32]> = group(0).binding(0);
    static sums: StorageMut<[f32]> = group(0).binding(1);
    const LANE: u32 = 2;

    #[entry_point(compute, threads(64))]
    fn reduce(global_invocation_id: Vec3<u32>, subgroup_size: u32, subgroup_invocation_id: u32) {
        let i = global_invocation_id.x;
        let value = sums.get_mut()[i];
        let total = subgroup_add(value) + subgroup_mul(value) + subgroup_min(value) + subgroup_max(value);
        let before = subgroup_exclusive_add(value) + subgroup_inclusive_mul(value);
        let bits = subgroup_and(i) | subgroup_or(i) ^ subgroup_xor(i);
        let votes = subgroup_ballot(value > 0.0);
        let leader = subgroup_broadcast_first(i) + subgroup_broadcast(i, LANE + 1);
        let moved = subgroup_shuffle(i, 0) + subgroup_shuffle_xor(i, 1) + subgroup_shuffle_up(i, 1) + subgroup_shuffle_down(i, 1);
        subgroup_barrier();
        if subgroup_all(value > 0.0) || subgroup_any(subgroup_invocation_id == 0) {
            totals.get_mut()[i] = bits + votes.x + leader + moved + subgroup_size;
            sums.get_mut()[i] = total + before;
        }
    }
"#;

#[test]
fn every_subgroup_operation_is_naga_s_own() {
    let wgsl = roundtrip_subgroup(REDUCE);
    for builtin in [
        "subgroupAdd(",
        "subgroupMul(",
        "subgroupMin(",
        "subgroupMax(",
        "subgroupExclusiveAdd(",
        "subgroupInclusiveMul(",
        "subgroupAnd(",
        "subgroupOr(",
        "subgroupXor(",
        "subgroupBallot(",
        "subgroupBroadcastFirst(",
        "subgroupShuffle(",
        "subgroupShuffleXor(",
        "subgroupShuffleUp(",
        "subgroupShuffleDown(",
        "subgroupAll(",
        "subgroupAny(",
        "subgroupBarrier();",
        "@builtin(subgroup_size)",
        "@builtin(subgroup_invocation_id)",
    ] {
        assert!(wgsl.contains(builtin), "{builtin}\n{wgsl}");
    }
    // The lane is folded, as a broadcast needs one known beforehand.
    assert!(
        wgsl.contains("subgroupBroadcast(global_invocation_id.x, 3u)"),
        "{wgsl}"
    );
}

#[test]
fn a_subgroup_operation_needs_the_capability() {
    let module = synaga::parse_str(REDUCE).unwrap();
    let err = synaga::validate_with(&module, ValidationFlags::all(), Capabilities::empty())
        .expect_err("validates without SUBGROUP");
    assert!(err.to_string().contains("SUBGROUP"), "{err}");
}

#[test]
fn a_fragment_shader_swaps_within_its_quad() {
    let wgsl = roundtrip_subgroup(
        r#"
        #[entry_point(fragment)]
        fn fs(#[location(0)] uv: Vec2, subgroup_size: u32) -> Vec4 {
            let across = quad_swap_x(uv) + quad_swap_y(uv) + quad_swap_diagonal(uv);
            let first = quad_broadcast(uv, 0);
            vec4(across.x, first.y, subgroup_size as f32, 1.0)
        }
        "#,
    );
    for builtin in [
        "quadSwapX(uv)",
        "quadSwapY(uv)",
        "quadSwapDiagonal(uv)",
        "quadBroadcast(uv, 0u)",
        "@builtin(subgroup_size)",
    ] {
        assert!(wgsl.contains(builtin), "{builtin}\n{wgsl}");
    }
}

#[test]
fn compute_has_the_subgroup_ids_too() {
    roundtrip_subgroup(
        r#"
        static out: StorageMut<[u32]> = group(0).binding(0);
        #[entry_point(compute, threads(64))]
        fn ids(num_subgroups: u32, subgroup_id: u32) {
            out.get_mut()[subgroup_id] = num_subgroups;
        }
        "#,
    );
}

#[test]
fn a_broadcast_lane_is_a_constant() {
    let err = reject(
        r#"
        fn f(x: f32, lane: u32) -> f32 { subgroup_broadcast(x, lane) }
        "#,
    );
    assert!(err.contains("`subgroup_broadcast` reads one lane"), "{err}");
}

#[test]
fn an_operation_takes_what_wgsl_says() {
    for (source, name) in [
        ("fn f(x: f32) -> f32 { subgroup_and(x) }", "subgroup_and"),
        ("fn f(x: u32) -> bool { subgroup_all(x) }", "subgroup_all"),
        ("fn f(x: bool) -> bool { subgroup_add(x) }", "subgroup_add"),
        (
            "fn f(x: bool) -> bool { subgroup_shuffle(x, 1) }",
            "subgroup_shuffle",
        ),
    ] {
        let err = reject(source);
        assert!(err.contains(name), "`{source}`: {err}");
    }
}
