//! The math runs on the CPU, and means there what it means in a shader.
//!
//! A method or operator named as Rust names it is Rust's; a free function
//! keeps WGSL's meaning. The GPU half of this, that the transpiler lowers each
//! to the builtin that means the same, is the transpiler's tests.

use synaga_shader::*;

#[test]
fn operators_work_lane_by_lane_and_against_a_scalar() {
    let a = vec3(1.0, 2.0, 3.0);
    let b = vec3(4.0, 5.0, 6.0);
    assert_eq!(a + b, vec3(5.0, 7.0, 9.0));
    assert_eq!(b - a, vec3(3.0, 3.0, 3.0));
    assert_eq!(a * b, vec3(4.0, 10.0, 18.0));
    assert_eq!(b / a, vec3(4.0, 2.5, 2.0));
    assert_eq!(a * 2.0, vec3(2.0, 4.0, 6.0));
    assert_eq!(2.0 * a, vec3(2.0, 4.0, 6.0));
    assert_eq!(1.0 - a, vec3(0.0, -1.0, -2.0));
    assert_eq!(-a, vec3(-1.0, -2.0, -3.0));
    let mut c = a;
    c += b;
    c *= 0.5;
    assert_eq!(c, vec3(2.5, 3.5, 4.5));

    let bits = vec4::<u32>(0x12, 0x34, 0x56, 0x78);
    assert_eq!(bits & 0xF, vec4(0x2, 0x4, 0x6, 0x8));
    assert_eq!(bits << 4, vec4(0x120, 0x340, 0x560, 0x780));
    assert_eq!(bits >> vec4(0, 1, 2, 3), vec4(0x12, 0x1A, 0x15, 0xF));
    assert_eq!(!vec2(true, false), vec2(false, true));
    assert_eq!(vec2(-7, 7) % 3, vec2(-1, 1));
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "overflow")]
fn an_integer_lane_overflows_as_the_integer_would() {
    // `u32::MAX + 1` panics under overflow checks, lane or not.
    let _ = vec2::<u32>(u32::MAX, 0) + vec2(1, 0);
}

#[test]
fn wrapping_arithmetic_wraps() {
    assert_eq!(
        vec2::<u32>(u32::MAX, 3).wrapping_add(vec2(1, 1)),
        vec2(0, 4)
    );
    assert_eq!(
        vec2::<u32>(0x8000_0000, 3).wrapping_mul(vec2(2, 3)),
        vec2(0, 9)
    );
    assert_eq!(vec2::<u32>(1, 0).wrapping_neg(), vec2(u32::MAX, 0));
    assert_eq!(vec2(i32::MIN, 1).wrapping_neg(), vec2(i32::MIN, -1));
    assert_eq!(
        vec2(i32::MIN, 0).wrapping_sub(vec2(1, 1)),
        vec2(i32::MAX, -1)
    );
}

#[test]
fn comparisons_are_per_lane() {
    // A method needs the scalar settled first, as for a bare integer literal.
    let a = vec3::<i32>(1, 5, 3);
    let b = vec3(2, 5, 1);
    assert_eq!(a.cmplt(b), vec3(true, false, false));
    assert_eq!(a.cmple(b), vec3(true, true, false));
    assert_eq!(a.cmpeq(b), vec3(false, true, false));
    assert_eq!(a.cmpne(b), vec3(true, false, true));
    assert!(a.cmpge(Vec3::ZERO).all());
    assert!(a.cmpgt(b).any() && !a.cmpgt(b).all());
}

#[test]
// Negating a comparison is the point here, which clippy warns about for any
// type that is only partially ordered, as a vector is.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn an_operator_compares_every_lane() {
    let p = vec2(3, -1);
    let extent = vec2(4, 4);
    assert!(p < extent && p <= extent && !(p >= Vec2::ZERO));
    // `!(a < b)` is "some lane is not less", not `a >= b`.
    let edge = vec2::<i32>(4, 2);
    assert!(!(edge < extent) && !(edge >= extent));
    assert_eq!(!(edge < extent), edge.cmpge(extent).any());
    // Equality is every lane, and so is the order where the lanes agree.
    assert!(vec3(1.0, 2.0, 3.0) == vec3(1.0, 2.0, 3.0));
    assert!(vec3(1.0, 2.0, 3.0) != vec3(1.0, 2.0, 4.0));
    use core::cmp::Ordering::*;
    assert_eq!(vec2(1, 2).partial_cmp(&vec2(1, 2)), Some(Equal));
    assert_eq!(vec2(1, 2).partial_cmp(&vec2(3, 4)), Some(Less));
    assert_eq!(vec2(5, 6).partial_cmp(&vec2(3, 4)), Some(Greater));
    assert_eq!(vec2(1, 6).partial_cmp(&vec2(3, 4)), None);
    // A NaN lane is neither, as a NaN is.
    let nan = vec2(f32::NAN, 0.0);
    assert!(!(nan < Vec2::ONE) && !(nan >= Vec2::ONE) && nan != nan);
}

#[test]
fn cast_converts_as_as_does() {
    // A float saturates into an integer, and NaN becomes zero.
    let v = vec4(-1.5, 2.9, 1e20, f32::NAN);
    assert_eq!(v.cast::<i32>(), vec4(-1, 2, i32::MAX, 0));
    assert_eq!(v.cast::<u32>(), vec4(0, 2, u32::MAX, 0));
    // Integers reinterpret each other's bits.
    assert_eq!(vec2(-1, 7).cast::<u32>(), vec2(u32::MAX, 7));
    // Into `bool`, anything but zero is `true`, NaN included.
    assert_eq!(
        vec4(0.0, -0.0, 0.5, f32::NAN).cast::<bool>(),
        vec4(false, false, true, true)
    );
    assert_eq!(vec2(true, false).cast::<f32>(), vec2(1.0, 0.0));
    assert_eq!(Vec2::<f32>::from(vec2(3u32, 4)), vec2(3.0, 4.0));
}

#[test]
fn the_math_methods_are_rusts() {
    let v = vec2(-1.25, 2.5);
    // Toward zero, as `f32::fract` is; the free function is the GPU's.
    assert_eq!(v.fract(), vec2(-0.25, 0.5));
    assert_eq!(fract(v), vec2(0.75, 0.5));
    // A half to the even neighbour.
    assert_eq!(vec3(0.5, 1.5, 2.5).round_ties_even(), vec3(0.0, 2.0, 2.0));
    assert_eq!(round(vec3(0.5, 1.5, -2.5)), vec3(0.0, 2.0, -2.0));
    assert_eq!(vec2(-3, 4).abs(), vec2(3, 4));
    assert_eq!(vec3(-3, 0, 4).signum(), vec3(-1, 0, 1));
    assert_eq!(
        vec2(1.0, 5.0).clamp(Vec2::splat(2.0), Vec2::splat(4.0)),
        vec2(2.0, 4.0)
    );
    assert_eq!(vec2::<i32>(1, 5).min(vec2(3, 3)), vec2(1, 3));
    assert_eq!(vec3::<i32>(1, 2, 3).dot(vec3(4, 5, 6)), 32);
    assert_eq!(vec4::<u32>(1, 2, 3, 4).element_sum(), 10);
    assert_eq!(vec2(3.0, 4.0).length(), 5.0);
    assert_eq!(vec2(3.0, 4.0).normalize(), vec2(0.6, 0.8));
    assert_eq!(vec2(1.0, 1.0).distance(vec2(4.0, 5.0)), 5.0);
    assert_eq!(
        vec2(0.0, 10.0).lerp(vec2(10.0, 20.0), 0.25),
        vec2(2.5, 12.5)
    );
    assert_eq!(vec2(1.0, 4.0).powf(0.5), vec2(1.0, 2.0));
    assert_eq!(
        vec2(2.0, 3.0).mul_add(vec2(4.0, 5.0), vec2(1.0, 1.0)),
        vec2(9.0, 16.0)
    );
    assert_eq!(
        vec3(1.0, 0.0, 0.0).cross(vec3(0.0, 1.0, 0.0)),
        vec3(0.0, 0.0, 1.0)
    );
    let down = vec3(1.0, -1.0, 0.0);
    let up = vec3(0.0, 1.0, 0.0);
    assert_eq!(down.reflect(up), vec3(1.0, 1.0, 0.0));
    assert_eq!(reflect(down, up), vec3(1.0, 1.0, 0.0));
    // Straight through at a ratio of one; nothing past the critical angle.
    assert_eq!(vec3(0.0, -1.0, 0.0).refract(up, 1.0), vec3(0.0, -1.0, 0.0));
    let grazing = vec3(1.0, -0.1, 0.0).normalize();
    assert_eq!(grazing.refract(up, 1.5), Vec3::ZERO);
    assert_eq!(refract(grazing, up, 1.5), Vec3::ZERO);
}

#[test]
fn the_free_functions_are_the_gpus() {
    // Zero's sign is zero, and the most negative `i32` has no magnitude.
    assert_eq!(sign(vec3(-2.0, 0.0, 3.0)), vec3(-1.0, 0.0, 1.0));
    assert_eq!(abs(i32::MIN), i32::MIN);
    // `clamp` is `min(max(..))`, which does not mind its bounds out of order.
    assert_eq!(clamp(5, 3, 1), 1);
    assert_eq!(
        clamp(vec2(-1.0, 2.0), Vec2::ZERO, Vec2::ONE),
        vec2(0.0, 1.0)
    );
    assert_eq!(saturate(vec2(-1.0, 0.5)), vec2(0.0, 0.5));
    assert_eq!(step(vec2(0.5, 0.5), vec2(0.25, 0.5)), vec2(0.0, 1.0));
    assert_eq!(smoothstep(0.0, 1.0, 0.5), 0.5);
    assert_eq!(smoothstep(0.0, 1.0, 2.0), 1.0);
    // A vector blended by one factor, or by one per lane.
    assert_eq!(
        mix(Vec3::ZERO, vec3(4.0, 8.0, 12.0), 0.25),
        vec3(1.0, 2.0, 3.0)
    );
    assert_eq!(
        mix(Vec2::ZERO, vec2(4.0, 8.0), vec2(0.5, 0.25)),
        vec2(2.0, 2.0)
    );
    assert_eq!(mix(0.0, 4.0, 0.25), 1.0);
    assert_eq!(inverse_sqrt(4.0), 0.5);
    assert_eq!(fma(2.0, 3.0, 1.0), 7.0);
    assert_eq!(dot(vec2(1, 2), vec2(3, 4)), 11);
    assert_eq!(length(vec2(3.0, 4.0)), 5.0);
    assert_eq!(normalize(vec2(0.0, 2.0)), vec2(0.0, 1.0));
    assert_eq!(faceforward(up(), vec3(0.0, -1.0, 0.0), up()), up());
    assert_eq!(faceforward(up(), up(), up()), -up());
}

fn up() -> Vec3 {
    vec3(0.0, 1.0, 0.0)
}

#[test]
fn select_takes_whole_values_or_lanes() {
    assert_eq!(select(1, 2, true), 2);
    assert_eq!(select(1, 2, false), 1);
    assert_eq!(select(Vec2::<f32>::ZERO, Vec2::ONE, true), Vec2::ONE);
    assert_eq!(
        select(vec3(1, 2, 3), vec3(4, 5, 6), vec3(true, false, true)),
        vec3(4, 2, 6)
    );
    assert!(all(vec2(true, true)) && !all(vec2(true, false)));
    assert!(any(vec2(false, true)) && !any(false));
}

#[test]
fn the_bit_builtins_count_as_the_gpu_does() {
    assert_eq!(count_one_bits(vec2::<u32>(0b1011, u32::MAX)), vec2(3, 32));
    assert_eq!(reverse_bits(1u32), 0x8000_0000);
    assert_eq!(count_leading_zeros(1u32), 31);
    assert_eq!(count_trailing_zeros(8u32), 3);
    // All ones where there is no bit to find.
    assert_eq!(
        first_leading_bit(vec3::<u32>(0, 1, 0x80)),
        vec3(u32::MAX, 0, 7)
    );
    assert_eq!(first_trailing_bit(vec2::<u32>(0, 0x80)), vec2(u32::MAX, 7));
    // A signed one looks for the highest bit unlike the sign bit.
    assert_eq!(first_leading_bit(vec4(0, -1, 1, -2)), vec4(-1, -1, 0, 0));
    assert_eq!(first_leading_bit(-256), 7);
    assert_eq!(first_trailing_bit(-256), 8);
}

#[test]
fn matrices_transform_as_the_gpu_does() {
    // Columns first: this is the matrix with rows (1 2) and (3 4).
    let m = mat2(vec2(1.0, 3.0), vec2(2.0, 4.0));
    assert_eq!(m * vec2(1.0, 1.0), vec2(3.0, 7.0));
    assert_eq!(vec2(1.0, 1.0) * m, vec2(4.0, 6.0));
    assert_eq!(m * m, mat2(vec2(7.0, 15.0), vec2(10.0, 22.0)));
    assert_eq!(transpose(m), mat2(vec2(1.0, 2.0), vec2(3.0, 4.0)));
    assert_eq!(determinant(m), -2.0);
    assert_eq!(m + m, m * 2.0);
    assert_eq!(2.0 * m - m, m);
    assert_eq!(Mat2::from_cols_array(&[1.0, 3.0, 2.0, 4.0]), m);

    // A 3x4 is three columns of four; its transpose four columns of three.
    let tall = mat3x4(
        vec4(1.0, 2.0, 3.0, 4.0),
        vec4(5.0, 6.0, 7.0, 8.0),
        vec4(9.0, 10.0, 11.0, 12.0),
    );
    assert_eq!(tall * vec3(1.0, 0.0, 0.0), vec4(1.0, 2.0, 3.0, 4.0));
    assert_eq!(vec4(1.0, 1.0, 1.0, 1.0) * tall, vec3(10.0, 26.0, 42.0));
    let wide = transpose(tall);
    assert_eq!(wide.x_axis, vec3(1.0, 5.0, 9.0));
    assert_eq!(wide.w_axis, vec3(4.0, 8.0, 12.0));
    // Each lane the dot product of two of `tall`'s columns.
    assert_eq!(
        wide * tall,
        mat3(
            vec3(30.0, 70.0, 110.0),
            vec3(70.0, 174.0, 278.0),
            vec3(110.0, 278.0, 446.0),
        )
    );

    let scale = mat3(
        vec3(2.0, 0.0, 0.0),
        vec3(0.0, 3.0, 0.0),
        vec3(0.0, 0.0, 4.0),
    );
    assert_eq!(determinant(scale), 24.0);
    let shear = mat4(
        vec4(1.0, 0.0, 0.0, 0.0),
        vec4(5.0, 2.0, 0.0, 0.0),
        vec4(0.0, 7.0, 3.0, 0.0),
        vec4(9.0, 0.0, 1.0, 4.0),
    );
    assert_eq!(determinant(shear), 24.0);
    assert_eq!(shear * Mat4::from(IDENTITY4), shear);
}

const IDENTITY4: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

#[test]
fn packing_rounds_and_clamps_as_the_gpu_does() {
    assert_eq!(pack4x8unorm(vec4(1.0, 0.0, 0.5, 2.0)), 0xFF80_00FF);
    assert_eq!(pack4x8snorm(vec4(1.0, -1.0, 0.0, 0.5)), 0x4000_817F);
    assert_eq!(
        unpack4x8unorm(0xFF80_00FF),
        vec4(1.0, 0.0, 128.0 / 255.0, 1.0)
    );
    assert_eq!(
        unpack4x8snorm(0x4000_817F),
        vec4(1.0, -1.0, 0.0, 64.0 / 127.0)
    );
    // -128 is past -127, which is -1 already.
    assert_eq!(unpack4x8snorm(0x80).x, -1.0);
    assert_eq!(pack2x16unorm(vec2(1.0, 0.5)), 0x8000_FFFF);
    assert_eq!(pack2x16snorm(vec2(-1.0, 1.0)), 0x7FFF_8001);
    assert_eq!(unpack2x16snorm(0x7FFF_8001), vec2(-1.0, 1.0));
    assert_eq!(unpack2x16unorm(0x0000_FFFF), vec2(1.0, 0.0));
}

#[test]
fn packed_bytes_multiply_and_pack_as_the_gpu_does() {
    // The first lane is the lowest byte.
    assert_eq!(dot4_u8_packed(0x0403_0201, 0x0807_0605), 5 + 12 + 21 + 32);
    assert_eq!(dot4_u8_packed(u32::MAX, u32::MAX), 4 * 255 * 255);
    // 0xFF is -1 and 0x80 is -128 to the signed one.
    assert_eq!(dot4_i8_packed(0x0000_80FF, 0x0000_0102), -2 - 128);
    assert_eq!(dot4_i8_packed(0x8080_8080, 0x8080_8080), 4 * 128 * 128);
    assert_eq!(pack4x_i8(vec4(-1, 2, -128, 127)), 0x7F80_02FF);
    assert_eq!(unpack4x_i8(0x7F80_02FF), vec4(-1, 2, -128, 127));
    // Each lane keeps its lowest eight bits, or is clamped to a byte first.
    assert_eq!(pack4x_i8(vec4(256 + 1, 0, 0, 0)), 0x0000_0001);
    assert_eq!(pack4x_i8_clamp(vec4(-200, 200, 0, 1)), 0x0100_7F80);
    assert_eq!(pack4x_u8(vec4(256 + 1, 2, 3, 4)), 0x0403_0201);
    assert_eq!(pack4x_u8_clamp(vec4(300, 2, 3, 4)), 0x0403_02FF);
    assert_eq!(unpack4x_u8(0x0403_02FF), vec4(255, 2, 3, 4));
}

#[test]
fn a_subgroup_on_the_cpu_is_the_one_invocation() {
    // A reduction or an inclusive scan of one lane is that lane, and an
    // exclusive scan is the operation's identity.
    assert_eq!(subgroup_add(vec3(1.0, 2.0, 3.0)), vec3(1.0, 2.0, 3.0));
    assert_eq!(subgroup_max(-4i32), -4);
    assert_eq!(subgroup_inclusive_add(7u32), 7);
    assert_eq!(subgroup_exclusive_add(5u32), 0);
    assert_eq!(subgroup_exclusive_mul(vec2(2, 3)), vec2(1, 1));
    assert_eq!(subgroup_xor(0b1010u32), 0b1010);
    assert!(subgroup_all(true) && !subgroup_any(false));
    // The one invocation is bit 0 of the first word.
    assert_eq!(subgroup_ballot(true), vec4(1, 0, 0, 0));
    assert_eq!(subgroup_ballot(false), Vec4::<u32>::ZERO);
    // There is no other lane to read from.
    assert_eq!(subgroup_broadcast(4.0, 0), 4.0);
    assert_eq!(subgroup_shuffle_xor(9u32, 1), 9);
    assert_eq!(quad_swap_diagonal(vec2(1.0, 2.0)), vec2(1.0, 2.0));
    subgroup_barrier();
}

#[cfg(feature = "f16")]
#[test]
fn an_f16_lane_rounds_to_the_nearest_even() {
    let h = f16::from_f32;
    // 2049 is halfway between 2048 and 2050, so it goes to the even one, from
    // a float or an integer alike.
    assert_eq!(
        vec2(2049.0, 2051.0).cast::<f16>(),
        vec2(h(2048.0), h(2052.0))
    );
    assert_eq!(vec2(2049u32, 3).cast::<f16>(), vec2(h(2048.0), h(3.0)));
    // Every `f16` is an `f32` exactly, and an integer saturates, as `as` does.
    // The `f16` nearest 0.1 is 819/8192.
    let v = vec3(h(0.1), f16::MAX, -f16::MAX);
    assert_eq!(v.cast::<f32>(), vec3(819.0 / 8192.0, 65504.0, -65504.0));
    assert_eq!(v.cast::<u32>(), vec3(0, 65504, 0));
    assert_eq!(vec2(true, false).cast::<f16>(), vec2(f16::ONE, f16::ZERO));

    // The arithmetic is lane by lane, and in half precision.
    let a = vec3(f16::ONE, h(2.0), h(-0.5));
    assert_eq!(a * h(2.0) + a, vec3(h(3.0), h(6.0), h(-1.5)));
    assert_eq!(-a, vec3(f16::NEG_ONE, h(-2.0), h(0.5)));
    assert_eq!(f16::MAX + f16::MAX, f16::INFINITY);
    assert_eq!(a.dot(a), h(5.25));
    assert_eq!(
        a.max(Vec3::splat(f16::ZERO)),
        vec3(f16::ONE, h(2.0), f16::ZERO)
    );
    assert_eq!(abs(a), vec3(f16::ONE, h(2.0), h(0.5)));
    assert_eq!(sign(a), vec3(f16::ONE, f16::ONE, f16::NEG_ONE));
    assert_eq!(clamp(h(3.0), f16::ZERO, f16::ONE), f16::ONE);
    assert_eq!(dot(a, a), h(5.25));
    assert!(vec2(f16::NAN, f16::ONE).cmplt(Vec2::ONE) == vec2(false, false));
    assert_eq!(subgroup_add(a), a);
}

#[test]
fn half_floats_round_to_the_nearest_even() {
    assert_eq!(pack2x16float(vec2(1.0, -2.0)), 0xC000_3C00);
    assert_eq!(pack2x16float(vec2(65504.0, 0.0)), 0x7BFF);
    // Halfway between the largest half and the next power of two, which is
    // odd, so it goes up, past the largest: to infinity.
    assert_eq!(pack2x16float(vec2(65520.0, 0.0)), 0x7C00);
    let tiny = 2f32.powi(-24);
    assert_eq!(pack2x16float(vec2(tiny, 0.5 * tiny)), 0x0000_0001);
    assert_eq!(pack2x16float(vec2(1.5 * tiny, 2.5 * tiny)), 0x0002_0002);
    assert_eq!(pack2x16float(vec2(f32::INFINITY, -0.0)), 0x8000_7C00);
    assert!(unpack2x16float(pack2x16float(vec2(f32::NAN, 0.0)))
        .x
        .is_nan());
    // Every half that is a number comes back as itself.
    for bits in 0..=u16::MAX {
        let exponent = (bits >> 10) & 0x1F;
        let mantissa = bits & 0x3FF;
        if exponent == 0x1F && mantissa != 0 {
            continue;
        }
        let packed = u32::from(bits) | u32::from(bits) << 16;
        assert_eq!(
            pack2x16float(unpack2x16float(packed)),
            packed,
            "{bits:#06x}"
        );
    }
}

#[test]
fn bitcast_keeps_the_bits() {
    assert_eq!(bitcast::<u32>(1.0), 0x3F80_0000);
    assert_eq!(bitcast::<f32>(0x4000_0000u32), 2.0);
    assert_eq!(bitcast::<i32>(u32::MAX), -1);
    assert_eq!(
        bitcast::<Vec2<u32>>(vec2(1.0, -0.0)),
        vec2(0x3F80_0000, 0x8000_0000)
    );
    assert_eq!(1.0f32.to_bits(), bitcast::<u32>(1.0f32));
}
