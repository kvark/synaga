//! What a host uses to fill a struct it shares with a shader. Unlike the rest
//! of this crate, these run on the CPU.
#![cfg(all(feature = "bytemuck", feature = "mint"))]

use synaga_shader::*;

#[test]
fn vectors_convert_from_and_to_arrays_and_mint() {
    let v: Vec3 = [1.0, 2.0, 3.0].into();
    assert_eq!(v, vec3(1.0, 2.0, 3.0));
    assert_eq!(<[f32; 3]>::from(v), [1.0, 2.0, 3.0]);
    let m: Vec4<u32> = mint::Vector4 {
        x: 1,
        y: 2,
        z: 3,
        w: 4,
    }
    .into();
    assert_eq!(m, vec4(1, 2, 3, 4));
    assert_eq!(
        mint::Vector4::from(m),
        mint::Vector4 {
            x: 1,
            y: 2,
            z: 3,
            w: 4
        }
    );
}

#[test]
fn matrices_convert_column_by_column() {
    let cols = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ];
    let m = Mat3x4::from(cols);
    assert_eq!(m.y_axis, vec4(5.0, 6.0, 7.0, 8.0));
    assert_eq!(<[[f32; 4]; 3]>::from(m), cols);
    // mint names it by rows first: four rows, three columns.
    let columns = mint::ColumnMatrix4x3::from(m);
    assert_eq!(
        columns.z,
        mint::Vector4 {
            x: 9.0,
            y: 10.0,
            z: 11.0,
            w: 12.0
        }
    );
    assert_eq!(Mat3x4::from(columns), m);
}

#[test]
fn a_shared_struct_is_its_bytes() {
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
    struct Params {
        transform: Mat4,
        tint: Vec4,
        size: Vec2<u32>,
        _pad: Vec2<u32>,
    }
    let params = Params {
        transform: Mat4::ZERO,
        tint: Vec4::ONE,
        size: vec2(3, 4),
        _pad: Vec2::ZERO,
    };
    let bytes = bytemuck::bytes_of(&params);
    assert_eq!(bytes.len(), 64 + 16 + 16);
    assert_eq!(bytes[80..84], 3u32.to_ne_bytes());
}

#[test]
fn a_three_lane_column_takes_four_as_on_the_gpu() {
    assert_eq!(size_of::<Mat3>(), 48);
    assert_eq!(size_of::<Mat4x3>(), 64);
    let m = mat3(
        vec3(1.0, 2.0, 3.0),
        vec3(4.0, 5.0, 6.0),
        vec3(7.0, 8.0, 9.0),
    );
    let lanes: [f32; 12] = bytemuck::cast(m);
    assert_eq!(
        lanes,
        [1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0, 7.0, 8.0, 9.0, 0.0]
    );
    assert_eq!(
        Mat3::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        m
    );
}

#[test]
fn lanes_and_columns_index_on_the_cpu() {
    let mut v = vec3(1.0, 2.0, 3.0);
    v[1] = 5.0;
    assert_eq!((v[0], v[1], v[2]), (1.0, 5.0, 3.0));
    let mut m = Mat4x3::ZERO;
    m[3] = v;
    assert_eq!(m.w_axis, v);
    assert_eq!(m[3][1], 5.0);
    let q: Vec4 = mint::Quaternion {
        v: [1.0, 2.0, 3.0].into(),
        s: 4.0,
    }
    .into();
    assert_eq!(q, vec4(1.0, 2.0, 3.0, 4.0));
}
