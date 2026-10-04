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
fn a_crate_that_knows_only_mint_finds_the_mint_type() {
    // What a crate generic over math libraries does with a field's type.
    fn to_mint<T: mint::IntoMint>(value: T) -> T::MintType {
        value.into()
    }
    let v: mint::Vector3<f32> = to_mint(vec3(1.0, 2.0, 3.0));
    assert_eq!(
        v,
        mint::Vector3 {
            x: 1.0,
            y: 2.0,
            z: 3.0
        }
    );
    let u: mint::Vector2<u32> = to_mint(vec2(4u32, 5));
    assert_eq!(u, mint::Vector2 { x: 4, y: 5 });
    let m: mint::ColumnMatrix2<f32> = to_mint(mat2(vec2(1.0, 0.0), vec2(0.0, 1.0)));
    assert_eq!(m.y, mint::Vector2 { x: 0.0, y: 1.0 });
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
fn one_derive_makes_a_struct_shared() {
    #[repr(C)]
    #[derive(Debug, PartialEq, Shared)]
    struct Params {
        tint: Vec4,
        size: Vec2<u32>,
        scale: f32,
        count: u32,
    }
    // `Default` is all zeroes, which is the GPU's too.
    let zero = Params::default();
    assert_eq!(zero, bytemuck::Zeroable::zeroed());
    assert_eq!(zero.tint, Vec4::ZERO);
    let params = Params {
        size: vec2(3, 4),
        ..zero
    };
    let copy = params;
    assert_eq!(bytemuck::bytes_of(&copy)[16..20], 3u32.to_ne_bytes());
    fn shared<T: Shared>() {}
    shared::<Params>();
}

#[test]
fn a_shared_struct_can_hold_an_enum() {
    #[repr(u32)]
    #[derive(Clone, Copy, Debug, PartialEq, bytemuck::NoUninit, bytemuck::Zeroable)]
    enum Mode {
        Final,
        Depth,
    }
    #[repr(C)]
    #[derive(Debug, PartialEq, Shared)]
    struct Params {
        mode: Mode,
        count: u32,
    }
    // The zero variant, as on the GPU.
    assert_eq!(Params::default().mode, Mode::Final);
    let params = Params {
        mode: Mode::Depth,
        count: 3,
    };
    let bytes = bytemuck::bytes_of(&params);
    assert_eq!(bytes[..4], 1u32.to_ne_bytes());
    assert_eq!(bytes[4..], 3u32.to_ne_bytes());
    fn shared<T: Shared>() {}
    shared::<Params>();
}

#[test]
fn a_shared_struct_can_hold_an_array_of_them() {
    #[repr(u32)]
    #[derive(Clone, Copy, Debug, PartialEq, bytemuck::NoUninit, bytemuck::Zeroable)]
    enum Kind {
        Point,
        Spot,
    }
    #[repr(C)]
    #[derive(Shared)]
    struct Light {
        color: Vec3,
        kind: Kind,
    }
    // `bytemuck` makes `[Light; 2]` `NoUninit` only for a `Pod` `Light`, which
    // one holding an enum is not, so the derive checks the element.
    #[repr(C)]
    #[derive(Shared)]
    struct Lights {
        lights: [Light; 2],
        count: u32,
        _pad: Vec3<u32>,
    }
    let mut lights = Lights::default();
    lights.lights[1] = Light {
        color: Vec3::ONE,
        kind: Kind::Spot,
    };
    lights.count = 2;
    assert_eq!(lights.lights[0].kind, Kind::Point);
    let bytes = bytemuck::bytes_of(&lights);
    assert_eq!(bytes.len(), 2 * 16 + 16);
    assert_eq!(bytes[28..32], 1u32.to_ne_bytes());
    assert_eq!(bytes[32..36], 2u32.to_ne_bytes());
}

#[test]
fn a_shared_struct_reads_back_with_its_enums_checked() {
    #[repr(u32)]
    #[derive(
        Clone,
        Copy,
        Debug,
        PartialEq,
        bytemuck::NoUninit,
        bytemuck::Zeroable,
        bytemuck::CheckedBitPattern,
    )]
    enum Mode {
        Final,
        Depth,
    }
    #[repr(C)]
    #[derive(Debug, PartialEq, Shared, bytemuck::CheckedBitPattern)]
    struct Params {
        tint: Vec4,
        mode: Mode,
        count: u32,
        _pad: Vec2<u32>,
    }
    let written = [
        Params {
            tint: Vec4::ONE,
            mode: Mode::Depth,
            count: 3,
            _pad: Vec2::ZERO,
        },
        Params::default(),
    ];
    // What a buffer holds, as a host maps it back.
    let mut words: Vec<u32> = bytemuck::cast_slice(&written).to_vec();
    let read: &[Params] = bytemuck::checked::cast_slice(&words);
    assert_eq!(read, &written);
    // Zeroes are the zero variant.
    assert_eq!(read[1].mode, Mode::Final);
    // A `u32` that is none of the variants is refused, not read.
    words[4] = 7;
    assert!(matches!(
        bytemuck::checked::try_cast_slice::<u32, Params>(&words),
        Err(bytemuck::checked::CheckedCastError::InvalidBitPattern)
    ));
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
