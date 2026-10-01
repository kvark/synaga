//! What a host sees when its layout of a shared struct is not the GPU's.
//!
//! This is the shape of the file the build script writes, with the numbers the
//! GPU lays `Globals` out at. Naga puts a `Vec3<f32>` at 16 bytes when the
//! buffer rounds each element up to four lanes, where Rust leaves it at 12, so
//! a `Mat4` followed by one puts the pair 4 bytes apart rather than flush — and
//! a real crate with a `Vec3` in its uniform gets exactly this diagnostic.
//!
//! `check_layout!` includes the generated file inside the module that lists the
//! shader modules, which is why these paths are written `self::`. That module is
//! spelled out here as `mod shaders`, which is what the include makes of it.


mod shaders {
    pub mod common {
        use synaga_shader::*;

        #[repr(C)]
        #[derive(Shared)]
        pub struct Globals {
            pub mvp_transform: Mat4,
            pub sprite_size: Vec3<f32>,
        }
    }
}

fn main() {
    // The generated file's body, with the GPU's numbers. Written out rather than
    // `check_layout!`'d because this is a crate of its own: a test of the
    // diagnostic, not of the include.
    #[allow(unused_qualifications)]
    const _: () = {
        assert!(::core::mem::size_of::<self::shaders::common::Globals>() == 80, "`common::Globals` is 80 bytes on the GPU");
        assert!(::core::mem::align_of::<self::shaders::common::Globals>() == 4, "`common::Globals` is aligned 4 on the GPU");
        assert!(::core::mem::offset_of!(self::shaders::common::Globals, sprite_size) == 64, "`common::Globals::sprite_size` is at byte 64 on the GPU");
    };
}
