//! Textures, samplers and ray queries, and what a shader does with them.
//!
//! Each operation is a method on the types it applies to, so `rustc` turns
//! away what WGSL would: sampling an integer texture, storing to a sampled
//! one, comparing depth without a comparison sampler, reading a write-only
//! storage texture. The names are WGSL's builtins in snake case, minus the
//! `texture` prefix: `textureSampleLevel(t, s, uv, 0.0)` is
//! `t.sample_level(&s, uv, 0.0)`.
//!
//! Each type is a marker with no data in it — the resource lives on the GPU —
//! which is why they are `Sync` by assertion rather than by their contents.

use core::marker::PhantomData;

use crate::resource::{Bindable, Resource};
use crate::unimplemented_on_cpu;
use crate::vector::{Vec2, Vec3, Vec4};

pub use access::*;
pub use format::*;

/// What a sampled texture's texels are made of: `f32`, `i32` or `u32`.
pub trait Component {
    /// Four of them, which is what a load or a sample produces.
    type Texel;
}
impl Component for f32 {
    type Texel = Vec4;
}
impl Component for i32 {
    type Texel = Vec4<i32>;
}
impl Component for u32 {
    type Texel = Vec4<u32>;
}

/// An integer WGSL takes as a mip level, array layer, or sample index.
pub trait TexelIndex: Copy {}
impl TexelIndex for i32 {}
impl TexelIndex for u32 {}

/// The coordinate of one texel in a one-dimensional texture.
pub trait TexelCoord1: Copy {}
impl TexelCoord1 for i32 {}
impl TexelCoord1 for u32 {}

/// The coordinate of one texel in a two-dimensional texture.
pub trait TexelCoord2: Copy {}
impl TexelCoord2 for Vec2<i32> {}
impl TexelCoord2 for Vec2<u32> {}

/// The coordinate of one texel in a three-dimensional texture.
pub trait TexelCoord3: Copy {}
impl TexelCoord3 for Vec3<i32> {}
impl TexelCoord3 for Vec3<u32> {}

/// Storage texture formats, as type arguments.
pub mod format {
    use crate::vector::Vec4;

    /// A storage texture format, and the texel a shader sees for it.
    pub trait Format {
        type Texel;
    }

    macro_rules! formats {
        ($texel:ty: $($name:ident),* $(,)?) => {
            $(
                /// A storage texture format.
                #[derive(Debug)]
                pub struct $name;
                impl Format for $name {
                    type Texel = $texel;
                }
            )*
        };
    }
    formats!(Vec4:
        R8Unorm, R8Snorm, R16Float, Rg8Unorm, Rg8Snorm, R32Float, Rg16Float,
        Rgba8Unorm, Rgba8Snorm, Rgb10a2Unorm, Rg11b10Float, Rg32Float,
        Rgba16Float, Rgba32Float,
    );
    formats!(Vec4<u32>:
        R8Uint, R16Uint, Rg8Uint, R32Uint, Rg16Uint, Rgba8Uint, Rg32Uint,
        Rgba16Uint, Rgba32Uint,
    );
    formats!(Vec4<i32>:
        R8Sint, R16Sint, Rg8Sint, R32Sint, Rg16Sint, Rgba8Sint, Rg32Sint,
        Rgba16Sint, Rgba32Sint,
    );
}

/// Storage texture access modes, as type arguments.
pub mod access {
    /// Read-only.
    #[derive(Debug)]
    pub struct Read;
    /// Write-only.
    #[derive(Debug)]
    pub struct Write;
    /// Both.
    #[derive(Debug)]
    pub struct ReadWrite;

    /// An access mode that allows `load`.
    pub trait Readable {}
    impl Readable for Read {}
    impl Readable for ReadWrite {}

    /// An access mode that allows `store`.
    pub trait Writable {}
    impl Writable for Write {}
    impl Writable for ReadWrite {}
}

macro_rules! marker {
    ($(#[$doc:meta])* $name:ident<$($param:ident),*>) => {
        $(#[$doc])*
        pub struct $name<$($param),*>(PhantomData<($($param,)*)>);
        impl<$($param),*> Resource for $name<$($param),*> {
            const BINDING: Self = $name(PhantomData);
        }
        impl<$($param),*> Bindable for $name<$($param),*> {}
        unsafe impl<$($param),*> Sync for $name<$($param),*> {}
        unsafe impl<$($param),*> Send for $name<$($param),*> {}
    };
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        pub struct $name;
        impl Resource for $name {
            const BINDING: Self = $name;
        }
        impl Bindable for $name {}
    };
}

/// A sampled texture.
macro_rules! sampled {
    ($($name:ident),* $(,)?) => {
        $(
            marker!(
                /// A sampled texture, of `f32`, `i32` or `u32` texels.
                $name<T>
            );
        )*
    };
}
sampled!(
    Texture1D,
    Texture2D,
    Texture2DArray,
    Texture3D,
    TextureCube,
    TextureCubeArray,
    TextureMultisampled2D,
);

/// A depth texture, whose texels are single `f32` depths.
macro_rules! depth {
    ($($name:ident),* $(,)?) => {
        $(
            marker!(
                /// A depth texture.
                $name
            );
        )*
    };
}
depth!(
    TextureDepth2D,
    TextureDepth2DArray,
    TextureDepthCube,
    TextureDepthCubeArray,
    TextureDepthMultisampled2D,
);

/// A storage texture, parameterised by format and access.
macro_rules! storage {
    ($($name:ident),* $(,)?) => {
        $(
            marker!(
                /// A storage texture: `TextureStorage2D<Rgba8Unorm, Write>`.
                $name<F, A>
            );
        )*
    };
}
storage!(
    TextureStorage1D,
    TextureStorage2D,
    TextureStorage2DArray,
    TextureStorage3D,
);

/// Samples a texture. `SamplerComparison` is its own type, so a depth
/// comparison cannot be handed an ordinary sampler.
#[derive(Clone, Copy)]
pub struct Sampler;
impl Resource for Sampler {
    const BINDING: Self = Sampler;
}
impl Bindable for Sampler {}

/// Samples a depth texture, comparing against a reference.
#[derive(Clone, Copy)]
pub struct SamplerComparison;
impl Resource for SamplerComparison {
    const BINDING: Self = SamplerComparison;
}
impl Bindable for SamplerComparison {}

/// What a ray query traces against.
#[derive(Clone, Copy)]
pub struct AccelerationStructure;
impl Resource for AccelerationStructure {
    const BINDING: Self = AccelerationStructure;
}
impl Bindable for AccelerationStructure {}

// The size queries. `dimensions` is the base level; `level_dimensions` is
// WGSL's `textureDimensions(t, level)`, which Rust cannot overload by arity.
macro_rules! sizes {
    ($size:ty; $([$($g:ident),*] $ty:ty),* $(,)?) => {
        $(
            impl<$($g),*> $ty {
                /// Size of the base mip level: `textureDimensions`.
                #[inline]
                pub fn dimensions(&self) -> $size {
                    unimplemented_on_cpu()
                }
            }
        )*
    };
}
macro_rules! mipped {
    ($size:ty; $([$($g:ident),*] $ty:ty),* $(,)?) => {
        $(
            impl<$($g),*> $ty {
                /// Size of one mip level: `textureDimensions(t, level)`.
                #[inline]
                pub fn level_dimensions(&self, level: impl TexelIndex) -> $size {
                    unimplemented_on_cpu()
                }

                /// Number of mip levels: `textureNumLevels`.
                #[inline]
                pub fn num_levels(&self) -> u32 {
                    unimplemented_on_cpu()
                }
            }
        )*
    };
}
macro_rules! layered {
    ($([$($g:ident),*] $ty:ty),* $(,)?) => {
        $(
            impl<$($g),*> $ty {
                /// Number of array layers: `textureNumLayers`.
                #[inline]
                pub fn num_layers(&self) -> u32 {
                    unimplemented_on_cpu()
                }
            }
        )*
    };
}

sizes!(u32; [T] Texture1D<T>, [F, A] TextureStorage1D<F, A>);
sizes!(Vec2<u32>;
    [T] Texture2D<T>,
    [T] Texture2DArray<T>,
    [T] TextureCube<T>,
    [T] TextureCubeArray<T>,
    [T] TextureMultisampled2D<T>,
    [] TextureDepth2D,
    [] TextureDepth2DArray,
    [] TextureDepthCube,
    [] TextureDepthCubeArray,
    [] TextureDepthMultisampled2D,
    [F, A] TextureStorage2D<F, A>,
    [F, A] TextureStorage2DArray<F, A>,
);
sizes!(Vec3<u32>; [T] Texture3D<T>, [F, A] TextureStorage3D<F, A>);

mipped!(u32; [T] Texture1D<T>);
mipped!(Vec2<u32>;
    [T] Texture2D<T>,
    [T] Texture2DArray<T>,
    [T] TextureCube<T>,
    [T] TextureCubeArray<T>,
    [] TextureDepth2D,
    [] TextureDepth2DArray,
    [] TextureDepthCube,
    [] TextureDepthCubeArray,
);
mipped!(Vec3<u32>; [T] Texture3D<T>);

layered!(
    [T] Texture2DArray<T>,
    [T] TextureCubeArray<T>,
    [] TextureDepth2DArray,
    [] TextureDepthCubeArray,
    [F, A] TextureStorage2DArray<F, A>,
);

impl<T> TextureMultisampled2D<T> {
    /// Samples per texel: `textureNumSamples`.
    #[inline]
    pub fn num_samples(&self) -> u32 {
        unimplemented_on_cpu()
    }
}

impl TextureDepthMultisampled2D {
    /// Samples per texel: `textureNumSamples`.
    #[inline]
    pub fn num_samples(&self) -> u32 {
        unimplemented_on_cpu()
    }
}

// Loads: one texel, no filtering. Every sampled texture but a cube has them.
impl<T: Component> Texture1D<T> {
    /// Read one texel at a mip level: `textureLoad`.
    #[inline]
    pub fn load(&self, coord: impl TexelCoord1, level: impl TexelIndex) -> T::Texel {
        unimplemented_on_cpu()
    }
}
impl<T: Component> Texture2D<T> {
    /// Read one texel at a mip level: `textureLoad`.
    #[inline]
    pub fn load(&self, coord: impl TexelCoord2, level: impl TexelIndex) -> T::Texel {
        unimplemented_on_cpu()
    }
}
impl<T: Component> Texture2DArray<T> {
    /// Read one texel of a layer at a mip level: `textureLoad`.
    #[inline]
    pub fn load(
        &self,
        coord: impl TexelCoord2,
        layer: impl TexelIndex,
        level: impl TexelIndex,
    ) -> T::Texel {
        unimplemented_on_cpu()
    }
}
impl<T: Component> Texture3D<T> {
    /// Read one texel at a mip level: `textureLoad`.
    #[inline]
    pub fn load(&self, coord: impl TexelCoord3, level: impl TexelIndex) -> T::Texel {
        unimplemented_on_cpu()
    }
}
impl<T: Component> TextureMultisampled2D<T> {
    /// Read one sample of one texel: `textureLoad`.
    #[inline]
    pub fn load(&self, coord: impl TexelCoord2, sample: impl TexelIndex) -> T::Texel {
        unimplemented_on_cpu()
    }
}
impl TextureDepth2D {
    /// Read one depth at a mip level: `textureLoad`.
    #[inline]
    pub fn load(&self, coord: impl TexelCoord2, level: impl TexelIndex) -> f32 {
        unimplemented_on_cpu()
    }
}
impl TextureDepth2DArray {
    /// Read one depth of a layer at a mip level: `textureLoad`.
    #[inline]
    pub fn load(
        &self,
        coord: impl TexelCoord2,
        layer: impl TexelIndex,
        level: impl TexelIndex,
    ) -> f32 {
        unimplemented_on_cpu()
    }
}
impl TextureDepthMultisampled2D {
    /// Read one sample of one depth: `textureLoad`.
    #[inline]
    pub fn load(&self, coord: impl TexelCoord2, sample: impl TexelIndex) -> f32 {
        unimplemented_on_cpu()
    }
}

// Sampling: filtered reads through a sampler. Only `f32` textures filter.
macro_rules! sample {
    ($ty:ty, $coord:ty, $grad:ty $(, $layer:ident)?) => {
        impl $ty {
            /// Filtered read at the level the hardware picks: `textureSample`.
            /// Fragment shaders only, since the level comes from derivatives.
            #[inline]
            pub fn sample(&self, with: &Sampler, coord: $coord $(, $layer: impl TexelIndex)?) -> Vec4 {
                unimplemented_on_cpu()
            }

            /// Filtered read at an explicit level: `textureSampleLevel`.
            #[inline]
            pub fn sample_level(
                &self,
                with: &Sampler,
                coord: $coord,
                $($layer: impl TexelIndex,)?
                level: f32,
            ) -> Vec4 {
                unimplemented_on_cpu()
            }

            /// Filtered read with the picked level shifted: `textureSampleBias`.
            #[inline]
            pub fn sample_bias(
                &self,
                with: &Sampler,
                coord: $coord,
                $($layer: impl TexelIndex,)?
                bias: f32,
            ) -> Vec4 {
                unimplemented_on_cpu()
            }

            /// Filtered read with explicit derivatives: `textureSampleGrad`.
            #[inline]
            pub fn sample_grad(
                &self,
                with: &Sampler,
                coord: $coord,
                $($layer: impl TexelIndex,)?
                ddx: $grad,
                ddy: $grad,
            ) -> Vec4 {
                unimplemented_on_cpu()
            }
        }
    };
}
sample!(Texture2D<f32>, Vec2, Vec2);
sample!(Texture2DArray<f32>, Vec2, Vec2, layer);
sample!(Texture3D<f32>, Vec3, Vec3);
sample!(TextureCube<f32>, Vec3, Vec3);
sample!(TextureCubeArray<f32>, Vec3, Vec3, layer);

impl Texture1D<f32> {
    /// Filtered read: `textureSample`.
    #[inline]
    pub fn sample(&self, with: &Sampler, coord: f32) -> Vec4 {
        unimplemented_on_cpu()
    }
}

// Depth sampling produces one `f32`, and a depth texture's level is an
// integer.
macro_rules! sample_depth {
    ($ty:ty, $coord:ty $(, $layer:ident)?) => {
        impl $ty {
            /// Read at the level the hardware picks: `textureSample`.
            #[inline]
            pub fn sample(&self, with: &Sampler, coord: $coord $(, $layer: impl TexelIndex)?) -> f32 {
                unimplemented_on_cpu()
            }

            /// Read at an explicit level: `textureSampleLevel`.
            #[inline]
            pub fn sample_level(
                &self,
                with: &Sampler,
                coord: $coord,
                $($layer: impl TexelIndex,)?
                level: impl TexelIndex,
            ) -> f32 {
                unimplemented_on_cpu()
            }

            /// Compare against `depth_ref` and filter the results:
            /// `textureSampleCompare`.
            #[inline]
            pub fn sample_compare(
                &self,
                with: &SamplerComparison,
                coord: $coord,
                $($layer: impl TexelIndex,)?
                depth_ref: f32,
            ) -> f32 {
                unimplemented_on_cpu()
            }

            /// The same, at the base level: `textureSampleCompareLevel`.
            #[inline]
            pub fn sample_compare_level(
                &self,
                with: &SamplerComparison,
                coord: $coord,
                $($layer: impl TexelIndex,)?
                depth_ref: f32,
            ) -> f32 {
                unimplemented_on_cpu()
            }
        }
    };
}
sample_depth!(TextureDepth2D, Vec2);
sample_depth!(TextureDepth2DArray, Vec2, layer);
sample_depth!(TextureDepthCube, Vec3);
sample_depth!(TextureDepthCubeArray, Vec3, layer);

// Storage textures: `load` needs an access mode that reads, `store` one that
// writes, and the texel is whatever the format says.
macro_rules! storage_access {
    ($ty:ident, $coord:ident $(, $layer:ident)?) => {
        impl<F: Format, A: Readable> $ty<F, A> {
            /// Read one texel: `textureLoad`.
            #[inline]
            pub fn load(&self, coord: impl $coord $(, $layer: impl TexelIndex)?) -> F::Texel {
                unimplemented_on_cpu()
            }
        }
        impl<F: Format, A: Writable> $ty<F, A> {
            /// Write one texel: `textureStore`.
            #[inline]
            pub fn store(&self, coord: impl $coord, $($layer: impl TexelIndex,)? value: F::Texel) {
                unimplemented_on_cpu()
            }
        }
    };
}
storage_access!(TextureStorage1D, TexelCoord1);
storage_access!(TextureStorage2D, TexelCoord2);
storage_access!(TextureStorage2DArray, TexelCoord2, layer);
storage_access!(TextureStorage3D, TexelCoord3);

/// A ray query: a local that traces one ray through an acceleration structure.
///
/// Start one with `RayQuery::default()`, point it at a ray with
/// [`initialize`](RayQuery::initialize), and [`proceed`](RayQuery::proceed)
/// until it says there is nothing more to consider. It cannot be copied,
/// which is WGSL's rule too.
#[derive(Default)]
pub struct RayQuery {
    _local: (),
}

impl RayQuery {
    /// Begin tracing `desc` through `scene`:
    /// `rayQueryInitialize`.
    #[inline]
    pub fn initialize(&mut self, scene: &AccelerationStructure, desc: RayDesc) {
        unimplemented_on_cpu()
    }

    /// Advance to the next candidate; `false` once there is none:
    /// `rayQueryProceed`.
    #[inline]
    pub fn proceed(&mut self) -> bool {
        unimplemented_on_cpu()
    }

    /// Offer a procedural hit at distance `hit_t`:
    /// `rayQueryGenerateIntersection`.
    #[inline]
    pub fn generate_intersection(&mut self, hit_t: f32) {
        unimplemented_on_cpu()
    }

    /// Keep the current triangle candidate as a committed hit:
    /// `rayQueryConfirmIntersection`.
    #[inline]
    pub fn confirm_intersection(&mut self) {
        unimplemented_on_cpu()
    }

    /// Stop considering candidates: `rayQueryTerminate`.
    #[inline]
    pub fn terminate(&mut self) {
        unimplemented_on_cpu()
    }

    /// The closest hit committed so far: `rayQueryGetCommittedIntersection`.
    #[inline]
    pub fn committed_intersection(&self) -> RayIntersection {
        unimplemented_on_cpu()
    }

    /// The candidate under consideration: `rayQueryGetCandidateIntersection`.
    #[inline]
    pub fn candidate_intersection(&self) -> RayIntersection {
        unimplemented_on_cpu()
    }
}

/// What to trace.
#[derive(Clone, Copy, Debug, Default)]
pub struct RayDesc {
    pub flags: RayFlag,
    pub cull_mask: u32,
    pub tmin: f32,
    pub tmax: f32,
    pub origin: Vec3,
    pub dir: Vec3,
}

/// What was found.
#[derive(Clone, Copy, Debug, Default)]
pub struct RayIntersection {
    pub kind: RayQueryIntersection,
    pub t: f32,
    pub instance_custom_data: u32,
    pub instance_index: u32,
    pub sbt_record_offset: u32,
    pub geometry_index: u32,
    pub primitive_index: u32,
    pub barycentrics: Vec2,
    pub front_face: bool,
    pub object_to_world: crate::matrix::Mat4x3,
    pub world_to_object: crate::matrix::Mat4x3,
}

bitflags::bitflags! {
    /// What a ray query skips, or takes as opaque: WGSL's `RAY_FLAG_*`, as a
    /// set. `RayFlag::TERMINATE_ON_FIRST_HIT | RayFlag::CULL_NO_OPAQUE`.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct RayFlag: u32 {
        const FORCE_OPAQUE = 0x1;
        const FORCE_NO_OPAQUE = 0x2;
        const TERMINATE_ON_FIRST_HIT = 0x4;
        const SKIP_CLOSEST_HIT_SHADER = 0x8;
        const CULL_BACK_FACING = 0x10;
        const CULL_FRONT_FACING = 0x20;
        const CULL_OPAQUE = 0x40;
        const CULL_NO_OPAQUE = 0x80;
        const SKIP_TRIANGLES = 0x100;
        const SKIP_AABBS = 0x200;
    }
}

/// What a ray query found, if anything: WGSL's `RAY_QUERY_INTERSECTION_*`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RayQueryIntersection {
    /// Nothing: the ray missed, or nothing was committed yet.
    #[default]
    None = 0,
    /// A triangle.
    Triangle = 1,
    /// A procedural hit, which `generate_intersection` offered.
    Generated = 2,
    /// A box of procedural geometry, whose hit is the shader's to find.
    Aabb = 3,
}
