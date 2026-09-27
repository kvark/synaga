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

use crate::resource::Resource;
use crate::unimplemented_on_cpu;
use crate::vector::{vec2, vec2i, vec2u, vec3, vec3i, vec3u, vec4, vec4i, vec4u};

pub use access::*;
pub use format::*;

/// What a sampled texture's texels are made of: `f32`, `i32` or `u32`.
pub trait Component {
    /// Four of them, which is what a load or a sample produces.
    type Texel;
}
impl Component for f32 {
    type Texel = vec4;
}
impl Component for i32 {
    type Texel = vec4i;
}
impl Component for u32 {
    type Texel = vec4u;
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
impl TexelCoord2 for vec2i {}
impl TexelCoord2 for vec2u {}

/// The coordinate of one texel in a three-dimensional texture.
pub trait TexelCoord3: Copy {}
impl TexelCoord3 for vec3i {}
impl TexelCoord3 for vec3u {}

/// Storage texture formats, as type arguments.
pub mod format {
    use crate::vector::{vec4, vec4i, vec4u};

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
    formats!(vec4:
        R8Unorm, R8Snorm, R16Float, Rg8Unorm, Rg8Snorm, R32Float, Rg16Float,
        Rgba8Unorm, Rgba8Snorm, Rgb10a2Unorm, Rg11b10Float, Rg32Float,
        Rgba16Float, Rgba32Float,
    );
    formats!(vec4u:
        R8Uint, R16Uint, Rg8Uint, R32Uint, Rg16Uint, Rgba8Uint, Rg32Uint,
        Rgba16Uint, Rgba32Uint,
    );
    formats!(vec4i:
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
        #[allow(non_camel_case_types)]
        pub struct $name<$($param),*>(PhantomData<($($param,)*)>);
        impl<$($param),*> Resource for $name<$($param),*> {
            const BINDING: Self = $name(PhantomData);
        }
        unsafe impl<$($param),*> Sync for $name<$($param),*> {}
        unsafe impl<$($param),*> Send for $name<$($param),*> {}
    };
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[allow(non_camel_case_types)]
        pub struct $name;
        impl Resource for $name {
            const BINDING: Self = $name;
        }
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
    texture_1d,
    texture_2d,
    texture_2d_array,
    texture_3d,
    texture_cube,
    texture_cube_array,
    texture_multisampled_2d,
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
    texture_depth_2d,
    texture_depth_2d_array,
    texture_depth_cube,
    texture_depth_cube_array,
    texture_depth_multisampled_2d,
);

/// A storage texture, parameterised by format and access.
macro_rules! storage {
    ($($name:ident),* $(,)?) => {
        $(
            marker!(
                /// A storage texture: `texture_storage_2d<Rgba8Unorm, Write>`.
                $name<F, A>
            );
        )*
    };
}
storage!(
    texture_storage_1d,
    texture_storage_2d,
    texture_storage_2d_array,
    texture_storage_3d,
);

/// Samples a texture. `sampler_comparison` is its own type, so a depth
/// comparison cannot be handed an ordinary sampler.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct sampler;
impl Resource for sampler {
    const BINDING: Self = sampler;
}

/// Samples a depth texture, comparing against a reference.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct sampler_comparison;
impl Resource for sampler_comparison {
    const BINDING: Self = sampler_comparison;
}

/// What a ray query traces against.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct acceleration_structure;
impl Resource for acceleration_structure {
    const BINDING: Self = acceleration_structure;
}

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

sizes!(u32; [T] texture_1d<T>, [F, A] texture_storage_1d<F, A>);
sizes!(vec2u;
    [T] texture_2d<T>,
    [T] texture_2d_array<T>,
    [T] texture_cube<T>,
    [T] texture_cube_array<T>,
    [T] texture_multisampled_2d<T>,
    [] texture_depth_2d,
    [] texture_depth_2d_array,
    [] texture_depth_cube,
    [] texture_depth_cube_array,
    [] texture_depth_multisampled_2d,
    [F, A] texture_storage_2d<F, A>,
    [F, A] texture_storage_2d_array<F, A>,
);
sizes!(vec3u; [T] texture_3d<T>, [F, A] texture_storage_3d<F, A>);

mipped!(u32; [T] texture_1d<T>);
mipped!(vec2u;
    [T] texture_2d<T>,
    [T] texture_2d_array<T>,
    [T] texture_cube<T>,
    [T] texture_cube_array<T>,
    [] texture_depth_2d,
    [] texture_depth_2d_array,
    [] texture_depth_cube,
    [] texture_depth_cube_array,
);
mipped!(vec3u; [T] texture_3d<T>);

layered!(
    [T] texture_2d_array<T>,
    [T] texture_cube_array<T>,
    [] texture_depth_2d_array,
    [] texture_depth_cube_array,
    [F, A] texture_storage_2d_array<F, A>,
);

impl<T> texture_multisampled_2d<T> {
    /// Samples per texel: `textureNumSamples`.
    #[inline]
    pub fn num_samples(&self) -> u32 {
        unimplemented_on_cpu()
    }
}

impl texture_depth_multisampled_2d {
    /// Samples per texel: `textureNumSamples`.
    #[inline]
    pub fn num_samples(&self) -> u32 {
        unimplemented_on_cpu()
    }
}

// Loads: one texel, no filtering. Every sampled texture but a cube has them.
impl<T: Component> texture_1d<T> {
    /// Read one texel at a mip level: `textureLoad`.
    #[inline]
    pub fn load(&self, coord: impl TexelCoord1, level: impl TexelIndex) -> T::Texel {
        unimplemented_on_cpu()
    }
}
impl<T: Component> texture_2d<T> {
    /// Read one texel at a mip level: `textureLoad`.
    #[inline]
    pub fn load(&self, coord: impl TexelCoord2, level: impl TexelIndex) -> T::Texel {
        unimplemented_on_cpu()
    }
}
impl<T: Component> texture_2d_array<T> {
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
impl<T: Component> texture_3d<T> {
    /// Read one texel at a mip level: `textureLoad`.
    #[inline]
    pub fn load(&self, coord: impl TexelCoord3, level: impl TexelIndex) -> T::Texel {
        unimplemented_on_cpu()
    }
}
impl<T: Component> texture_multisampled_2d<T> {
    /// Read one sample of one texel: `textureLoad`.
    #[inline]
    pub fn load(&self, coord: impl TexelCoord2, sample: impl TexelIndex) -> T::Texel {
        unimplemented_on_cpu()
    }
}
impl texture_depth_2d {
    /// Read one depth at a mip level: `textureLoad`.
    #[inline]
    pub fn load(&self, coord: impl TexelCoord2, level: impl TexelIndex) -> f32 {
        unimplemented_on_cpu()
    }
}
impl texture_depth_2d_array {
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
impl texture_depth_multisampled_2d {
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
            pub fn sample(&self, with: &sampler, coord: $coord $(, $layer: impl TexelIndex)?) -> vec4 {
                unimplemented_on_cpu()
            }

            /// Filtered read at an explicit level: `textureSampleLevel`.
            #[inline]
            pub fn sample_level(
                &self,
                with: &sampler,
                coord: $coord,
                $($layer: impl TexelIndex,)?
                level: f32,
            ) -> vec4 {
                unimplemented_on_cpu()
            }

            /// Filtered read with the picked level shifted: `textureSampleBias`.
            #[inline]
            pub fn sample_bias(
                &self,
                with: &sampler,
                coord: $coord,
                $($layer: impl TexelIndex,)?
                bias: f32,
            ) -> vec4 {
                unimplemented_on_cpu()
            }

            /// Filtered read with explicit derivatives: `textureSampleGrad`.
            #[inline]
            pub fn sample_grad(
                &self,
                with: &sampler,
                coord: $coord,
                $($layer: impl TexelIndex,)?
                ddx: $grad,
                ddy: $grad,
            ) -> vec4 {
                unimplemented_on_cpu()
            }
        }
    };
}
sample!(texture_2d<f32>, vec2, vec2);
sample!(texture_2d_array<f32>, vec2, vec2, layer);
sample!(texture_3d<f32>, vec3, vec3);
sample!(texture_cube<f32>, vec3, vec3);
sample!(texture_cube_array<f32>, vec3, vec3, layer);

impl texture_1d<f32> {
    /// Filtered read: `textureSample`.
    #[inline]
    pub fn sample(&self, with: &sampler, coord: f32) -> vec4 {
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
            pub fn sample(&self, with: &sampler, coord: $coord $(, $layer: impl TexelIndex)?) -> f32 {
                unimplemented_on_cpu()
            }

            /// Read at an explicit level: `textureSampleLevel`.
            #[inline]
            pub fn sample_level(
                &self,
                with: &sampler,
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
                with: &sampler_comparison,
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
                with: &sampler_comparison,
                coord: $coord,
                $($layer: impl TexelIndex,)?
                depth_ref: f32,
            ) -> f32 {
                unimplemented_on_cpu()
            }
        }
    };
}
sample_depth!(texture_depth_2d, vec2);
sample_depth!(texture_depth_2d_array, vec2, layer);
sample_depth!(texture_depth_cube, vec3);
sample_depth!(texture_depth_cube_array, vec3, layer);

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
storage_access!(texture_storage_1d, TexelCoord1);
storage_access!(texture_storage_2d, TexelCoord2);
storage_access!(texture_storage_2d_array, TexelCoord2, layer);
storage_access!(texture_storage_3d, TexelCoord3);

/// A ray query: a local that traces one ray through an acceleration structure.
///
/// Start one with `ray_query::default()`, point it at a ray with
/// [`initialize`](ray_query::initialize), and [`proceed`](ray_query::proceed)
/// until it says there is nothing more to consider. It cannot be copied,
/// which is WGSL's rule too.
#[allow(non_camel_case_types)]
#[derive(Default)]
pub struct ray_query {
    _local: (),
}

impl ray_query {
    /// Begin tracing `desc` through `scene`:
    /// `rayQueryInitialize`.
    #[inline]
    pub fn initialize(&mut self, scene: &acceleration_structure, desc: RayDesc) {
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
    pub flags: u32,
    pub cull_mask: u32,
    pub tmin: f32,
    pub tmax: f32,
    pub origin: vec3,
    pub dir: vec3,
}

/// What was found.
#[derive(Clone, Copy, Debug, Default)]
pub struct RayIntersection {
    pub kind: u32,
    pub t: f32,
    pub instance_custom_data: u32,
    pub instance_index: u32,
    pub sbt_record_offset: u32,
    pub geometry_index: u32,
    pub primitive_index: u32,
    pub barycentrics: vec2,
    pub front_face: bool,
    pub object_to_world: crate::matrix::mat4x3,
    pub world_to_object: crate::matrix::mat4x3,
}
