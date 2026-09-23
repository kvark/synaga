//! Textures and samplers, and the operations on them.
//!
//! The types are opaque: a shader binds one and passes it to the builtins,
//! and there is nothing to read out of it directly.
//!
//! Each is a marker with no data in it — the resource lives on the GPU — which
//! is why they are `Sync` by assertion rather than by their contents.

use core::marker::PhantomData;

use crate::resource::Resource;
use crate::unimplemented_on_cpu;
use crate::vector::{vec2, vec2i, vec2u, vec3, vec3i, vec3u, vec4};

pub use access::*;
pub use format::*;

/// Storage texture formats, as type arguments.
pub mod format {
    macro_rules! formats {
        ($($name:ident),* $(,)?) => {
            $(
                /// A storage texture format.
                #[derive(Debug)]
                pub struct $name;
            )*
        };
    }
    formats!(
        R8Unorm,
        R8Snorm,
        R8Uint,
        R8Sint,
        R16Uint,
        R16Sint,
        R16Float,
        Rg8Unorm,
        Rg8Snorm,
        Rg8Uint,
        Rg8Sint,
        R32Uint,
        R32Sint,
        R32Float,
        Rg16Uint,
        Rg16Sint,
        Rg16Float,
        Rgba8Unorm,
        Rgba8Snorm,
        Rgba8Uint,
        Rgba8Sint,
        Rgb10a2Unorm,
        Rg11b10Float,
        Rg32Uint,
        Rg32Sint,
        Rg32Float,
        Rgba16Uint,
        Rgba16Sint,
        Rgba16Float,
        Rgba32Uint,
        Rgba32Sint,
        Rgba32Float,
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
}

macro_rules! sampled {
    ($($name:ident),* $(,)?) => {
        $(
            /// A sampled texture.
            #[allow(non_camel_case_types)]
            pub struct $name<T>(PhantomData<T>);
            impl<T> Resource for $name<T> {
                const BINDING: Self = $name(PhantomData);
            }
            unsafe impl<T> Sync for $name<T> {}
            unsafe impl<T> Send for $name<T> {}
        )*
    };
}
sampled!(
    texture_1d,
    texture_2d,
    texture_3d,
    texture_cube,
    texture_2d_array,
    texture_cube_array,
    texture_multisampled_2d,
);

macro_rules! plain {
    ($($name:ident),* $(,)?) => {
        $(
            /// A texture or sampler with nothing to parameterise.
            #[allow(non_camel_case_types)]
            #[derive(Clone, Copy)]
            pub struct $name;
            impl Resource for $name {
                const BINDING: Self = $name;
            }
        )*
    };
}
plain!(
    sampler,
    sampler_comparison,
    texture_depth_2d,
    texture_depth_cube,
    texture_depth_2d_array,
    texture_depth_multisampled_2d,
    acceleration_structure,
);

macro_rules! storage_texture {
    ($($name:ident),* $(,)?) => {
        $(
            /// A storage texture, parameterised by format and access.
            #[allow(non_camel_case_types)]
            pub struct $name<F, A>(PhantomData<(F, A)>);
            impl<F, A> Resource for $name<F, A> {
                const BINDING: Self = $name(PhantomData);
            }
            unsafe impl<F, A> Sync for $name<F, A> {}
            unsafe impl<F, A> Send for $name<F, A> {}
        )*
    };
}
storage_texture!(
    texture_storage_1d,
    texture_storage_2d,
    texture_storage_3d,
    texture_storage_2d_array,
);

/// A ray query, declared as a local and driven by the `rayQuery*` builtins.
///
/// It is passed by value. The value is a marker — the query lives in the
/// invocation — and copying it still names that same local, which is what the
/// operations write.
#[derive(Clone, Copy, Default)]
#[allow(non_camel_case_types)]
pub struct ray_query;

/// Begin a query. `query` is the local the later operations name.
#[inline]
#[allow(non_snake_case)]
pub fn rayQueryInitialize(
    _query: ray_query,
    _acceleration_structure: &acceleration_structure,
    _desc: RayDesc,
) {
}

/// Advance to the next candidate. `true` while one remains.
#[inline]
#[allow(non_snake_case)]
pub fn rayQueryProceed(_query: ray_query) -> bool {
    unimplemented_on_cpu()
}

/// The closest hit committed so far.
#[inline]
#[allow(non_snake_case)]
pub fn rayQueryGetCommittedIntersection(_query: ray_query) -> RayIntersection {
    unimplemented_on_cpu()
}

/// The candidate currently under consideration.
#[inline]
#[allow(non_snake_case)]
pub fn rayQueryGetCandidateIntersection(_query: ray_query) -> RayIntersection {
    unimplemented_on_cpu()
}

/// Keep the current triangle candidate as a committed hit.
#[inline]
#[allow(non_snake_case)]
pub fn rayQueryConfirmIntersection(_query: ray_query) {}

/// Offer a generated intersection at distance `hit_t`.
#[inline]
#[allow(non_snake_case)]
pub fn rayQueryGenerateIntersection(_query: ray_query, _hit_t: f32) {}

/// Stop considering candidates.
#[inline]
#[allow(non_snake_case)]
pub fn rayQueryTerminate(_query: ray_query) {}

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

/// A coordinate a texture can be addressed with.
pub trait TexelCoord {}
impl TexelCoord for i32 {}
impl TexelCoord for u32 {}
impl TexelCoord for vec2i {}
impl TexelCoord for vec2u {}
impl TexelCoord for vec3i {}
impl TexelCoord for vec3u {}

/// Anything the sampling builtins accept as a texture.
pub trait Sampled {
    /// What a sample of it produces.
    type Texel;
}
impl<T> Sampled for texture_1d<T> {
    type Texel = vec4;
}
impl<T> Sampled for texture_2d<T> {
    type Texel = vec4;
}
impl<T> Sampled for texture_3d<T> {
    type Texel = vec4;
}
impl<T> Sampled for texture_cube<T> {
    type Texel = vec4;
}
impl<T> Sampled for texture_2d_array<T> {
    type Texel = vec4;
}
impl<T> Sampled for texture_multisampled_2d<T> {
    type Texel = vec4;
}
impl Sampled for texture_depth_2d {
    type Texel = f32;
}
impl<F, A> Sampled for texture_storage_2d<F, A> {
    type Texel = vec4;
}

macro_rules! sample_fns {
    ($($wgsl:ident / $snake:ident ($($arg:ident: $ty:ty),*);)*) => {
        $(
            #[inline]
            #[allow(non_snake_case)]
            pub fn $wgsl<T: Sampled>($($arg: $ty),*) -> T::Texel { unimplemented_on_cpu() }
            #[inline]
            pub fn $snake<T: Sampled>($($arg: $ty),*) -> T::Texel { unimplemented_on_cpu() }
        )*
    };
}

sample_fns! {
    textureSample / texture_sample (t: &T, s: &sampler, coord: vec2);
    textureSampleLevel / texture_sample_level (t: &T, s: &sampler, coord: vec2, level: f32);
}

/// Read a texel, at a mip level.
#[inline]
#[allow(non_snake_case)]
pub fn textureLoad<T: Sampled, C: TexelCoord>(_t: &T, _coord: C, _level: i32) -> T::Texel {
    unimplemented_on_cpu()
}

#[inline]
pub fn texture_load<T: Sampled, C: TexelCoord>(_t: &T, _coord: C, _level: i32) -> T::Texel {
    unimplemented_on_cpu()
}

/// Read a texel from a storage texture, which has no mip levels.
///
/// WGSL calls this `textureLoad` too, at one argument fewer. Rust cannot give
/// one name two arities, so the storage form takes its own.
#[inline]
#[allow(non_snake_case)]
pub fn textureLoadStorage<F, A, C: TexelCoord>(_t: &texture_storage_2d<F, A>, _coord: C) -> vec4 {
    unimplemented_on_cpu()
}

#[inline]
pub fn texture_load_storage<F, A, C: TexelCoord>(_t: &texture_storage_2d<F, A>, _coord: C) -> vec4 {
    unimplemented_on_cpu()
}

#[inline]
#[allow(non_snake_case)]
pub fn textureStore<F, A, C: TexelCoord>(_t: &texture_storage_2d<F, A>, _coord: C, _value: vec4) {
    unimplemented_on_cpu()
}

#[inline]
pub fn texture_store<F, A, C: TexelCoord>(_t: &texture_storage_2d<F, A>, _coord: C, _value: vec4) {
    unimplemented_on_cpu()
}

/// Size of a texture, which is a `vec2<u32>` for the two-dimensional ones.
#[inline]
#[allow(non_snake_case)]
pub fn textureDimensions<T>(_t: &T) -> vec2u {
    unimplemented_on_cpu()
}

/// Size of one mip level. WGSL spells this as `textureDimensions` with another
/// argument; Rust cannot give one name two arities.
#[inline]
#[allow(non_snake_case)]
pub fn textureDimensionsLevel<T>(_t: &T, _level: i32) -> vec2u {
    unimplemented_on_cpu()
}

#[inline]
pub fn texture_dimensions<T>(_t: &T) -> vec2u {
    unimplemented_on_cpu()
}

#[inline]
#[allow(non_snake_case)]
pub fn textureNumLevels<T>(_t: &T) -> u32 {
    unimplemented_on_cpu()
}

#[inline]
#[allow(non_snake_case)]
pub fn textureSampleCompare<T>(
    _t: &T,
    _s: &sampler_comparison,
    _coord: vec2,
    _depth_ref: f32,
) -> f32 {
    unimplemented_on_cpu()
}
