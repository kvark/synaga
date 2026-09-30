//! Resources: what a shader binds, and where it lives.
//!
//! The address space is part of the type rather than an attribute, and the
//! initialiser says where the resource binds, so a global is checkable with
//! nothing but Rust:
//!
//! ```ignore
//! static camera: Uniform<Camera> = group(0).binding(0);
//! static counters: StorageMut<[u32]> = group(0).binding(1);
//! static albedo: Texture2D<f32> = group(1).binding(0);
//! ```
//!
//! `group(0).binding(1)` is WGSL's `@group(0) @binding(1)`. A host that
//! assigns bindings itself, matching globals up by name as Blade does, has
//! them written `= binding()` instead.
//!
//! Each derefs to what it holds, so `camera.view` reads through it.
//!
//! None of them is a `static mut`: edition 2024 refuses even a read through
//! one. A writable resource is a plain `static`, its atomics take `&self`, and
//! a write goes through [`StorageMut::get_mut`]:
//!
//! ```ignore
//! let slot = counters[0];
//! counters.get_mut()[1] = slot + 1;
//! ```
//!
//! Other invocations run at the same time and may be writing the same memory,
//! as they may in WGSL, and the shader keeps the writes that matter apart, as
//! it has to in WGSL. `get_mut` is not `unsafe` for it, because what `unsafe`
//! would guard against, two `&mut` to one place on the CPU, cannot happen:
//! the body never returns there.

use core::marker::PhantomData;
use core::ops::{Deref, Index, IndexMut};

use crate::unimplemented_on_cpu;

/// What a shader global holds: a resource, or the shader's own memory.
pub trait Resource {
    /// The value a `static` is initialised with. Nothing reads it: what the
    /// global stands for is on the GPU.
    const BINDING: Self;
}

/// A resource bound from outside the shader: a buffer, a texture, a sampler,
/// an acceleration structure, or an array of them. [`Workgroup`] and
/// [`Private`] memory is the shader's own, so it binds to nothing.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a resource a host binds",
    note = "buffers, textures, samplers, acceleration structures and binding arrays bind; \
            `Workgroup` and `Private` memory is the shader's own, and is initialised with \
            `binding()`"
)]
pub trait Bindable: Resource {}

/// The initialiser for a global whose binding the host assigns, and for
/// workgroup and private memory, which has none.
#[inline]
pub const fn binding<T: Resource>() -> T {
    T::BINDING
}

/// The initialiser for a global that says where it binds: `group(0).binding(1)`
/// is WGSL's `@group(0) @binding(1)`.
///
/// ```
/// use synaga_shader::*;
///
/// // A `const` works too, and the host can use the same one.
/// pub const MATERIAL: u32 = 1;
///
/// pub static camera: Uniform<Mat4> = group(0).binding(0);
/// pub static albedo: Texture2D<f32> = group(MATERIAL).binding(0);
/// pub static linear: Sampler = group(MATERIAL).binding(1);
/// ```
///
/// Only a resource the host binds takes one:
///
/// ```compile_fail,E0277
/// use synaga_shader::*;
///
/// pub static tile: Workgroup<[f32; 64]> = group(0).binding(0);
/// ```
#[inline]
pub const fn group(group: u32) -> Group {
    Group(())
}

/// A bind group, as [`group`] names it, waiting for the binding within it.
#[derive(Clone, Copy, Debug)]
pub struct Group(());

impl Group {
    /// The resource at `binding` in this group.
    #[inline]
    pub const fn binding<T: Bindable>(self, binding: u32) -> T {
        T::BINDING
    }
}

macro_rules! space {
    ($(#[$doc:meta])* $name:ident, $mutable:literal) => {
        $(#[$doc])*
        pub struct $name<T: ?Sized>(PhantomData<*const T>);

        impl<T: ?Sized> Resource for $name<T> {
            const BINDING: Self = $name(PhantomData);
        }

        // A `static` has to be `Sync`, and this is a marker with no data in
        // it: the resource itself lives on the GPU.
        unsafe impl<T: ?Sized> Sync for $name<T> {}
        unsafe impl<T: ?Sized> Send for $name<T> {}

        impl<T: ?Sized> Deref for $name<T> {
            type Target = T;
            #[inline]
            fn deref(&self) -> &T { unimplemented_on_cpu() }
        }
    };
}

space!(
    /// A uniform buffer: read-only, and small enough to fit the stricter
    /// layout rules that come with the space.
    Uniform,
    false
);
space!(
    /// A read-only storage buffer, which is where a runtime-sized `[T]` lives.
    Storage,
    false
);
space!(
    /// A read-write storage buffer. Write through [`StorageMut::get_mut`].
    StorageMut,
    true
);
space!(
    /// Memory shared across a workgroup, zeroed each dispatch. Write through
    /// [`Workgroup::get_mut`].
    Workgroup,
    true
);
space!(
    /// Memory private to each invocation. Write through [`Private::get_mut`].
    Private,
    true
);

// Buffers bind; workgroup and private memory is the shader's own.
impl<T: ?Sized> Bindable for Uniform<T> {}
impl<T: ?Sized> Bindable for Storage<T> {}
impl<T: ?Sized> Bindable for StorageMut<T> {}

macro_rules! writable {
    ($name:ident) => {
        impl<T: ?Sized> $name<T> {
            /// What this holds, to write to: `buf.get_mut().x = 1`.
            ///
            /// Other invocations may be reading or writing the same memory at
            /// the same time. The shader keeps the writes that matter apart,
            /// by giving each invocation its own part of the memory or by a
            /// barrier between them, as it has to in WGSL.
            ///
            /// `&mut` from `&self` would be unsound if it returned on the CPU,
            /// where two of them could alias; it never does.
            #[inline]
            #[allow(clippy::mut_from_ref)]
            pub fn get_mut(&self) -> &mut T {
                unimplemented_on_cpu()
            }

            /// Write `value` over all of it: `emit_end.set(n)`.
            ///
            /// This is `*emit_end.get_mut() = n`. Rust cannot assign through
            /// `*emit_end`, which would borrow a `static` mutably, so this is
            /// the `Cell::set` of a resource.
            #[inline]
            pub fn set(&self, value: T)
            where
                T: Sized,
            {
                unimplemented_on_cpu()
            }
        }
    };
}
writable!(StorageMut);
writable!(Workgroup);
writable!(Private);

/// An array of resources bound as one, indexed in the shader.
pub struct BindingArray<T: ?Sized, const N: usize = 0>(PhantomData<T>);

impl<T: ?Sized, const N: usize> Resource for BindingArray<T, N> {
    const BINDING: Self = BindingArray(PhantomData);
}

impl<T: ?Sized + Bindable, const N: usize> Bindable for BindingArray<T, N> {}

unsafe impl<T: ?Sized, const N: usize> Sync for BindingArray<T, N> {}
unsafe impl<T: ?Sized, const N: usize> Send for BindingArray<T, N> {}

impl<T: ?Sized, const N: usize> Index<u32> for BindingArray<T, N> {
    type Output = T;
    #[inline]
    fn index(&self, index: u32) -> &T {
        unimplemented_on_cpu()
    }
}

impl<T: ?Sized, const N: usize> Index<usize> for BindingArray<T, N> {
    type Output = T;
    #[inline]
    fn index(&self, index: usize) -> &T {
        unimplemented_on_cpu()
    }
}

impl<T: ?Sized, const N: usize> IndexMut<usize> for BindingArray<T, N> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        unimplemented_on_cpu()
    }
}
