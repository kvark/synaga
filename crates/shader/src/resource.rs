//! Resources: what a shader binds, and where it lives.
//!
//! The address space is part of the type rather than an attribute, so a global
//! needs nothing but a type to be checkable:
//!
//! ```ignore
//! static camera: Uniform<Camera> = binding();
//! static mut counters: StorageMut<[u32]> = binding();
//! static albedo: texture_2d<f32> = binding();
//! ```
//!
//! Each derefs to what it holds, so `camera.view` reads through it.
//!
//! A writable resource is a `static mut`, because assigning through a shared
//! `static` is not something Rust allows however the type is arranged. The
//! stage attributes wrap function bodies in `unsafe` so the shader source does
//! not have to say it.

use core::marker::PhantomData;
use core::ops::{Deref, DerefMut, Index, IndexMut};

use crate::unimplemented_on_cpu;

/// A resource a shader binds. Every one is written `= binding()`.
pub trait Resource {
    /// The value a `static` is initialised with. Nothing reads it: the binding
    /// is supplied by the host at pipeline creation.
    const BINDING: Self;
}

/// The initialiser for any shader global.
#[inline]
pub const fn binding<T: Resource>() -> T {
    T::BINDING
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
    /// A read-write storage buffer. Declare it `static mut`.
    StorageMut,
    true
);
space!(
    /// Memory shared across a workgroup, zeroed each dispatch. Declare it
    /// `static mut`.
    Workgroup,
    true
);
space!(
    /// Memory private to each invocation. Declare it `static mut`.
    Private,
    true
);

macro_rules! writable {
    ($name:ident) => {
        impl<T: ?Sized> DerefMut for $name<T> {
            #[inline]
            fn deref_mut(&mut self) -> &mut T {
                unimplemented_on_cpu()
            }
        }
    };
}
writable!(StorageMut);
writable!(Workgroup);
writable!(Private);

/// An atomic in a storage or workgroup buffer.
#[derive(Clone, Copy, Debug, Default)]
#[allow(non_camel_case_types)]
pub struct atomic<T>(PhantomData<T>);

impl<T> Resource for atomic<T> {
    const BINDING: Self = atomic(PhantomData);
}

unsafe impl<T> Sync for atomic<T> {}
unsafe impl<T> Send for atomic<T> {}

/// An array of resources bound as one, indexed in the shader.
#[allow(non_camel_case_types)]
pub struct binding_array<T: ?Sized, const N: usize = 0>(PhantomData<T>);

impl<T: ?Sized, const N: usize> Resource for binding_array<T, N> {
    const BINDING: Self = binding_array(PhantomData);
}

unsafe impl<T: ?Sized, const N: usize> Sync for binding_array<T, N> {}
unsafe impl<T: ?Sized, const N: usize> Send for binding_array<T, N> {}

impl<T: ?Sized, const N: usize> Index<u32> for binding_array<T, N> {
    type Output = T;
    #[inline]
    fn index(&self, index: u32) -> &T {
        unimplemented_on_cpu()
    }
}

impl<T: ?Sized, const N: usize> Index<usize> for binding_array<T, N> {
    type Output = T;
    #[inline]
    fn index(&self, index: usize) -> &T {
        unimplemented_on_cpu()
    }
}

impl<T: ?Sized, const N: usize> IndexMut<usize> for binding_array<T, N> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        unimplemented_on_cpu()
    }
}

/// Number of elements in a runtime-sized array.
#[inline]
#[allow(non_snake_case)]
pub fn arrayLength<T>(_array: &[T]) -> u32 {
    unimplemented_on_cpu()
}

/// Number of elements in a runtime-sized array.
#[inline]
pub fn array_length<T>(_array: &[T]) -> u32 {
    unimplemented_on_cpu()
}

/// Read an atomic. The argument is the atomic itself: the operation can only
/// mean that storage, so the `&` WGSL writes is left off.
#[inline]
#[allow(non_snake_case)]
pub fn atomicLoad<T>(_atomic: atomic<T>) -> T {
    unimplemented_on_cpu()
}

/// Store `value` into an atomic.
#[inline]
#[allow(non_snake_case)]
pub fn atomicStore<T>(_atomic: atomic<T>, _value: T) {}

/// Add `value` and return what was there.
#[inline]
#[allow(non_snake_case)]
pub fn atomicAdd<T>(_atomic: atomic<T>, _value: T) -> T {
    unimplemented_on_cpu()
}

/// Subtract `value` and return what was there.
#[inline]
#[allow(non_snake_case)]
pub fn atomicSub<T>(_atomic: atomic<T>, _value: T) -> T {
    unimplemented_on_cpu()
}
