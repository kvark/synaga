//! Resources: what a shader binds, and where it lives.
//!
//! The address space is part of the type rather than an attribute, so a global
//! needs nothing but a type to be checkable:
//!
//! ```ignore
//! static camera: Uniform<Camera> = binding();
//! static counters: StorageMut<[u32]> = binding();
//! static albedo: texture_2d<f32> = binding();
//! ```
//!
//! Each derefs to what it holds, so `camera.view` reads through it.
//!
//! None of them is a `static mut`. Other invocations run at the same time and
//! may be writing the same memory, which Rust calls shared mutable state; a
//! `static mut` says so too, but edition 2024 refuses even a read through one.
//! So a writable resource is a plain `static`: reading it is safe, its atomics
//! take `&self`, and a write goes through [`StorageMut::get_mut`], which is
//! `unsafe` because the shader, not the compiler, keeps the invocations apart:
//!
//! ```ignore
//! let slot = counters[0];
//! unsafe { counters.get_mut()[1] = slot + 1 };
//! ```

use core::marker::PhantomData;
use core::ops::{Deref, Index, IndexMut};

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

macro_rules! writable {
    ($name:ident) => {
        impl<T: ?Sized> $name<T> {
            /// What this holds, to write to: `unsafe { buf.get_mut().x = 1 }`.
            ///
            /// # Safety
            ///
            /// Other invocations may be reading or writing the same memory at
            /// the same time. The shader has to keep the writes that matter
            /// apart, by giving each invocation its own part of the memory or
            /// by a barrier between them, as it has to on a GPU. On the CPU,
            /// two of these must not be alive at once.
            #[inline]
            #[allow(clippy::mut_from_ref)]
            pub unsafe fn get_mut(&self) -> &mut T {
                unimplemented_on_cpu()
            }
        }
    };
}
writable!(StorageMut);
writable!(Workgroup);
writable!(Private);

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
