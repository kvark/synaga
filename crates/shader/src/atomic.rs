//! Atomics: `core::sync::atomic`'s types, minus what a GPU does not have.
//!
//! WGSL's atomics are relaxed and nothing stronger, so there is no `Ordering`
//! to pass, and `compare_exchange_weak` hands back WGSL's pair of old value
//! and outcome rather than a `Result`. Otherwise these are the standard types,
//! and on the CPU they are exactly that: every method is a relaxed operation
//! on a real atomic.
//!
//! Every method takes `&self`, as the standard ones do. An atomic changes
//! through a shared reference, so a buffer that is only ever updated through
//! its atomics can be a plain `static`:
//!
//! ```ignore
//! static counters: StorageMut<Counters> = group(0).binding(0);
//! let slot = counters.next.fetch_add(1);
//! ```

use core::fmt;
use core::sync::atomic::{self, Ordering::Relaxed};

/// What [`AtomicU32::compare_exchange_weak`] and its siblings hand back:
/// WGSL's `__atomic_compare_exchange_result`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompareExchange<T> {
    /// What the atomic held before the operation.
    pub old_value: T,
    /// Whether it held `current`, and so now holds `new`. A weak exchange may
    /// fail even then, so a loop retries until this is set.
    pub exchanged: bool,
}

macro_rules! atomic {
    ($(#[$doc:meta])* $name:ident, $int:ty) => {
        $(#[$doc])*
        #[repr(transparent)]
        #[derive(Default)]
        pub struct $name(atomic::$name);

        impl $name {
            /// An atomic holding `value`. Only the CPU makes one; on the GPU an
            /// atomic lives in a buffer or in workgroup memory.
            pub const fn new(value: $int) -> Self {
                Self(atomic::$name::new(value))
            }

            /// The value, once nothing else can reach the atomic.
            pub fn into_inner(self) -> $int {
                self.0.into_inner()
            }

            /// Read the value. WGSL's `atomicLoad`.
            #[inline]
            pub fn load(&self) -> $int {
                self.0.load(Relaxed)
            }

            /// Write `value`. WGSL's `atomicStore`.
            #[inline]
            pub fn store(&self, value: $int) {
                self.0.store(value, Relaxed)
            }

            /// Write `value` and return what was there. WGSL's `atomicExchange`.
            #[inline]
            pub fn swap(&self, value: $int) -> $int {
                self.0.swap(value, Relaxed)
            }

            /// Add, wrapping, and return what was there. WGSL's `atomicAdd`.
            #[inline]
            pub fn fetch_add(&self, value: $int) -> $int {
                self.0.fetch_add(value, Relaxed)
            }

            /// Subtract, wrapping, and return what was there. WGSL's
            /// `atomicSub`.
            #[inline]
            pub fn fetch_sub(&self, value: $int) -> $int {
                self.0.fetch_sub(value, Relaxed)
            }

            /// Keep the larger and return what was there. WGSL's `atomicMax`.
            #[inline]
            pub fn fetch_max(&self, value: $int) -> $int {
                self.0.fetch_max(value, Relaxed)
            }

            /// Keep the smaller and return what was there. WGSL's `atomicMin`.
            #[inline]
            pub fn fetch_min(&self, value: $int) -> $int {
                self.0.fetch_min(value, Relaxed)
            }

            /// Bitwise and, returning what was there. WGSL's `atomicAnd`.
            #[inline]
            pub fn fetch_and(&self, value: $int) -> $int {
                self.0.fetch_and(value, Relaxed)
            }

            /// Bitwise or, returning what was there. WGSL's `atomicOr`.
            #[inline]
            pub fn fetch_or(&self, value: $int) -> $int {
                self.0.fetch_or(value, Relaxed)
            }

            /// Bitwise exclusive or, returning what was there. WGSL's
            /// `atomicXor`.
            #[inline]
            pub fn fetch_xor(&self, value: $int) -> $int {
                self.0.fetch_xor(value, Relaxed)
            }

            /// Write `new` if the atomic holds `current`. WGSL's
            /// `atomicCompareExchangeWeak`, which may fail spuriously; so may
            /// this.
            #[inline]
            pub fn compare_exchange_weak(&self, current: $int, new: $int) -> CompareExchange<$int> {
                match self.0.compare_exchange_weak(current, new, Relaxed, Relaxed) {
                    Ok(old_value) => CompareExchange {
                        old_value,
                        exchanged: true,
                    },
                    Err(old_value) => CompareExchange {
                        old_value,
                        exchanged: false,
                    },
                }
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Debug::fmt(&self.load(), f)
            }
        }

        impl From<$int> for $name {
            fn from(value: $int) -> Self {
                Self::new(value)
            }
        }
    };
}

atomic!(
    /// WGSL's `atomic<u32>`.
    AtomicU32,
    u32
);
atomic!(
    /// WGSL's `atomic<i32>`.
    AtomicI32,
    i32
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cpu_atomic_is_a_real_one() {
        let a = AtomicU32::new(5);
        assert_eq!(a.fetch_add(3), 5);
        assert_eq!(a.fetch_sub(1), 8);
        assert_eq!(a.fetch_max(10), 7);
        assert_eq!(a.fetch_min(2), 10);
        assert_eq!(a.swap(9), 2);
        assert_eq!(a.load(), 9);
        a.store(0);
        assert_eq!(a.fetch_add(u32::MAX), 0, "adds wrap, as on a GPU");
        assert_eq!(a.load(), u32::MAX);

        let b = AtomicI32::new(-1);
        assert_eq!(b.fetch_and(6), -1);
        assert_eq!(b.fetch_or(1), 6);
        assert_eq!(b.fetch_xor(7), 7);
        assert_eq!(b.into_inner(), 0);
    }

    #[test]
    fn compare_exchange_says_what_it_found() {
        let a = AtomicU32::new(1);
        let failed = a.compare_exchange_weak(2, 3);
        assert_eq!(
            failed,
            CompareExchange {
                old_value: 1,
                exchanged: false
            }
        );
        // A weak exchange may fail spuriously, so retry as a shader would.
        loop {
            let r = a.compare_exchange_weak(1, 3);
            assert_eq!(r.old_value, 1);
            if r.exchanged {
                break;
            }
        }
        assert_eq!(a.load(), 3);
    }
}
