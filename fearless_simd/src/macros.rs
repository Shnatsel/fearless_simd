// Copyright 2024 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Macros publicly exported

/// Access the applicable [`Simd`] for a given `level`, and perform an operation using it.
///
/// This macro is the root of how any explicitly written SIMD functions in this crate are
/// called from a non-SIMD context.
///
/// The first parameter to the macro is the [`Level`].
/// You should prefer to construct a [`Level`] once and pass it around, rather than
/// frequently calling [`Level::new()`].
/// This is because `Level::new` has to detect which target features are available, which can be slow.
///
/// The code of the operation will be repeated literally several times in the output, so you should prefer
/// to keep this code small (as it will be type-checked, etc. for each supported SIMD level on your target).
/// In most cases, it should be a single call to a function which is generic over `Simd` implementations,
/// as seen in [the examples](#examples).
/// For clarity, it will only be executed once per execution of `dispatch`.
///
/// To guarantee target-feature-specific code generation, any functions called within the operation should
/// be `#[inline(always)]`.
///
/// Note that as an implementation detail of this macro, the operation will be executed inside a closure.
/// This is what enables the target features to be enabled for the code inside the operation.
/// A consequence of this is that early `return` and `?` will not work as expected.
/// Note that in cases where you use `dispatch` to call a single function (which we expect to be the
/// majority of cases), you can use `?` on the return value of dispatch instead.
/// To emulate early return, you can use [`ControlFlow`](core::ops::ControlFlow) instead.
///
/// # Example
///
/// ```rust
/// use fearless_simd::{Level, Simd, dispatch};
///
/// #[inline(always)]
/// fn sigmoid<S: Simd>(simd: S, x: &[f32], out: &mut [f32]) { /* ... */ }
///
/// let level = Level::new();
///
/// dispatch!(level, simd => sigmoid(simd, &[/*...*/], &mut [/*...*/]));
/// ```
///
/// [`Level`]: crate::Level
/// [`Level::new()`]: crate::Level::new
/// [`Simd`]: crate::Simd
#[macro_export]
macro_rules! dispatch {
    // This falls through to the next branch, but with `forced_fallback_arm` turned into a boolean literal
    // indicating whether or not the `force_support_fallback` crate feature is enabled.
    ($level:expr, $simd:pat => $op:expr) => {{ $crate::internal_unstable_dispatch_inner!($level, $simd => $op) }};
    (@impl $level:expr, $simd:pat => $op:expr; $forced_fallback_arm: literal) => {{
        /// Convert the `Simd` value into an `impl Simd`, which enforces that
        /// it is correctly handled.
        // TODO: Just make into a `pub` function in fearless_simd itself?
        #[inline(always)]
        fn launder<S: $crate::Simd>(x: S) -> impl $crate::Simd {
            x
        }

        match $level {
            #[cfg(target_arch = "aarch64")]
            $crate::Level::Neon(neon) => {
                let $simd = launder(neon);
                $crate::Simd::vectorize(
                    neon,
                    #[inline(always)]
                    || $op,
                )
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            $crate::Level::WasmSimd128(wasm) => {
                let $simd = launder(wasm);
                $crate::Simd::vectorize(
                    wasm,
                    #[inline(always)]
                    || $op,
                )
            }
            // This fallthrough logic is documented at the definition site of `Level`.
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                not(all(target_feature = "avx2", target_feature = "fma"))
            ))]
            $crate::Level::Sse4_2(sse4_2) => {
                let $simd = launder(sse4_2);
                $crate::Simd::vectorize(
                    sse4_2,
                    #[inline(always)]
                    || $op,
                )
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            $crate::Level::Avx2(avx2) => {
                let $simd = launder(avx2);
                $crate::Simd::vectorize(
                    avx2,
                    #[inline(always)]
                    || $op,
                )
            }
            #[cfg(any(
                all(target_arch = "aarch64", not(target_feature = "neon")),
                all(
                    any(target_arch = "x86", target_arch = "x86_64"),
                    not(target_feature = "sse4.2")
                ),
                all(target_arch = "wasm32", not(target_feature = "simd128")),
                not(any(
                    target_arch = "x86",
                    target_arch = "x86_64",
                    target_arch = "aarch64",
                    target_arch = "wasm32"
                )),
                $forced_fallback_arm
            ))]
            $crate::Level::Fallback(fb) => {
                let $simd = launder(fb);
                // This vectorize call does nothing, but it is reasonable to be consistent here.
                $crate::Simd::vectorize(
                    fb,
                    #[inline(always)]
                    || $op,
                )
            }
            _ => unreachable!(),
        }
    }};
}

/// Returns a function pointer to a SIMD-enabled function for later use.
///
/// This macro is similar to [`dispatch!`], but instead of immediately executing the operation,
/// it returns a function pointer that can be stored and called later. This is useful when you
/// want to detect SIMD capabilities once at initialization time and then use the appropriate
/// function throughout your application without repeated dispatch overhead.
///
/// The first parameter to the macro is the [`Level`].
/// The second parameter is a function path (not a function call or closure).
/// The third parameter is the function signature `(ArgTypes...) -> ReturnType`.
///
/// The returned function pointer has the signature `fn(ArgTypes...) -> ReturnType`.
/// Note that unlike [`dispatch!`], the function does NOT receive the `Simd` type as an argument -
/// the SIMD implementation is baked into the returned function pointer.
///
/// Due to the way Rust handles function argument patterns in macros, you must provide
/// argument names along with types using the syntax `(name1: Type1, name2: Type2, ...) -> ReturnType`.
///
/// # Example
///
/// ```rust
/// use fearless_simd::{Level, Simd, dispatch_for_later};
///
/// #[inline(always)]
/// fn add_slices<S: Simd>(_simd: S, a: &[f32], b: &[f32], out: &mut [f32]) {
///     for ((a, b), out) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
///         *out = a + b;
///     }
/// }
///
/// let level = Level::new();
///
/// // Get a function pointer for later use
/// let add_fn = dispatch_for_later!(level, add_slices, (a: &[f32], b: &[f32], out: &mut [f32]) -> ());
///
/// // Call the function pointer later, without needing to dispatch again
/// let a = [1.0f32, 2.0, 3.0];
/// let b = [4.0f32, 5.0, 6.0];
/// let mut out = [0.0f32; 3];
/// add_fn(&a, &b, &mut out);
/// assert_eq!(out, [5.0, 7.0, 9.0]);
/// ```
///
/// [`Level`]: crate::Level
/// [`Simd`]: crate::Simd
#[macro_export]
macro_rules! dispatch_for_later {
    ($level:expr, $func:path, ($($arg_name:ident : $arg_ty:ty),*) -> $ret:ty) => {{
        $crate::internal_unstable_dispatch_for_later_inner!($level, $func, ($($arg_name : $arg_ty),*) -> $ret)
    }};
    (@impl $level:expr, $func:path, ($($arg_name:ident : $arg_ty:ty),*) -> $ret:ty; $forced_fallback_arm:literal) => {{
        match $level {
            #[cfg(target_arch = "aarch64")]
            $crate::Level::Neon(_neon) => {
                #[target_feature(enable = "neon")]
                unsafe fn inner($($arg_name : $arg_ty),*) -> $ret {
                    // SAFETY: We're inside a target_feature(enable = "neon") function,
                    // so it's safe to create a Neon token.
                    $func(unsafe { $crate::aarch64::Neon::new_unchecked() }, $($arg_name),*)
                }
                fn wrapper($($arg_name : $arg_ty),*) -> $ret {
                    // SAFETY: We checked that the neon feature is available via Level::Neon
                    unsafe { inner($($arg_name),*) }
                }
                wrapper as fn($($arg_ty),*) -> $ret
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            $crate::Level::WasmSimd128(_wasm) => {
                // WASM doesn't need target_feature wrapper since simd128 is compile-time
                // WasmSimd128::new_unchecked() is not unsafe on WASM
                fn wrapper($($arg_name : $arg_ty),*) -> $ret {
                    $func($crate::wasm32::WasmSimd128::new_unchecked(), $($arg_name),*)
                }
                wrapper as fn($($arg_ty),*) -> $ret
            }
            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                not(all(target_feature = "avx2", target_feature = "fma"))
            ))]
            $crate::Level::Sse4_2(_sse4_2) => {
                #[target_feature(enable = "sse4.2")]
                unsafe fn inner($($arg_name : $arg_ty),*) -> $ret {
                    // SAFETY: We're inside a target_feature(enable = "sse4.2") function,
                    // so it's safe to create an Sse4_2 token.
                    $func(unsafe { $crate::x86::Sse4_2::new_unchecked() }, $($arg_name),*)
                }
                fn wrapper($($arg_name : $arg_ty),*) -> $ret {
                    // SAFETY: We checked that the sse4.2 feature is available via Level::Sse4_2
                    unsafe { inner($($arg_name),*) }
                }
                wrapper as fn($($arg_ty),*) -> $ret
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            $crate::Level::Avx2(_avx2) => {
                #[target_feature(enable = "avx2", enable = "fma")]
                unsafe fn inner($($arg_name : $arg_ty),*) -> $ret {
                    // SAFETY: We're inside a target_feature(enable = "avx2", enable = "fma") function,
                    // so it's safe to create an Avx2 token.
                    $func(unsafe { $crate::x86::Avx2::new_unchecked() }, $($arg_name),*)
                }
                fn wrapper($($arg_name : $arg_ty),*) -> $ret {
                    // SAFETY: We checked that avx2 and fma features are available via Level::Avx2
                    unsafe { inner($($arg_name),*) }
                }
                wrapper as fn($($arg_ty),*) -> $ret
            }
            #[cfg(any(
                all(target_arch = "aarch64", not(target_feature = "neon")),
                all(
                    any(target_arch = "x86", target_arch = "x86_64"),
                    not(target_feature = "sse4.2")
                ),
                all(target_arch = "wasm32", not(target_feature = "simd128")),
                not(any(
                    target_arch = "x86",
                    target_arch = "x86_64",
                    target_arch = "aarch64",
                    target_arch = "wasm32"
                )),
                $forced_fallback_arm
            ))]
            $crate::Level::Fallback(_fb) => {
                fn wrapper($($arg_name : $arg_ty),*) -> $ret {
                    $func($crate::Fallback::new(), $($arg_name),*)
                }
                wrapper as fn($($arg_ty),*) -> $ret
            }
            _ => unreachable!(),
        }
    }};
}

// This macro turns whether the `force_support_fallback` macro is enabled into a boolean literal
// in `dispatch`, which allows it to be used correctly cross-crate.
// This trickery is required because macros are expanded in the context of the calling crate, including for
// evaluating `cfg`s.

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(feature = "force_support_fallback")]
macro_rules! internal_unstable_dispatch_inner {
    ($level:expr, $simd:pat => $op:expr) => {
        $crate::dispatch!(
            @impl $level, $simd => $op; true
        )
    };
}

/// Implementation detail of [`crate::dispatch`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(not(feature = "force_support_fallback"))]
macro_rules! internal_unstable_dispatch_inner {
    ($level:expr, $simd:pat => $op:expr) => {
        $crate::dispatch!(@impl $level, $simd => $op; false)
    };
}

/// Implementation detail of [`crate::dispatch_for_later`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(feature = "force_support_fallback")]
macro_rules! internal_unstable_dispatch_for_later_inner {
    ($level:expr, $func:path, ($($arg_name:ident : $arg_ty:ty),*) -> $ret:ty) => {
        $crate::dispatch_for_later!(
            @impl $level, $func, ($($arg_name : $arg_ty),*) -> $ret; true
        )
    };
}

/// Implementation detail of [`crate::dispatch_for_later`]; this is not public API.
#[macro_export]
#[doc(hidden)]
#[cfg(not(feature = "force_support_fallback"))]
macro_rules! internal_unstable_dispatch_for_later_inner {
    ($level:expr, $func:path, ($($arg_name:ident : $arg_ty:ty),*) -> $ret:ty) => {
        $crate::dispatch_for_later!(@impl $level, $func, ($($arg_name : $arg_ty),*) -> $ret; false)
    };
}

#[cfg(test)]
// This expect also validates that we haven't missed any levels!
#[expect(
    unreachable_patterns,
    reason = "Level is non_exhaustive, but you must be exhaustive within the same crate."
)]
mod tests {
    use crate::{Level, Simd};

    #[allow(dead_code, reason = "Compile test")]
    fn dispatch_generic() {
        fn generic<S: Simd, T>(_: S, x: T) -> T {
            x
        }
        dispatch!(Level::new(), simd => generic::<_, ()>(simd, ()));
    }

    #[allow(dead_code, reason = "Compile test")]
    fn dispatch_value() {
        fn make_fn<S: Simd>() -> impl FnOnce(S) {
            |_| ()
        }
        dispatch!(Level::new(), simd => (make_fn())(simd));
    }

    #[test]
    fn dispatch_output() {
        assert_eq!(42, dispatch!(Level::new(), _simd => 42));
    }

    mod no_import_simd {
        /// We should be able to use [`dispatch`] in a scope which doesn't import anything.
        #[test]
        fn dispatch_with_no_imports() {
            let res = dispatch!(crate::Level::new(), _ => 1 + 2);
            assert_eq!(res, 3);
        }
    }

    // Tests for dispatch_for_later!

    #[inline(always)]
    fn add_values<S: Simd>(_simd: S, a: i32, b: i32) -> i32 {
        a + b
    }

    #[test]
    fn dispatch_for_later_basic() {
        let level = Level::new();
        let add_fn = dispatch_for_later!(level, add_values, (a: i32, b: i32) -> i32);
        assert_eq!(add_fn(2, 3), 5);
        assert_eq!(add_fn(10, 20), 30);
    }

    #[inline(always)]
    fn multiply_slice<S: Simd>(_simd: S, slice: &mut [i32], factor: i32) {
        for x in slice.iter_mut() {
            *x *= factor;
        }
    }

    #[test]
    fn dispatch_for_later_with_mutable_ref() {
        let level = Level::new();
        let mul_fn =
            dispatch_for_later!(level, multiply_slice, (slice: &mut [i32], factor: i32) -> ());
        let mut data = [1, 2, 3, 4, 5];
        mul_fn(&mut data, 2);
        assert_eq!(data, [2, 4, 6, 8, 10]);
    }

    #[inline(always)]
    fn no_args<S: Simd>(_simd: S) -> i32 {
        42
    }

    #[test]
    fn dispatch_for_later_no_args() {
        let level = Level::new();
        let fn_ptr = dispatch_for_later!(level, no_args, () -> i32);
        assert_eq!(fn_ptr(), 42);
    }

    #[inline(always)]
    fn return_unit<S: Simd>(_simd: S, _x: i32) {}

    #[test]
    fn dispatch_for_later_unit_return() {
        let level = Level::new();
        let fn_ptr = dispatch_for_later!(level, return_unit, (x: i32) -> ());
        fn_ptr(123); // Should compile and run without panicking
    }

    /// Test that the function pointer can be stored and called later.
    #[test]
    fn dispatch_for_later_store_and_call() {
        let level = Level::new();
        let stored_fn: fn(i32, i32) -> i32 =
            dispatch_for_later!(level, add_values, (a: i32, b: i32) -> i32);

        // Store and call multiple times
        assert_eq!(stored_fn(0, 0), 0);
        assert_eq!(stored_fn(1, 2), 3);
        assert_eq!(stored_fn(2, 4), 6);
        assert_eq!(stored_fn(3, 6), 9);
        assert_eq!(stored_fn(4, 8), 12);
    }

    mod no_import_simd_for_later {
        /// We should be able to use [`dispatch_for_later`] in a scope which doesn't import anything.
        #[inline(always)]
        fn add<S: crate::Simd>(_simd: S, a: i32, b: i32) -> i32 {
            a + b
        }

        #[test]
        fn dispatch_for_later_with_no_imports() {
            let fn_ptr = dispatch_for_later!(crate::Level::new(), add, (a: i32, b: i32) -> i32);
            assert_eq!(fn_ptr(1, 2), 3);
        }
    }
}
