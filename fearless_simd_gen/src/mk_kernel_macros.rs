// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{level::Level, mk_neon::Neon, mk_wasm::WasmSimd128, mk_x86::X86};

/// This emits a String rather than a TokenStream
/// because rustfmt just gives up formatting macros
/// and we end up with a completely unreadable token soup
/// if we don't impose formatting on it manually.
pub(crate) fn mk_kernel_macros() -> String {
    [
        kernel_macro(&Neon),
        kernel_macro(&WasmSimd128),
        kernel_macro(&X86::Sse4_2),
        kernel_macro(&X86::Avx2),
    ]
    .join("\n")
}

fn kernel_macro(level: &dyn Level) -> String {
    let macro_name = format!("{}_kernel", snake_case(level.name()));
    let name = level.name();
    let cfg = level
        .availability_cfg()
        .expect("kernel macros should only be generated for cfg-gated SIMD levels");
    let body = kernel_body(level, KernelGenerics::None);
    let generic_body = kernel_body(level, KernelGenerics::Simple);

    KERNEL_MACRO_TEMPLATE
        .replace("@MACRO_NAME@", &macro_name)
        .replace("@LEVEL_NAME@", name)
        .replace("@CFG@", cfg)
        .replace("@BODY@", &body)
        .replace("@GENERIC_BODY@", &generic_body)
}

#[derive(Clone, Copy)]
enum KernelGenerics {
    None,
    Simple,
}

impl KernelGenerics {
    fn declaration(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Simple => {
                r#"<
                $($generic $(: $generic_bound)?),+
            >"#
            }
        }
    }

    fn call(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Simple => "::<$($generic),+>",
        }
    }
}

fn kernel_body(level: &dyn Level, generics: KernelGenerics) -> String {
    let (attrs, call) = if let Some(features) = level.enabled_target_features() {
        (
            format!(
                r#"            #[inline]
            #[target_feature(enable = "{features}")]"#
            ),
            KERNEL_CALL_WITH_TARGET_FEATURES.replace("@LEVEL_NAME@", level.name()),
        )
    } else {
        ("            #[inline]".to_string(), KERNEL_CALL.to_string())
    };

    KERNEL_BODY_TEMPLATE
        .replace("@ATTRS@", &attrs)
        .replace("@GENERICS@", generics.declaration())
        .replace("@CALL@", &call)
        .replace("@CALL_GENERICS@", generics.call())
}

fn snake_case(name: &str) -> String {
    let mut result = String::new();
    let mut prev_was_lowercase = false;
    for ch in name.chars() {
        if ch == '_' {
            result.push(ch);
            prev_was_lowercase = false;
        } else if ch.is_uppercase() {
            if prev_was_lowercase {
                result.push('_');
            }
            result.extend(ch.to_lowercase());
            prev_was_lowercase = false;
        } else {
            result.push(ch);
            prev_was_lowercase = ch.is_lowercase();
        }
    }
    result
}

const KERNEL_MACRO_TEMPLATE: &str = r#"
#[doc = "Defines a safe kernel for `@LEVEL_NAME@`."]
#[doc = ""]
#[doc = "Generic kernels only accept type parameters with no bound or one path-like bound, such as `<S: Simd, T: Copy>`."]
#[doc = ""]
#[doc = "Kernel macros only accept safe functions."]
#[doc = ""]
#[doc = "```compile_fail"]
#[doc = "fearless_simd::@MACRO_NAME@! {"]
#[doc = "    unsafe fn should_not_compile() {}"]
#[doc = "}"]
#[doc = "```"]
#[doc = ""]
#[doc = "```compile_fail"]
#[doc = "fearless_simd::@MACRO_NAME@! {"]
#[doc = "    fn should_not_compile<T: Copy + Clone>(x: T) -> T { x }"]
#[doc = "}"]
#[doc = "```"]
#[macro_export]
macro_rules! @MACRO_NAME@ {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident <
            $($generic:ident $(: $generic_bound:path)?),+ $(,)?
        >(
            $($arg:ident : $arg_ty:ty),* $(,)?
        ) $(-> $ret:ty)? {
            $($kernel_body:tt)*
        }
    ) => {
        #[cfg(@CFG@)]
        $(#[$meta])*
        $vis fn $name<
            $($generic $(: $generic_bound)?),+
        >(
            _simd: $crate::@LEVEL_NAME@,
            $($arg: $arg_ty),*
        ) $(-> $ret)? {
@GENERIC_BODY@
        }
    };

    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident(
            $($arg:ident : $arg_ty:ty),* $(,)?
        ) $(-> $ret:ty)? {
            $($kernel_body:tt)*
        }
    ) => {
        #[cfg(@CFG@)]
        $(#[$meta])*
        $vis fn $name(
            _simd: $crate::@LEVEL_NAME@,
            $($arg: $arg_ty),*
        ) $(-> $ret)? {
@BODY@
        }
    };

    ($($unsupported:tt)*) => {
        compile_error!(
            "kernel macros support only safe functions with non-generic signatures or simple type generics like `<S: Simd, T: Copy>`; use a wrapper function for lifetimes, const generics, `where` clauses, or multiple bounds"
        );
    };
}
"#;

const KERNEL_BODY_TEMPLATE: &str = r#"@ATTRS@
            fn __fearless_simd_kernel@GENERICS@(
                $($arg: $arg_ty),*
            ) $(-> $ret)? {
                $($kernel_body)*
            }

@CALL@"#;

const KERNEL_CALL_WITH_TARGET_FEATURES: &str = r#"            // SAFETY: the `@LEVEL_NAME@` token proves that the required target features are available.
            unsafe { __fearless_simd_kernel@CALL_GENERICS@($($arg),*) }"#;

const KERNEL_CALL: &str = r#"            let _ = _simd;
            __fearless_simd_kernel@CALL_GENERICS@($($arg),*)"#;
