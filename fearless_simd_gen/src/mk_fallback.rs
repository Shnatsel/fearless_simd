// Copyright 2025 the Fearless_SIMD Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::arch::fallback;
use crate::generic::{
    generic_from_bytes, generic_mask_from_bitmask, generic_mask_set, generic_mask_to_bitmask,
    generic_op_name, generic_to_bytes, integer_lane_mask_splat_arg, scalar_binary_op,
    scalar_compare_op, scalar_cvt, scalar_load_interleaved, scalar_store_interleaved,
    scalar_unary_op, scalar_unzip,
};
use crate::level::Level;
use crate::ops::{Op, OpSig, RefKind, valid_reinterpret};
use crate::types::{ScalarType, VecType};
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Clone, Copy)]
pub(crate) struct Fallback;

impl Level for Fallback {
    fn name(&self) -> &'static str {
        "Fallback"
    }

    fn native_width(&self) -> usize {
        128
    }

    fn max_block_size(&self) -> usize {
        512
    }

    fn enabled_target_features(&self) -> Option<&'static str> {
        None
    }

    fn arch_ty(&self, vec_ty: &VecType) -> TokenStream {
        let scalar_rust = vec_ty.scalar.rust(vec_ty.scalar_bits);
        let len = vec_ty.len;
        quote!([#scalar_rust; #len])
    }

    fn token_doc(&self) -> &'static str {
        r#"A token for scalar fallback SIMD, representing the "fallback" level."#
    }

    fn make_module_prelude(&self) -> TokenStream {
        quote! {
            use core::ops::*;

            #[cfg(all(feature = "libm", not(feature = "std")))]
            trait FloatExt {
                fn floor(self) -> Self;
                fn ceil(self) -> Self;
                fn round_ties_even(self) -> Self;
                fn fract(self) -> Self;
                fn sqrt(self) -> Self;
                fn trunc(self) -> Self;
            }
            #[cfg(all(feature = "libm", not(feature = "std")))]
            impl FloatExt for f32 {
                #[inline(always)]
                fn floor(self) -> f32 {
                    libm::floorf(self)
                }
                #[inline(always)]
                fn ceil(self) -> f32 {
                    libm::ceilf(self)
                }
                #[inline(always)]
                fn round_ties_even(self) -> f32 {
                    libm::rintf(self)
                }
                #[inline(always)]
                fn sqrt(self) -> f32 {
                    libm::sqrtf(self)
                }
                #[inline(always)]
                fn fract(self) -> f32 {
                    self - self.trunc()
                }
                #[inline(always)]
                fn trunc(self) -> f32 {
                    libm::truncf(self)
                }
            }

            #[cfg(all(feature = "libm", not(feature = "std")))]
            impl FloatExt for f64 {
                #[inline(always)]
                fn floor(self) -> f64 {
                    libm::floor(self)
                }
                #[inline(always)]
                fn ceil(self) -> f64 {
                    libm::ceil(self)
                }
                #[inline(always)]
                fn round_ties_even(self) -> f64 {
                    libm::rint(self)
                }
                #[inline(always)]
                fn sqrt(self) -> f64 {
                    libm::sqrt(self)
                }
                #[inline(always)]
                fn fract(self) -> f64 {
                    self - self.trunc()
                }
                #[inline(always)]
                fn trunc(self) -> f64 {
                    libm::trunc(self)
                }
            }
        }
    }

    fn make_level_body(&self) -> TokenStream {
        let level_tok = Self.token();
        quote! {
            #[cfg(feature = "force_support_fallback")]
            return Level::#level_tok(self);
            #[cfg(not(feature = "force_support_fallback"))]
            Level::baseline()
        }
    }

    fn make_impl_body(&self) -> TokenStream {
        quote! {
            #[inline]
            pub const fn new() -> Self {
                Self { _private: () }
            }
        }
    }

    fn make_method(&self, op: Op, vec_ty: &VecType) -> TokenStream {
        let Op { sig, method, .. } = op;
        let method_sig = op.simd_trait_method_sig(vec_ty);

        match sig {
            OpSig::Splat => {
                let num_elements = vec_ty.len;
                let normalize_mask = integer_lane_mask_splat_arg(vec_ty);
                quote! {
                    #method_sig {
                        #normalize_mask
                        [val; #num_elements].simd_into(self)
                    }
                }
            }
            OpSig::Unary => {
                if method == "approximate_recip" {
                    return quote! {
                        #method_sig {
                            1.0 / a
                        }
                    };
                }

                scalar_unary_op(method_sig, method, vec_ty)
            }
            OpSig::WidenNarrow { target_ty } => scalar_cvt(method_sig, vec_ty, &target_ty),
            OpSig::Binary => scalar_binary_op(method_sig, method, vec_ty),
            OpSig::Shift => {
                let items = make_list(
                    (0..vec_ty.len)
                        .map(|idx| {
                            let args = [lane(quote! { a }, vec_ty, idx), quote! { shift }];
                            let expr = fallback::expr(method, vec_ty, &args);
                            quote! { #expr }
                        })
                        .collect::<Vec<_>>(),
                );

                quote! {
                    #method_sig {
                        #items.simd_into(self)
                    }
                }
            }
            OpSig::Ternary => {
                if method == "mul_add" {
                    quote! {
                        #method_sig {
                            a.mul(b).add(c)
                        }
                    }
                } else if method == "mul_sub" {
                    quote! {
                        #method_sig {
                            a.mul(b).sub(c)
                        }
                    }
                } else {
                    let args = [
                        quote! { a.into() },
                        quote! { b.into() },
                        quote! { c.into() },
                    ];

                    let expr = fallback::expr(method, vec_ty, &args);
                    quote! {
                        #method_sig {
                            #expr.simd_into(self)
                        }
                    }
                }
            }
            OpSig::Compare => scalar_compare_op(method_sig, method, vec_ty),
            OpSig::Select => {
                let mask_type = vec_ty.mask_ty();
                let items = make_list(
                    (0..vec_ty.len)
                        .map(|idx| {
                            let a = lane(quote! { a }, &mask_type, idx);
                            let b = lane(quote! { b }, vec_ty, idx);
                            let c = lane(quote! { c }, vec_ty, idx);
                            quote! { if #a != 0 { #b } else { #c } }
                        })
                        .collect::<Vec<_>>(),
                );

                quote! {
                    #method_sig {
                        #items.simd_into(self)
                    }
                }
            }
            OpSig::Combine { combined_ty } => {
                let n = vec_ty.len;
                let n2 = combined_ty.len;
                let default = match vec_ty.scalar {
                    ScalarType::Float => quote! { 0.0 },
                    _ => quote! { 0 },
                };
                quote! {
                    #method_sig {
                        let mut result = [#default; #n2];
                        result[0..#n].copy_from_slice(&a.val.0);
                        result[#n..#n2].copy_from_slice(&b.val.0);
                        result.simd_into(self)
                    }
                }
            }
            OpSig::Split { half_ty } => {
                let n = vec_ty.len;
                let nhalf = half_ty.len;
                let default = match vec_ty.scalar {
                    ScalarType::Float => quote! { 0.0 },
                    _ => quote! { 0 },
                };
                quote! {
                    #method_sig {
                        let mut b0 = [#default; #nhalf];
                        let mut b1 = [#default; #nhalf];
                        b0.copy_from_slice(&a.val.0[0..#nhalf]);
                        b1.copy_from_slice(&a.val.0[#nhalf..#n]);
                        (b0.simd_into(self), b1.simd_into(self))
                    }
                }
            }
            OpSig::Zip { select_low } => {
                let indices = if select_low {
                    0..vec_ty.len / 2
                } else {
                    (vec_ty.len / 2)..vec_ty.len
                };

                let zip = make_list(
                    indices
                        .map(|idx| {
                            let a = lane(quote! { a }, vec_ty, idx);
                            let b = lane(quote! { b }, vec_ty, idx);
                            quote! { #a, #b }
                        })
                        .collect::<Vec<_>>(),
                );

                quote! {
                    #method_sig {
                        #zip.simd_into(self)
                    }
                }
            }
            OpSig::Unzip { select_even } => scalar_unzip(method_sig, vec_ty, select_even),
            OpSig::Slide { .. } => {
                let n = vec_ty.len;
                quote! {
                    #method_sig {
                        let mut dest = [Default::default(); #n];
                        dest[..#n - SHIFT].copy_from_slice(&a.val.0[SHIFT..]);
                        dest[#n - SHIFT..].copy_from_slice(&b.val.0[..SHIFT]);
                        dest.simd_into(self)
                    }
                }
            }
            OpSig::Cvt {
                target_ty,
                scalar_bits,
                precise,
            } => {
                if precise {
                    let non_precise =
                        generic_op_name(method.strip_suffix("_precise").unwrap(), vec_ty);
                    quote! {
                        #method_sig {
                            self.#non_precise(a)
                        }
                    }
                } else {
                    let to_ty = vec_ty.reinterpret(target_ty, scalar_bits);
                    scalar_cvt(method_sig, vec_ty, &to_ty)
                }
            }
            OpSig::Reinterpret {
                target_ty,
                scalar_bits,
            } => {
                if valid_reinterpret(vec_ty, target_ty, scalar_bits) {
                    quote! {
                        #method_sig {
                            a.bitcast()
                        }
                    }
                } else {
                    quote! {}
                }
            }
            OpSig::MaskReduce {
                quantifier,
                condition,
            } => {
                let check = if condition {
                    quote! { != }
                } else {
                    quote! { == }
                };

                let expr = match quantifier {
                    crate::ops::Quantifier::Any => {
                        let lanes = (0..vec_ty.len).map(|idx| lane(quote! { a }, vec_ty, idx));
                        quote! { #(#lanes #check 0)||* }
                    }
                    crate::ops::Quantifier::All => {
                        let lanes = (0..vec_ty.len).map(|idx| lane(quote! { a }, vec_ty, idx));
                        quote! { #(#lanes #check 0)&&* }
                    }
                };

                quote! {
                    #method_sig {
                        #expr
                    }
                }
            }
            OpSig::MaskFromBitmask => generic_mask_from_bitmask(method_sig, vec_ty),
            OpSig::MaskToBitmask => generic_mask_to_bitmask(method_sig, vec_ty),
            OpSig::MaskSet => generic_mask_set(method_sig, vec_ty),
            OpSig::LoadInterleaved {
                block_size,
                block_count,
            } => scalar_load_interleaved(method_sig, vec_ty, block_size, block_count),
            OpSig::StoreInterleaved {
                block_size,
                block_count,
            } => scalar_store_interleaved(method_sig, vec_ty, block_size, block_count),
            OpSig::FromArray { kind } => {
                let vec_rust = vec_ty.rust();
                let wrapper = vec_ty.aligned_wrapper();
                let expr = match kind {
                    RefKind::Value => quote! { val },
                    RefKind::Ref | RefKind::Mut => quote! { *val },
                };
                quote! {
                    #method_sig {
                        #vec_rust { val: #wrapper(#expr), simd: self }
                    }
                }
            }
            OpSig::AsArray { kind } => {
                let ref_tok = kind.token();
                quote! {
                    #method_sig {
                        #ref_tok a.val.0
                    }
                }
            }
            OpSig::StoreArray => {
                quote! {
                    #method_sig {
                        *dest = a.val.0;
                    }
                }
            }
            OpSig::FromBytes => generic_from_bytes(method_sig, vec_ty),
            OpSig::ToBytes => generic_to_bytes(method_sig, vec_ty),
            OpSig::Interleave => {
                let zip_low = generic_op_name("zip_low", vec_ty);
                let zip_high = generic_op_name("zip_high", vec_ty);
                quote! {
                    #method_sig {
                        (self.#zip_low(a, b), self.#zip_high(a, b))
                    }
                }
            }
            OpSig::Deinterleave => {
                let unzip_low = generic_op_name("unzip_low", vec_ty);
                let unzip_high = generic_op_name("unzip_high", vec_ty);
                quote! {
                    #method_sig {
                        (self.#unzip_low(a, b), self.#unzip_high(a, b))
                    }
                }
            }
        }
    }

    fn make_type_impl(&self) -> TokenStream {
        TokenStream::new()
    }
}

fn lane(value: TokenStream, vec_ty: &VecType, idx: usize) -> TokenStream {
    if vec_ty.scalar == ScalarType::Mask {
        quote! { #value.val.0[#idx] }
    } else {
        quote! { #value[#idx] }
    }
}

fn make_list(items: Vec<TokenStream>) -> TokenStream {
    quote!([#( #items, )*])
}
