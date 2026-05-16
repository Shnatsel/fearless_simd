use fearless_simd::{Fallback, SimdCvtFloat, SimdCvtTruncate, f32x4, i32x4};

pub struct DownstreamInput;

impl SimdCvtTruncate<DownstreamInput> for i32x4<Fallback> {
    fn truncate_from(_: DownstreamInput) -> Self {
        unimplemented!()
    }

    fn truncate_from_precise(_: DownstreamInput) -> Self {
        unimplemented!()
    }
}

impl SimdCvtFloat<DownstreamInput> for f32x4<Fallback> {
    fn float_from(_: DownstreamInput) -> Self {
        unimplemented!()
    }
}
