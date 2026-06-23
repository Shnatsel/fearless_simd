I wrote a temporary probe crate here: [lib.rs](/tmp/fearless_simd_asm_probe/src/lib.rs:1).

Compiled with:

```sh
RUSTFLAGS="-C target-cpu=x86-64 -C codegen-units=1 -C llvm-args=-x86-asm-syntax=intel" \
cargo rustc --release --lib -- --emit=asm
```

Assembly output: `/tmp/fearless_simd_asm_probe/target/release/deps/fearless_simd_asm_probe-*.s`

**Findings**
For the common straight-line ops, fallback autovectorization is excellent. SSE2 and fallback generate identical or effectively identical assembly for:

| Probe | Fallback | SSE2 | Result |
|---|---:|---:|---|
| `f32x4` load/store | 3 instr | 3 instr | identical `movups` |
| `f32x4` add | 5 | 5 | identical `addps` |
| `f32x4` mul-add | 7 | 7 | identical `mulps` + `addps` |
| `i32x4` add | 5 | 5 | identical `paddd` |
| `i32x4` mul | 11 | 11 | identical SSE2 `pmuludq` shuffle sequence |
| `i32x4` max | 9 | 9 | same compare/select shape |
| `u8x16` add/max | 5 | 5 | identical `paddb` / `pmaxub` |

Places where SSE2 is clearly better:

| Probe | Fallback | SSE2 | Notes |
|---|---:|---:|---|
| `f32x4` compare+select | 25 | 10 | SSE2 emits clean `cmpltps` + bitwise select; fallback partially scalarizes |
| `i32x4` compare+select | 20 | 9 | SSE2 emits clean `pcmpgtd` + bitwise select |
| `f32x16` interleaved load | 29 | 21 | SSE2 unpack sequence is cleaner |
| `f32x16` interleaved store | 29 | 21 | same improvement |

Important sore spot:

| Probe | Fallback | SSE2 | Notes |
|---|---:|---:|---|
| `u8x64` interleaved load | 241 | alias to fallback | huge scalar byte shuffle |
| `u8x64` interleaved store | 245 | 245 | same huge scalar byte shuffle |

So: the SSE2 backend mostly does **not** improve basic f32/i32/u8 arithmetic over fallback autovectorization, because LLVM already nails those. It **does** help for mask select and f32 interleaving. The big remaining opportunity is 8-bit/16-bit interleaved load/store: our scalar fallback there is very expensive, and SSE2 probably deserves a real unpack-based implementation if those APIs matter.
