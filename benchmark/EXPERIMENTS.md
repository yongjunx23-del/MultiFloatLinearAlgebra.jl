# Rejected GEMM experiments

Date: 2026-08-12

These experiments are kept as negative evidence so future optimization work does not repeat approaches that failed the 512/1024 throughput gate on the GitHub-hosted x86-64 runner (Julia 1.12.6, four Julia threads).

## Dense A+B packing

A specialization that packed every complete four-row A group into `MultiFloatVec` storage while also packing B was numerically correct, but it repacked A on every call and made the 512/1024 calibration campaign several times slower before establishing any stable throughput benefit. It was removed. The retained packed implementation therefore packs B only and reuses A loads inside the register microkernel.

## Float64x3 three-column microkernel

The existing generic `Val(3)` path was first tested and was clearly unsuitable as a hot microkernel: at n=512 it took roughly 5.3-6.0 s versus about 0.34 s for the existing 32x2 packed route because the generic `ntuple` formulation did not optimize like the straight-line Val2/Val4 kernels.

A fair follow-up used an explicit straight-line three-accumulator `MultiFloatVec{4,Float64,3}` kernel. Every candidate produced exactly the same per-output reduction result as direct GEMM.

| n | route | time | relative to direct |
|---:|---|---:|---:|
| 512 | direct 32x2 | 0.310819 s | 1.000x |
| 512 | packed 32x2 | 0.311742 s | 0.997x |
| 512 | packed 24x3 | 0.326696 s | 0.951x |
| 512 | packed 30x3 | 0.337457 s | 0.921x |
| 512 | packed 32x3 | 0.310337 s | 1.002x |
| 1024 | direct 32x2 | 2.480616 s | 1.000x |
| 1024 | packed 32x2 | 2.541883 s | 0.976x |
| 1024 | packed 24x3 | 2.511187 s | 0.988x |
| 1024 | packed 30x3 | 2.524392 s | 0.983x |
| 1024 | packed 32x3 | 2.517041 s | 0.986x |

Conclusion: micro=3 does not clear the package's 5% calibration margin at either scale and regresses at n=1024. It is not promoted into `KernelConfig` or the production microkernel set. On this runner Float64x3 remains on the direct route; further x3 work should target arithmetic scheduling / raw MultiFloat kernel cost rather than additional panel packing or output-column fanout.
