# Raw MultiFloat SIMD arithmetic profile

Date: 2026-08-12

## Purpose

This report isolates the arithmetic cost underneath dense MultiFloat GEMM before
changing any production arithmetic network. The goal is to distinguish four
possible bottlenecks:

1. missing inlining or compiler calls;
2. cache/panel traffic;
3. register pressure and spills from multiple GEMM accumulators;
4. the length and dependency depth of the MultiFloat EFT/compression network
   itself.

The profiling code lives in `benchmark/arithmetic_profile.jl` and
`benchmark/accumulator_pressure.jl`. Measurements below are directional results
from GitHub-hosted Ubuntu 24.04 x86-64 runners with Julia 1.12.6. Runners are
not dedicated machines, so small differences should not be interpreted as
portable architecture constants.

## Source-level arithmetic growth

`MultiFloats.jl` v3.2.6 implements x2/x3/x4 arithmetic as explicit fixed
networks of `two_prod`, `two_sum`, `fast_two_sum`, ordinary products/additions,
and FMA-based error recovery. The network grows rapidly with limb count:

- x2 multiplication starts from one `two_prod`, two cross-products, and a
  short compression sequence;
- x3 adds the next convolution diagonal plus substantially more error
  propagation/compression;
- x4 adds another diagonal and a much longer set of `two_prod`, `two_sum`, and
  `fast_two_sum` dependencies.

The same pattern appears in `mfadd`: x3 and especially x4 spend much more work
compressing the result back into a normalized fixed-width expansion.

Importantly, ordinary GEMM arithmetic does **not** call the generic iterative
`renormalize` routine after every scalar operation. `mfadd`/`mfmul` already
contain their own fixed compression networks. The explicit renormalization
measurements below are therefore diagnostic bounds rather than a decomposition
that should be added directly to GEMM time.

## Runtime arithmetic cost

Representative single-thread profile. Values are normalized per logical scalar
term even for `MultiFloatVec{4}` so scalar and four-lane SIMD paths can be
compared directly.

| operation | Float64x2 | Float64x3 | Float64x4 |
|---|---:|---:|---:|
| scalar add | 1.42 ns | 3.98 ns | 7.48 ns |
| scalar mul | 1.28 ns | 2.83 ns | 7.60 ns |
| scalar mul + add | 2.15 ns | 7.47 ns | 16.80 ns |
| scalar dot-style accumulate | 11.38 ns | 26.84 ns | 64.22 ns |
| Vec4 add | 1.12 ns | 2.99 ns | 6.21 ns |
| Vec4 mul | 1.52 ns | 3.97 ns | 6.92 ns |
| Vec4 mul + add | 1.85 ns | 6.49 ns | 16.17 ns |
| Vec4 dot-style accumulate | 2.34 ns | 6.35 ns | 16.10 ns |
| Vec4 renormalize, already clean | 0.90 ns | 1.38 ns | 1.83 ns |
| Vec4 renormalize, deliberately overlapping | 1.15 ns | 1.94 ns | 4.13 ns |

The dominant observation is not memory traffic: x3/x4 arithmetic itself grows
very quickly. Relative to x2, the representative Vec4 dot-style accumulation
cost is about 2.7x for x3 and 6.9x for x4.

## Native-code inspection

`code_native`/`code_llvm` probes were collected for scalar and `MultiFloatVec`
add, multiply, multiply-plus-add, and renormalization operations.

### Inlining

The optimized x2/x3/x4 add/mul/mul+add wrappers contained no arithmetic helper
function calls. LLVM inlined the MultiFloat networks into straight-line native
code. Therefore the x3/x4 slowdown is **not** primarily an abstraction or
function-call overhead problem.

### Instruction and register growth

Approximate native instruction counts for the four-lane operations on the AVX2
runner were:

| Vec4 operation | x2 | x3 | x4 |
|---|---:|---:|---:|
| add | ~30 | ~76 | ~140 |
| mul | ~17 | ~57 | ~143 |
| mul + add | ~39 | ~123 | ~271 |

The maximum YMM register indices used by the single-operation probes also grew
strongly:

- x2 mul+add: roughly `ymm0` through `ymm6`;
- x3 mul+add: roughly `ymm0` through `ymm9`;
- x4 mul+add: roughly `ymm0` through `ymm14`.

Single arithmetic operations themselves did not show meaningful spill/reload
traffic. x4 nevertheless comes very close to exhausting the 16 architectural
YMM registers available on AVX2.

A rough dependency-depth analysis of the generated LLVM shows the same trend:
the x4 multiply-plus-add chain is several times deeper than x2. The exact cycle
numbers are only heuristics (not `llvm-mca` throughput predictions), but the
qualitative conclusion is stable: x3/x4 are increasingly **EFT critical-path
bound**.

## GEMM accumulator-pressure experiment

To separate register pressure from dependency latency, a dot-like inner loop
was measured with one accumulator, two interleaved accumulators, two accumulators
computed in separate loops, and four interleaved accumulators. Times are per
scalar product.

| arithmetic | one accumulator | two interleaved | two split loops | four interleaved |
|---|---:|---:|---:|---:|
| Float64x2 | 2.63 ns | 1.38 ns | 2.71 ns | 1.31 ns |
| Float64x3 | 7.25 ns | 6.66 ns | 7.16 ns | 6.42 ns |
| Float64x4 | 17.64 ns | 16.19 ns | 17.51 ns | 15.70 ns |

The multi-accumulator x4 loops do contain substantial spill/reload traffic, but
reducing the number of accumulators is still slower. Independent accumulators
supply instruction-level parallelism that hides part of the long EFT latency.

Therefore:

- register spilling is a cost, but it is **not** the dominant reason x4 is
  slow;
- a single-accumulator x4 GEMM would be the wrong optimization;
- future arithmetic work should shorten the product-accumulation dependency
  chain while preserving enough independent accumulators for ILP.

## Rejected primitive rewrite: magnitude-ordered FastTwoSum

A low-risk experiment replaced general `TwoSum(a,b)` calls inside x3/x4 mfadd
with a lane-wise magnitude comparison followed by `FastTwoSum(large, small)`.
On the finite stress sets, including cancellation-heavy inputs, the primitive
returned the same `(sum,error)` bitwise, and the reconstructed x3/x4 additions
were bitwise equal to the current `mfadd` and remained normalized.

Performance rejected the approach:

| arithmetic | current Vec4 add | ordered-FastTwoSum add | speedup |
|---|---:|---:|---:|
| Float64x3 | 3.32 ns | 11.53 ns | 0.29x |
| Float64x4 | 6.89 ns | 6.69 ns | 1.03x |

The compare/select overhead is catastrophic for x3 and the x4 gain is only
about 3%, below the package's 5% optimization gate. This rewrite is not a
production candidate.

## Direct GEMM scheduling experiment: two versus four output columns

The accumulator-pressure result suggested that more independent output
accumulators could hide EFT latency. A benchmark-only direct GEMM therefore
computed four output columns at once while preserving the exact ascending-`k`
reduction order for every output element. All candidates were required to be
bitwise equal to the established direct GEMM result.

### n=512

| arithmetic | existing two-column direct | four-column direct | speedup |
|---|---:|---:|---:|
| Float64x2 | 83.468 ms | 73.113 ms | 1.142x |
| Float64x3 | 327.903 ms | 323.281 ms | 1.014x |
| Float64x4 | 704.741 ms | 693.365 ms | 1.016x |

### n=1024

| arithmetic | existing two-column direct | four-column direct | speedup |
|---|---:|---:|---:|
| Float64x2 | 755.027 ms | 654.427 ms | 1.154x |
| Float64x3 | 2790.931 ms | 2782.394 ms | 1.003x |
| Float64x4 | 6126.323 ms | 6142.099 ms | 0.997x |

This is a genuine scheduling opportunity for x2, but not for x3/x4. The x2
candidate should be compared against B-packed GEMM in the **same benchmark
run** before changing the machine calibration policy because hosted-runner
variance is large enough to invalidate comparisons across runs. A production
x2 direct-four kernel is therefore a separate follow-up, not part of this raw
x3/x4 arithmetic decision.

For x3/x4, simple output-column unrolling has now also been exhausted: it does
not clear the 5% gate at either 512 or 1024.

## Decision

The profiling campaign rules out several attractive but incorrect diagnoses:

1. **Not missing inlining.** The arithmetic networks are already inlined.
2. **Not primarily panel/cache traffic.** Previous B-packing barely changes
   x3/x4 throughput at 512/1024.
3. **Not simply too many accumulators.** Multiple accumulators improve x3/x4
   throughput despite spill traffic.
4. **Not fixed by a local TwoSum substitution.** Magnitude-ordered FastTwoSum
   fails the performance gate.
5. **Not fixed by more output-column ILP in GEMM.** Four-column scheduling is
   valuable for x2 only.

The next meaningful x3/x4 experiment is therefore a **specialized fused
multiply-accumulate / dot arithmetic network** that avoids fully compressing an
N-limb product and then immediately feeding that product through a second full
N-limb addition/compression network:

```text
current:
    p   = mfmul(x, y)      # product -> N-limb compression
    acc = mfadd(acc, p)    # second N-limb compression

target experiment:
    acc = fused_mulacc(acc, x, y)
          # one combined expansion/compression network
```

This changes intermediate rounding semantics, so bitwise equality is no longer
the right acceptance criterion. A fused network must instead pass:

- normalized output invariants;
- BigFloat differential error no worse than the current path;
- random, dynamic-range, and cancellation-heavy adversarial sets;
- dot/GEMM backward-error tests;
- stable >5% improvement in Vec4 dot accumulation and 512/1024 GEMM before it
  enters production.

`MultiFloats.jl` already contains the `scripts/MFIR.jl` arithmetic-network IR,
including explicit `TWO_SUM`, `FAST_TWO_SUM`, `MUL`, `FMA`, `TWO_PROD`,
`definition_map`, and `use_counts`. It is a good starting representation for a
critical-path/live-range-aware fused-network prototype. MFIR currently provides
program representation and mutation utilities, not a complete scheduler, so a
small dependency/live-range scheduler will still need to be added around it.

## Recommended next implementation

1. Build an MFIR representation of the existing x3 `mfmul -> mfadd` reference
   path and compute dependency depth / use counts / peak live temporaries.
2. Construct an x3 fused `mulacc` candidate that carries product residual terms
   directly into the accumulator compression network instead of materializing
   a normalized x3 product first.
3. Generate a benchmark-only Julia `MultiFloatVec{4,Float64,3}` kernel.
4. Validate against BigFloat and current x3 dot/GEMM over adversarial inputs.
5. Only after x3 clears correctness and the 5% throughput gate, repeat the
   procedure for x4.
6. Keep the production `MultiFloats.jl` arithmetic untouched until this evidence
   exists.
