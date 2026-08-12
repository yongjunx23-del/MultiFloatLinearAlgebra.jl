# Float64x3 MFIR fused `mulacc` prototype

Date: 2026-08-12

## Scope

This experiment is benchmark-only. It does **not** modify the production
`MultiFloats.jl` arithmetic or the `MultiFloatLinearAlgebra.jl` GEMM kernels.

The target operation is the hot dot/GEMM recurrence

```text
current:
    product = x * y
    acc     = acc + product
```

for `MultiFloatVec{4,Float64,3}`.  The current path fully compresses the x3
product and then runs the full x3 addition/compression network.  The prototype
keeps the established x3 multiplication prefix but cuts away a suffix of the
product-only compression before feeding the remaining limbs into the normal x3
addition network.

A small benchmark-local MFIR-compatible subset models the tail DAG.  It mirrors
the relevant `MultiFloats.jl/scripts/MFIR.jl` operation/program/use-count
semantics without adding scripts or an IR dependency to the package runtime.

## MFIR fusion cuts

The common multiplication prefix is excluded from the table because it is
identical for all candidates.

| stage | tail MFIR instructions | peak live values | macro critical depth |
|---|---:|---:|---:|
| current | 20 | 6 | 14 |
| conservative | 18 | 6 | 12 |
| mid | 17 | 6 | 11 |
| aggressive | 16 | 6 | 10 |

`aggressive` removes the final four product-compression EFT operations before
entering the x3 `mfadd` network.  `mid` restores one `TwoSum`; `conservative`
restores that `TwoSum` plus one `FastTwoSum`.

## BigFloat differential validation

Validation uses 512-bit BigFloat references and true three-limb inputs generated
from BigFloat values rather than Float64-only inputs.  The one-step test covers:

- random moderate-scale inputs;
- wide dynamic range;
- product/accumulator cancellation;
- alternating signs;
- overflow/underflow-safe edge exponents.

Errors are scaled by `|acc| + |x*y|`.  Dot tests use the same five modes and
scale by the exact BigFloat sum of product magnitudes, giving a backward-error
style comparison.

### Aggressive fusion cut

Across all five one-step suites:

- maximum scaled error ratio versus the current path: **1.000**;
- cases with larger error than the current path: **0**;
- non-normalized outputs: **0**.

Representative maximum scaled errors were identical to the current path:

| mode | current | aggressive |
|---|---:|---:|
| random | 1.030e-48 | 1.030e-48 |
| wide | 8.027e-49 | 8.027e-49 |
| cancellation | 2.954e-49 | 2.954e-49 |
| alternating | 8.821e-49 | 8.821e-49 |
| edge | 8.730e-49 | 8.730e-49 |

The 512-term Vec4 dot differential test also produced a maximum error ratio of
**1.000** in every mode and all outputs remained normalized:

| mode | current | aggressive |
|---|---:|---:|
| random | 2.044e-48 | 2.044e-48 |
| wide | 7.067e-49 | 7.067e-49 |
| cancellation | 3.898e-50 | 3.898e-50 |
| alternating | 3.048e-48 | 3.048e-48 |
| edge | 3.158e-49 | 3.158e-49 |

`mid` and `conservative` produced the same differential-error results on these
suites.  This does not yet constitute a formal proof or a claim of universal
bitwise equivalence; it is evidence that the removed product-tail compression is
redundant for the tested product-then-add use case.

## Vec4 dot throughput

GitHub-hosted Ubuntu x86-64 runner, Julia 1.12.6, one Julia thread, median of
nine samples over 8192 `MultiFloatVec{4}` terms.  Times are normalized per
logical scalar product.

| candidate | current | fused | speedup | production gate |
|---|---:|---:|---:|---|
| aggressive | 6.125 ns | 5.648 ns | **1.084x** | PASS |
| mid | 6.236 ns | 5.857 ns | **1.065x** | PASS |
| conservative | 6.453 ns | 6.151 ns | 1.049x | FAIL |

The production gate for this experiment requires both:

1. differential accuracy no worse than 1.05x the current maximum scaled error,
   with normalized outputs; and
2. more than 5% Vec4 dot speedup.

The aggressive and mid cuts pass; aggressive is the stronger candidate.

## Native code

For the one-step Vec4 `mulacc` wrapper on the same runner:

| path | approximate native instructions | spill/reload | calls | YMM registers used |
|---|---:|---:|---:|---:|
| current | 128 | 0 | 0 | 10 |
| aggressive | 113 | 0 | 0 | 10 |
| mid | 119 | 0 | 0 | 10 |
| conservative | 122 | 0 | 0 | 10 |

This supports the intended mechanism: the gain comes from removing redundant
arithmetic/compression work, not from changing packing, allocation, or function
inlining behavior.

## Decision

The x3 fused `mulacc` research hypothesis is **validated at the dot-kernel
level**.  The aggressive cut is the preferred candidate for the next gate.

It is intentionally **not** wired into production GEMM yet.  The next required
experiment is a real x3 GEMM A/B at `n = 256, 512, 1024`, using the exact same
matrix data and threading configuration for current and fused kernels.  Only if
matrix-level throughput clears a stable 5% gate while BigFloat/backward-error
checks remain unchanged should a production `mulacc_x3` hook be introduced.

Reproduction:

```bash
julia -t 1 --project=benchmark benchmark/fused_mulacc_x3.jl
```
