# MultiFloatLinearAlgebra benchmark report

Date: 2026-08-12

## Scope and environment

This report records the first reproducible CPU results for the standalone
`MultiFloats.jl` backend. The measurements were produced on GitHub-hosted
Ubuntu 24.04 runners with Julia 1.12.6, four Julia threads, and one BLAS
thread. GitHub runners are not dedicated machines, so the numbers are a
route-selection and regression baseline rather than an HPC-node performance
claim.

The benchmarked fixed-width arithmetic types are `Float64x2`, `Float64x3`,
and `Float64x4`. Every optimized kernel preserves each output element's
ascending reduction order and is checked against the direct backend result.

## Dense baseline at n=256

| Operation | Arithmetic | Generic `LinearAlgebra` | MFLA | Speedup |
|---|---|---:|---:|---:|
| GEMM | Float64x2 | 143.178 ms | 9.902 ms | 14.46x |
| GEMM | Float64x3 | 546.941 ms | 38.884 ms | 14.07x |
| GEMM | Float64x4 | 1151.840 ms | 95.980 ms | 12.00x |
| TRSM | Float64x2 | 5.875 ms | 1.323 ms | 4.44x |
| TRSM | Float64x3 | 27.150 ms | 5.562 ms | 4.88x |
| TRSM | Float64x4 | 66.849 ms | 8.486 ms | 7.88x |
| Cholesky | Float64x2 | 14.413 ms | 6.395 ms | 2.25x |
| Cholesky | Float64x3 | 67.604 ms | 13.115 ms | 5.15x |
| Cholesky | Float64x4 | 171.889 ms | 28.143 ms | 6.11x |
| LU | Float64x2 | 31.382 ms | 11.914 ms | 2.63x |
| LU | Float64x3 | 135.084 ms | 20.839 ms | 6.48x |
| LU | Float64x4 | 344.422 ms | 45.793 ms | 7.52x |

## Packed-panel GEMM

The retained packed path packs only the B panel into caller-owned reusable
storage. An experimental dense A+B packing specialization was removed because
it allocated and repacked A on every call and made the 512/1024 calibration
campaign several times slower. The B-only route retains the straight-line
`Val{1}`, `Val{2}`, and `Val{4}` microkernels.

### n=512

| Arithmetic | Generic `LinearAlgebra` | Direct MFLA | Packed MFLA | Packed/direct |
|---|---:|---:|---:|---:|
| Float64x2 | 1.168148 s | 0.087943 s | 0.075653 s | 1.16x |
| Float64x3 | 4.399648 s | 0.312891 s | 0.313175 s | 1.00x |
| Float64x4 | 8.451718 s | 0.701760 s | 0.690191 s | 1.02x |

### n=1024

| Arithmetic | Direct MFLA | Packed MFLA | Packed/direct |
|---|---:|---:|---:|
| Float64x2 | 0.741636 s | 0.665380 s | 1.11x |
| Float64x3 | 2.738718 s | 2.781304 s | 0.98x |
| Float64x4 | 6.167091 s | 6.140189 s | 1.00x |

### Routing decision

The built-in uncalibrated profile deliberately keeps `gemm_strategy=:auto`
on the direct route. `calibrate_gemm` is explicit, has no process-global
state, and by default tests 512 and 1024 with a 1.05 minimum speedup. It chooses
a candidate by the largest tested size and accepts a crossover only when all
larger tested points form a stable winning suffix.

On this runner the resulting policy is:

| Arithmetic | Candidate geometry | Calibrated route |
|---|---|---|
| Float64x2 | panel 32, micro 4 | packed from n=512 |
| Float64x3 | panel 32, micro 2 | direct |
| Float64x4 | panel 16, micro 2 | direct |

This is machine evidence, not a portable architecture default. Apple Silicon
and EPYC nodes should run the manual calibration workflow and apply the
returned `GemmProfile` with `with_gemm_profile`.

## Blocked symmetric-indefinite LDLT

The blocked path retains Bunch--Kaufman-style 1x1/2x2 pivot decisions. A panel
computes `W = L*D`, and the trailing symmetric update is delegated to the
shared lower-triangular `gemmt!` primitive:

```text
A22_lower <- A22_lower - L21 * W21'
```

Only the lower triangle performs multiplication; an inexpensive mirror copy
restores the dense symmetric view used by subsequent pivot searches. The
measured default panel widths are 16/12/8 for x2/x3/x4, and `:auto` switches
from unblocked to blocked at n=512.

### Saddle-point KKT benchmark, n=512

| Arithmetic | Unblocked | Auto blocked | Speedup | Relative residual |
|---|---:|---:|---:|---:|
| Float64x2 | 0.112946 s | 0.063480 s | 1.78x | 8.49e-32 |
| Float64x3 | 0.213872 s | 0.167799 s | 1.27x | 3.62e-48 |
| Float64x4 | 0.428969 s | 0.314672 s | 1.36x | 6.96e-64 |

### Saddle-point KKT benchmark, n=1024

| Arithmetic | Unblocked | Auto blocked | Speedup | Relative residual |
|---|---:|---:|---:|---:|
| Float64x2 | 1.818292 s | 0.402603 s | 4.52x | 1.30e-31 |
| Float64x3 | 1.688233 s | 0.987449 s | 1.71x | 6.31e-48 |
| Float64x4 | 3.330110 s | 2.005588 s | 1.66x | 3.18e-64 |

Independent indefinite scaling sweeps selected the same 16/12/8 panel widths
at both n=512 and n=1024. Pathological matrices that force adjacent 2x2 pivots
also pass the factor/solve residual tests.

## Repeated-call allocation audit

`allocation_audit.jl` measures kernel calls after warmup with all matrix inputs,
outputs, right-hand sides, and the packed GEMM workspace preallocated. Copying
the pristine input back into an in-place factorization buffer is included.

On Apple M4 with Julia 1.12.6, four available Julia threads, and `n=128`:

| Operation | 1 thread | 4 threads | Interpretation |
|---|---:|---:|---|
| direct/auto GEMM | 0 B | 2.3–2.5 KiB | task objects only on threaded route |
| packed GEMM with workspace | 0 B | 2.3–2.5 KiB | packed buffers are fully reused |
| GEMMT / SYRK / packed SYRK | 0 B | 2.0–2.3 KiB | task objects only on threaded route |
| TRSM | 48 B | 48 B | fixed call overhead |
| TRSV / right TRMM | 0 B | 0 B | allocation free |
| Cholesky | 336 B | 26.7–27.5 KiB | factor wrapper/views plus per-panel tasks |
| LU | 1.4 KiB | 29.5–35.3 KiB | pivot vector plus per-panel tasks |
| LDLT | 3.3–5.3 KiB | same | pivot/block/weighted-panel metadata; n=128 uses unblocked route |

The ranges span Float64x2/x3/x4. This audit identifies no missing reusable
numeric buffer in the single-thread kernels. A future general workspace should
focus on caller-owned residual/correction storage and factor metadata reuse;
thread-task allocation is a separate scheduling concern and should not force a
large monolithic workspace design.

## Stable TRSV versus one-column TRSM

`trsv_stability.jl` uses BenchmarkTools with 30 samples per case and measures
the substitution kernel separately from an end-to-end call that copies the
input RHS. Results below are Apple M4, Julia 1.12.6, one Julia thread, lower
non-unit `trans=:N`, and times in milliseconds. The ratio is TRSV/TRSM, so a
value below one favors TRSV.

| Type | n | Kernel TRSV / TRSM | Ratio | End-to-end TRSV / TRSM | Ratio |
|---|---:|---:|---:|---:|---:|
| x2 | 64 | 0.011 / 0.011 | 0.98x | 0.011 / 0.011 | 1.00x |
| x2 | 128 | 0.045 / 0.045 | 0.99x | 0.045 / 0.046 | 0.98x |
| x2 | 256 | 0.186 / 0.185 | 1.01x | 0.187 / 0.188 | 1.00x |
| x2 | 512 | 0.750 / 0.765 | 0.98x | 0.768 / 0.767 | 1.00x |
| x2 | 1024 | 3.334 / 3.289 | 1.01x | 3.395 / 3.561 | 0.95x |
| x3 | 64 | 0.034 / 0.034 | 1.01x | 0.034 / 0.034 | 1.00x |
| x3 | 128 | 0.134 / 0.132 | 1.01x | 0.136 / 0.133 | 1.02x |
| x3 | 256 | 0.523 / 0.548 | 0.96x | 0.532 / 0.535 | 0.99x |
| x3 | 512 | 2.202 / 2.129 | 1.03x | 2.174 / 2.175 | 1.00x |
| x3 | 1024 | 10.865 / 10.691 | 1.02x | 10.368 / 10.449 | 0.99x |
| x4 | 64 | 0.054 / 0.054 | 1.00x | 0.054 / 0.055 | 0.98x |
| x4 | 128 | 0.208 / 0.208 | 1.00x | 0.204 / 0.205 | 1.00x |
| x4 | 256 | 0.799 / 0.813 | 0.98x | 0.808 / 0.808 | 1.00x |
| x4 | 512 | 3.231 / 3.192 | 1.01x | 3.255 / 3.242 | 1.00x |
| x4 | 1024 | 13.766 / 13.678 | 1.01x | 13.584 / 13.815 | 0.98x |

Across all sizes and arithmetic types, the kernel ratio was 0.96x--1.03x and
the end-to-end ratio was 0.95x--1.02x. There is no stable material regression.
The TRSV kernel allocated 0 B versus TRSM's fixed 48 B; including the RHS copy,
TRSV allocated 64 B less in every case. Vector factor solves therefore remain
on the direct TRSV path.

## Explicit mixed-precision residual

`mixed_residual.jl` compares ordinary same-type residual evaluation with the
explicit x2-to-x3/x4 and x3-to-x4 paths. Inputs and outputs are preallocated;
the shape is 257-by-129 with one and four right-hand sides. Measurements below
were collected on Apple M4, Julia 1.12.6, one BLAS thread, and five samples.

| Source -> residual | RHS | 1 thread | 4 threads | 1-thread alloc | 4-thread alloc |
|---|---:|---:|---:|---:|---:|
| x2 -> x3 | 1 | 0.544 ms | 0.208 ms | 0 B | 2016 B |
| x2 -> x3 | 4 | 2.088 ms | 0.933 ms | 0 B | 2016 B |
| x2 -> x4 | 1 | 0.793 ms | 0.480 ms | 0 B | 2016 B |
| x2 -> x4 | 4 | 3.216 ms | 1.702 ms | 0 B | 2016 B |
| x3 -> x4 | 1 | 0.814 ms | 0.535 ms | 0 B | 2016 B |
| x3 -> x4 | 4 | 3.214 ms | 1.825 ms | 0 B | 2016 B |

The ordinary same-type vector baselines were 0.066--0.174 ms with one thread.
Mixed residuals are intentionally more expensive because every operand is
converted and multiplied at the requested higher precision. Single-thread
paths are allocation-free; the four-thread allocation is task scheduling
overhead. No result from this benchmark changes a production route.

## Reusable workspace cycles

`workspace_cycles.jl` measures repeated in-place factor, solve, residual, and
one-correction cycles with all numerical input/output arrays preallocated.
Results below are Apple M4, Julia 1.12.6, one Julia thread, one BLAS thread,
`n=128`, and four right-hand sides.

| Cycle | Arithmetic | Independently owned | `MFWorkspace` reuse |
|---|---|---:|---:|
| LU, direct GEMM | x2 / x3 / x4 | 1648 B | 560 B |
| LU, packed GEMM | x2 | 64,240 B | 560 B |
| LU, packed GEMM | x3 | 94,960 B | 560 B |
| LU, packed GEMM | x4 | 125,680 B | 560 B |
| blocked LDLT | x2 | 52,848 B | 224 B |
| blocked LDLT | x3 | 70,256 B | 224 B |
| blocked LDLT | x4 | 87,664 B | 224 B |
| RRQR factor | x2 / x3 / x4 | 3232 / 4256 / 5280 B | 32 B |

With a live factor and preallocated destination, general multi-RHS residual
evaluation measured 0 B and one correction measured 96 B for all three types.
The remaining workspace-cycle allocations are factor/view wrapper objects and
fixed TRSM call overhead; numeric metadata, the weighted panel, and packed
GEMM storage are reused. The workspace deliberately does not own residual or
correction outputs because those public APIs already require caller storage.

## Solver-shape tuning campaign

`shape_tuning.jl` compares the real direct-family baseline (fused direct for
x3, standard direct otherwise) against caller-forced packed GEMM. All outputs
must be exactly equal before timing. The following median speedups are from
Apple M4, Julia 1.12.6, seven samples, one BLAS thread; values above one favor
packing.

| Arithmetic | Shape `(m,k,n)` | 1 thread | 4 threads |
|---|---|---:|---:|
| x2 | square `(256,256,256)` | 1.037x | 1.152x |
| x2 | tall `(1024,128,64)` | 1.111x | 1.097x |
| x2 | wide `(64,128,1024)` | 1.017x | 1.011x |
| x2 | moderate `(384,192,256)` | 1.030x | 1.059x |
| x3 | square | 0.958x | 0.847x |
| x3 | tall | 0.936x | 0.969x |
| x3 | wide | 0.917x | 0.893x |
| x3 | moderate | 0.943x | 0.893x |
| x4 | square | 1.031x | 1.019x |
| x4 | tall | 1.028x | 1.056x |
| x4 | wide | 1.028x | 0.996x |
| x4 | moderate | 1.034x | 0.976x |

Only the x2 tall point clears five percent in both thread configurations; the
wide point does not, so one rectangular bucket would overgeneralize. x3
packing consistently regresses, while x4 is below the meaningful threshold or
changes sign. No production route changed: strongly rectangular `:auto` GEMM
remains outside square calibration, and machine profiles remain explicit.

`structured_x3.jl` uses benchmark-local GEMMT/SYRK loops with either standard
`acc+x*y` or `mulacc_x3`, then compares every Float64 limb bit. On this
Darwin/AArch64 machine, single-thread fused speedups over rows 128/256 and
reduction widths 8/16/32 were 1.017x--1.062x for GEMMT and 1.022x--1.061x for
SYRK. Four-thread large cases were generally positive, but short GEMMT cases
were noisy enough to change sign. This supports the current static
Darwin+AArch64 structured fusion without adding a runtime profile or a
thread/size policy. The x86 regression evidence still keeps x86 on standard
structured accumulation.

The KKT LDLT check at `n=512` again favored the existing blocked route by
3.04x/1.51x/1.60x for x2/x3/x4, with residuals of approximately
`8.5e-32`/`3.6e-48`/`7.0e-64`. No LDLT crossover change was justified.

## Multi-RHS solves

`multi_rhs.jl` measures Cholesky, LU, and LDLT solves at `n=256` with 1, 2, 4,
and 8 right-hand sides. All calls allocate 96 B after input/output
preallocation. The four-RHS lane layout is materially better than linear
per-column scaling; setting `thread_count=4` does not materially change solve
time because triangular dependencies remain serial. Backward errors remained
at x2/x3/x4 scales of about `1e-32`/`1e-48`/`1e-64`. There is no second
multi-RHS implementation worth routing, so no planner was added.

## Correctness gates

The package test matrix covers:

- direct and packed GEMM equality, including reusable workspace;
- GEMV, lower SYRK, and BLAS-like left/right TRSM variants;
- blocked Cholesky reconstruction and vector/multi-RHS solves;
- blocked partial-pivoting LU and vector/multi-RHS solves;
- LDLT 1x1 and forced 2x2 pivots;
- blocked LDLT panel-boundary 2x2 cases and multi-RHS residuals.

The suite has passed on Julia 1.10, 1.11, and 1.12 for Float64x2/x3/x4.

## Reproduction

Instantiate the benchmark environment:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Run the ordinary dense benchmark:

```bash
julia -t 4 --project=benchmark benchmark/benchmarks.jl 256 --bigfloat
```

Run direct-versus-packed GEMM throughput gates:

```bash
julia -t 4 --project=benchmark benchmark/packed_gemm.jl 512 --generic
julia -t 4 --project=benchmark benchmark/packed_gemm.jl 1024
```

Run explicit machine calibration:

```bash
julia -t 4 --project=benchmark benchmark/calibrate_gemm.jl 512 1024 3 1.05
```

Run LDLT scaling and saddle-point KKT benchmarks:

```bash
julia -t 4 --project=benchmark benchmark/ldlt_scaling.jl 512
julia -t 4 --project=benchmark benchmark/ldlt_scaling.jl 1024
julia -t 4 --project=benchmark benchmark/kkt_ldlt.jl 512
julia -t 4 --project=benchmark benchmark/kkt_ldlt.jl 1024
```

Run the repeated-call allocation audit:

```bash
julia -t 4 --project=. benchmark/allocation_audit.jl 128
```

Run explicit mixed-residual timing and allocation measurements:

```bash
julia -t 4 --project=. benchmark/mixed_residual.jl 257 129 4 5
```

Run repeated factor/solve/residual/correction workspace cycles:

```bash
julia --project=. benchmark/workspace_cycles.jl 128 4
```

Run solver-shape, structured-arithmetic, and multi-RHS tuning campaigns:

```bash
julia -t 4 --project=. benchmark/shape_tuning.jl 7
julia -t 4 --project=. benchmark/structured_x3.jl 7
julia -t 4 --project=. benchmark/multi_rhs.jl 256 5
julia -t 4 --project=. benchmark/kkt_ldlt.jl 512
```

Run the durable solver-relevant suite (optional third argument writes TSV):

```bash
julia -t 4 --project=. benchmark/solver_suite.jl 96 3 /tmp/mfla-solver-suite.tsv
```

The manual `Solver-Relevant Benchmark Suite` GitHub workflow runs the same
command on Ubuntu x86-64 and macOS ARM and uploads each TSV. It is deliberately
not a push-time calibration or route selector.

The suite applies a correctness gate before timing and emits one stable TSV
schema for normal/transpose GEMV, SYMV, GEMM, GEMMT, SYRK, packed SYRK, TRSV,
TRSM, TRMM, Cholesky, LU, KKT LDLT, equality RRQR, predictor/corrector solves,
residuals, one explicit correction, and complete repeated cycles. Every row
records median wall time, allocations, the monotonic process peak RSS, thread
count, shape, arithmetic type, route, workspace capacity where applicable,
quality metrics, and factor diagnostics. It uses no implicit calibration and
does not alter a production route.

### Solver suite baseline, n=96

Apple M4, Julia 1.12.6, four available Julia threads, one BLAS thread, three
samples, and identical seeded inputs for the 1-thread and 4-thread cases. The
run emitted 198 metric rows and every correctness gate passed.

Maximum relative result errors over all reported kernel/factor/solve rows were
`1.15e-30`, `3.45e-47`, and `2.07e-62` for x2/x3/x4. Maximum normwise backward
errors were `2.23e-32`, `6.82e-49`, and `3.68e-64`, respectively.

Representative kernel medians in milliseconds:

| Kernel | x2 1T / 4T | x3 1T / 4T | x4 1T / 4T |
|---|---:|---:|---:|
| GEMV normal | 0.010 / 0.045 | 0.027 / 0.049 | 0.046 / 0.058 |
| GEMV transpose | 0.008 / 0.038 | 0.018 / 0.045 | 0.040 / 0.057 |
| SYMV | 0.052 / 0.059 | 0.182 / 0.087 | 0.228 / 0.229 |
| GEMM | 0.344 / 0.240 | 1.258 / 0.550 | 3.207 / 1.289 |
| GEMMT | 0.267 / 0.171 | 0.762 / 0.408 | 1.802 / 0.771 |
| SYRK | 0.275 / 0.219 | 0.780 / 0.393 | 1.775 / 0.795 |
| TRSV, one RHS | 0.029 / 0.031 | 0.086 / 0.091 | 0.132 / 0.134 |
| TRSM, four RHS | 0.046 / 0.047 | 0.109 / 0.113 | 0.228 / 0.239 |

Single-thread GEMV, GEMV-transpose, SYMV, GEMM, GEMMT, SYRK, packed SYRK,
TRSV, and TRMM were allocation-free; TRSM retained its fixed 48-byte overhead.
Thread task creation accounted for the multi-thread allocation. At this small
shape, task overhead dominates GEMV/GEMV-transpose and some factorizations,
while cubic GEMM/GEMMT/SYRK work benefits. The suite records this evidence but
does not add a size/thread planner.

The complete factor/predictor/corrector/residual/one-correction cycles used the
same RHS for owned and workspace-backed measurements. Single-thread allocation
changed as follows:

| Cycle | Arithmetic | Owned | `MFWorkspace` | Time owned / reuse |
|---|---|---:|---:|---:|
| LU | x2 | 1248 B | 464 B | 0.614 / 0.630 ms |
| LU | x3 | 1248 B | 464 B | 1.838 / 1.845 ms |
| LU | x4 | 1264 B | 480 B | 4.420 / 4.288 ms |
| LDLT | x2 | 2848 B | 304 B | 0.508 / 0.552 ms |
| LDLT | x3 | 3808 B | 304 B | 1.609 / 1.677 ms |
| LDLT | x4 | 4336 B | 320 B | 3.456 / 3.545 ms |

RRQR workspace reuse reduced factor metadata allocation from
1328/1760/2144 B to 32 B for x2/x3/x4, with no material timing change. Process
peak RSS rose from roughly 597 MB to 790 MB over the complete run; this is a
monotonic process-lifetime peak (including compilation and all three types),
not an operation-level delta.

The 512/1024 workflows are manual by design so ordinary documentation and
small code changes do not repeatedly consume large benchmark jobs.
