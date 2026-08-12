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

The 512/1024 workflows are manual by design so ordinary documentation and
small code changes do not repeatedly consume large benchmark jobs.
