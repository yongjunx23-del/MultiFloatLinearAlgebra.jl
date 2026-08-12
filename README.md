# MultiFloatLinearAlgebra.jl

A standalone CPU linear-algebra backend designed specifically for
[MultiFloats.jl](https://github.com/dzhang314/MultiFloats.jl).

The package is intentionally solver-independent: it does **not** depend on
SDPX.jl or any optimization model. Solver packages can call this package as a
backend and keep planning, sparse symbolic analysis, KKT assembly, precision
selection, and certification in their own layers.

## Status

The current CPU backend implements:

- deterministic `mfdot`;
- SIMD/threaded `gemv!`;
- SIMD/threaded dense `gemm!`;
- lower-triangular SIMD/threaded `syrk!`;
- BLAS-like left/right `trsm!` with lower/upper and transpose support;
- blocked lower Cholesky using the shared `trsm!` + `syrk!` kernels;
- blocked partial-pivoting LU using panel factorization + `trsm!` + `gemm!`;
- symmetric-indefinite LDLᵀ with Bunch--Kaufman-style 1×1/2×2 pivots;
- vector and multi-RHS solves for Cholesky, LU, and LDLᵀ factors.

The SIMD kernels map four independent matrix rows or right-hand sides into
`MultiFloatVec{4,T,N}` lanes. This reuses MultiFloats.jl's native arithmetic
networks instead of repeatedly converting through BigFloat.

GPU kernels and sparse symbolic/supernodal factorization remain separate future
milestones.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yongjunx23-del/MultiFloatLinearAlgebra.jl")
```

For development:

```julia
Pkg.develop(path="MultiFloatLinearAlgebra.jl")
```

## Backend API

```julia
using MultiFloats
using MultiFloatLinearAlgebra

T = Float64x4
config = KernelConfig(thread_count=Threads.nthreads())

A = T.(randn(256, 256))
B = T.(randn(256, 256))
C = zeros(T, 256, 256)
gemm!(C, A, B; config)

# SPD factor/solve
R = T.(randn(256, 256))
S = R * transpose(R) + T(256) * I
Fc = MultiFloatLinearAlgebra.cholesky!(copy(Matrix(S)); config)
x = solve(Fc, T.(randn(256)); config)

# General dense factor/solve
Flu = MultiFloatLinearAlgebra.lu!(copy(A); config)
y = solve(Flu, T.(randn(256, 8)); config)

# Symmetric-indefinite factor/solve (KKT-style backend primitive)
K = copy(A + transpose(A))
Fldl = MultiFloatLinearAlgebra.ldlt!(K)
z = solve(Fldl, T.(randn(256)); config)
```

The package intentionally does not pirate `LinearAlgebra.mul!`,
`LinearAlgebra.cholesky!`, `LinearAlgebra.lu!`, or factorization dispatch. The
backend choice stays explicit so callers such as SDPX.jl can A/B or route it
without load-order-dependent behavior.

## Numerical backend structure

```text
mfdot
  │
  ├── gemv!
  ├── gemm! ───────────────┐
  ├── syrk! ────────┐      │
  └── trsm! ───┐    │      │
               │    │      │
          Cholesky  │      │
                    │      │
                blocked LU ┘

          LDLᵀ (1×1 / 2×2 D)
               │
             solve
```

The important design rule is that O(n³) factorization work is pushed into the
same shared kernels exposed to callers. Cholesky uses TRSM/SYRK; blocked LU
uses TRSM/GEMM. This avoids solver-specific or factorization-specific matrix
multiply implementations.

## Benchmark

Instantiate the benchmark environment once:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Then run, for example:

```bash
julia -t 4 --project=benchmark benchmark/benchmarks.jl 256
```

Add `--bigfloat` to include a 256-bit BigFloat GEMM reference.

### Reproducible CI baseline

GitHub-hosted Ubuntu runner, Julia 1.12.6, 4 Julia threads, 1 BLAS thread.
These are directional CI numbers rather than dedicated-HPC-node claims.

| operation | n | arithmetic | generic LinearAlgebra | MFLA | speedup |
|---|---:|---|---:|---:|---:|
| GEMM | 256 | Float64x2 | 143.178 ms | 9.902 ms | 14.46× |
| GEMM | 256 | Float64x3 | 546.941 ms | 38.884 ms | 14.07× |
| GEMM | 256 | Float64x4 | 1151.840 ms | 95.980 ms | 12.00× |
| TRSM | 256 | Float64x2 | 5.875 ms | 1.323 ms | 4.44× |
| TRSM | 256 | Float64x3 | 27.150 ms | 5.562 ms | 4.88× |
| TRSM | 256 | Float64x4 | 66.849 ms | 8.486 ms | 7.88× |
| Cholesky | 256 | Float64x2 | 14.413 ms | 6.395 ms | 2.25× |
| Cholesky | 256 | Float64x3 | 67.604 ms | 13.115 ms | 5.15× |
| Cholesky | 256 | Float64x4 | 171.889 ms | 28.143 ms | 6.11× |
| LU | 256 | Float64x2 | 31.382 ms | 11.914 ms | 2.63× |
| LU | 256 | Float64x3 | 135.084 ms | 20.839 ms | 6.48× |
| LU | 256 | Float64x4 | 344.422 ms | 45.793 ms | 7.52× |

The first LDLᵀ implementation currently measures approximately 1.20 ms,
4.05 ms, and 8.27 ms at n=128 for Float64x2/x3/x4 respectively. Its current
benchmark is backend-only because Julia's generic symmetric-indefinite path is
not used as a stable comparison contract here.

For scale context, the same CI run measured one-thread Float64 BLAS GEMM near
0.83 ms at n=256. The primary metric for this project remains specialized
MultiFloat backend versus generic MultiFloat `LinearAlgebra`.

## Validation

CI runs on Julia 1.10, 1.11, and 1.12. Current tests cover Float64x2/x3/x4:

- dot/GEMV/GEMM/SYRK;
- TRSM on left/right, lower/upper, normal/transposed systems;
- unit and non-unit triangular solves;
- blocked Cholesky reconstruction and multi-RHS residuals;
- blocked partial-pivoting LU vector/multi-RHS residuals;
- LDLᵀ 1×1 pivot cases;
- forced LDLᵀ 2×2 pivot cases and multi-RHS residuals.

All three Julia-version jobs pass on the current backend commit.

## SDPX integration boundary

SDPX.jl should treat this package as a numerical backend only. A clean caller
boundary is:

```text
SDPX planner / sparse symbolic / KKT assembly / certification
                         │
                         ▼
             MultiFloatLinearAlgebra
       GEMM / SYRK / TRSM / factor / solve
                         │
                         ▼
                    MultiFloats.jl
```

This package should not absorb SDPX's route selection, cone logic, tolerances,
precision policy, or certification.

## Roadmap

1. Packed-panel GEMM and architecture-specific cache/microkernel calibration.
2. Blocked/parallel symmetric-indefinite LDLᵀ trailing updates.
3. Optional mixed-precision residual/refinement primitives without solver
   policy.
4. Sparse supernodal **numeric** kernels on top of the dense backend while
   leaving symbolic analysis to the caller or a separate package.
5. GPU GEMM/SYRK/TRSM after the CPU numerical contract is stable.
6. A proof-friendly arithmetic-network boundary so core kernels can track
   formal guarantees from MultiFloatProofs.

## Numerical contract

The current kernels support one- through four-limb `MultiFloat{T,N}` values.
They preserve each SIMD lane's scalar dependency/reduction order. The package
does not claim support for experimental x5-x8 arithmetic.
