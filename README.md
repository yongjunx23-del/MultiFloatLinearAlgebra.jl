# MultiFloatLinearAlgebra.jl

A standalone CPU linear-algebra backend designed specifically for
[MultiFloats.jl](https://github.com/dzhang314/MultiFloats.jl).

The package is intentionally solver-independent: it does **not** depend on
SDPX.jl or any optimization model. Solver packages can call this package as a
backend and keep planning, sparse symbolic analysis, KKT assembly, and
certification in their own layers.

## Status

The initial backend implements:

- deterministic `mfdot`;
- SIMD/threaded `gemv!`;
- SIMD/threaded dense `gemm!`;
- lower-triangular SIMD/threaded `syrk!`;
- blocked lower Cholesky using the same `syrk!` backend;
- partial-pivoting dense LU;
- Cholesky/LU triangular solves.

The SIMD kernels map four independent matrix rows into
`MultiFloatVec{4,T,N}` lanes. This reuses MultiFloats.jl's native arithmetic
networks instead of repeatedly converting through BigFloat.

GPU kernels, sparse symbolic factorization, and supernodal sparse numeric
factorization are deliberately outside the first milestone.

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
A = T.(randn(128, 128))
B = T.(randn(128, 128))
C = zeros(T, 128, 128)

config = KernelConfig(thread_count=Threads.nthreads())

gemm!(C, A, B; config)

S = A * transpose(A)
F = cholesky!(copy(S); config)
x = solve(F, T.(randn(128)))
```

The package intentionally does not pirate `LinearAlgebra.mul!`,
`LinearAlgebra.cholesky!`, or `LinearAlgebra.lu!`. This makes the backend
choice explicit and allows callers to A/B it against Julia's generic
`LinearAlgebra` implementation.

## Benchmark

Instantiate the benchmark environment once:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Then run, for example:

```bash
julia -t 4 --project=benchmark benchmark/benchmarks.jl 128
```

Add `--bigfloat` to include a 256-bit BigFloat GEMM reference.

The benchmark reports:

- Float64 BLAS (one BLAS thread);
- Julia generic `LinearAlgebra` on Float64x2/x3/x4;
- this package's specialized MultiFloat backend;
- optional BigFloat.

This separation is important: the primary speedup metric is specialized
MultiFloat backend versus generic MultiFloat `LinearAlgebra`; Float64 and
BigFloat show the surrounding performance/precision envelope.

## Design lineage

The first CPU kernels are extracted conceptually from extended-precision work
developed and benchmarked inside SDPX.jl: disjoint output ownership,
four-lane `MultiFloatVec` execution, cache/block-aware GEMM/SYRK, and blocked
Cholesky. Solver-specific Schur/KKT code and automatic route selection are not
part of this package.

## Roadmap

1. Validate and tune x2/x3/x4 CPU kernels on x86-64 and Apple Silicon.
2. Add packed-panel GEMM microkernels and machine calibration.
3. Add symmetric-indefinite LDLᵀ with 1×1/2×2 pivoting.
4. Add mixed-precision iterative refinement as an optional factor/solve layer.
5. Add GPU GEMM/SYRK kernels.
6. Add a proof-friendly arithmetic-network boundary so core kernels can track
   formal guarantees from MultiFloatProofs.

## Numerical contract

The current kernels support one- through four-limb `MultiFloat{T,N}` values.
They preserve each SIMD lane's ascending reduction order. Correctness tests
compare against Julia's generic MultiFloat linear algebra and high-precision
residual checks.

The package does not claim support for experimental x5-x8 arithmetic.
