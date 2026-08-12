# MultiFloatLinearAlgebra.jl

A standalone CPU linear-algebra backend designed specifically for
[MultiFloats.jl](https://github.com/dzhang314/MultiFloats.jl).

The package is intentionally solver-independent. It does **not** depend on
SDPX.jl or any optimization model. Solver packages can call it for dense
numeric kernels and factorizations while retaining problem analysis, sparse
symbolics, KKT assembly, precision policy, fallback policy, and certification
in their own layers.

## Implemented backend

- deterministic `mfdot`;
- SIMD/threaded `gemv!`;
- direct and B-panel-packed `gemm!`;
- lower-triangular `syrk!` and `gemmt!`;
- BLAS-like left/right `trsm!`;
- blocked lower Cholesky using TRSM/SYRK;
- blocked partial-pivoting LU using TRSM/GEMM;
- Bunch--Kaufman-style symmetric-indefinite LDLT with 1x1/2x2 pivots;
- vector and multi-RHS solves for Cholesky, LU, and LDLT;
- explicit x2/x3/x4 GEMM calibration and inspectable route plans.

The hot kernels map four independent matrix rows or right-hand sides into
`MultiFloatVec{4,T,N}` lanes. They use MultiFloats.jl's native fixed-width
arithmetic rather than converting through `BigFloat`.

The package supports one- through four-limb `MultiFloat{T,N}` values. It does
not claim support for experimental x5-x8 arithmetic.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yongjunx23-del/MultiFloatLinearAlgebra.jl")
```

For development:

```julia
Pkg.develop(path="MultiFloatLinearAlgebra.jl")
```

## Basic backend use

```julia
using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra

T = Float64x4
A = T.(randn(256, 256))
B = T.(randn(256, 256))
C = zeros(T, 256, 256)

config = KernelConfig(thread_count=Threads.nthreads())
gemm!(C, A, B; config)

# SPD factor/solve
R = T.(randn(256, 256))
S = R * transpose(R) + T(256) * I
Fc = MultiFloatLinearAlgebra.cholesky!(copy(Matrix(S)); config)
x = solve(Fc, T.(randn(256)); config)

# General dense factor/solve
Flu = MultiFloatLinearAlgebra.lu!(copy(A); config)
y = solve(Flu, T.(randn(256, 8)); config)

# Symmetric-indefinite KKT-style factor/solve
K = copy(A + transpose(A))
Fldl = MultiFloatLinearAlgebra.ldlt!(K; config)
z = solve(Fldl, T.(randn(256)); config)
```

The package deliberately does not pirate `LinearAlgebra.mul!`,
`LinearAlgebra.cholesky!`, or `LinearAlgebra.lu!`. Backend selection remains
explicit, so callers can A/B the specialized implementation against Julia's
generic algorithms.

## Packed GEMM and machine calibration

The retained packed route packs only B panels into reusable caller-owned
storage. This avoids repacking A on every call and preserves the direct
kernel's per-output ascending reduction order.

```julia
T = Float64x2
n = 1024
A = T.(randn(n, n))
B = T.(randn(n, n))
C = zeros(T, n, n)

workspace = GemmWorkspace(
    T;
    thread_count=Threads.nthreads(),
    capacity=n * 32,
)

packed = KernelConfig(
    thread_count=Threads.nthreads(),
    gemm_strategy=:packed,
    gemm_panel_columns=32,
    gemm_micro_columns=4,
)

gemm!(C, A, B; config=packed, workspace)
```

Uncalibrated `gemm_strategy=:auto` deliberately stays on the direct route.
Calibration is explicit and has no process-global state:

```julia
calibration = calibrate_gemm(
    Float64x2;
    sizes=(512, 1024),
    samples=3,
    thread_count=Threads.nthreads(),
    minimum_speedup=1.05,
)

profile = calibration.profile
config = with_gemm_profile(KernelConfig(), profile)
plan = gemm_plan(Float64x2, 1024, 1024, 1024, config)
```

`GemmCalibration` records the complete measurements, selected geometry,
required minimum speedup, machine fingerprint, and crossover. The candidate
is selected at the largest tested size and is enabled only over a stable
winning suffix of the requested sizes.

On the documented four-thread GitHub x86-64 runner, B-panel packing is stable
for Float64x2: about 1.16x at n=512 and 1.11x at n=1024. Float64x3 and
Float64x4 do not clear the five-percent gate at n=1024 and therefore retain
the direct route. These are machine results, not portable hard-coded profiles.

## Symmetric-indefinite LDLT

The blocked path retains Bunch--Kaufman-style 1x1/2x2 pivot decisions. Each
panel forms `W = L*D`, then delegates the lower-triangular trailing update to
`gemmt!`:

```text
A22_lower <- A22_lower - L21 * W21'
```

Only the lower triangle performs multiplication; an inexpensive mirror copy
restores the dense symmetric view used by subsequent pivot searches.

```julia
K = T.(your_symmetric_kkt_matrix)
config = KernelConfig(thread_count=Threads.nthreads())
plan = ldlt_plan(T, size(K, 1), config)
F = MultiFloatLinearAlgebra.ldlt!(K; config)
x = solve(F, T.(rhs); config)
```

The default `:auto` route uses the established unblocked factorization below
n=512 and blocked LDLT at or above n=512. Measured default panel widths are:

| Arithmetic | Panel width |
|---|---:|
| Float64x2 | 16 |
| Float64x3 | 12 |
| Float64x4 | 8 |

On saddle-point KKT matrices, the blocked route measured 1.78x/1.27x/1.36x
at n=512 and 4.52x/1.71x/1.66x at n=1024 for x2/x3/x4, respectively. The
corresponding relative solve residuals were approximately 1e-31, 1e-48, and
1e-64.

## Solver integration boundary

A solver adapter can remain thin:

```julia
# SPD system
factor = MultiFloatLinearAlgebra.cholesky!(matrix; config)
MultiFloatLinearAlgebra.ldiv!(rhs, factor; config)

# General dense system
factor = MultiFloatLinearAlgebra.lu!(matrix; config)
MultiFloatLinearAlgebra.ldiv!(rhs, factor; config)

# Symmetric-indefinite KKT system
factor = MultiFloatLinearAlgebra.ldlt!(matrix; config)
MultiFloatLinearAlgebra.ldiv!(rhs, factor; config)
```

The caller remains responsible for choosing the arithmetic type, deciding
whether a fallback is allowed, assembling sparse or structured systems, and
validating the final residual or certificate.

## Validation

The correctness suite covers Float64x2/x3/x4 on Julia 1.10, 1.11, and 1.12:

- direct and packed GEMM equality with reusable workspace;
- GEMV, lower SYRK, GEMMT, and left/right TRSM variants;
- Cholesky reconstruction and vector/multi-RHS residuals;
- blocked LU and vector/multi-RHS residuals;
- LDLT 1x1 and forced 2x2 pivots;
- blocked LDLT panel-boundary 2x2 cases and multi-RHS residuals.

## Benchmarks

The complete environment, A/B tables, KKT residuals, route decisions, and
reproduction commands are recorded in
[`benchmark/RESULTS.md`](benchmark/RESULTS.md).

Ordinary dense benchmark:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia -t 4 --project=benchmark benchmark/benchmarks.jl 256 --bigfloat
```

Large throughput and calibration workflows are manual by design:

```bash
julia -t 4 --project=benchmark benchmark/packed_gemm.jl 512 --generic
julia -t 4 --project=benchmark benchmark/packed_gemm.jl 1024
julia -t 4 --project=benchmark benchmark/calibrate_gemm.jl 512 1024 3 1.05
julia -t 4 --project=benchmark benchmark/ldlt_scaling.jl 1024
julia -t 4 --project=benchmark benchmark/kkt_ldlt.jl 1024
```

## Roadmap

1. Run explicit calibration on Apple Silicon and target EPYC nodes; keep
   profiles machine-local unless multiple machines support a portable rule.
2. Add optional mixed-precision residual/refinement primitives without
   importing solver policy.
3. Add sparse supernodal numeric kernels on top of the dense backend while
   leaving symbolic analysis to the caller or a separate package.
4. Add GPU GEMM/SYRK/GEMMT/TRSM after the CPU numerical contract is stable.
5. Add proof-friendly arithmetic-network boundaries tied to
   MultiFloatProofs.
