# MultiFloatLinearAlgebra.jl

A standalone CPU linear-algebra backend designed specifically for
[MultiFloats.jl](https://github.com/dzhang314/MultiFloats.jl).

The package is intentionally solver-independent. It does **not** depend on
SDPX.jl or any optimization model. Solver packages can call it for dense
numeric kernels and factorizations while retaining problem analysis, sparse
symbolics, KKT assembly, precision policy, fallback policy, and certification
in their own layers.

## Implemented backend

- deterministic vector/Frobenius `mfdot`;
- SIMD/threaded `gemv!` with transpose (`trans=:T`) support;
- direct and B-panel-packed `gemm!`;
- lower-triangular `syrk!`/`gemmt!` and packed-lower `syrk_packed!`;
- BLAS-like left/right `trsm!`;
- single-RHS vector `trsv!`;
- authoritative-triangle `symv!`;
- blocked lower Cholesky using TRSM/SYRK;
- blocked partial-pivoting LU using TRSM/GEMM;
- Bunch--Kaufman-style symmetric-indefinite LDLT with 1x1/2x2 pivots;
- vector and multi-RHS solves for Cholesky, LU, and LDLT;
- deterministic column-pivoted Householder RRQR with caller-controlled rank,
  Q/Q' application, and leading-R solves;
- stable `factor_diagnostics` facts for Cholesky, LU, LDLT, and QR;
- general/symmetric residuals, normwise backward error, and one-step
  factor-based refinement correction;
- explicit higher-limb residuals for x2-to-x3/x4 and x3-to-x4 evaluation;
- explicit x2/x3/x4 GEMM calibration and inspectable route plans.

The hot kernels map four independent matrix rows or right-hand sides into
`MultiFloatVec{4,T,N}` lanes. They use MultiFloats.jl's native fixed-width
arithmetic rather than converting through `BigFloat`.

The production provider contract is `MultiFloat{Float64,N}` for `N=1:4`.
Although kernel implementations retain internal generic structure, other base
types and experimental x5-x8 arithmetic are not benchmarked or supported.

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

# Rank-revealing equality factorization (rank policy remains caller-owned)
Fqr = rrqr!(copy(A))
rank = numerical_rank(Fqr; rtol=T(256) * eps(T))
qt_rhs = T.(randn(size(A, 1)))
apply_q!(qt_rhs, Fqr; trans=:T)
leading_solution = copy(qt_rhs[1:rank])
solve_r!(leading_solution, Fqr, rank; config)
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

A configuration returned by `with_gemm_profile` is bound to the profile's
exact MultiFloat arithmetic type. `gemm_plan` and `gemm!` reject applying it
to another arithmetic type instead of silently reusing incompatible
calibration data. Ordinary `KernelConfig()` and `:auto` behavior are unchanged.

`GemmCalibration` records the complete measurements, selected geometry,
required minimum speedup, machine fingerprint, and crossover. The candidate
is selected at the largest tested size and is enabled only over a stable
winning suffix of the requested sizes.

On the documented four-thread GitHub x86-64 runner, B-panel packing is stable
for Float64x2: about 1.16x at n=512 and 1.11x at n=1024. Float64x3 and
Float64x4 do not clear the five-percent gate at n=1024 and therefore retain
the direct route. These are machine results, not portable hard-coded profiles.

## Reusable solver workspace

`MFWorkspace` groups only material reusable storage: packed GEMM panels, LU
pivots, LDLT metadata and weighted panels, and RRQR reflector/permutation plus
hybrid norm-state scratch. Residual and correction arrays remain explicit
caller-owned outputs.

```julia
workspace = MFWorkspace(
    T;
    factor_capacity=1024,
    ldlt_block_capacity=16,
    thread_count=Threads.nthreads(),
    gemm_capacity=1024 * 32,
)

capacity = workspace_capacity(workspace)
ensure_workspace_capacity!(
    workspace;
    factor_capacity=2048,
    ldlt_block_capacity=16,
    gemm_workers=Threads.nthreads(),
    gemm_capacity=2048 * 32,
)

Flu = MultiFloatLinearAlgebra.lu!(copy(A); config, workspace)
Fldl = MultiFloatLinearAlgebra.ldlt!(copy(K); config, workspace)
Fqr = rrqr!(copy(E); workspace)
```

LU, LDLT, and RRQR factors created with `workspace=` own snapshots of the
metadata needed by solve and diagnostics. Multiple factors may remain live;
later workspace reuse or growth does not invalidate them. Factorization
scratch in one `MFWorkspace` is not safe for concurrent factorization calls.
Packed GEMM calls sharing one `GemmWorkspace` (including the GEMM component of
an `MFWorkspace`) are safe and serialized by an object-local lock. Separate
workspaces retain call-level concurrency. No process-global state is involved.

Existing calls without `workspace=` retain independently owned metadata.
`GemmWorkspace` also remains supported by `gemm!` and `residual!`.

## Capability query

`capabilities(T)` is a pure, immutable description of the public dense
backend. It never benchmarks, calibrates, inspects a solver, or changes state.

```julia
facts = capabilities(Float64x3)
facts.supported                  # true
facts.transpose_gemv             # true
facts.rrqr                       # true
facts.mixed_residual_targets     # (x2=false, x3=false, x4=true)
facts.mixed_residual_target_types # (Float64x4,)
facts.factor_metadata_ownership  # :factor_owned
facts.factor_matrix_ownership    # :borrowed_input
facts.factorization_destructive  # true
facts.factor_solve_mutates_factor # false
facts.shared_gemm_workspace_concurrency # :serialized_safe
facts.reusable_workspace         # true
```

The fixed keys cover the exact scalar type and provider, kernels,
factorizations, vector/multi-RHS support, residual targets, workspace
ownership/concurrency, SYRK storage, and threading. Unsupported limb counts
report every operation as false; production calls still fail explicitly and
are never replaced by another algorithm.

The final read-only SDPX operation map and provider-readiness conclusion are
in [`docs/SDPX_PROVIDER_READINESS.md`](docs/SDPX_PROVIDER_READINESS.md).

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
MultiFloatLinearAlgebra.ldiv!(solution, factor, rhs; config)

# General dense system
factor = MultiFloatLinearAlgebra.lu!(matrix; config)
MultiFloatLinearAlgebra.ldiv!(solution, factor, rhs; config)

# Symmetric-indefinite KKT system
factor = MultiFloatLinearAlgebra.ldlt!(matrix; config)
MultiFloatLinearAlgebra.ldiv!(solution, factor, rhs; config)
```

The caller remains responsible for choosing the arithmetic type, deciding
whether a fallback is allowed, assembling sparse or structured systems, and
validating the final residual or certificate.

For QR, `factor_permutation(F)` returns `p` such that `A[:, p] = Q*R`, and
`factor_rdiag(F)` returns a copy of the signed R diagonal. Rank deficiency is
a successful factorization. `numerical_rank(F; atol, rtol)` evaluates only the
threshold supplied by the caller; its zero defaults mean exact nonzero rank.
`factor_matrix(F)` remains borrowed compact storage and must not be mutated.

For LDLT, `factor_pivots(F)` and `factor_blocks(F)` return copies of the raw
Bunch-Kaufman metadata, while `factor_permutation(F)` returns `p` such that
`A_original[p, p] = L*D*L'`. `factor_inertia(F)` reports the inertia by scanning
only D's 1x1 and 2x2 blocks, avoiding the comprehensive O(n^2) finite-factor
scan in `factor_diagnostics` when a solver needs only this structural fact.

`factor_state`, `factor_precision`, and `factor_provider` provide stable
symbolic state, exact scalar type, and `:mfla` identity.
`factor_diagnostics(F)` returns a stable NamedTuple of numerical facts. It
reports failure locations, factor scales, LU/LDLT growth metrics, LDLT pivot
counts and inertia, and QR rank at an explicitly supplied threshold. Returned
pivot/permutation vectors are copies. Diagnostics never select a fallback or
precision.

Residual arithmetic uses one sign convention throughout:

```text
r = b - A*x
```

`residual!` supports vectors and multiple right-hand sides; `uplo=:lower` or
`:upper` reads only an authoritative symmetric triangle.
`normwise_backward_error(A, x, b, r)` reports
`||r||inf / (||A||inf*||x||inf + ||b||inf)` using an overflow-safe scaled
evaluation. `refinement_correction!(delta, F, r)` computes exactly one
`delta = F \\ r`. MFLA does not decide whether to accept or repeat it.

Higher-precision residual arithmetic is a separate, explicit operation:

```julia
residual_mixed!(r_x3, A_x2, x_x2, b_x2)
residual_mixed!(r_x4, A_x2, x_x2, b_x2)
residual_mixed!(r_x4, A_x3, x_x3, b_x3)
```

Vector and matrix right-hand sides are supported, including `uplo=:lower` or
`:upper`. Every source value is converted before multiplication and
subtraction; MFLA never computes a low-precision residual and then promotes
it. Ordinary kernels remain strict same-type operations. There is no hidden
precision escalation and no production `BigFloat` conversion.

## Validation

The correctness suite covers Float64x2/x3/x4 on Julia 1.10, 1.11, and 1.12:

- direct and packed GEMM equality with reusable workspace;
- GEMV, lower SYRK, GEMMT, and left/right TRSM variants;
- Cholesky reconstruction and vector/multi-RHS residuals;
- blocked LU and vector/multi-RHS residuals;
- LDLT 1x1 and forced 2x2 pivots;
- blocked LDLT panel-boundary 2x2 cases and multi-RHS residuals.
- RRQR reconstruction and orthogonality for tall, wide, and square matrices;
- duplicate/zero/nearly dependent/scaled QR columns, Q/Q' application, and
  vector/matrix leading-R solves.

## Benchmarks

The complete environment, A/B tables, KKT residuals, route decisions, and
reproduction commands are recorded in
[`benchmark/RESULTS.md`](benchmark/RESULTS.md).

The solver-facing ownership, failure, concurrency, precision, and symmetric
storage rules are frozen in
[`docs/SOLVER_BACKEND_CONTRACT.md`](docs/SOLVER_BACKEND_CONTRACT.md).

Ordinary dense benchmark:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia -t 4 --project=benchmark benchmark/benchmarks.jl 256 --bigfloat
```

Large throughput and calibration workflows are manual by design:

```bash
julia -t 4 --project=benchmark benchmark/packed_gemm.jl 512 --generic
julia -t 4 --project=benchmark benchmark/packed_gemm.jl 1024
julia -t 4 --project=benchmark benchmark/ldlt_scaling.jl 1024
julia -t 4 --project=benchmark benchmark/kkt_ldlt.jl 1024
julia -t 4 --project=. benchmark/mixed_residual.jl 257 129 4 5
julia --project=. benchmark/workspace_cycles.jl 128 4
julia -t 4 --project=. benchmark/shape_tuning.jl 7
julia -t 4 --project=. benchmark/structured_x3.jl 7
julia -t 4 --project=. benchmark/multi_rhs.jl 256 5
julia -t 4 --project=. benchmark/solver_suite.jl 16,32,64,96,128,192,256,384,512 3 solver-suite.tsv
```

## Roadmap

The immediate goal is to become the authoritative MultiFloats dense backend
that solver packages such as SDPX can rely on, not a general-purpose BLAS
replacement. Work is sequenced in three stages.

### Completed

- Factorization public protocol: `AbstractMFFactorization`,
  `factor_kind`/`factor_status`/`factor_matrix`, `size`/`eltype`, `issuccess`.
- Machine-compatible GEMM calibration (`GemmProfile`, `profile_compatible`,
  strict `with_gemm_profile`), inspectable route reasons, near-square auto
  routing, and one-based-indexing / no-aliasing kernel contracts.
- Float64x3 fused `mulacc_x3` direct GEMM, productionized on both AArch64 and
  x86_64; structured GEMMT/SYRK x3 fusion conservatively gated to
  Darwin + AArch64.
- Transpose GEMV (`gemv!(...; trans=:T)`), vector TRSV (`trsv!`), and
  authoritative-triangle SYMV (`symv!`), with factor vector solves routed
  through TRSV.
- Authoritative-triangle TRMM (`trmm!`) for left/right, lower/upper,
  transposed/non-transposed, and unit/non-unit triangular products.
- Downstream dense-kernel inventory, narrow packed-lower `syrk_packed!` with
  reduction slices, and repeated-call allocation audit; no speculative
  `syr2k!` or solver-scatter API was added.
- Rank-revealing column-pivoted Householder QR (`rrqr!`/`MFQR`) with
  caller-controlled `numerical_rank`, Q/Q' application, and leading-R solves.
- Factor diagnostics for Cholesky/LU/LDLT/QR, including LU growth, LDLT
  1x1/2x2 counts and inertia, and QR rank at a caller-supplied threshold.
- Residual `b-A*x`, overflow-safe normwise backward error, and explicit
  one-correction factor solves for vector and multi-RHS systems.
- Explicit deterministic mixed residual evaluation for x2-to-x3/x4 and
  x3-to-x4, with authoritative symmetric-triangle support.
- Caller-owned `MFWorkspace` for packed GEMM, LU/LDLT/RRQR metadata, and LDLT
  weighted panels, with explicit capacity, factor-owned metadata snapshots,
  and safe multiple-live-factor semantics.
- Pure machine-readable `capabilities(T)` facts, including exact mixed
  residual target types.
- Aqua type-piracy checking enabled.

### Completed roadmap review

Proof-boundary and adversarial validation, the durable solver-cycle benchmark,
and the read-only SDPX provider-readiness review are complete. The reproducible
arithmetic contract is in `docs/MULACC_X3_PROOF_CONTRACT.md`, the benchmark
entry point is `benchmark/solver_suite.jl`, and the downstream operation map is
in `docs/SDPX_PROVIDER_READINESS.md`.

### Deferred

- Sparse supernodal and GPU kernels, x5–x8 arithmetic, and BigFloat runtime
  fallback remain out of scope until the dense CPU contract is frozen.
- Proof integration lives in a companion project; MFLA keeps stable
  arithmetic-network boundaries for it.
