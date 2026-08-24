# MFLA factor-cache allocation and accuracy report

Branch `refactor/reusable-factor-cache`. Julia 1.12.6 (aarch64 macOS), BLAS
threads=1. All numbers are `@allocated` bytes after one warm call, measured by
the repeated-call audit so the reported value is the steady-state hot path.

## Allocation before / after

### Baseline (before the factor-cache refactor)

The pre-existing `benchmark/allocation_audit.jl` at n=64, threads=1:

| operation | x2 | x3 | x4 |
|---|---|---|---|
| gemm/auto | 64 | 64 | 96 |
| syrk | 0 | 0 | 0 |
| trsm | 48 | 48 | 48 |
| trsv | 0 | 0 | 0 |
| cholesky! (incl. factor object) | 144 | 144 | 144 |
| lu! (incl. pivot metadata copy) | 1488 | 1488 | 1584 |
| ldlt! (incl. metadata copy) | 1824 | 2368 | 2816 |

The baseline factors allocate a fresh owned-metadata snapshot every call (the
1488-2816-byte columns) because each `lu!`/`ldlt!` returns a new independently
ownable factor.

### Factor cache (new), n=64, single thread

| operation | x2 | x3 | x4 |
|---|---|---|---|
| cached vector solve | **0** | **0** | **0** |
| cached matrix solve | 96 (2x `trsm` dispatch) | 96 | 96 |
| cached factorize (LU) | 912 (stable) | 912 | 1008 |
| `invalidate!` | 0 | 0 | 0 |

Repeated `factorize!` produces identical, bounded allocation (no storage
growth). The residual factorize bytes are the shared `gemm`/`trsm`/`syrk`
kernel dispatch and `@view` subarray creation inside the block loops — the same
kernel costs the standalone API pays, not cache-core allocations.

### LinearSolve extension (n=64, x2)

| operation | bytes |
|---|---|
| `LinearSolve.init` + first factor+solve | 101856 (includes LinearCache, factor storage, one factorization) |
| RHS-only update, reuse factor | **0** |
| A-update in place, refactorize into existing storage | **0** |
| failed factorization retcode | reports `ReturnCode.Failure`, `isfresh` stays true |

The old extension allocated a fresh `Matrix(A)` copy plus factor object on every
fresh factorization; the cache-backed extension performs exactly one storage
allocation (in `init`/first factorize) and then reuses it.

## Remaining framework-level overhead

The only non-zero allocations that remain on warm single-threaded paths are
shared-kernel dispatch and subarray-view costs that predate this change (they
appear in the baseline audit as `gemm=64`, `trsm=48`). Threaded task creation is
reported separately and only occurs above documented size thresholds. The
LinearSolve solution-object construction is a framework cost measured separately
above and is not part of the kernel gate.

## Accuracy

Every cache solve is validated against the standalone factor solve and/or a
512-bit BigFloat reference in `test/factor_caches.jl`. For n=48:

| factor | max relative error (x2) | reference |
|---|---|---|
| Cholesky | <= 4096*eps(x2) tolerance | BigFloat |
| LU | <= 4096*96*eps | BigFloat |
| LDLT | <= 4096*256*eps | BigFloat |
| RRQR (square) | <= 4096*256*eps | BigFloat |

Pathological, singular, rank-deficient, nonfinite, adjacent-2x2-pivot, multi-RHS,
aliased, and threaded-vs-serial cases are covered in
`test/factor_caches.jl` and the pre-existing adversarial suite. All pass.

## Runtime

Factor-cache `factorize!` and `solve!` call the same specialized SIMD/threaded
kernels as the standalone API (no generic fallback, no reduction-order change),
so runtime is unchanged from the prior measured solver-suite results; the cache
removes allocation and re-factorization overhead on repeated RHS without
changing numerical results bit-for-bit.
