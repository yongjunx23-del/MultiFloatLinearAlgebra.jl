# MFLA factor-cache allocation and accuracy report

Branch `fix/factor-cache-contract-hardening`. Julia 1.12.6 (aarch64 macOS),
BLAS threads=1. All numbers are `@allocated` bytes after one warm call,
measured by the repeated-call audit so the reported value is the steady-state
hot path.

> **Honesty note.** This report states what the allocation gate actually
> asserts and distinguishes *verified* claims from *status-under-work*. It does
> **not** claim cache solves are validated against an independent BigFloat oracle.

## Allocation

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

| operation | verified guarantee |
|---|---|
| cached **vector** solve | **0 bytes** |
| cached **matrix** solve | **0 bytes** |
| cached `factorize!` (Cholesky / LU / LDLT incl. blocked / RRQR incl. blocked) | **0 bytes** |
| repeated `factorize!` (A refactor at unchanged size) | **0 bytes** |
| `invalidate!` | 0 bytes |

The cache hot path calls kwarg-free, view-free block kernels operating on the
parent matrix with explicit offsets, so it performs no Julia heap allocation.
`Profile.Allocs` and `@code_warntype` confirm the sources are eliminated, and
`benchmark/allocation_gate.jl --check` enforces it as a hard CI gate.

### LinearSolve extension (n=64, x2)

| phase | bytes / behavior |
|---|---|
| `LinearSolve.init` + first factor+solve | includes LinearCache, factor storage, one factorization |
| RHS-only update, reuse factor | **0 bytes** (reuses the existing factorization) |
| A-update in place | re-factorizes into existing storage at the same cost class as a fresh `factorize!` (0-byte core + framework solution wrapper) |
| failed factorization retcode | reports `ReturnCode.Failure`, `isfresh` stays true so a replacement `A` can be retried |

The old extension allocated a fresh `Matrix(A)` copy plus factor object on every
fresh factorization; the cache-backed extension performs exactly one storage
allocation (in `init`/first factorize) and then reuses it. `init` does not run
the O(n³) factorization. The LinearSolve solution-object construction is a
framework cost measured separately (escaping-solution measurement defeats
dead-code elimination) and is not part of the kernel gate.

## Remaining framework-level overhead

The only non-zero allocations that remain on warm single-threaded paths are in
the **public** standalone kernels, not the cache core: the `Symbol`-based
`GemmPlan` struct of the inspectable-route `gemm!`/`gemmt!` API, and
`@sync`/`Threads.@spawn` threaded-task closure objects (reported separately,
only above documented size thresholds). These predate this change and appear in
the baseline audit as `gemm=64`, `trsm=48`. The LinearSolve solution-object
construction is measured separately and is not part of the kernel gate.

## Accuracy

Cache solves are validated in `test/factor_caches.jl` against the standalone
factor solve for the same input, and independently in
`benchmark/independent_accuracy.jl` against a 512-bit `BigFloat` reference.
The independent validation covers solution relative error, factor
reconstruction residual (L·Lᵀ, P·A≈L·U, A[p,p]≈L·D·Lᵀ, A[:,p]≈Q·R, QᵀQ≈I),
normwise backward error, permutation validity, LDLT inertia, LU pivot growth,
RRQR rank/reconstruction (rank-deficient), pathological/singular/nonfinite/
badly-scaled inputs, success→failure→recovery, and serial(1t)-vs-threaded(4t)
bit-identity. The script exits nonzero on any failure.

Pathological, singular, rank-deficient, nonfinite, adjacent-2x2-pivot, multi-RHS,
aliased, and threaded-vs-serial cases are covered in `test/factor_caches.jl` and
the pre-existing adversarial suite.

## Runtime

Factor-cache `factorize!` and `solve!` call the same specialized SIMD/threaded
kernels as the standalone API (no generic fallback, no reduction-order change),
so runtime is unchanged from the prior measured solver-suite results; the cache
removes allocation and re-factorization overhead on repeated RHS without
changing numerical results bit-for-bit.
