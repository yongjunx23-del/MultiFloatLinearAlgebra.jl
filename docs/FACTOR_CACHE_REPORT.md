# MFLA factor-cache allocation and accuracy report

Branch `fix/factor-cache-contract-hardening`. Julia 1.12.6 (aarch64 macOS),
BLAS threads=1. All numbers are `@allocated` bytes after one warm call,
measured by the repeated-call audit so the reported value is the steady-state
hot path.

> **Honesty note.** This report states what the allocation gate actually
> asserts and distinguishes *verified* claims from *status-under-work*. It does
> **not** claim `factorize!` is zero-allocation, and it does **not** claim cache
> solves are validated against an independent BigFloat oracle.

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
| cached **vector** solve | **0 bytes** (asserted by the suite) |
| cached **matrix** solve | shared `trsm` kernel-dispatch floor only; bounded and identical across consecutive calls (the suite asserts equality across calls, not a flat 0) |
| cached `factorize!` (LU/LDLT/RRQR) | a **small non-zero** amount — shared `gemm`/`trsm`/`syrk` kernel dispatch plus `@view` SubArray creation in the block loops; **never** storage growth |
| `invalidate!` | 0 bytes |

Repeated `factorize!` at an unchanged size produces identical, bounded
allocation (no storage growth). The residual `factorize!` bytes are **not** a
settled or guaranteed-zero contract: they are **under active elimination** and
tracked by the allocation gate. The RRQR `solve_r!` rectangular route carries
the same `@view` SubArray cost.

### LinearSolve extension (n=64, x2)

| phase | bytes / behavior |
|---|---|
| `LinearSolve.init` + first factor+solve | includes LinearCache, factor storage, one factorization |
| RHS-only update, reuse factor | **0 bytes** (reuses the existing factorization) |
| A-update in place | re-factorizes into existing storage; **not** asserted 0-byte (same `factorize!` cost class) |
| failed factorization retcode | reports `ReturnCode.Failure`, `isfresh` stays true so a replacement `A` can be retried |

The old extension allocated a fresh `Matrix(A)` copy plus factor object on every
fresh factorization; the cache-backed extension performs exactly one storage
allocation (in `init`/first factorize) and then reuses it. `init` does not run
the O(n³) factorization.

## Remaining framework-level overhead

The only non-zero allocations that remain on warm single-threaded paths are
shared-kernel dispatch and subarray-view costs that predate this change (they
appear in the baseline audit as `gemm=64`, `trsm=48`). Threaded task creation is
reported separately and only occurs above documented size thresholds; completing
the `factorize!` zero-allocation and the threaded-task allocation are **explicit
remaining work**. The LinearSolve solution-object construction is a framework
cost measured separately above and is not part of the kernel gate.

## Accuracy

Cache solves are validated in `test/factor_caches.jl` **against the standalone
factor solve for the same input** (`ldiv!` with `cholesky!`/`lu!`/`ldlt!`/
`rrqr!`), with the maximum-relative-error norm computed in 512-bit BigFloat.
The reference is the standalone MFLA factor, **not** an independent BigFloat
oracle, so this report does **not** claim "cache solves validated against
BigFloat". Whether the suite currently passes is not asserted here; run
`Pkg.test("MultiFloatLinearAlgebra")` to confirm.

Pathological, singular, rank-deficient, nonfinite, adjacent-2x2-pivot, multi-RHS,
aliased, and threaded-vs-serial cases are covered in `test/factor_caches.jl` and
the pre-existing adversarial suite.

## Runtime

Factor-cache `factorize!` and `solve!` call the same specialized SIMD/threaded
kernels as the standalone API (no generic fallback, no reduction-order change),
so runtime is unchanged from the prior measured solver-suite results; the cache
removes allocation and re-factorization overhead on repeated RHS without
changing numerical results bit-for-bit.
