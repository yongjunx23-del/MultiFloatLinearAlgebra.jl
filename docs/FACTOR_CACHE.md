# Reusable factor-cache API and lifecycle

The factor-cache layer gives solver packages (and SDPX) a reusable, low-
allocation route to repeated factorization and solve without abandoning the
safe standalone factor API. A cache owns every array its factorization and
solve need, so the warm hot path never allocates RHS scratch, factor wrappers,
metadata copies, or factor storage.

This is a *borrowed* storage contract: a cache is invalidated by the next
`factorize!`, and is deliberately not a long-lived, independently ownable
factor. Callers that need a persistent factor continue to use the standalone
`lu!` / `ldlt!` / `cholesky!` / `rrqr!` API.

## Types

| Cache | Factorization | Owned storage |
|---|---|---|
| `MFCholeskyCache{MF}` | lower Cholesky | factor matrix, status |
| `MFLUCache{MF}` | partial-pivoting LU | factor matrix, pivots |
| `MFLDLTCache{MF}` | symmetric-indefinite LDLT | factor matrix, `dsub`, Bunch-Kaufman pivots/blocks, weighted panel |
| `MFRRQRCache{MF}` | pivoted Householder QR | factor matrix, reflectors, permutation + cycle leaders, norm scratch, GEMM workspace |

Every cache implements `AbstractMFFactorCache{MF}` and the uniform interface
below. `MF` is always a Float64-based `MultiFloat` (`Float64x2/3/4`); other base
types and x5-x8 are rejected at construction.

## Uniform interface

```julia
cache = MFCholeskyCache(T)                    # or MFLUCache / MFLDLTCache / MFRRQRCache
prepare!(cache, n; nrhs=1)                    # explicit reserve (RRQR also accepts (m, n))
factorize!(cache, A; check=true, config=...)  # overwrite owned storage
solve!(x, cache, b; config=...)               # vector RHS, caller-owned x
solve!(X, cache, B; config=...)               # multi-RHS, caller-owned X
factor_status(cache)                          # 0 success; -1 nonfinite; -2 invalidated
factor_diagnostics(cache)                     # stable NamedTuple snapshot
invalidate!(cache)                            # explicit refresh marker, allocation-free
```

`factor_kind`, `factor_state`, `factor_precision`, `factor_provider`,
`factor_matrix`, `size`, `eltype`, and `issuccess` mirror the standalone
factor accessors. `factor_state` adds `:invalidated` for a cache that has not
been factorized since `prepare!`/`invalidate!`.

## Lifecycle contract

- **Ownership.** The cache owns its factor matrix and every metadata array.
- **Warm zero allocation.** A successful vector `solve!` allocates **0 bytes**.
  Matrix `solve!` allocates only the shared triangular-kernel dispatch floor
  (the baseline audit reports `trsm=48`); the cache itself creates no RHS,
  factor wrapper, or metadata copy. Repeated `factorize!` never grows owned
  storage: its allocation is bounded and identical across consecutive calls.
- **Growth is explicit.** Storage grows only in `prepare!` (or the
  `prepare!(cache, m, n)` overload for RRQR). A `factorize!` at a size larger
  than the prepared capacity throws `ArgumentError` and tells you to call
  `prepare!` first. The factorization hot path never silently `resize!`.
- **A-update in place.** Changing `A` values at the same size only overwrites
  the owned factor matrix; the storage object is preserved (`factor_matrix(cache)
  === same_object`).
- **Borrowed factor.** `factor_matrix(cache)` is borrowed, read-only, cache-
  owned storage and is invalidated by the next `factorize!`. It must not be
  mutated and must not be treated as a permanent independent factor.
- **Invalidation.** `invalidate!(cache)` sets `:invalidated` and makes the next
  `solve!` throw until `factorize!` runs again. This is the explicit refresh
  signal after a caller mutates `A` in place.
- **Concurrency.** A single cache is not concurrency-safe: one cache must not
  be used by concurrent factorization or solve calls. Distinct caches retain
  call-level concurrency. The embedded GEMM workspace of an `MFRRQRCache` is
  internally serialized by its own lock.

## Failure semantics

With `check=false`, a numerical failure leaves the cache in a well-defined,
queryable failed state rather than throwing:

- nonfinite input -> `factor_status == -1`, `factor_state == :nonfinite_input`;
- Cholesky not-PD -> `factor_status > 0`, `factor_state == :not_posdef`;
- LU/LDLT singular -> `factor_status > 0`, `factor_state == :singular`.

`issuccess(cache)` is false in every failed state and `solve!` throws before
mutating its destination. Replacing `A` (same size) and re-running `factorize!`
recovers a successful cache from any failed state. RRQR rank deficiency is not
a failed factorization (callers choose a threshold through
`factor_diagnostics(cache; atol, rtol)`).

## Pre-allocation queries

```julia
workspace_requirements(operation, shape, config)   # MFWorkspace capacity
factor_cache_requirements(kind, shape, config)     # exact cache storage sizes
```

Both are pure (never benchmark, calibrate, mutate, or read solver state) and
let a caller reserve all storage up front so the warm core path allocates
nothing.

## Zero-allocation status

Measured with the repeated-call audit (`@allocated` after one warm call), a
cache that is `prepare!`d for the matrix size:

| operation | allocation |
|---|---|
| cached vector solve | 0 bytes |
| cached matrix solve | shared `trsm` kernel dispatch floor only |
| cached factorize (any kind) | shared `gemm`/`trsm`/`syrk` kernel dispatch bytes only; never storage growth |
| `invalidate!` | 0 bytes |

The only remaining bytes come from the shared dense kernels' dispatch (present
in the baseline allocation audit before this change: `gemm=64`, `trsm=48`) and
from `@view` subarray creation inside the factorization block loops. They are
framework/kernel-level costs, not cache-core allocations, and are reported
separately from the kernel gate.

## Migration

The standalone factor API is unchanged. Adopting a cache is a pure performance
path: replace

```julia
F = lu!(copy(A); config=cfg)      # fresh factor each call, allocates matrix copy
ldiv!(x, F, b; config=cfg)
```

with

```julia
cache = MFLUCache(T; config=cfg)
prepare!(cache, n)
factorize!(cache, A)              # reuses owned storage, no matrix copy
solve!(x, cache, b; config=cfg)   # 0-byte vector solve
```

For an unchanged `A` with repeated RHS, call `factorize!` once and `solve!`
many times; update `A` values in place and call `factorize!` again without a
new `prepare!` when the size is unchanged.
