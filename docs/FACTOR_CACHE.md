# Reusable factor-cache API and lifecycle

The factor-cache layer gives solver packages (and SDPX) a reusable, low-
allocation route to repeated factorization and solve without abandoning the
safe standalone factor API. A cache owns every array its factorization and
solve need, so the warm hot path does not create RHS scratch, factor wrappers,
or metadata copies.

This is a *borrowed* storage contract: a cache is invalidated by the next
`factorize!`, and is deliberately not a long-lived, independently ownable
factor. Callers that need a persistent factor continue to use the standalone
`lu!` / `ldlt!` / `cholesky!` / `rrqr!` API.

## Types

| Cache | Factorization | Owned storage |
|---|---|---|
| `MFCholeskyCache{MF}` | lower Cholesky | factor matrix, status, frozen `KernelConfig` |
| `MFLUCache{MF}` | partial-pivoting LU | factor matrix, pivots, frozen `KernelConfig` |
| `MFLDLTCache{MF}` | symmetric-indefinite LDLT | factor matrix, `dsub`, Bunch-Kaufman pivots/blocks, weighted panel, frozen `KernelConfig` |
| `MFRRQRCache{MF}` | pivoted Householder QR | factor matrix, reflectors, permutation + cycle leaders, norm scratch, frozen `KernelConfig` |

Every cache implements `AbstractMFFactorCache{MF}` and the uniform interface
below. `MF` is always a Float64-based `MultiFloat` (`Float64x2/3/4`); other base
types and x5-x8 are rejected at construction.

## Uniform interface

```julia
cache = MFCholeskyCache(T; config=KernelConfig())   # or MFLUCache / MFLDLTCache / MFRRQRCache
prepare!(cache, n; nrhs=1)                           # explicit reserve; RRQR also prepare!(cache, m, n)
factorize!(cache, A; check=true, config=cache.config) # overwrite owned storage
solve!(x, cache, b)                                  # vector RHS, caller-owned x
solve!(X, cache, B)                                  # multi-RHS, caller-owned X
factor_status(cache)                                 # 0 success; -1 nonfinite; >0 failure location; -2 invalidated
factor_diagnostics(cache)                            # stable NamedTuple snapshot (RRQR: factor_diagnostics(cache; atol, rtol))
factor_cache_capacity(cache)                         # exact currently-allocated owned-buffer capacities
invalidate!(cache)                                   # explicit refresh marker, allocation-free
reconfigure!(cache, new_config)                      # change the frozen config; then prepare! again
```

`factor_kind`, `factor_state`, `factor_precision`, `factor_provider`,
`factor_matrix`, `size`, `eltype`, and `issuccess` mirror the standalone
factor accessors. `factor_state` adds `:invalidated` for a cache that has not
been factorized since construction/`prepare!`/`invalidate!`/`reconfigure!`.

The RRQR cache additionally exposes `apply_q!`, `solve_r!`, `numerical_rank`,
`factor_permutation`, and `factor_rdiag` (see
[`SOLVER_BACKEND_CONTRACT.md`](SOLVER_BACKEND_CONTRACT.md) for the standalone
analogues).

### Frozen configuration contract

A cache's `KernelConfig` is **frozen** at construction (or at the last
`reconfigure!`). The hot path (`factorize!` / `solve!` / `apply_q!` /
`solve_r!`) throws an `ArgumentError` if you pass a `config` that differs from
`cache.config`. To change the configuration:

```julia
reconfigure!(cache, new_config)   # replaces cache.config and invalidates the cache
prepare!(cache, n)                # required before the next warm factorize!/solve!
```

A changed config can require a larger GEMM/LDLT/RRQR workspace, which is why
`prepare!` must run again after `reconfigure!`.

## Lifecycle contract

- **Ownership.** The cache owns its factor matrix and every metadata array.
- **Fail-closed.** `factorize!` invalidates the cache **before** touching its
  numerical storage, so an unexpected exception can never leave a stale
  success behind. Only a complete, full success sets `issuccess == true`.
- **Solve refuses after failure.** Every `solve!` (and `apply_q!`/`solve_r!`/
  `numerical_rank`) throws unless `issuccess(cache)` is true; a refused solve
  never mutates its destination.
- **Replacement-A recovery.** From any failed state, replacing `A` at the same
  size and re-running `factorize!` recovers a successful cache.
- **Growth is explicit.** Storage grows only in `prepare!` (or the
  `prepare!(cache, m, n)` overload for RRQR). A `factorize!` at a size larger
  than the prepared capacity throws `ArgumentError` and tells you to call
  `prepare!` first. The factorization hot path never silently `resize!`.
- **A-update in place.** Changing `A` values at the same size only overwrites
  the owned factor matrix; the storage object is preserved
  (`factor_matrix(cache) === same_object`).
- **Borrowed factor.** `factor_matrix(cache)` is borrowed, read-only, cache-
  owned storage and is invalidated by the next `factorize!`. It must not be
  mutated and must not be treated as a permanent independent factor.
- **Invalidation.** `invalidate!(cache)` sets `:invalidated` and makes the next
  `solve!` throw until `factorize!` runs again. This is the explicit refresh
  signal after a caller mutates `A` in place.
- **Concurrency.** A single cache is not concurrency-safe: one cache must not
  be used by concurrent factorization or solve calls. Distinct caches retain
  call-level concurrency. The embedded GEMM workspace of an
  `MFLUCache`/`MFLDLTCache`/`MFRRQRCache` is internally serialized by its own
  lock.

## Failure semantics

With `check=false`, a numerical failure leaves the cache in a well-defined,
queryable failed state rather than throwing:

- nonfinite input -> `factor_status == -1`, `factor_state == :nonfinite_input`;
- Cholesky not-PD -> `factor_status > 0`, `factor_state == :not_posdef`;
- LU/LDLT singular -> `factor_status > 0`, `factor_state == :singular`.

With `check=true` (the default), the same numerical failure throws after the
cache has been left in that queryable failed state. `issuccess(cache)` is false
in every failed state and `solve!` throws before mutating its destination.
Replacing `A` (same size) and re-running `factorize!` recovers a successful
cache from any failed state. RRQR rank deficiency is not a failed
factorization (callers choose a threshold through `numerical_rank` or
`factor_diagnostics(cache; atol, rtol)`).

## Pre-allocation queries

```julia
workspace_requirements(::Type{MF}, operation, shape, config)   # MFWorkspace capacity
factor_cache_requirements(::Type{MF}, kind, shape, config)     # exact cache storage sizes
factor_cache_capacity(cache)                                   # what is currently allocated
```

Both requirement queries are **exact**: they are parameterized by the exact
MultiFloat scalar type `MF`, the operation/kind, the `shape` `NamedTuple`, and
the frozen `config`, so the reported capacities match what the warm
`factorize!`/`solve!` actually consume (including the resolved GEMM/LDLT/RRQR
route, worker count, and panel/microkernel widths). They are pure — they never
benchmark, calibrate, mutate storage, or read solver state. `shape` carries the
operation's dimensions: `(n=)` for Cholesky/LU/LDLT and `(m=, n=)` for RRQR /
GEMM.

Calling these once, before any numerical work, lets a caller reserve every
array the warm path needs and then guarantee that `factorize!`/`solve!` never
grow storage.

## Zero-allocation status

This status is what the current allocation gate actually asserts. Measured
with the repeated-call audit (`@allocated` after one warm call, single thread),
a cache that is `prepare!`d for the matrix size allocates **0 bytes** on every
warm hot path (verified with `Profile.Allocs` and `@code_warntype`, and
enforced by the `benchmark/allocation_gate.jl --check` CI gate):

| operation | allocation |
|---|---|
| cached **vector** solve | **0 bytes** |
| cached **matrix** solve | **0 bytes** |
| cached `factorize!` (Cholesky / LU / LDLT incl. blocked / RRQR incl. blocked) | **0 bytes** |
| repeated `factorize!` (A refactor at unchanged size) | **0 bytes** |
| success → failure → recovery refactor | **0 bytes** |
| `invalidate!` | 0 bytes |

The factorization cores and cache solves call kwarg-free, view-free block
kernels (operating on the parent matrix with explicit offsets), so the warm
path performs no Julia heap allocation. The standalone factor API (and the
public `gemm!`/`gemmt!` inspectable-route kernels) may still allocate a small
amount — the `Symbol`-based `GemmPlan` struct of the route API and the
`@sync`/`Threads.@spawn` threaded-task closure — which is reported separately
and is not part of the cache-core gate. Threaded task spawn/fetch allocation is
also reported separately (task objects, not numerical buffers).

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
solve!(x, cache, b)               # 0-byte vector solve
```

For an unchanged `A` with repeated RHS, call `factorize!` once and `solve!`
many times; update `A` values in place and call `factorize!` again without a
new `prepare!` when the size is unchanged. To change the configuration later,
call `reconfigure!(cache, new_config)` followed by `prepare!(cache, n)` — never
pass a different `config` to the hot path.

See [`FACTOR_CACHE_LIFECYCLE.md`](FACTOR_CACHE_LIFECYCLE.md) for the state
diagrams, and [`FACTOR_CACHE_REPORT.md`](FACTOR_CACHE_REPORT.md) for the
allocation/accuracy detail.
