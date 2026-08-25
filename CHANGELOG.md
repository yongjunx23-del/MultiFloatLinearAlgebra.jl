# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-08-25

### Final contract cleanup

- **Exact `workspace_requirements`.** `workspace_requirements(::Type{MF},
  operation, shape, config)` now reports the exact scratch the standalone
  `lu!` / `ldlt!` / `rrqr!` cores need so `ensure_workspace_capacity!(ws;
  req...)` makes the first call not grow: blocked LDLT reports
  `ldlt_block_capacity`, packed LU reports the trailing-update GEMM
  workers/capacity, and blocked RRQR reports `qr_ftranspose_rows/cols` and
  `qr_aux`. `MFWorkspace` gained `qr_ftranspose`/`qr_aux` capacity tracking,
  and the field names were unified (`factor_capacity` / `ldlt_block_capacity` /
  `gemm_capacity`) across `workspace_requirements`, `ensure_workspace_capacity!`,
  and `workspace_capacity` so the splat contract holds.
- **Deterministic failure diagnostics.** Cache metadata (LU `ipiv`, LDLT
  `dsub`/`pivots`/`blocks`, RRQR `tau`/`permutation`) is now identity/zero
  initialized before the nonfinite check, so a nonfinite failure no longer
  leaks stale pivots/blocks/inertia/permutation into `factor_diagnostics`.
  Nonfinite diagnostics return `nothing` for the metadata fields (with
  `finite=false`); LU singular diagnostics expose an identity pivot tail
  instead of uninitialized garbage.
- **Regression tests.** Added `workspace_requirements` route coverage
  (blocked/unblocked LDLT, direct/packed GEMM, RRQR square/tall/wide),
  cache-reuse tests (large→small, LDLT blocked→unblocked) asserting
  `capacity >= requirements`, and gated the x1 zero-allocation capability facts
  to the multi-limb types the allocation gate verifies (x2/x3/x4). The `nrhs`
  `prepare!` argument is documented as a reserved no-op.

### Factor-cache contract hardening

- **Fail-closed lifecycle.** `factorize!` now invalidates the cache *before*
  touching its numerical storage, so an unexpected exception can never leave a
  stale success behind; only a complete, full success sets `issuccess == true`.
  Every `solve!` / `apply_q!` / `solve_r!` / `numerical_rank` refuses (throws,
  leaving its destination untouched) unless the cache is successful, and a
  failed cache is recovered by replacing `A` (same size) and re-running
  `factorize!`.
- **Frozen configuration contract.** A cache's `KernelConfig` is fixed at
  construction (or the last `reconfigure!`); the hot path throws if passed a
  config differing from `cache.config`. Configuration changes are made only
  through `reconfigure!(cache, new_config)` followed by `prepare!`.
- **Unified cache core.** `MFCholeskyCache`, `MFLUCache`, `MFLDLTCache`, and
  `MFRRQRCache` implement a uniform `AbstractMFFactorCache` surface
  (`prepare!` / `factorize!` / `solve!` / `invalidate!` / `reconfigure!`,
  `factor_status`, `issuccess`, `factor_state` including `:invalidated`,
  `factor_diagnostics`, `factor_cache_capacity`).
- **Exact requirements / capacity API.** `workspace_requirements(::Type{MF},
  operation, shape, config)` and `factor_cache_requirements(::Type{MF}, kind,
  shape, config)` are exact, type- and config-dependent, pure queries; the
  resolved GEMM/LDLT/RRQR route, worker count, and panel widths are reflected
  so a caller can reserve once and guarantee no warm-path storage growth.
- **Rectangular RRQR cache route.** The `MFRRQRCache` exposes `apply_q!`,
  `solve_r!`, `numerical_rank`, `factor_permutation`, and `factor_rdiag` for
  tall/wide factors, in addition to the square full-rank `solve!` path.
- **LinearSolve tightening.** The weak-dep extension (`MultiFloatLU`,
  `MultiFloatCholesky`) now builds the cache at the matrix size in
  `init_cacheval` (reserving O(n²) storage only, no O(n³) factorization at
  init), so the first `solve!` does not grow storage; reuses the factor across
  RHS updates, re-factorizes into existing storage on an `A` update, and
  `refresh!` is now a real exported public API — a core generic
  `refresh!(cache::AbstractMFFactorCache)` (alias of `invalidate!`) exported
  from the main module, plus a `refresh!(cache::LinearSolve.LinearCache)`
  method in the extension for in-place `A` mutation — and reports
  `ReturnCode.Failure` on a failed factorization while keeping the cache fresh
  (never retaining a prior success) so a replacement `A` can be retried.

### Zero-allocation cache core

All cache hot paths now allocate **0 bytes** on the warm single-thread path
(verified with `Profile.Allocs` / `@code_warntype` and enforced by
`benchmark/allocation_gate.jl --check`):

- cached `factorize!` for Cholesky / LU / LDLT (incl. blocked) / RRQR (incl.
  blocked);
- cached vector and matrix `solve!`;
- repeated `factorize!` (A refactor at unchanged size);
- success → failure → recovery refactor;
- `invalidate!`.

This was achieved by replacing the factorization block-loop `@view` SubArray
creation and keyword-argument dispatch with kwarg-free, view-free block kernels
that operate on the parent matrix with explicit offsets. The only warm-path
allocations that remain are threaded-task (`@sync`/`Threads.@spawn`) objects on
multi-worker routes and the public `gemm!`/`gemmt!` inspectable-route `GemmPlan`
struct — both reported separately and not part of the cache-core gate.

### Independent numerical validation

`benchmark/independent_accuracy.jl` validates the cache layer against an
independent 512-bit `BigFloat` reference: solution relative error, factor
reconstruction residual, normwise backward error, permutation validity, LDLT
inertia, LU pivot growth, RRQR rank/reconstruction (incl. rank-deficient),
pathological/singular/nonfinite/badly-scaled inputs, success→failure→recovery,
and serial(1t)-vs-threaded(4t) bit-identity. The script exits nonzero on any
failure.

### Versioning

This release (0.3.0) is the minor bump warranted once the validation suite
passes on the hardened factor-cache contract.
