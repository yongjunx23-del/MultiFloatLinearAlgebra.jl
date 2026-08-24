# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  RHS updates, re-factorizes into existing storage on an `A` update, exposes a
  public `refresh!` for in-place `A` mutation, and reports `ReturnCode.Failure`
  on a failed factorization while keeping the cache fresh (never retaining a
  prior success) so a replacement `A` can be retried.

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

A **minor version bump** (e.g. `0.2.0` → `0.3.0`) is warranted once the
validation suite passes on the hardened factor-cache contract.
