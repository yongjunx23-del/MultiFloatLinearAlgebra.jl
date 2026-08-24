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
  `MultiFloatCholesky`) now initializes an empty cache (no double numerical
  factorization), reuses the factor across RHS updates, re-factorizes into
  existing storage on an `A` update, and reports `ReturnCode.Failure` on a
  failed factorization while keeping the cache fresh so a replacement `A` can
  be retried.

### Allocation-status clarification

Vector solves are asserted 0-byte; matrix solves allocate only the shared
triangular-kernel dispatch floor (bounded and identical across calls); and
`factorize!` still allocates a small non-zero amount (kernel dispatch plus
`@view` SubArray creation in the block loops). That `factorize!` byte count is
under active elimination and is tracked by the allocation gate — it is not yet
a guaranteed-zero contract. Documentation was corrected to stop claiming
`factorize!` is zero-allocation and to stop asserting that cache solves are
validated against an independent BigFloat oracle (cache accuracy is validated
against the standalone factor solve, with a 512-bit BigFloat relative-error
norm).

### Versioning

A **minor version bump** (e.g. `0.2.0` → `0.3.0`) is warranted once the
validation suite passes on the hardened factor-cache contract.
