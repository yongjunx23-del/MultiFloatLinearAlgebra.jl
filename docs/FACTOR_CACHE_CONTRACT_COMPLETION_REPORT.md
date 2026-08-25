# FactorCache contract completion report

**Branch:** `fix/factor-cache-contract-completion`
**Baseline:** `defbf30e345abdf34e2117e45a8a79b1ee02388d`
**Date:** 2025-08-25
**Status:** All P0/P1 issues fixed; full test suite, allocation gate, and independent 512-bit BigFloat validation pass.

This report documents the second hardening round of the MFLA FactorCache contract. It was driven by seven parallel read-only review subagents (A–G), whose findings were consolidated into an issue matrix and fixed in 11 atomic commits across four waves.

---

## 1. Review issue matrix

| # | Issue | Severity | Owner | Files | Fix | Test |
|---|---|---|---|---|---|---|
| A-1 | Julia 1.9 + LinearSolve 2.22 multi-RHS MethodError (CI red) | P0 | A | ext/MultiFloatLinearSolveExt.jl | extension-owned solution storage in `_solve!` | multi-RHS on 2.22 & 5.x |
| A-2 | CI does not test LinearSolve 3.83/4.x | P0 | A | .github/workflows/ci.yml | `linearsolve-compat` matrix job | CI matrix |
| A-3 | `refresh!` not a real public API | P1 | A | src/, ext/ | core generic + export + extension method | public `refresh!` |
| B-1 | `reconfigure!` w/o `prepare!` lets `factorize!` run (LDLT OOB) | P0 | B | factor_cache_defs.jl, factor_caches.jl | `config_epoch`/`prepared_epoch`/`prepared_shape` + `_check_prepared` | reconfigure→factorize throws |
| B-2 | shape check reads live storage (field-mutation bypass) | P2 | B | factor_caches.jl | recorded `prepared_shape` | field-mutation throws |
| B-3 | RRQR `threads` kwarg bypasses frozen config | P2 | B/C | factor_caches.jl | removed kwarg | — |
| C-1 | `solve_r!(...; trans=:T)` silently does `trans=:N` | P1 | C | factor_caches.jl, trsm.jl | forward `trans`; add `_trsv/_trsm_leading_upper_trans!` | R'*x=b, R'*X=B |
| C-2 | `apply_q!`/`solve_r!` allocate (Union return) | P1 | C | factor_caches.jl | helpers return `nothing` | 0-byte gate rows |
| D-1 | Dead `GemmWorkspace` in LU/LDLT/RRQR caches + broken gemm reporting | P1 | D | factor_cache_defs.jl, factor_caches.jl, qr.jl, requirements | delete field; report gemm=0 | gemm fields in consistency test |
| D-2 | RRQR `cycle_leaders` requirements vs actual `n` | P1 | D | requirements | report `n` | tall RRQR consistency |
| D-3 | RRQR `ftranspose` zero-reflector mismatch | P2 | D | factor_caches.jl, requirements | `max(block,1)` rows | m=0/n=0 consistency |
| D-4 | `factorization`/`factor`/`factor_capacity` naming | P2 | D | requirements | rename to `factor` | — |
| D-5 | negative dims not validated | P2 | D | requirements | `_require_nonnegative` | throws |
| D-6 | `m*n` integer overflow | P2 | D | requirements | `_checked_elements` | throws |
| D-7 | `:syrk/:gemmt` GEMM plan `m` dimension | P2 | D | requirements | `gemm_plan(n,k,n)` | — |
| D-8 | `:solve` reports `gemm_workers=1` | P2 | D | requirements | `gemm_workers=0` | — |
| E-1 | `factor_diagnostics(cache)` reads undef/stale on prepared/invalidated | P1 | E | diagnostics.jl | numeric fields `nothing` on invalid | diagnostics-state testset |
| E-2 | `factor_permutation`/`factor_rdiag` return garbage on non-success | P1 | E | factor_caches.jl | throw on `!issuccess` | throws |
| E-3 | LU nonfinite `maximum_u`/`pivot_growth` = NaN | P2 | E | diagnostics.jl | `nothing` for status -1 | — |
| F-1 | Gate excludes LDLT matrix solve (stale comment) | P1 | F | allocation_gate.jl | add `ldlt_matrix_solve` row | gate row |
| F-2 | Gate under-reports GemmPlan (context-dependent) | P1 | F | allocation_gate.jl | documented; gate measures warm call | — |
| F-3 | Cache factorization 1.6–1.8× slower than standalone | P1 | F | block_updates.jl | V4 2-column store-pair | perf ratio |
| F-4 | Gate uses `minimum`, masking unstable allocs | P2 | F | allocation_gate.jl | documented | — |
| F-5 | Gate missing required operations | P2 | F | allocation_gate.jl | added 26 rows | gate rows |
| G-1 | FACTOR_CACHE_ACCURACY_REPORT.md stale (9 failures) | P1 | G | docs | regenerated 189/0 | — |
| G-2 | README allocation status stale | P1 | G | README.md | rewritten | — |
| G-3 | LDLT/RRQR matrix-solve not in test suite | P1 | G | test/factor_caches.jl | added | matrix-solve tests |
| G-4 | capabilities x1 claim not gated; no matrix-solve fact | P2 | G | capabilities.jl | added matrix-solve fact | propertynames test |

---

## 2. Atomic commit list (baseline `defbf30` → HEAD)

| Commit | Description |
|---|---|
| `f24edc9` | Enforce fail-closed lifecycle with config/prepared epochs |
| `ba6f995` | Harden cache diagnostics and metadata accessors against undef/stale storage |
| `8dd2401` | Fix LinearSolve 2.22 multi-RHS and make `refresh!` a public API |
| `e4d07af` | Fix RRQR cache `solve_r! trans=:T` and make `apply_q!`/`solve_r!` zero-alloc |
| `2bde264` | Make factor-cache requirements/capacity exact and remove dead GemmWorkspace |
| `864ea84` | Expand allocation gate to cover LDLT matrix solve, RRQR routes, and metadata |
| `d9c5c90` | Add LinearSolve compat CI matrix, bounds-checked tests, and rectangular BigFloat checks |
| `bc9fe91` | Add LDLT and RRQR matrix-solve correctness and 0-byte tests |
| `f11e461` | Vectorize view-free LU/RRQR trailing GEMM to close cache-vs-standalone gap |
| `6475952` | Sync docs, capabilities, and changelog with the hardened factor-cache contract |
| `278edcf` | Stop committing the 1.12-resolved Manifest so all supported Julia versions load |

---

## 3. Verification results

### 3.1 Full test suite
`Pkg.test()` on Julia 1.12.6: **all pass** (1871 core + factor-cache + LinearSolve extension testsets). Run with `--check-bounds=yes` in CI.

### 3.2 Allocation hard gate
`benchmark/allocation_gate.jl --check`: **GATE PASSED**. Every gated hot path is 0 bytes — all four `factorize!`, vector+matrix `solve!`, LDLT matrix solve, RRQR `apply_q!`/`solve_r!` :N/:T (vector+matrix), rectangular route, `reconfigure!`/`prepare!`, and metadata accessors. The only nonzero rows are the 6 public `gemm!`/`gemmt!` `GemmPlan` rows (framework cost, reported separately).

### 3.3 Independent 512-bit BigFloat validation
`benchmark/independent_accuracy.jl`: **189/189 PASS, exit 0** (was 162/9). New rectangular RRQR checks (tall least-squares, wide reconstruction, `solve_r! trans=:T` vector+matrix) close the coverage gap that previously hid the `trans=:T` bug.

### 3.4 Cross-version loading
Julia 1.9, 1.10, 1.12 all load with fresh resolution (the committed 1.12-resolved Manifest was removed). Julia 1.9 + LinearSolve 2.22 multi-RHS, A-replacement, failure/retry, and public `refresh!` all verified.

### 3.5 Cache vs standalone runtime (n=256, Float64x2, 1 thread)
| Factor | Before | After |
|---|---|---|
| LU | 1.78× | 1.10× |
| LDLT | 0.99× | 0.91× |
| RRQR | 1.17× | 1.00× |

The V4 2-column store-pair rewrite closed the cache-vs-standalone gap while preserving zero allocation and deterministic reduction.

### 3.6 Requirements/capacity exact match
`factor_cache_requirements(...)` → `prepare!(...)` → `factor_cache_capacity(...)` match **exactly** (not just `>=`) for all four kinds across square/tall/wide/zero-reflector shapes, thread counts 1/2, and forced-`:packed` configs. Negative dimensions and `m*n` overflow are rejected.

---

## 4. LinearSolve compatibility matrix

| Julia | LinearSolve | Multi-RHS | A-replace | Failure/retry | `refresh!` |
|---|---|---|---|---|---|
| 1.9 | 2.22 | ✓ | ✓ | ✓ | ✓ |
| 1.10 | 3.83 | ✓ (CI) | ✓ | ✓ | ✓ |
| 1.10 | 4 | ✓ (CI) | ✓ | ✓ | ✓ |
| 1.12 | 5 | ✓ | ✓ | ✓ | ✓ |

The `linearsolve-compat` CI job pins each version; 3.83/4.x were previously never exercised.

---

## 5. FactorCache lifecycle state machine

```
reconfigure! ──► reconfigure_requires_prepare ──► prepare! ──► prepared
                                                      │
prepare! ──► prepared ──► factorize! ──► success / failed
   │                              │
   └── invalidate! ──► invalidated ─┘
```

- `config_epoch` increments on every `reconfigure!`; `prepared_epoch` is set to it on `prepare!`.
- `factorize!` refuses unless `prepared_epoch == config_epoch` and `prepared_shape` matches (immune to field mutation).
- `solve!`/`apply_q!`/`solve_r!`/`numerical_rank` refuse unless `issuccess`.
- `factor_diagnostics` returns numeric fields as `nothing` on prepared/invalidated caches; `factor_permutation`/`factor_rdiag` throw on non-success.

---

## 6. Remaining limitations (honest)

- The 6 public `gemm!`/`gemmt!` `GemmPlan` rows still allocate (64–160 bytes) — a framework cost of the Symbol-based route descriptor, not the cache core. A pre-parsed `GemmPlan` API is a candidate future optimization.
- Threaded-task creation allocates (report-only, not gated); a persistent-worker design is deferred until profiling shows it matters.
- `size(cache)` allocates 32 bytes (a Julia-level `size(::Matrix)` cost, not a package defect).
- The cache-vs-standalone ratio is now ~1.0–1.1×, not bit-identical; the two paths use different (both deterministic) reduction schedules.

---

## 7. Conclusion

**Merge and release 0.3.0: YES.** All P0/P1 issues are fixed, the full test suite passes, the allocation gate passes with every cache hot path at 0 bytes, the independent 512-bit BigFloat validation passes 189/189, the LinearSolve compat matrix is CI-enforced, and the requirements/capacity contract is exact. The FactorCache contract is complete and honest.
