# Final contract cleanup report

**Branch:** `fix/final-contract-cleanup`
**Base:** `main` @ `2eee7e1` (merge of PR #7)
**Date:** 2025-08-25
**Status:** All confirmed contract defects fixed; full suite, allocation gate, and independent 512-bit BigFloat validation pass.

This round fixed only confirmed contract defects (no new algorithms, no speculative features). Three parallel read-only review subagents (A: MFWorkspace requirements, B: failure diagnostics, C: contract/regression tests) produced an issue matrix; the main agent fixed all P0/P1 issues in 3 atomic commits.

---

## Issue matrix

| # | Issue | Severity | Fix |
|---|---|---|---|
| A-1 | `workspace_requirements(:ldlt)` always returned `ldlt_block=0`; blocked route needs `plan.block_size` | P1 | report `ldlt_block_capacity = plan.block_size` when blocked |
| A-2 | `workspace_requirements(:lu)` returned `gemm_workers=0/gemm_capacity=0`; packed trailing GEMM needs the plan's capacity | P1 | report the trailing-update GEMM plan's workers/capacity |
| A-3 | Blocked RRQR `qr_ftranspose`/`qr_aux` grown on the hot path; API could not express them | P1 | add `qr_ftranspose_rows/cols` + `qr_aux` capacity tracking to `MFWorkspace`, `ensure_workspace_capacity!`, `workspace_requirements`, `workspace_capacity` |
| A-4 | `workspace_capacity` field names (`factor`/`ldlt_block`/`gemm_elements_per_worker`) mismatched `ensure_workspace_capacity!` keywords, breaking the splat contract | P0 | unify to `factor_capacity`/`ldlt_block_capacity`/`gemm_capacity`; non-packed ops report `gemm_workers=1` |
| B-1 | Cache metadata not deterministically initialized before the nonfinite check; nonfinite failure left stale pivots/blocks/inertia/permutation | P1 | identity/zero-init LU ipiv, LDLT dsub/blocks/pivots, RRQR tau/permutation before the core (allocation-free) |
| B-2 | `factor_diagnostics` returned stale metadata on `:nonfinite_input` | P1 | return `nothing` for metadata fields on nonfinite (matching invalidated), `finite=false` |
| B-3 | LU singular diagnostics exposed uninitialized garbage in the pivot tail | P2 | identity-init makes the tail deterministic |
| C-1 | `workspace_requirements` tests missing (LDLT blocked/unblocked, direct/packed GEMM, RRQR square/tall/wide) | P1 | added route-coverage testset |
| C-2 | Cache-reuse tests missing (large→small, blocked→unblocked) | P1 | added reuse testset |
| C-3 | `capacity >= requirements` semantics untested (only `==` at exact shape) | P1 | reuse test asserts `>=` |
| C-4 | x1 zero-alloc capability claimed but never gated | P1 | gate to `N >= 2`; update capabilities test |
| C-5 | `nrhs` docstring overclaimed "records intended multi-RHS capacity" | P2 | document as reserved no-op |

---

## Atomic commits

| Commit | Description |
|---|---|
| `dcce7d6` | Deterministically initialize cache metadata and harden nonfinite diagnostics |
| `5c73cd1` | Make workspace_requirements exact for blocked LDLT, packed LU, and blocked RRQR |
| `0a83f5d` | Add workspace-requirements route tests, cache-reuse tests, and gate x1 capability |

---

## Verification

- **Full test suite**: passes (with `--check-bounds=yes` in CI).
- **Allocation gate**: `GATE PASSED` — every cache hot path 0 bytes (the metadata init is allocation-free).
- **Independent 512-bit BigFloat validation**: **189/189**.
- **Cache/standalone runtime**: LU ratio 1.1× at n=256 (no regression).
- **LinearSolve compat**: Julia 1.9 + LinearSolve 2.22 extension tests 161/161.
- **No-grow contract**: `ensure_workspace_capacity!(ws; workspace_requirements(...)...)` then blocked LDLT / packed LU / blocked RRQR run without growing the workspace (verified).

---

## Conclusion

The confirmed contract defects are fixed. The `workspace_requirements`/`ensure_workspace_capacity!` contract is now exact (no-grow on first call) for all four standalone routes, failure diagnostics are deterministic and stale-free, and the regression tests lock the contract. No solver policy, fallback, KKT, IPM, or SDPX-specific logic was added. Per the task, MFLA optimization stops here — no pre-parsed GemmPlan, persistent worker pool, or new factorization was implemented.
