# SDPX dense MultiFloat provider readiness

Date: 2026-08-13

## Audit scope

This is a read-only comparison of:

- MFLA `main` after the dense-provider roadmap through Phase O;
- local SDPX checkout commit `b1fe70e79e3ae69d8ed1da8132ec1f32ad7b306f`.

The SDPX worktree contained pre-existing uncommitted changes in solver,
workspace, Schur, pipeline, and Q3 files. MFLA did not modify that repository.
The table reflects the inspected worktree and cites its source locations so a
future adapter review can refresh the evidence.

MFLA supports `Float64x2`, `Float64x3`, and `Float64x4` for every ordinary
operation below. The same-type dense surface also supports `Float64x1`, but x1
is not an SDPX replacement target. `capabilities(T)` is the authoritative pure
machine-readable query. This Float64-based x1-x4 range is the complete
production scalar contract; other base types and x5-x8 are unsupported.

Capability facts make factor storage ownership explicit: metadata is
factor-owned, the factor matrix borrows the destructive input, factorization
is destructive, and factor solves do not mutate the factor.

## Provider matrix

| SDPX operation and evidence | MFLA API | Workspace | Diagnostics | Remaining gap / ownership |
|---|---|---|---|---|
| Dense vector/Frobenius dot (`src/kernels/api.jl:13`, `src/kernels/generic.jl:20`, Schur and threaded reductions) | `mfdot(x, y)` | No scratch required | Caller receives scalar fact | Complete for dense real MultiFloat arrays. COO/CSC traversal stays in SDPX. |
| Normal and transpose matvec (`src/schur.jl:36`, `src/step.jl:45`, `src/kkt.jl:2140`) | `gemv!(...; trans=:N/:T)` | Caller-owned output | Deterministic route contract | Thin signature/ownership adapter; formulation and matrix layout stay in SDPX. |
| General dense matrix product (`src/kernels/api.jl:22`, `src/kernels/extended_precision_blas/gemm.jl:9`) | `gemm!`; `gemm_plan`; explicit `GemmProfile` | `GemmWorkspace` or `MFWorkspace` for packed panels | Inspectable strategy/reason and machine/type-compatible profile | Replace duplicated fixed-MultiFloat GEMM dispatch. SDPX retains mutable BigFloat-owned kernels. |
| Symmetric matrix-vector (`src/kkt.jl:2940`) | `symv!(...; uplo=:lower/:upper)` | No numerical scratch | Authoritative triangle and deterministic output ownership | Thin adapter; SDPX retains Schur storage and worker policy. |
| Dense Gram/lower SYRK (`src/schur.jl:388`, `src/kernels/extended_precision_blas/syrk.jl:308`) | `syrk!` | Caller-owned matrix | Lower-authoritative contract | Replace duplicated fixed-MultiFloat SYRK. SDPX decides dense/sparse route and merging. |
| Compact packed Gram (`src/schur.jl:854`, `src/kernels/extended_precision_blas/syrk.jl:629`) | `syrk_packed!` with explicit reduction range | Caller-owned packed vector | Deterministic packed layout | Complete for block-local fixed-MultiFloat panels. |
| Global-ID Schur scatter (`src/kernels/extended_precision_blas/syrk.jl:700`, `src/schur.jl:868`) | No public scatter API | Not applicable | Not applicable | Deliberately remains in SDPX: global IDs, collisions, ownership, and assembly order are solver storage policy. |
| Triangular vector/matrix solve (`src/kernels/api.jl:49`, `src/kkt.jl:1061`) | `trsv!`, `trsm!` | Caller-owned RHS | Explicit lower/upper, transpose, unit diagonal | Thin signature adapter. Vector factor solves already route through TRSV. |
| Triangular matrix multiply (`src/kernels/api.jl:69`, `src/schur.jl:479`) | `trmm!` | Caller-owned RHS | Authoritative triangle | Complete numerical primitive; SDPX retains block transform sequence. |
| SPD factor and solve (`src/kernels/api.jl:78`, `src/kkt.jl:702`) | `cholesky!`, `ldiv!`, `solve` | Factor matrix caller-owned; RHS caller-owned | `factor_status`, diagonal range/spread, failure pivot, finite fact | Adapter maps factor object/status to SDPX storage. Regularization and retry stay in SDPX. |
| General factor and solve | `lu!`, `ldiv!`, `solve` | `MFWorkspace` reuses pivots/GEMM panels | Pivot vector, accepted pivot range, growth, failure location | Available as explicit caller-selected alternative; MFLA never selects it after another factor fails. |
| Symmetric-indefinite KKT (`src/kkt.jl` dense KKT paths) | `ldlt!`, `ldlt_plan`, `ldiv!`, `solve`; lightweight `factor_pivots`, `factor_blocks`, `factor_permutation`, `factor_inertia` | `MFWorkspace` reuses pivots, blocks, weighted panel, GEMM buffers | O(n) structural metadata and inertia; comprehensive block quality/growth and failure location remain in `factor_diagnostics` | Dense primitive complete. SDPX must choose formulation, LDLT route, regularization, and fallback. |
| Equality/null-space rank route (`src/kkt.jl:680`) | `rrqr!`, `factor_permutation`, `numerical_rank`, `apply_q!`, `solve_r!` | `MFWorkspace` reuses reflector/permutation metadata | R diagonal range, permutation, caller-threshold rank | Factor-object adapter and formulation assembly only. Rank threshold and QR-vs-normal-equation choice stay in SDPX. |
| General/symmetric residual (`src/kernels/mixed_precision_kkt.jl` residual paths) | `residual!` with `r=b-A*x` | General multi-RHS may reuse GEMM workspace; output caller-owned | Authoritative triangle | Dense primitive complete. |
| Explicit promoted residual | `residual_mixed!` for x2->x3/x4 and x3->x4 | Output caller-owned | Exact supported pairs in `capabilities` | Dense primitive complete. SDPX chooses whether and when to request it. |
| Backward error and one correction | `normwise_backward_error`, `refinement_correction!` | Output caller-owned | Overflow-safe normwise fact | MFLA performs exactly one requested correction. SDPX owns stopping, stagnation, repetition, acceptance, and certification. |
| Repeated factor/solve cycle | `MFWorkspace`, `workspace_capacity`, `ensure_workspace_capacity!` | Reusable factorization scratch and packed/weighted panels; returned factor metadata is owned | Live factors survive workspace reuse and growth | One workspace may serve sequential factorizations with multiple live factors; concurrent factorization calls use distinct workspaces. |
| Capability selection | `capabilities(T)` | None | Immutable operation facts | SDPX may inspect these facts, but chooses provider and records fallback reason. |

## Code that remains in SDPX

The following are not missing dense MFLA primitives and should not be moved
into this package:

- COO/CSC coefficient traversal, sparse contractions, Schur block incidence,
  global-ID scatter, arrow layouts, and deterministic block merge policy
  (`src/schur.jl` and `src/kernels/extended_precision_blas/syrk.jl`);
- Q3/SOC coordinate products, Jordan algebra, inverses, Nesterov--Todd/HKM
  scaling, cone residuals, and line search (`src/soc_q3_kernels.jl`,
  `src/soc_native_q3.jl`);
- elementwise iterate update and trial construction (`kaxpby!`), raw solver
  norms (`knrmInf`), fraction-to-boundary, PSD trial acceptance, and
  certificate logic (`src/step.jl`, `src/kernels/generic.jl`);
- precision promotion/conversion guards, Float64 preconditioner selection,
  adaptive refinement, regularization loops, and all fallback decisions
  (`src/kernels/mixed_precision_kkt.jl` and KKT orchestration);
- KKT formulation and assembly, equality normal-equation versus QR choice,
  convergence tolerances, and provider reporting.

MFLA intentionally does not add generic public `axpby!` or raw array infinity
norm APIs merely to replace short solver-local loops. Those operations do not
constitute a dense factor/solve backend gap. A future concrete performance
profile could justify a Level-1 kernel, but solver stopping semantics would
still remain downstream.

## Remaining dense numerical gaps

For the intended solver-independent, fixed-MultiFloat dense provider layer,
no mandatory primitive remains for SDPX's existing normal-equation,
augmented-KKT, or equality-RRQR routes. The remaining work is an SDPX adapter
and deletion/A-B validation of its duplicated fixed-MultiFloat GEMM, SYRK,
packed SYRK, symmetric matvec, and triangular wrappers.

Two optional future kernels may be considered only with new measured callers:

1. a generalized symmetric rank update or scatter-neutral batched Gram API;
2. a Level-1 array `axpby!`/maximum-absolute-value kernel if profiling proves
   solver-local loops material at x2/x3/x4.

Neither is required to call MFLA as SDPX's complete dense MultiFloat numerical
provider today. A global-ID scatter API remains explicitly rejected because it
would encode SDPX assembly policy.

## Required adapter behavior

The adapter should preserve this direction:

```text
SDPX ExecutionPlan chooses formulation, provider, factor kind, and precision
    -> MFLA executes the explicitly requested factor/solve/residual operation
    -> MFLA returns factors, route facts, diagnostics, residual, and correction
    -> SDPX decides fallback, refinement, convergence, and certification
```

It must never turn an MFLA failure into an implicit fallback inside MFLA. In
particular, these transitions remain visible SDPX decisions:

```text
LDLT -> QR -> LU
x2 -> x3 -> x4
ordinary residual -> promoted residual
one correction -> another correction
```

## Readiness conclusion

MFLA is ready to serve as SDPX's complete **dense fixed-MultiFloat numerical
provider** for x2/x3/x4. This means dense kernels, Cholesky/LU/LDLT/RRQR,
vector/multi-RHS solves, numerical diagnostics, residual/backward error,
explicit promoted residual, one requested correction, workspace reuse, and
capability facts are present and adversarially tested.

It is not, and should not become, a complete SDPX kernel namespace. SDPX still
requires its solver-specific sparse assembly, Q3/SOC algebra, iterate updates,
line search, precision/fallback orchestration, and certification code.

The next logical development step is therefore the downstream SDPX adapter and
duplicate-kernel A/B removal, not speculative expansion of MFLA. If MFLA work
continues independently, evidence should choose among generalized update
closure, more formal arithmetic proof, or a concrete new provider caller; no
current audit evidence justifies hidden policy or a broad architecture rewrite.
