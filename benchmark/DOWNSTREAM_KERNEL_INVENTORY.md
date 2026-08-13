# Downstream dense-kernel inventory

Date: 2026-08-13

This inventory is based on a read-only inspection of the local SDPX checkout.
It records concrete dense MultiFloat duplication without making MFLA depend on
SDPX or moving solver storage policy into the numerical backend.

| Downstream operation | Current SDPX implementation | Nearest MFLA primitive | Remaining capability | Expected call shape |
|---|---|---|---|---|
| Dense Gram update `S += P'P` | `ksyrk!` / `ExtendedPrecisionBLAS.syrk!` | `syrk!(S, P, 1, 1)` | None for authoritative-lower storage | Reduction-by-Schur-width panel, square lower output |
| Compact block Gram triangle | `syrk_packed_triangle!` | `syrk_packed!` | None for fixed MultiFloat arithmetic; explicit reduction slices support deterministic row-bin partials | Small/medium block-local panel, packed vector destination |
| Global Schur scatter | `syrk_scatter_triangle!` and arrow-specific scatter | `syrk!` | Deliberately retained in SDPX: global IDs, collision ownership, and arrow layout are solver assembly policy | Block-local panel scattered into a global lower triangle |
| Triangular matrix multiply | `ktrmm!` in Schur panel transforms | `trmm!` | None | Primarily right multiplication by a lower triangular block matrix |
| Equality rank reduction | Pivoted generic `qr` | `rrqr!`, `apply_q!`, `solve_r!`, `numerical_rank` | Thin factor-object adapter only | Tall equality matrix; vector and matrix Q/Q' application |
| Solve residual / validation | Hand-written matvec, axpby, and infinity norms | `residual!`, `residual_mixed!`, `normwise_backward_error`, `refinement_correction!` | SDPX retains convergence, retry, and certification policy | Dense vector and modest multi-RHS solves |
| Dense Frobenius/vector dot | Generic `kdot` | `mfdot` | None for dense real MultiFloat arrays; COO/sparse traversal remains downstream | Dense block diagnostics and reductions |

No concrete SDPX use of a symmetric rank-2k update was found. Consequently,
MFLA does not add `syr2k!` or a public scatter API in Phase D1. Global-ID
scatter can be reconsidered only with a caller whose contract is independent
of solver formulation and storage ownership.

The final read-only comparison and complete provider table are recorded in
[`../docs/SDPX_PROVIDER_READINESS.md`](../docs/SDPX_PROVIDER_READINESS.md).

The existing `syrk!` and `gemmt!` lower-triangle forms already cover MFLA's
Cholesky and LDLT updates and SDPX's ordinary dense Gram update. Upper and
no-transpose variants are not added solely for BLAS surface completeness.
