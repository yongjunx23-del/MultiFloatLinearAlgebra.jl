"""
    capabilities(T::Type{<:MultiFloat}) -> NamedTuple

Return immutable, machine-readable facts about the dense backend surface for a
Float64-based MultiFloat type. This query never benchmarks, calibrates, reads
solver state, or mutates package state.

One- through four-limb types report the ordinary same-type kernel and factor
surface. `mixed_residual_targets` states exactly which explicitly higher-limb
residual output types are accepted. Unsupported limb counts return
`supported=false` and false for every operation; calls to those operations
continue to fail explicitly rather than selecting a fallback.
"""
function capabilities(
    ::Type{MultiFloat{Float64,N}},
) where {N}
    supported = 1 <= N <= 4
    mixed_residual_targets = (
        x2=false,
        x3=supported && N == 2,
        x4=supported && N in (2, 3),
    )
    mixed_precision_residual = any(values(mixed_residual_targets))
    return (
        supported=supported,
        limb_count=N,
        dot=supported,
        gemv=supported,
        transpose_gemv=supported,
        symv=supported,
        gemm=supported,
        gemmt=supported,
        syrk=supported,
        syrk_packed=supported,
        trsv=supported,
        trsm=supported,
        trmm=supported,
        cholesky=supported,
        lu=supported,
        ldlt=supported,
        rrqr=supported,
        factor_diagnostics=supported,
        apply_q=supported,
        solve_r=supported,
        vector_rhs=supported,
        multi_rhs=supported,
        residual=supported,
        mixed_precision_residual=mixed_precision_residual,
        mixed_residual_targets=mixed_residual_targets,
        refinement_correction=supported,
        reusable_workspace=supported,
        threading=supported,
    )
end
