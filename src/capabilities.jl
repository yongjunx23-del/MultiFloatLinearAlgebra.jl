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
    ::Type{MF},
) where {N,MF<:MultiFloat{Float64,N}}
    supported = 1 <= N <= 4
    # The allocation gate (benchmark/allocation_gate.jl) verifies the warm-path
    # zero-allocation facts for x2/x3/x4 only; it does not test x1, so the
    # claims are gated to the multi-limb types (N >= 2).
    zero_alloc_gated = supported && N >= 2
    mixed_residual_targets = (
        x2=false,
        x3=supported && N == 2,
        x4=supported && N in (2, 3),
    )
    mixed_precision_residual = any(values(mixed_residual_targets))
    mixed_residual_target_types = if supported && N == 2
        (MultiFloat{Float64,3}, MultiFloat{Float64,4})
    elseif supported && N == 3
        (MultiFloat{Float64,4},)
    else
        ()
    end
    return (
        provider=:mfla,
        scalar_type=MF,
        base_type=Float64,
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
        ldlt_lightweight_metadata=supported,
        factor_diagnostics=supported,
        apply_q=supported,
        solve_r=supported,
        vector_rhs=supported,
        multi_rhs=supported,
        residual=supported,
        mixed_precision_residual=mixed_precision_residual,
        mixed_residual_targets=mixed_residual_targets,
        mixed_residual_target_types=mixed_residual_target_types,
        refinement_correction=supported,
        reusable_workspace=supported,
        factor_cache=supported,
        factor_cache_kinds=(
            cholesky=supported, lu=supported, ldlt=supported, rrqr=supported,
        ),
        factor_cache_ownership=:cache_owned,
        # Warm-path zero-allocation facts, gated to the multi-limb types the
        # allocation gate actually verifies (x2/x3/x4); x1 is not claimed.
        factor_cache_warm_vector_solve_zero_alloc=zero_alloc_gated,
        factor_cache_warm_matrix_solve_zero_alloc=zero_alloc_gated,
        factor_metadata_ownership=:factor_owned,
        factor_matrix_ownership=:borrowed_input,
        factorization_destructive=true,
        factor_solve_mutates_factor=false,
        shared_gemm_workspace_concurrency=:serialized_safe,
        concurrent_factor_workspace=false,
        syrk_authoritative_triangle=:lower,
        syrk_inactive_triangle=:preserved,
        threading=supported,
    )
end
