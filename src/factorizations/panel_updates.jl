# M03 — LDLT panel updates: the trailing-update mechanics, and the
# lower-only vs mirrored comparison the task card asks for.
#
# STATUS: add-only module file written by the M03 worker; not in the include
# graph at the time of writing (probed — see the header of `pivot_policy.jl`).
#
# WHAT THIS FILE MEASURES AND WHY IT EXISTS
#
# MFLA's blocked LDLT ends every trailing update with a full lower-to-upper
# mirror (`_mirror_lower_to_upper_block!`, called from
# `_ldlt_block_trailing_update_viewfree!`), because the *next* panel's pivot
# search reads a symmetric trailing block. The task card asks for a
# lower-only-accessor comparison against that mirrored baseline, "not changing
# all pivot logic at once", and warns that an O(n^2) copy should not be removed
# on the assumption that it is free.
#
# So this file implements the SAME blocked algorithm twice, changing exactly
# one thing: `:mirror` performs the full symmetric trailing update (rank-1 and
# rank-2 outer products written to both triangles, which is what the stored
# rank-2 panel actually computes), while `:lower_only` writes only `row >=
# column`. Panel factorization, pivot selection, block grammar and the solve
# path are byte-identical between the two arms. The driver
#
#   * asserts the two arms produce bit-identical lower triangles and identical
#     grammars (so the comparison is a like-for-like cost comparison), and
#   * asserts `:lower_only` leaves the trailing UPPER triangle bit-unchanged
#     (so "it never reads the mirror" is a measured statement, not a claim), and
#   * reports the wall-time delta as a fraction of the end-to-end total, not as
#     a single-kernel speedup.
#
# This is an EXPERIMENT, not a second production factorization: neither routine
# is reachable from `MultiFloatLinearAlgebra.ldlt!`, nothing in the package
# calls it, and the driver asserts that. `LDLTPlan`/`KernelConfig` are not
# honoured here on purpose — a config surface would make it look selectable.

const M03_PANEL_UPDATES_VERSION = v"1.0.0"

# Same include-time guard as pivot_policy.jl: this file must land in the package
# namespace, which is why it also `include`s pivot_policy.jl (guarded) so it is
# usable stand-alone without duplicating `M03PivotGrammar`.
@isdefined(MultiFloatLinearAlgebra) || error(
    "M03: include this file into MultiFloatLinearAlgebra, not into a sandbox " *
    "module: Base.include(MultiFloatLinearAlgebra, " *
    "\"src/factorizations/panel_updates.jl\")")

@isdefined(M03PivotGrammar) || Base.include(
    @__MODULE__, joinpath(@__DIR__, "pivot_policy.jl"))

const M03_DEFAULT_ALPHA_NUMERATOR = 17

"""
    m03_bk_alpha(::Type{MF}) -> MF

The Bunch--Kaufman threshold `(1 + sqrt(17)) / 8`, identical to the constant
`_ldlt_factorize_core!` builds, so both experiment arms use the production
pivot threshold.
"""
function m03_bk_alpha(::Type{MF}) where {MF<:MultiFloat}
    return (one(MF) + sqrt(MF(M03_DEFAULT_ALPHA_NUMERATOR))) / MF(8)
end

"""
    m03_mirror_lower_to_upper!(A, first) -> A

Independent re-implementation of the mirror step, restricted to
`A[first:end, first:end]`. Kept local so that the `:lower_only` arm is not
accidentally calling a mirrored helper, and so the driver can diff the two
arms' behaviour rather than their source text.
"""
function m03_mirror_lower_to_upper!(A::AbstractMatrix, first::Int)
    n = size(A, 1)
    @inbounds for column in first:n
        for row in first:column
            A[column, row] = A[row, column]
        end
    end
    return A
end

function _m03_trailing_rank1!(A::AbstractMatrix{MF}, k::Int, d::MF,
                              trailing_first::Int,
                              lower_only::Bool) where {MF<:MultiFloat}
    n = size(A, 1)
    @inbounds for column in trailing_first:n
        coefficient = d * A[column, k]
        for row in trailing_first:n
            (!lower_only || row >= column) || continue
            A[row, column] -= A[row, k] * coefficient
        end
    end
    return A
end

function _m03_trailing_rank2!(A::AbstractMatrix{MF}, k::Int,
                              d11::MF, d21::MF, d22::MF,
                              trailing_first::Int,
                              lower_only::Bool) where {MF<:MultiFloat}
    n = size(A, 1)
    @inbounds for column in trailing_first:n
        coefficient_first = d11 * A[column, k] + d21 * A[column, k + 1]
        coefficient_second = d21 * A[column, k] + d22 * A[column, k + 1]
        for row in trailing_first:n
            (!lower_only || row >= column) || continue
            A[row, column] -= A[row, k] * coefficient_first +
                              A[row, k + 1] * coefficient_second
        end
    end
    return A
end

"""
    m03_ldlt_blocked_experiment(A, dsub, pivots, blocks; panel_width=16,
                                variant=:mirror) -> info

Blocked Bunch--Kaufman `L D L'` on the lower triangle of `A`, written in place.

`variant=:mirror` mirrors each trailing update into the upper triangle;
`variant=:lower_only` writes the lower triangle only. Everything else — the
panel pivot search (`_select_bk_panel_pivot`), the lazy-panel Schur evaluation
(`_ldlt_panel_entry`), the symmetric swaps (`_ldlt_symmetric_swap!`) and the
panel row solve (`_m03_solve_2x2`) — is shared code, so the only difference
between the arms is the storage the trailing update touches.

Returns the `info` status: `0` on success, or the 1-based index of the pivot at
which the factorization stopped (matching `ldlt!`'s convention).
"""
function m03_ldlt_blocked_experiment!(A::AbstractMatrix{MF},
                                     dsub::AbstractVector{MF},
                                     pivots::AbstractVector{Int},
                                     blocks::AbstractVector{UInt8};
                                     panel_width::Int=16,
                                     variant::Symbol=:mirror) where {MF<:MultiFloat}
    variant in (:mirror, :lower_only) ||
        throw(ArgumentError("variant must be :mirror or :lower_only"))
    panel_width >= 1 || throw(ArgumentError("panel_width must be positive"))
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("blocked LDLT needs a square matrix"))
    length(dsub) == n && length(pivots) == n && length(blocks) == n ||
        throw(DimensionMismatch("metadata length mismatch"))
    lower_only = variant === :lower_only

    fill!(dsub, zero(MF))
    fill!(blocks, UInt8(0))
    @inbounds for index in 1:n
        pivots[index] = index
    end
    alpha = m03_bk_alpha(MF)

    panel_first = 1
    while panel_first <= n
        requested_last = min(panel_first + panel_width - 1, n)
        info, panel_last = _factor_ldlt_panel!(
            A, panel_first, requested_last, dsub, pivots, blocks, alpha,
        )
        !iszero(info) && return info
        trailing_first = panel_last + 1
        if trailing_first <= n
            q = panel_first
            @inbounds while q <= panel_last
                if blocks[q] == UInt8(1)
                    d = A[q, q]
                    iszero(d) && return q
                    _m03_trailing_rank1!(A, q, d, trailing_first, lower_only)
                    q += 1
                else
                    d11 = A[q, q]
                    d21 = dsub[q]
                    d22 = A[q + 1, q + 1]
                    _m03_trailing_rank2!(
                        A, q, d11, d21, d22, trailing_first, lower_only)
                    q += 2
                end
            end
            lower_only || m03_mirror_lower_to_upper!(A, trailing_first)
        end
        panel_first = panel_last + 1
    end
    return 0
end

"""
    m03_panel_operation_counts(g) -> (rank1, rank2, trailing_elements)

The operation inventory of a factorization with grammar `g`, used to show that
the two experiment arms do the same arithmetic: `rank1`/`rank2` are the number
of block updates, and `trailing_elements` is the total number of stored
elements either arm writes when it is not restricted to the lower triangle.
"""
function m03_panel_operation_counts(g::M03PivotGrammar)
    starts = m03_grammar_block_starts(g)
    rank1 = 0
    rank2 = 0
    trailing_elements = 0
    for k in starts
        if g.blocks[k] == UInt8(1)
            rank1 += 1
            width = g.n - k
        else
            rank2 += 1
            width = g.n - (k + 1)
        end
        trailing_elements += width * width
    end
    return (rank1=rank1, rank2=rank2, trailing_elements=trailing_elements)
end

"""
    m03_iterative_refinement!(x, A, b, factors, dsub, g, p; max_iterations=5,
                              tolerance) -> (iterations, errors)

A caller-side driver loop: solve, measure the normwise backward error, correct,
repeat. Each outer iteration is one `m03_solve!` plus one `residual!` recompute
against the ORIGINAL operator `A` (never against the factor), so the iteration
count is a property of the factor and the arithmetic, not of the harness.

Returns the number of corrections actually applied and the backward error
observed after each of them, starting from the zero correction.
"""
function m03_iterative_refinement!(x::AbstractVector{MF},
                                   A::AbstractMatrix{MF},
                                   b::AbstractVector{MF},
                                   factors::AbstractMatrix{MF},
                                   dsub::AbstractVector{MF},
                                   g::M03PivotGrammar,
                                   p::AbstractVector{<:Integer};
                                   max_iterations::Int=5,
                                   tolerance::Real=0.0) where {MF<:MultiFloat}
    max_iterations >= 0 || throw(ArgumentError("max_iterations must be nonnegative"))
    residual = similar(b)
    errors = Float64[]
    iterations = 0
    x .= zero(MF)
    residual!(residual, A, x, b; uplo=:lower)
    push!(errors, Float64(normwise_backward_error(A, x, b, residual; uplo=:lower)))
    while iterations < max_iterations
        correction = copy(residual)
        m03_solve!(correction, factors, dsub, g, p; trans=:N) || break
        x .+= correction
        residual!(residual, A, x, b; uplo=:lower)
        push!(errors, Float64(
            normwise_backward_error(A, x, b, residual; uplo=:lower)))
        iterations += 1
        errors[end] <= tolerance && break
    end
    return iterations, errors
end
