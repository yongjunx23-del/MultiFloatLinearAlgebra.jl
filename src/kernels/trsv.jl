"""
    trsv!(x, A, alpha=one(eltype(x));
          uplo=:lower, trans=:N, diag=:nonunit, config=KernelConfig())

Solve the single right-hand-side triangular system

`op(A) * x = alpha * b`

in place, overwriting `x` (which holds `b` on entry) with the solution. `op(A)`
is `A` or `transpose(A)` according to `trans`. `uplo` selects which triangle of
`A` is authoritative and `diag=:unit` treats the diagonal as unit.

This is the single-column counterpart of [`trsm!`](@ref) and follows the same
numerical ordering: forward substitution for the effective lower triangle and
backward substitution for the effective upper triangle. The triangular
dependency is inherently serial, so there is no parallel path.
`x` must not alias `A`.
"""
function trsv!(
    x::AbstractVector{MF},
    A::AbstractMatrix{MF},
    alpha::MF=one(MF);
    uplo::Symbol=:lower,
    trans::Symbol=:N,
    diag::Symbol=:nonunit,
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    n, m = size(A)
    n == m || throw(DimensionMismatch("trsv! requires a square triangular matrix"))
    length(x) == n ||
        throw(DimensionMismatch("trsv! right-hand side length differs"))
    uplo in (:lower, :upper) || throw(ArgumentError("uplo must be :lower or :upper"))
    trans in (:N, :T) || throw(ArgumentError("trans must be :N or :T"))
    diag in (:unit, :nonunit) || throw(ArgumentError("diag must be :unit or :nonunit"))
    _check_supported(MF)
    Base.require_one_based_indexing(x, A)
    _require_no_output_alias("trsv!", x, A)

    if alpha != one(MF)
        @inbounds for index in eachindex(x)
            x[index] *= alpha
        end
    end

    transposed = trans === :T
    effective_lower = (uplo === :lower) == !transposed
    unit_diagonal = diag === :unit

    if effective_lower
        @inbounds for row in 1:n
            value = x[row]
            for k in 1:(row - 1)
                value -= _trsm_coefficient(A, row, k, transposed) * x[k]
            end
            if !unit_diagonal
                value /= A[row, row]
            end
            x[row] = value
        end
    else
        @inbounds for row in n:-1:1
            value = x[row]
            for k in (row + 1):n
                value -= _trsm_coefficient(A, row, k, transposed) * x[k]
            end
            if !unit_diagonal
                value /= A[row, row]
            end
            x[row] = value
        end
    end
    return x
end
