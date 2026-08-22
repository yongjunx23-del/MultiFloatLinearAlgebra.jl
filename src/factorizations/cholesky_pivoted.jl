"""
    MFCholeskyPivoted

Rank-revealing lower Cholesky factorization with symmetric diagonal
pivoting.  On return, the lower triangle of `factors` stores `L` and
`factors[permutation, permutation]` has the leading representation
`L[:, 1:rank] * transpose(L[:, 1:rank])` up to the retained rank.
"""
struct MFCholeskyPivoted{
    MF<:MultiFloat,
    M<:AbstractMatrix{MF},
    P<:AbstractVector{Int},
} <: AbstractMFFactorization{MF}
    factors::M
    permutation::P
    rank::Int
    info::Int
end

factor_kind(::MFCholeskyPivoted) = :cholesky_pivoted
factor_status(F::MFCholeskyPivoted) = F.info
factor_matrix(F::MFCholeskyPivoted) = F.factors

function factor_state(F::MFCholeskyPivoted)
    F.info == 0 && return :success
    F.info == -1 && return :nonfinite_input
    return :not_posdef
end

"""
    factor_permutation(F::MFCholeskyPivoted) -> Vector{Int}

Return the symmetric permutation `p` for which the leading retained block of
`A[p, p]` is represented by the Cholesky factor in `F`.
"""
function factor_permutation(F::MFCholeskyPivoted)
    return copy(F.permutation)
end

"""Return a copy of the stored diagonal pivots of `L`."""
function factor_rdiag(F::MFCholeskyPivoted{MF}) where {MF<:MultiFloat}
    count = min(size(F.factors)...)
    diagonal = Vector{MF}(undef, count)
    @inbounds for index in 1:count
        diagonal[index] = F.factors[index, index]
    end
    return diagonal
end

"""
    cholesky_pivoted!(A; tol=zero(eltype(A)), check=false)

Compute a dense rank-revealing pivoted Cholesky factorization of a symmetric
positive-semidefinite MultiFloat matrix.  The lower triangle is authoritative
on input; it is mirrored before factorization.  At each step the largest
remaining diagonal is selected, rows and columns are swapped symmetrically,
and the trailing Schur complement receives a rank-one update.

`tol` is a relative pivot threshold.  A pivot is retained when it is strictly
larger than `tol * first_pivot`; a tolerance stop is a successful rank reveal
and is reported by `rank`, not by `factor_status`.  A non-positive pivot is a
numerical factorization failure (`factor_status > 0`) unless `check=true`, in
which case `LinearAlgebra.PosDefException` is thrown.
"""
function cholesky_pivoted!(
    A::AbstractMatrix{MF};
    tol::Real=zero(MF),
    check::Bool=true,
) where {MF<:MultiFloat}
    n, m = size(A)
    n == m || throw(DimensionMismatch(
        "cholesky_pivoted! requires a square matrix",
    ))
    _check_supported(MF)
    Base.require_one_based_indexing(A)
    relative_tolerance = MF(tol)
    isfinite(relative_tolerance) && relative_tolerance >= zero(MF) ||
        throw(ArgumentError("tol must be finite and nonnegative"))
    if !_lower_triangle_finite(A)
        check && throw(DomainError(A,
            "cholesky_pivoted!: input matrix contains non-finite entries"))
        return MFCholeskyPivoted{MF,typeof(A),Vector{Int}}(
            A, collect(1:n), 0, -1,
        )
    end

    # Only the lower triangle is part of the input contract.  Mirroring once
    # makes all subsequent symmetric swaps and Schur updates deterministic.
    @inbounds for column in 1:n, row in 1:(column - 1)
        A[row, column] = A[column, row]
    end

    permutation = collect(1:n)
    rank = n
    info = 0
    first_pivot = zero(MF)

    @inbounds for k in 1:n
        pivot_index = k
        pivot_diagonal = A[k, k]
        for index in (k + 1):n
            candidate = A[index, index]
            if candidate > pivot_diagonal
                pivot_index = index
                pivot_diagonal = candidate
            end
        end

        if k == 1
            pivot_diagonal > zero(MF) || begin
                rank = 0
                info = 1
                break
            end
            first_pivot = sqrt(pivot_diagonal)
        end

        if pivot_diagonal <= zero(MF)
            # A PSD rank reveal can leave a tiny negative Schur pivot after
            # high-precision cancellation.  A caller-supplied tolerance makes
            # that roundoff-sized pivot a legitimate rank stop; without one,
            # retain the fail-closed non-positive-pivot status.
            threshold = relative_tolerance * first_pivot
            if !iszero(relative_tolerance) &&
               abs(pivot_diagonal) <= threshold * threshold
                rank = k - 1
                break
            end
            rank = k - 1
            info = k
            break
        end
        pivot = sqrt(pivot_diagonal)
        if !iszero(relative_tolerance) &&
           pivot <= relative_tolerance * first_pivot
            rank = k - 1
            break
        end

        if pivot_index != k
            for column in 1:n
                A[k, column], A[pivot_index, column] =
                    A[pivot_index, column], A[k, column]
            end
            for row in 1:n
                A[row, k], A[row, pivot_index] =
                    A[row, pivot_index], A[row, k]
            end
            permutation[k], permutation[pivot_index] =
                permutation[pivot_index], permutation[k]
        end

        A[k, k] = pivot
        for row in (k + 1):n
            A[row, k] /= pivot
        end

        # Full symmetric rank-one update.  The pivot column is untouched, so
        # its entries remain available for the next update and the row/column
        # ordering is exactly the PSTRF max-residual rule.
        for column in (k + 1):n
            factor = A[column, k]
            for row in (k + 1):n
                A[row, column] -= A[row, k] * factor
            end
        end
    end

    if !iszero(info) && check
        throw(LinearAlgebra.PosDefException(info))
    end
    return MFCholeskyPivoted{MF,typeof(A),typeof(permutation)}(
        A, permutation, rank, info,
    )
end
