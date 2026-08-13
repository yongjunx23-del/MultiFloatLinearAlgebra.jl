function _trmm_left!(
    B::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    alpha::MF,
    effective_lower::Bool,
    transposed::Bool,
    unit_diagonal::Bool,
) where {MF<:MultiFloat}
    m = size(A, 1)
    n = size(B, 2)
    @inbounds for j in 1:n
        for i in (effective_lower ? (m:-1:1) : (1:m))
            accumulator = zero(MF)
            if effective_lower
                for k in 1:(i - 1)
                    accumulator += _trsm_coefficient(A, i, k, transposed) * B[k, j]
                end
                if unit_diagonal
                    accumulator += B[i, j]
                else
                    accumulator += A[i, i] * B[i, j]
                end
            else
                if unit_diagonal
                    accumulator += B[i, j]
                else
                    accumulator += A[i, i] * B[i, j]
                end
                for k in (i + 1):m
                    accumulator += _trsm_coefficient(A, i, k, transposed) * B[k, j]
                end
            end
            B[i, j] = alpha * accumulator
        end
    end
    return B
end

function _trmm_right!(
    B::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    alpha::MF,
    effective_lower::Bool,
    transposed::Bool,
    unit_diagonal::Bool,
) where {MF<:MultiFloat}
    m = size(B, 1)
    n = size(A, 1)
    @inbounds for i in 1:m
        # Columns needed by the current result must still contain their input
        # values: lower products consume columns to the right, upper products
        # consume columns to the left.
        for j in (effective_lower ? (1:n) : (n:-1:1))
            accumulator = zero(MF)
            if effective_lower
                if unit_diagonal
                    accumulator += B[i, j]
                else
                    accumulator += B[i, j] * A[j, j]
                end
                for k in (j + 1):n
                    accumulator += B[i, k] * _trsm_coefficient(A, k, j, transposed)
                end
            else
                for k in 1:(j - 1)
                    accumulator += B[i, k] * _trsm_coefficient(A, k, j, transposed)
                end
                if unit_diagonal
                    accumulator += B[i, j]
                else
                    accumulator += B[i, j] * A[j, j]
                end
            end
            B[i, j] = alpha * accumulator
        end
    end
    return B
end

"""
    trmm!(B, A, alpha=one(eltype(B));
          side=:left, uplo=:lower, trans=:N, diag=:nonunit,
          config=KernelConfig())

Triangular matrix multiply. Computes

`B = alpha * op(A) * B` when `side=:left`, or
`B = alpha * B * op(A)` when `side=:right`,

in place, where `op(A)` is `A` or `transpose(A)` according to `trans`. `uplo`
selects the authoritative triangle and `diag=:unit` treats the diagonal of
`op(A)` as unit. Only the selected triangle is read; the inactive triangle may
contain arbitrary values.

The reduction is ascending in the inner dimension, matching the package's
deterministic ordering convention. There is no parallel path because the
operation is inherently triangular. `B` must not alias `A`.
"""
function trmm!(
    B::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    alpha::MF=one(MF);
    side::Symbol=:left,
    uplo::Symbol=:lower,
    trans::Symbol=:N,
    diag::Symbol=:nonunit,
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    n, m = size(A)
    n == m || throw(DimensionMismatch("trmm! requires a square triangular matrix"))
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    uplo in (:lower, :upper) || throw(ArgumentError("uplo must be :lower or :upper"))
    trans in (:N, :T) || throw(ArgumentError("trans must be :N or :T"))
    diag in (:unit, :nonunit) || throw(ArgumentError("diag must be :unit or :nonunit"))
    _check_supported(MF)
    Base.require_one_based_indexing(B, A)
    _require_no_output_alias("trmm!", B, A)

    transposed = trans === :T
    effective_lower = (uplo === :lower) == !transposed
    unit_diagonal = diag === :unit

    if side === :left
        size(B, 1) == n ||
            throw(DimensionMismatch("trmm! left dimensions differ"))
        return _trmm_left!(B, A, alpha, effective_lower, transposed, unit_diagonal)
    end
    size(B, 2) == n ||
        throw(DimensionMismatch("trmm! right dimensions differ"))
    return _trmm_right!(B, A, alpha, effective_lower, transposed, unit_diagonal)
end
