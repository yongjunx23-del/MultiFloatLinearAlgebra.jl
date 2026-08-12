struct MFLDLT{MF<:MultiFloat,M<:AbstractMatrix{MF}}
    factors::M
    dsub::Vector{MF}
    pivots::Vector{Int}
    blocks::Vector{UInt8}
    info::Int
end

issuccess(F::MFLDLT) = iszero(F.info)

function _mirror_lower_to_upper!(A::AbstractMatrix)
    n = size(A, 1)
    @inbounds for column in 1:n
        for row in (column + 1):n
            A[column, row] = A[row, column]
        end
    end
    return A
end

function _ldlt_symmetric_swap!(
    A::AbstractMatrix,
    first::Int,
    second::Int,
    active_first::Int,
)
    first == second && return nothing
    n = size(A, 1)

    # Previously computed L columns move with the active row only.
    @inbounds for column in 1:(active_first - 1)
        A[first, column], A[second, column] =
            A[second, column], A[first, column]
    end

    # The active Schur complement is still stored symmetrically, so apply
    # P*A*P' to that submatrix by one row swap followed by one column swap.
    @inbounds for column in active_first:n
        A[first, column], A[second, column] =
            A[second, column], A[first, column]
    end
    @inbounds for row in active_first:n
        A[row, first], A[row, second] =
            A[row, second], A[row, first]
    end
    return nothing
end

function _select_bk_pivot(
    A::AbstractMatrix{MF},
    k::Int,
    alpha::MF,
) where {MF<:MultiFloat}
    n = size(A, 1)
    absakk = abs(A[k, k])
    k == n && return iszero(absakk) ? (0, k) : (1, k)

    imax = k + 1
    colmax = abs(A[imax, k])
    @inbounds for row in (k + 2):n
        candidate = abs(A[row, k])
        if candidate > colmax
            colmax = candidate
            imax = row
        end
    end

    max(absakk, colmax) == zero(MF) && return (0, k)
    absakk >= alpha * colmax && return (1, k)

    rowmax = zero(MF)
    @inbounds for column in k:(imax - 1)
        rowmax = max(rowmax, abs(A[imax, column]))
    end
    @inbounds for row in (imax + 1):n
        rowmax = max(rowmax, abs(A[row, imax]))
    end

    if absakk >= alpha * colmax * (colmax / rowmax)
        return (1, k)
    elseif abs(A[imax, imax]) >= alpha * rowmax
        return (1, imax)
    else
        return (2, imax)
    end
end

function _ldlt_rank1_update!(
    A::AbstractMatrix{MF},
    k::Int,
    d::MF,
) where {T,N,MF<:MultiFloat{T,N}}
    n = size(A, 1)
    V4 = MultiFloatVec{4,T,N}

    @inbounds for column in (k + 1):n
        coefficient = d * A[column, k]
        row = column
        while row + 3 <= n
            values = V4(
                A[row, column],
                A[row + 1, column],
                A[row + 2, column],
                A[row + 3, column],
            )
            factors = V4(
                A[row, k],
                A[row + 1, k],
                A[row + 2, k],
                A[row + 3, k],
            )
            values -= factors * V4(coefficient)
            for lane in 1:4
                i = row + lane - 1
                value = values[lane]
                A[i, column] = value
                A[column, i] = value
            end
            row += 4
        end
        while row <= n
            value = A[row, column] - A[row, k] * coefficient
            A[row, column] = value
            A[column, row] = value
            row += 1
        end
    end
    return nothing
end

function _ldlt_rank2_update!(
    A::AbstractMatrix{MF},
    k::Int,
    d11::MF,
    d21::MF,
    d22::MF,
) where {T,N,MF<:MultiFloat{T,N}}
    n = size(A, 1)
    V4 = MultiFloatVec{4,T,N}

    @inbounds for column in (k + 2):n
        l1 = A[column, k]
        l2 = A[column, k + 1]
        coefficient1 = d11 * l1 + d21 * l2
        coefficient2 = d21 * l1 + d22 * l2
        row = column
        while row + 3 <= n
            values = V4(
                A[row, column],
                A[row + 1, column],
                A[row + 2, column],
                A[row + 3, column],
            )
            first_factors = V4(
                A[row, k],
                A[row + 1, k],
                A[row + 2, k],
                A[row + 3, k],
            )
            second_factors = V4(
                A[row, k + 1],
                A[row + 1, k + 1],
                A[row + 2, k + 1],
                A[row + 3, k + 1],
            )
            values -= first_factors * V4(coefficient1) +
                      second_factors * V4(coefficient2)
            for lane in 1:4
                i = row + lane - 1
                value = values[lane]
                A[i, column] = value
                A[column, i] = value
            end
            row += 4
        end
        while row <= n
            value = A[row, column] -
                    A[row, k] * coefficient1 -
                    A[row, k + 1] * coefficient2
            A[row, column] = value
            A[column, row] = value
            row += 1
        end
    end
    return nothing
end

"""
    ldlt!(A; check=true)

Symmetric-indefinite `L*D*L'` factorization for dense MultiFloat matrices.
The lower triangle of `A` is authoritative on input. Pivot selection follows
the Bunch--Kaufman threshold strategy and may use 1x1 or 2x2 diagonal blocks.

On return, the strict lower triangle stores unit-lower `L`, the diagonal stores
the diagonal of `D`, and `F.dsub[k]` stores the off-diagonal of a 2x2 `D`
block starting at `k`. `F.blocks[k]` is 1 or 2 at a block start (and zero at
the second index of a 2x2 block). The recorded pivot sequence is sufficient
to apply and undo the symmetric permutation without forming a permutation
matrix.
"""
function ldlt!(
    A::AbstractMatrix{MF};
    check::Bool=true,
) where {MF<:MultiFloat}
    n, m = size(A)
    n == m || throw(DimensionMismatch("ldlt! requires a square matrix"))
    _check_supported(MF)
    _mirror_lower_to_upper!(A)

    dsub = zeros(MF, n)
    pivots = collect(1:n)
    blocks = zeros(UInt8, n)
    alpha = (one(MF) + sqrt(MF(17))) / MF(8)
    info = 0
    k = 1

    while k <= n
        block_size, pivot = _select_bk_pivot(A, k, alpha)
        if block_size == 0
            info = k
            break
        elseif block_size == 1
            pivots[k] = pivot
            blocks[k] = UInt8(1)
            _ldlt_symmetric_swap!(A, k, pivot, k)

            d = A[k, k]
            if iszero(d)
                info = k
                break
            end
            @inbounds for row in (k + 1):n
                A[row, k] /= d
            end
            k < n && _ldlt_rank1_update!(A, k, d)
            k += 1
        else
            k < n || begin
                info = k
                break
            end
            pivots[k] = pivot
            blocks[k] = UInt8(2)
            blocks[k + 1] = UInt8(0)
            _ldlt_symmetric_swap!(A, k + 1, pivot, k)

            d11 = A[k, k]
            d21 = A[k + 1, k]
            d22 = A[k + 1, k + 1]
            determinant = d11 * d22 - d21 * d21
            if iszero(determinant)
                info = k
                break
            end

            @inbounds for row in (k + 2):n
                first = A[row, k]
                second = A[row, k + 1]
                A[row, k] =
                    (first * d22 - second * d21) / determinant
                A[row, k + 1] =
                    (second * d11 - first * d21) / determinant
            end

            dsub[k] = d21
            if k + 1 < n
                _ldlt_rank2_update!(A, k, d11, d21, d22)
            end
            # The internal subdiagonal belongs to D, not L. Keeping it in a
            # separate vector makes the stored L directly usable by unit TRSM.
            A[k + 1, k] = zero(MF)
            A[k, k + 1] = zero(MF)
            k += 2
        end
    end

    if !iszero(info) && check
        throw(LinearAlgebra.SingularException(info))
    end
    return MFLDLT{MF,typeof(A)}(A, dsub, pivots, blocks, info)
end
