struct MFLU{MF<:MultiFloat,M<:AbstractMatrix{MF}}
    factors::M
    ipiv::Vector{Int}
    info::Int
end

issuccess(F::MFLU) = iszero(F.info)

@inline function _swap_rows!(
    A::AbstractMatrix,
    first::Int,
    second::Int,
)
    first == second && return nothing
    @inbounds for column in axes(A, 2)
        A[first, column], A[second, column] =
            A[second, column], A[first, column]
    end
    return nothing
end

function _lu_rank1_update!(
    A::AbstractMatrix{MF},
    k::Int,
) where {T,N,MF<:MultiFloat{T,N}}
    m, n = size(A)
    V4 = MultiFloatVec{4,T,N}
    @inbounds for column in (k + 1):n
        multiplier = V4(A[k, column])
        row = k + 1
        while row + 3 <= m
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
            values -= factors * multiplier
            for lane in 1:4
                A[row + lane - 1, column] = values[lane]
            end
            row += 4
        end
        while row <= m
            A[row, column] -= A[row, k] * A[k, column]
            row += 1
        end
    end
    return nothing
end

"""
    lu!(A; check=true)

Dense partial-pivoting LU specialized for MultiFloat matrices. Row updates are
vectorized across four independent rows. The factorization stores unit-lower
`L` below the diagonal and `U` on/above the diagonal.
"""
function lu!(
    A::AbstractMatrix{MF};
    check::Bool=true,
) where {MF<:MultiFloat}
    m, n = size(A)
    _check_supported(MF)
    kmax = min(m, n)
    ipiv = Vector{Int}(undef, kmax)
    info = 0

    @inbounds for k in 1:kmax
        pivot = k
        pivot_value = abs(A[k, k])
        for row in (k + 1):m
            candidate = abs(A[row, k])
            if candidate > pivot_value
                pivot = row
                pivot_value = candidate
            end
        end
        ipiv[k] = pivot
        _swap_rows!(A, k, pivot)

        if iszero(A[k, k])
            info = k
            break
        end

        inverse_pivot = inv(A[k, k])
        for row in (k + 1):m
            A[row, k] *= inverse_pivot
        end
        k < n && _lu_rank1_update!(A, k)
    end

    if !iszero(info) && check
        throw(LinearAlgebra.SingularException(info))
    end
    return MFLU{MF,typeof(A)}(A, ipiv, info)
end
