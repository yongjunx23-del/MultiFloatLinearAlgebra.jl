function ldiv!(
    destination::AbstractVector{MF},
    F::MFCholesky{MF},
) where {MF<:MultiFloat}
    issuccess(F) || throw(LinearAlgebra.PosDefException(F.info))
    A = F.factors
    n = size(A, 1)
    length(destination) == n ||
        throw(DimensionMismatch("right-hand side length differs"))

    @inbounds for row in 1:n
        value = destination[row]
        for column in 1:(row - 1)
            value -= A[row, column] * destination[column]
        end
        destination[row] = value / A[row, row]
    end

    @inbounds for row in n:-1:1
        value = destination[row]
        for column in (row + 1):n
            value -= A[column, row] * destination[column]
        end
        destination[row] = value / A[row, row]
    end
    return destination
end

function ldiv!(
    destination::AbstractMatrix{MF},
    F::MFCholesky{MF},
) where {MF<:MultiFloat}
    size(destination, 1) == size(F.factors, 1) ||
        throw(DimensionMismatch("right-hand side dimensions differ"))
    @inbounds for column in axes(destination, 2)
        ldiv!(@view(destination[:, column]), F)
    end
    return destination
end

function _apply_pivots!(
    destination::AbstractVector,
    ipiv::AbstractVector{Int},
)
    @inbounds for k in eachindex(ipiv)
        pivot = ipiv[k]
        if pivot != k
            destination[k], destination[pivot] =
                destination[pivot], destination[k]
        end
    end
    return destination
end

function ldiv!(
    destination::AbstractVector{MF},
    F::MFLU{MF},
) where {MF<:MultiFloat}
    issuccess(F) || throw(LinearAlgebra.SingularException(F.info))
    A = F.factors
    n, m = size(A)
    n == m || throw(DimensionMismatch("solve requires a square LU factor"))
    length(destination) == n ||
        throw(DimensionMismatch("right-hand side length differs"))
    _apply_pivots!(destination, F.ipiv)

    @inbounds for row in 1:n
        value = destination[row]
        for column in 1:(row - 1)
            value -= A[row, column] * destination[column]
        end
        destination[row] = value
    end

    @inbounds for row in n:-1:1
        value = destination[row]
        for column in (row + 1):n
            value -= A[row, column] * destination[column]
        end
        destination[row] = value / A[row, row]
    end
    return destination
end

function ldiv!(
    destination::AbstractMatrix{MF},
    F::MFLU{MF},
) where {MF<:MultiFloat}
    size(destination, 1) == size(F.factors, 1) ||
        throw(DimensionMismatch("right-hand side dimensions differ"))
    @inbounds for column in axes(destination, 2)
        ldiv!(@view(destination[:, column]), F)
    end
    return destination
end

solve(F::Union{MFCholesky,MFLU}, rhs::AbstractVector) =
    ldiv!(copy(rhs), F)
solve(F::Union{MFCholesky,MFLU}, rhs::AbstractMatrix) =
    ldiv!(copy(rhs), F)
