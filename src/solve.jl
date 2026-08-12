"""
    ldiv!(destination, F; config=KernelConfig())

Solve the factorized system represented by `F` in place, overwriting
`destination` with the solution. A single right-hand side vector or a matrix
of right-hand sides is accepted. `F` must be successful (`issuccess(F)`),
otherwise a `PosDefException` or `SingularException` is thrown.

Dispatch selects the triangular solve for `MFCholesky`, `MFLU`, or `MFLDLT`.
Vector right-hand sides are solved through the same package `trsm!` kernel as
matrix right-hand sides by treating them as a single column.
"""
function ldiv!(
    destination::AbstractVector{MF},
    F::MFCholesky{MF};
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    issuccess(F) || throw(LinearAlgebra.PosDefException(F.info))
    n = size(F.factors, 1)
    length(destination) == n ||
        throw(DimensionMismatch("right-hand side length differs"))

    rhs = reshape(destination, n, 1)
    trsm!(
        rhs,
        F.factors;
        side=:left,
        uplo=:lower,
        trans=:N,
        diag=:nonunit,
        config=config,
    )
    trsm!(
        rhs,
        F.factors;
        side=:left,
        uplo=:lower,
        trans=:T,
        diag=:nonunit,
        config=config,
    )
    return destination
end

function ldiv!(
    destination::AbstractMatrix{MF},
    F::MFCholesky{MF};
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    issuccess(F) || throw(LinearAlgebra.PosDefException(F.info))
    size(destination, 1) == size(F.factors, 1) ||
        throw(DimensionMismatch("right-hand side dimensions differ"))
    trsm!(
        destination,
        F.factors;
        side=:left,
        uplo=:lower,
        trans=:N,
        diag=:nonunit,
        config=config,
    )
    trsm!(
        destination,
        F.factors;
        side=:left,
        uplo=:lower,
        trans=:T,
        diag=:nonunit,
        config=config,
    )
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

function _apply_pivots!(
    destination::AbstractMatrix,
    ipiv::AbstractVector{Int},
)
    @inbounds for k in eachindex(ipiv)
        pivot = ipiv[k]
        if pivot != k
            for column in axes(destination, 2)
                destination[k, column], destination[pivot, column] =
                    destination[pivot, column], destination[k, column]
            end
        end
    end
    return destination
end

function ldiv!(
    destination::AbstractVector{MF},
    F::MFLU{MF};
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    issuccess(F) || throw(LinearAlgebra.SingularException(F.info))
    A = F.factors
    n, m = size(A)
    n == m || throw(DimensionMismatch("solve requires a square LU factor"))
    length(destination) == n ||
        throw(DimensionMismatch("right-hand side length differs"))
    _apply_pivots!(destination, F.ipiv)

    rhs = reshape(destination, n, 1)
    trsm!(
        rhs,
        F.factors;
        side=:left,
        uplo=:lower,
        trans=:N,
        diag=:unit,
        config=config,
    )
    trsm!(
        rhs,
        F.factors;
        side=:left,
        uplo=:upper,
        trans=:N,
        diag=:nonunit,
        config=config,
    )
    return destination
end

function ldiv!(
    destination::AbstractMatrix{MF},
    F::MFLU{MF};
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    issuccess(F) || throw(LinearAlgebra.SingularException(F.info))
    n, m = size(F.factors)
    n == m || throw(DimensionMismatch("solve requires a square LU factor"))
    size(destination, 1) == n ||
        throw(DimensionMismatch("right-hand side dimensions differ"))
    _apply_pivots!(destination, F.ipiv)
    trsm!(
        destination,
        F.factors;
        side=:left,
        uplo=:lower,
        trans=:N,
        diag=:unit,
        config=config,
    )
    trsm!(
        destination,
        F.factors;
        side=:left,
        uplo=:upper,
        trans=:N,
        diag=:nonunit,
        config=config,
    )
    return destination
end

@inline function _ldlt_swap_rhs_rows!(
    destination::AbstractVector,
    first::Int,
    second::Int,
)
    first == second && return nothing
    destination[first], destination[second] =
        destination[second], destination[first]
    return nothing
end

function _ldlt_swap_rhs_rows!(
    destination::AbstractMatrix,
    first::Int,
    second::Int,
)
    first == second && return nothing
    @inbounds for column in axes(destination, 2)
        destination[first, column], destination[second, column] =
            destination[second, column], destination[first, column]
    end
    return nothing
end

function _apply_ldlt_pivots_forward!(destination, F::MFLDLT)
    n = length(F.blocks)
    k = 1
    while k <= n
        block = F.blocks[k]
        if block == UInt8(1)
            _ldlt_swap_rhs_rows!(destination, k, F.pivots[k])
            k += 1
        elseif block == UInt8(2)
            _ldlt_swap_rhs_rows!(destination, k + 1, F.pivots[k])
            k += 2
        else
            throw(ArgumentError("invalid LDLT block structure"))
        end
    end
    return destination
end

function _apply_ldlt_pivots_reverse!(destination, F::MFLDLT)
    k = length(F.blocks)
    while k >= 1
        if F.blocks[k] == UInt8(1)
            _ldlt_swap_rhs_rows!(destination, k, F.pivots[k])
            k -= 1
        elseif F.blocks[k] == UInt8(0) && k > 1 &&
               F.blocks[k - 1] == UInt8(2)
            start = k - 1
            _ldlt_swap_rhs_rows!(destination, start + 1, F.pivots[start])
            k -= 2
        else
            throw(ArgumentError("invalid LDLT block structure"))
        end
    end
    return destination
end

function _ldlt_solve_d!(destination::AbstractVector{MF}, F::MFLDLT{MF}) where {MF<:MultiFloat}
    n = length(F.blocks)
    k = 1
    @inbounds while k <= n
        if F.blocks[k] == UInt8(1)
            destination[k] /= F.factors[k, k]
            k += 1
        else
            d11 = F.factors[k, k]
            d21 = F.dsub[k]
            d22 = F.factors[k + 1, k + 1]
            determinant = d11 * d22 - d21 * d21
            first = destination[k]
            second = destination[k + 1]
            destination[k] = (d22 * first - d21 * second) / determinant
            destination[k + 1] = (d11 * second - d21 * first) / determinant
            k += 2
        end
    end
    return destination
end

function _ldlt_solve_d!(destination::AbstractMatrix{MF}, F::MFLDLT{MF}) where {MF<:MultiFloat}
    n = length(F.blocks)
    k = 1
    @inbounds while k <= n
        if F.blocks[k] == UInt8(1)
            d = F.factors[k, k]
            for column in axes(destination, 2)
                destination[k, column] /= d
            end
            k += 1
        else
            d11 = F.factors[k, k]
            d21 = F.dsub[k]
            d22 = F.factors[k + 1, k + 1]
            determinant = d11 * d22 - d21 * d21
            for column in axes(destination, 2)
                first = destination[k, column]
                second = destination[k + 1, column]
                destination[k, column] =
                    (d22 * first - d21 * second) / determinant
                destination[k + 1, column] =
                    (d11 * second - d21 * first) / determinant
            end
            k += 2
        end
    end
    return destination
end

function ldiv!(
    destination::AbstractVector{MF},
    F::MFLDLT{MF};
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    issuccess(F) || throw(LinearAlgebra.SingularException(F.info))
    n = size(F.factors, 1)
    length(destination) == n ||
        throw(DimensionMismatch("right-hand side length differs"))
    _apply_ldlt_pivots_forward!(destination, F)

    rhs = reshape(destination, n, 1)
    trsm!(
        rhs,
        F.factors;
        side=:left,
        uplo=:lower,
        trans=:N,
        diag=:unit,
        config=config,
    )
    _ldlt_solve_d!(destination, F)
    trsm!(
        rhs,
        F.factors;
        side=:left,
        uplo=:lower,
        trans=:T,
        diag=:unit,
        config=config,
    )

    _apply_ldlt_pivots_reverse!(destination, F)
    return destination
end

function ldiv!(
    destination::AbstractMatrix{MF},
    F::MFLDLT{MF};
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    issuccess(F) || throw(LinearAlgebra.SingularException(F.info))
    size(destination, 1) == size(F.factors, 1) ||
        throw(DimensionMismatch("right-hand side dimensions differ"))
    _apply_ldlt_pivots_forward!(destination, F)
    trsm!(
        destination,
        F.factors;
        side=:left,
        uplo=:lower,
        trans=:N,
        diag=:unit,
        config=config,
    )
    _ldlt_solve_d!(destination, F)
    trsm!(
        destination,
        F.factors;
        side=:left,
        uplo=:lower,
        trans=:T,
        diag=:unit,
        config=config,
    )
    _apply_ldlt_pivots_reverse!(destination, F)
    return destination
end

"""
    solve(F, rhs; config=KernelConfig())

Solve the factorized system `F` for `rhs` and return a new array. This is the
non-mutating wrapper around `ldiv!` that copies `rhs` first.
"""
solve(
    F::Union{MFCholesky,MFLU,MFLDLT},
    rhs::AbstractVector;
    config::KernelConfig=KernelConfig(),
) = ldiv!(copy(rhs), F; config=config)

solve(
    F::Union{MFCholesky,MFLU,MFLDLT},
    rhs::AbstractMatrix;
    config::KernelConfig=KernelConfig(),
) = ldiv!(copy(rhs), F; config=config)
