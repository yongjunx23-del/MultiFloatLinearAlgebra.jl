"""
    ldiv!(destination, F; config=KernelConfig())

Solve the factorized system represented by `F` in place, overwriting
`destination` with the solution. A single right-hand side vector or a matrix
of right-hand sides is accepted. `F` must be successful (`issuccess(F)`),
otherwise a `PosDefException` or `SingularException` is thrown.

Dispatch selects the triangular solve for `MFCholesky`, `MFLU`, or `MFLDLT`.
Vector right-hand sides use the dedicated [`trsv!`](@ref) single-column kernel,
while matrix right-hand sides use [`trsm!`](@ref).
"""
@inline function _check_factor_solve_success(F::MFCholesky)
    issuccess(F) || throw(LinearAlgebra.PosDefException(factor_status(F)))
    return nothing
end

@inline function _check_factor_solve_success(F::Union{MFLU,MFLDLT})
    issuccess(F) || throw(LinearAlgebra.SingularException(factor_status(F)))
    return nothing
end

function ldiv!(
    destination::AbstractVector{MF},
    F::MFCholesky{MF};
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    _check_factor_solve_success(F)
    Base.require_one_based_indexing(destination, F.factors)
    n = size(F.factors, 1)
    length(destination) == n ||
        throw(DimensionMismatch("right-hand side length differs"))
    _require_no_output_alias("ldiv!", destination, F.factors)

    trsv!(
        destination,
        F.factors;
        uplo=:lower,
        trans=:N,
        diag=:nonunit,
        config=config,
    )
    trsv!(
        destination,
        F.factors;
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
    _check_factor_solve_success(F)
    Base.require_one_based_indexing(destination, F.factors)
    size(destination, 1) == size(F.factors, 1) ||
        throw(DimensionMismatch("right-hand side dimensions differ"))
    _require_no_output_alias("ldiv!", destination, F.factors)
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
    _check_factor_solve_success(F)
    Base.require_one_based_indexing(destination, F.factors)
    A = F.factors
    n, m = size(A)
    n == m || throw(DimensionMismatch("solve requires a square LU factor"))
    length(destination) == n ||
        throw(DimensionMismatch("right-hand side length differs"))
    _require_no_output_alias("ldiv!", destination, F.factors)
    _apply_pivots!(destination, F.ipiv)

    trsv!(
        destination,
        F.factors;
        uplo=:lower,
        trans=:N,
        diag=:unit,
        config=config,
    )
    trsv!(
        destination,
        F.factors;
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
    _check_factor_solve_success(F)
    Base.require_one_based_indexing(destination, F.factors)
    n, m = size(F.factors)
    n == m || throw(DimensionMismatch("solve requires a square LU factor"))
    size(destination, 1) == n ||
        throw(DimensionMismatch("right-hand side dimensions differ"))
    _require_no_output_alias("ldiv!", destination, F.factors)
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
    _check_factor_solve_success(F)
    Base.require_one_based_indexing(destination, F.factors)
    n = size(F.factors, 1)
    length(destination) == n ||
        throw(DimensionMismatch("right-hand side length differs"))
    _require_no_output_alias("ldiv!", destination, F.factors)
    _apply_ldlt_pivots_forward!(destination, F)

    trsv!(
        destination,
        F.factors;
        uplo=:lower,
        trans=:N,
        diag=:unit,
        config=config,
    )
    _ldlt_solve_d!(destination, F)
    trsv!(
        destination,
        F.factors;
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
    _check_factor_solve_success(F)
    Base.require_one_based_indexing(destination, F.factors)
    size(destination, 1) == size(F.factors, 1) ||
        throw(DimensionMismatch("right-hand side dimensions differ"))
    _require_no_output_alias("ldiv!", destination, F.factors)
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
    ldiv!(x, F, b; config=KernelConfig())
    ldiv!(X, F, B; config=KernelConfig())

Copy a vector or multi-RHS source into a caller-owned destination, then solve
with a successful Cholesky, LU, or LDLT factor. All validation occurs before
the copy, so a failed factor or invalid destination leaves both arrays
unchanged. Exact source/destination identity is allowed; other overlapping
views are rejected.

Square, exactly full-rank RRQR factors have a corresponding wrapper below;
rectangular/rank-selected routes use `apply_q!`, `solve_r!`, and
`factor_permutation` explicitly.
"""
function ldiv!(
    destination::AbstractVector{MF},
    F::Union{MFCholesky{MF},MFLU{MF},MFLDLT{MF}},
    source::AbstractVector{MF};
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    _check_factor_solve_success(F)
    Base.require_one_based_indexing(destination, source, factor_matrix(F))
    n = size(F, 1)
    size(F, 2) == n || throw(DimensionMismatch("solve requires a square factor"))
    length(destination) == n && length(source) == n ||
        throw(DimensionMismatch("right-hand side length differs"))
    _require_no_output_alias("ldiv!", destination, factor_matrix(F))
    destination === source || begin
        Base.mightalias(destination, source) &&
            throw(ArgumentError("ldiv! does not support partially overlapping right-hand sides"))
        copyto!(destination, source)
    end
    return ldiv!(destination, F; config=config)
end

function ldiv!(
    destination::AbstractMatrix{MF},
    F::Union{MFCholesky{MF},MFLU{MF},MFLDLT{MF}},
    source::AbstractMatrix{MF};
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    _check_factor_solve_success(F)
    Base.require_one_based_indexing(destination, source, factor_matrix(F))
    n = size(F, 1)
    size(F, 2) == n || throw(DimensionMismatch("solve requires a square factor"))
    size(destination) == size(source) ||
        throw(DimensionMismatch("source and destination dimensions differ"))
    size(destination, 1) == n ||
        throw(DimensionMismatch("right-hand side dimensions differ"))
    _require_no_output_alias("ldiv!", destination, factor_matrix(F))
    destination === source || begin
        Base.mightalias(destination, source) &&
            throw(ArgumentError("ldiv! does not support partially overlapping right-hand sides"))
        copyto!(destination, source)
    end
    return ldiv!(destination, F; config=config)
end

function _check_qr_square_solve(F::MFQR, destination)
    issuccess(F) ||
        throw(ArgumentError("ldiv! requires a successful QR factorization"))
    n, m = size(F)
    n == m || throw(DimensionMismatch(
        "QR ldiv! is defined only for square factors; use apply_q! and solve_r! for caller-selected rectangular/rank routes",
    ))
    size(destination, 1) == n ||
        throw(DimensionMismatch("right-hand side dimensions differ"))
    @inbounds for index in 1:n
        iszero(factor_matrix(F)[index, index]) &&
            throw(LinearAlgebra.SingularException(index))
    end
    return n
end

function _permute_qr_solution!(
    destination::AbstractVector,
    permutation,
    cycle_leaders,
)
    @inbounds for start in cycle_leaders
        value = destination[start]
        current = start
        while true
            target = permutation[current]
            if target == start
                destination[target] = value
                break
            end
            next_value = destination[target]
            destination[target] = value
            value = next_value
            current = target
        end
    end
    return destination
end
function _permute_qr_solution!(
    destination::AbstractMatrix,
    permutation,
    cycle_leaders,
)
    @inbounds for column in axes(destination, 2), start in cycle_leaders
        value = destination[start, column]
        current = start
        while true
            target = permutation[current]
            if target == start
                destination[target, column] = value
                break
            end
            next_value = destination[target, column]
            destination[target, column] = value
            value = next_value
            current = target
        end
    end
    return destination
end

"""
    ldiv!(destination, F::MFQR; config=KernelConfig())

Solve a square, exactly full-rank pivoted-QR system in place. This wrapper
performs `Q'`, leading-`R`, and column-permutation operations without choosing
a numerical-rank threshold. Rectangular and caller-selected rank routes must
use `apply_q!`, `solve_r!`, and `factor_permutation` explicitly.
"""
function ldiv!(
    destination::Union{AbstractVector{MF},AbstractMatrix{MF}},
    F::MFQR{MF};
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    Base.require_one_based_indexing(destination, factor_matrix(F))
    n = _check_qr_square_solve(F, destination)
    _require_no_output_alias("ldiv!", destination, factor_matrix(F))
    apply_q!(destination, F; trans=:T)
    solve_r!(destination, F, n; config=config)
    return _permute_qr_solution!(
        destination, F.permutation, F.permutation_cycle_leaders,
    )
end

function ldiv!(
    destination::AbstractVector{MF},
    F::MFQR{MF},
    source::AbstractVector{MF};
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    Base.require_one_based_indexing(destination, source, factor_matrix(F))
    n = _check_qr_square_solve(F, destination)
    length(source) == n ||
        throw(DimensionMismatch("right-hand side length differs"))
    _require_no_output_alias("ldiv!", destination, factor_matrix(F))
    destination === source || begin
        Base.mightalias(destination, source) &&
            throw(ArgumentError("ldiv! does not support partially overlapping right-hand sides"))
        copyto!(destination, source)
    end
    return ldiv!(destination, F; config=config)
end

function ldiv!(
    destination::AbstractMatrix{MF},
    F::MFQR{MF},
    source::AbstractMatrix{MF};
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    Base.require_one_based_indexing(destination, source, factor_matrix(F))
    n = _check_qr_square_solve(F, destination)
    size(source) == size(destination) ||
        throw(DimensionMismatch("source and destination dimensions differ"))
    size(source, 1) == n ||
        throw(DimensionMismatch("right-hand side dimensions differ"))
    _require_no_output_alias("ldiv!", destination, factor_matrix(F))
    destination === source || begin
        Base.mightalias(destination, source) &&
            throw(ArgumentError("ldiv! does not support partially overlapping right-hand sides"))
        copyto!(destination, source)
    end
    return ldiv!(destination, F; config=config)
end

"""
    solve(F, rhs; config=KernelConfig())

Solve the factorized system `F` for `rhs` and return a new array. This is the
non-mutating wrapper around `ldiv!` that copies `rhs` first.
"""
solve(
    F::Union{MFCholesky,MFLU,MFLDLT,MFQR},
    rhs::AbstractVector;
    config::KernelConfig=KernelConfig(),
) = ldiv!(copy(rhs), F; config=config)

solve(
    F::Union{MFCholesky,MFLU,MFLDLT,MFQR},
    rhs::AbstractMatrix;
    config::KernelConfig=KernelConfig(),
) = ldiv!(copy(rhs), F; config=config)
