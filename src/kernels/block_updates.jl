"""
Offset-based, kwarg-free block kernels used by the factorization cores.

The factorization block loops need to update sub-blocks of the parent matrix
(`A`) without creating `SubArray` views (which allocate on the heap) and
without the keyword-argument dispatch box of the public kernels. These kernels
take the parent matrix plus explicit row/column offsets and the resolved
`KernelConfig` as a positional argument, so the warm path performs no Julia
heap allocation. They implement the exact same deterministic reduction order as
the public kernels they replace.
"""

################################################################################
# Cholesky trailing update: solve A21 = A21 * inv(L11)' (right, lower, trans=T)
################################################################################

# Solves the right-lower-transpose block triangular system that Cholesky's
# panel solve needs, operating on the parent `A` directly. This reproduces the
# public `trsm!(...; side=:right, uplo=:lower, trans=:T, diag=:nonunit)` call
# bit-for-bit (ascending column, ascending k reduction, final division by the
# diagonal) but with zero allocation. The destination rows are `A[(last+1):n, :]`
# and the triangular block is the lower `L11 = A[first:last, first:last]`.
function _cholesky_panel_trsm!(A::AbstractMatrix{MF}, first::Int, last::Int) where {MF<:MultiFloat}
    n = size(A, 1)
    # effective_lower=false, transposed=true => for column ascending, k ascending,
    # coefficient is A[column, k] (lower triangle of L11).
    @inbounds for column in first:last
        for row in (last + 1):n
            value = A[row, column]
            for k in first:(column - 1)
                value -= A[row, k] * A[column, k]
            end
            A[row, column] = value / A[column, column]
        end
    end
    return nothing
end

# Lower-trailing SYRK update: C <- C - P * P' where C = A[(last+1):n,(last+1):n]
# (lower triangle) and P = A[(last+1):n, first:last]. Same reduction order as
# the public `syrk!(...; uplo=:lower)` call, zero allocation.
function _cholesky_trailing_syrk!(A::AbstractMatrix{MF}, first::Int, last::Int) where {T,N,MF<:MultiFloat{T,N}}
    n = size(A, 1)
    V4 = MultiFloatVec{4,T,N}
    trailing_first = last + 1
    # The trailing block is (n - last) square. Local column `lc` corresponds to
    # parent column `trailing_first + lc - 1`; only rows >= that column.
    for local_column in 1:(n - last)
        parent_column = trailing_first + local_column - 1
        row = local_column
        @inbounds while row + 3 <= (n - last)
            accumulator = zero(V4)
            for k in first:last
                values = V4(
                    A[trailing_first + row - 1, k],
                    A[trailing_first + row, k],
                    A[trailing_first + row + 1, k],
                    A[trailing_first + row + 2, k],
                )
                accumulator = _structured_mulacc(accumulator, values, V4(A[parent_column, k]))
            end
            result = -accumulator
            for lane in 1:4
                parent_row = trailing_first + row + lane - 2
                A[parent_row, parent_column] += result[lane]
            end
            row += 4
        end
        while row <= (n - last)
            accumulator = zero(MF)
            for k in first:last
                accumulator += A[trailing_first + row - 1, k] * A[parent_column, k]
            end
            parent_row = trailing_first + row - 1
            A[parent_row, parent_column] -= accumulator
            row += 1
        end
    end
    return nothing
end

################################################################################
# LU trailing update: U12 <- L11 \ U12 (left, lower, unit); A22 <- A22 - A21*U12
################################################################################

function _lu_panel_trsm!(A::AbstractMatrix{MF}, first::Int, last::Int) where {MF<:MultiFloat}
    m, n = size(A)
    # Solve U12 = L11 \ U12 (left, lower, unit) in place, where L11 is the lower
    # unit block A[first:last, first:last] and U12 = A[first:last, (last+1):n].
    # Matches `trsm!(...; side=:left, uplo=:lower, trans=:N, diag=:unit)`:
    # for row ascending, k ascending, subtract L11[row,k]*U12[k,col], unit diag.
    @inbounds for column in (last + 1):n
        for row in first:last
            value = A[row, column]
            for k in first:(row - 1)
                value -= A[row, k] * A[k, column]
            end
            A[row, column] = value
        end
    end
    return nothing
end

function _lu_trailing_gemm!(A::AbstractMatrix{MF}, first::Int, last::Int) where {T,N,MF<:MultiFloat{T,N}}
    m, n = size(A)
    V4 = MultiFloatVec{4,T,N}
    # A22 <- A22 - A21 * U12, where A21 = A[(last+1):m, first:last] (unit lower
    # multipliers), U12 = A[first:last, (last+1):n] (upper).
    for column in (last + 1):n
        row = last + 1
        @inbounds while row + 3 <= m
            acc0 = zero(MF)
            acc1 = zero(MF)
            acc2 = zero(MF)
            acc3 = zero(MF)
            for k in first:last
                m0 = A[row, k]; m1 = A[row + 1, k]; m2 = A[row + 2, k]; m3 = A[row + 3, k]
                u = A[k, column]
                acc0 -= m0 * u
                acc1 -= m1 * u
                acc2 -= m2 * u
                acc3 -= m3 * u
            end
            A[row, column] += acc0
            A[row + 1, column] += acc1
            A[row + 2, column] += acc2
            A[row + 3, column] += acc3
            row += 4
        end
        while row <= m
            acc = zero(MF)
            for k in first:last
                acc -= A[row, k] * A[k, column]
            end
            A[row, column] += acc
            row += 1
        end
    end
    return nothing
end

# View-free, kwarg-free LU factorization core used by the cache hot path. It
# reuses the exact same pivoting (`_factor_lu_panel!`) and block updates as the
# standalone core but performs the trailing trsm/gemm directly on the parent
# matrix, so the warm factorize allocates 0 bytes.
function _lu_factorize_core_viewfree!(
    A::Matrix{MF},
    pivots::AbstractVector{Int},
    config::KernelConfig,
    check::Bool,
) where {MF<:MultiFloat}
    m, n = size(A)
    kmax = min(m, n)
    if !_all_finite(A)
        check && throw(DomainError(A, "lu!: input matrix contains non-finite entries"))
        return -1
    end
    info = 0
    block = max(config.lu_block, 1)
    for first in 1:block:kmax
        last = min(first + block - 1, kmax)
        info = _factor_lu_panel!(A, first, last, pivots)
        if !iszero(info)
            check && throw(LinearAlgebra.SingularException(info))
            return info
        end
        if last < n
            _lu_panel_trsm!(A, first, last)
            if last < m
                _lu_trailing_gemm!(A, first, last)
            end
        end
    end
    return info
end
