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

# View-free, kwarg-free LDLT trailing update used by the cache hot path. Builds
# the weighted panel into `weighted_storage` (rows 1:trailing_count) and applies
# the lower-triangular GEMM output = alpha*left*right' + beta*output directly on
# the parent `A` (trailing block), matching `gemmt!` / `_gemmt_column!`
# bit-for-bit (ascending k reduction, lower triangle only). Zero allocation.
function _ldlt_block_trailing_update_viewfree!(
    A::AbstractMatrix{MF},
    panel_first::Int,
    panel_last::Int,
    weighted_storage::Matrix{MF},
    dsub::AbstractVector{MF},
    blocks::AbstractVector{UInt8},
) where {T,N,MF<:MultiFloat{T,N}}
    n = size(A, 1)
    panel_last < n || return nothing
    trailing_count = n - panel_last
    panel_width = panel_last - panel_first + 1
    # weighted panel build (same as _build_ldlt_weighted_panel!, no view)
    q = panel_first
    @inbounds while q <= panel_last
        local_column = q - panel_first + 1
        if blocks[q] == UInt8(1)
            d = A[q, q]
            for local_row in 1:trailing_count
                row = panel_last + local_row
                weighted_storage[local_row, local_column] = A[row, q] * d
            end
            q += 1
        else
            d11 = A[q, q]; d21 = dsub[q]; d22 = A[q + 1, q + 1]
            for local_row in 1:trailing_count
                row = panel_last + local_row
                first = A[row, q]; second = A[row, q + 1]
                weighted_storage[local_row, local_column] = first * d11 + second * d21
                weighted_storage[local_row, local_column + 1] = first * d21 + second * d22
            end
            q += 2
        end
    end
    # lower-triangular GEMM: output[row,column] (row>=column) over the trailing
    # block = sum_k A[trailing_row, panel_first+k-1] * weighted[column_local, k]
    V4 = MultiFloatVec{4,T,N}
    trailing_first = panel_last + 1
    for local_column in 1:trailing_count
        parent_column = trailing_first + local_column - 1
        row = local_column
        @inbounds while row + 3 <= trailing_count
            accumulator = zero(V4)
            for k in 1:panel_width
                parent_row0 = trailing_first + row - 1
                values = V4(
                    A[parent_row0, panel_first + k - 1],
                    A[parent_row0 + 1, panel_first + k - 1],
                    A[parent_row0 + 2, panel_first + k - 1],
                    A[parent_row0 + 3, panel_first + k - 1],
                )
                accumulator = _structured_mulacc(accumulator, values, V4(weighted_storage[local_column, k]))
            end
            result = -accumulator
            for lane in 1:4
                parent_row = trailing_first + row + lane - 2
                A[parent_row, parent_column] += result[lane]
            end
            row += 4
        end
        while row <= trailing_count
            accumulator = zero(MF)
            for k in 1:panel_width
                parent_row = trailing_first + row - 1
                accumulator += A[parent_row, panel_first + k - 1] * weighted_storage[local_column, k]
            end
            parent_row = trailing_first + row - 1
            A[parent_row, parent_column] -= accumulator
            row += 1
        end
    end
    # restore the upper mirror (read-only), matching the original trailing update
    _mirror_lower_to_upper_block!(A, trailing_first, n)
    return nothing
end

# Mirror lower->upper only within the trailing block A[trailing_first:n, trailing_first:n].
function _mirror_lower_to_upper_block!(A::AbstractMatrix, trailing_first::Int, n::Int)
    @inbounds for column in trailing_first:n
        for row in trailing_first:n
            row > column || continue
            A[column, row] = A[row, column]
        end
    end
    return A
end

# View-free blocked LDLT factorization core used by the cache hot path. Same
# Bunch-Kaufman panel factorization and pivot selection as the standalone
# `_factor_ldlt_blocked!`, but the trailing update runs on the parent `A` with
# no SubArray views, so the warm path allocates 0 bytes.
function _ldlt_factorize_blocked_viewfree!(
    A::AbstractMatrix{MF},
    dsub::AbstractVector{MF},
    pivots::AbstractVector{Int},
    blocks::AbstractVector{UInt8},
    weighted_storage::Matrix{MF},
    alpha::MF,
    plan::LDLTPlan,
    config::KernelConfig,
) where {MF<:MultiFloat}
    n = size(A, 1)
    panel_first = 1
    while panel_first <= n
        requested_last = min(panel_first + plan.block_size - 1, n)
        info, panel_last = _factor_ldlt_panel!(
            A, panel_first, requested_last, dsub, pivots, blocks, alpha,
        )
        !iszero(info) && return info
        _ldlt_block_trailing_update_viewfree!(
            A, panel_first, panel_last, weighted_storage, dsub, blocks,
        )
        panel_first = panel_last + 1
    end
    return 0
end

# View-free, kwarg-free trailing GEMM for blocked RRQR: computes
#   C[trailing, trailing] = -A[trailing, panel] * F + C[trailing, trailing]
# where F is the caller-owned `ftranspose` matrix (rows 1:actual, columns
# (block_end+1):n), operating on the parent `A` with no SubArray views. Matches
# `gemm!(C, A, B, -one, one)` / `_gemm_direct_column_range!` bit-for-bit
# (ascending k reduction, 2-column V4 store pair). Zero allocation.
function _rrqr_trailing_gemm_viewfree!(
    A::AbstractMatrix{MF},
    block_start::Int,
    block_end::Int,
    ftranspose::AbstractMatrix{MF},
    actual::Int,
    n::Int,
) where {T,N,MF<:MultiFloat{T,N}}
    m = size(A, 1)
    trailing_first = block_end + 1
    trailing_last = n
    trailing_rows = m - block_end
    trailing_cols = n - block_end
    panel_cols = actual
    V4 = MultiFloatVec{4,T,N}
    # iterate output columns of the trailing block, two at a time (matches the
    # direct GEMM 2-column V4 store-pair schedule)
    col = 1
    @inbounds while col + 1 <= trailing_cols
        parent_c1 = trailing_first + col - 1
        parent_c2 = parent_c1 + 1
        row = 1
        while row + 3 <= trailing_rows
            parent_r0 = trailing_first + row - 1
            acc0a = zero(MF); acc1a = zero(MF); acc2a = zero(MF); acc3a = zero(MF)
            acc0b = zero(MF); acc1b = zero(MF); acc2b = zero(MF); acc3b = zero(MF)
            for k in 1:panel_cols
                a0 = A[parent_r0, block_start + k - 1]
                a1 = A[parent_r0 + 1, block_start + k - 1]
                a2 = A[parent_r0 + 2, block_start + k - 1]
                a3 = A[parent_r0 + 3, block_start + k - 1]
                f1 = ftranspose[k, parent_c1]
                f2 = ftranspose[k, parent_c2]
                acc0a += a0 * f1; acc1a += a1 * f1; acc2a += a2 * f1; acc3a += a3 * f1
                acc0b += a0 * f2; acc1b += a1 * f2; acc2b += a2 * f2; acc3b += a3 * f2
            end
            A[parent_r0, parent_c1] -= acc0a
            A[parent_r0 + 1, parent_c1] -= acc1a
            A[parent_r0 + 2, parent_c1] -= acc2a
            A[parent_r0 + 3, parent_c1] -= acc3a
            A[parent_r0, parent_c2] -= acc0b
            A[parent_r0 + 1, parent_c2] -= acc1b
            A[parent_r0 + 2, parent_c2] -= acc2b
            A[parent_r0 + 3, parent_c2] -= acc3b
            row += 4
        end
        while row <= trailing_rows
            parent_r = trailing_first + row - 1
            acca = zero(MF); accb = zero(MF)
            for k in 1:panel_cols
                a = A[parent_r, block_start + k - 1]
                acca += a * ftranspose[k, parent_c1]
                accb += a * ftranspose[k, parent_c2]
            end
            A[parent_r, parent_c1] -= acca
            A[parent_r, parent_c2] -= accb
            row += 1
        end
        col += 2
    end
    if col <= trailing_cols
        parent_c = trailing_first + col - 1
        row = 1
        while row + 3 <= trailing_rows
            accumulator = zero(V4)
            for k in 1:panel_cols
                parent_r0 = trailing_first + row - 1
                values = V4(
                    A[parent_r0, block_start + k - 1],
                    A[parent_r0 + 1, block_start + k - 1],
                    A[parent_r0 + 2, block_start + k - 1],
                    A[parent_r0 + 3, block_start + k - 1],
                )
                accumulator += values * V4(ftranspose[k, parent_c])
            end
            result = -accumulator
            for lane in 1:4
                A[trailing_first + row + lane - 2, parent_c] += result[lane]
            end
            row += 4
        end
        while row <= trailing_rows
            parent_r = trailing_first + row - 1
            accumulator = zero(MF)
            for k in 1:panel_cols
                accumulator += A[parent_r, block_start + k - 1] * ftranspose[k, parent_c]
            end
            A[parent_r, parent_c] -= accumulator
            row += 1
        end
    end
    return nothing
end
