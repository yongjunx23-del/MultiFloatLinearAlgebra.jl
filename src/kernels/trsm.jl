@inline function _trsm_coefficient(
    A::AbstractMatrix,
    row::Int,
    column::Int,
    transposed::Bool,
)
    return transposed ? A[column, row] : A[row, column]
end

function _trsm_scale!(B::AbstractMatrix{MF}, alpha::MF) where {MF<:MultiFloat}
    alpha == one(MF) && return B
    @inbounds for index in eachindex(B)
        B[index] *= alpha
    end
    return B
end

function _trsm_left_columns!(
    B::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    first_column::Int,
    last_column::Int,
    effective_lower::Bool,
    transposed::Bool,
    unit_diagonal::Bool,
) where {T,N,MF<:MultiFloat{T,N}}
    n = size(A, 1)
    V4 = MultiFloatVec{4,T,N}
    column = first_column

    @inbounds while column + 3 <= last_column
        if effective_lower
            for row in 1:n
                values = V4(
                    B[row, column],
                    B[row, column + 1],
                    B[row, column + 2],
                    B[row, column + 3],
                )
                for k in 1:(row - 1)
                    values -= V4(_trsm_coefficient(A, row, k, transposed)) * V4(
                        B[k, column],
                        B[k, column + 1],
                        B[k, column + 2],
                        B[k, column + 3],
                    )
                end
                if !unit_diagonal
                    values /= V4(A[row, row])
                end
                for lane in 1:4
                    B[row, column + lane - 1] = values[lane]
                end
            end
        else
            for row in n:-1:1
                values = V4(
                    B[row, column],
                    B[row, column + 1],
                    B[row, column + 2],
                    B[row, column + 3],
                )
                for k in (row + 1):n
                    values -= V4(_trsm_coefficient(A, row, k, transposed)) * V4(
                        B[k, column],
                        B[k, column + 1],
                        B[k, column + 2],
                        B[k, column + 3],
                    )
                end
                if !unit_diagonal
                    values /= V4(A[row, row])
                end
                for lane in 1:4
                    B[row, column + lane - 1] = values[lane]
                end
            end
        end
        column += 4
    end

    for rhs_column in column:last_column
        if effective_lower
            for row in 1:n
                value = B[row, rhs_column]
                for k in 1:(row - 1)
                    value -=
                        _trsm_coefficient(A, row, k, transposed) *
                        B[k, rhs_column]
                end
                if !unit_diagonal
                    value /= A[row, row]
                end
                B[row, rhs_column] = value
            end
        else
            for row in n:-1:1
                value = B[row, rhs_column]
                for k in (row + 1):n
                    value -=
                        _trsm_coefficient(A, row, k, transposed) *
                        B[k, rhs_column]
                end
                if !unit_diagonal
                    value /= A[row, row]
                end
                B[row, rhs_column] = value
            end
        end
    end
    return nothing
end

function _trsm_right_rows!(
    B::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    first_row::Int,
    last_row::Int,
    effective_lower::Bool,
    transposed::Bool,
    unit_diagonal::Bool,
) where {T,N,MF<:MultiFloat{T,N}}
    n = size(A, 1)
    V4 = MultiFloatVec{4,T,N}
    row = first_row

    @inbounds while row + 3 <= last_row
        if effective_lower
            for column in n:-1:1
                values = V4(
                    B[row, column],
                    B[row + 1, column],
                    B[row + 2, column],
                    B[row + 3, column],
                )
                for k in (column + 1):n
                    values -= V4(
                        B[row, k],
                        B[row + 1, k],
                        B[row + 2, k],
                        B[row + 3, k],
                    ) * V4(_trsm_coefficient(A, k, column, transposed))
                end
                if !unit_diagonal
                    values /= V4(A[column, column])
                end
                for lane in 1:4
                    B[row + lane - 1, column] = values[lane]
                end
            end
        else
            for column in 1:n
                values = V4(
                    B[row, column],
                    B[row + 1, column],
                    B[row + 2, column],
                    B[row + 3, column],
                )
                for k in 1:(column - 1)
                    values -= V4(
                        B[row, k],
                        B[row + 1, k],
                        B[row + 2, k],
                        B[row + 3, k],
                    ) * V4(_trsm_coefficient(A, k, column, transposed))
                end
                if !unit_diagonal
                    values /= V4(A[column, column])
                end
                for lane in 1:4
                    B[row + lane - 1, column] = values[lane]
                end
            end
        end
        row += 4
    end

    for scalar_row in row:last_row
        if effective_lower
            for column in n:-1:1
                value = B[scalar_row, column]
                for k in (column + 1):n
                    value -=
                        B[scalar_row, k] *
                        _trsm_coefficient(A, k, column, transposed)
                end
                if !unit_diagonal
                    value /= A[column, column]
                end
                B[scalar_row, column] = value
            end
        else
            for column in 1:n
                value = B[scalar_row, column]
                for k in 1:(column - 1)
                    value -=
                        B[scalar_row, k] *
                        _trsm_coefficient(A, k, column, transposed)
                end
                if !unit_diagonal
                    value /= A[column, column]
                end
                B[scalar_row, column] = value
            end
        end
    end
    return nothing
end

"""
    trsm!(B, A, alpha=one(eltype(B)); side=:left, uplo=:lower,
          trans=:N, diag=:nonunit, config=KernelConfig())

Solve the BLAS-like triangular system

`op(A) * X = alpha * B` (`side=:left`) or
`X * op(A) = alpha * B` (`side=:right`)

in place, where `op(A)` is `A` or `transpose(A)`. The implementation is
specialized for MultiFloat matrices, maps four independent right-hand sides
(or rows for a right-side solve) into `MultiFloatVec` lanes, and lets threads
own disjoint RHS tiles. `B` must not alias `A`.
"""
function trsm!(
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
    n == m || throw(DimensionMismatch("trsm! requires a square triangular matrix"))
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    uplo in (:lower, :upper) || throw(ArgumentError("uplo must be :lower or :upper"))
    trans in (:N, :T) || throw(ArgumentError("trans must be :N or :T"))
    diag in (:unit, :nonunit) || throw(ArgumentError("diag must be :unit or :nonunit"))
    side === :left ?
        (size(B, 1) == n || throw(DimensionMismatch("trsm! left dimensions differ"))) :
        (size(B, 2) == n || throw(DimensionMismatch("trsm! right dimensions differ")))
    _check_supported(MF)
    Base.require_one_based_indexing(B, A)
    _require_no_output_alias("trsm!", B, A)

    _trsm_scale!(B, alpha)
    transposed = trans === :T
    effective_lower = (uplo === :lower) == !transposed
    unit_diagonal = diag === :unit

    if side === :left
        rhs_columns = size(B, 2)
        tile = max(config.column_tile, 4)
        jobs = cld(rhs_columns, tile)
        workers = _workers(config, jobs)
        if workers == 1 || jobs <= 1
            _trsm_left_columns!(
                B, A, 1, rhs_columns,
                effective_lower, transposed, unit_diagonal,
            )
        else
            @sync for worker in 1:workers
                Threads.@spawn begin
                    for job in worker:workers:jobs
                        first_column = (job - 1) * tile + 1
                        last_column = min(job * tile, rhs_columns)
                        _trsm_left_columns!(
                            B, A, first_column, last_column,
                            effective_lower, transposed, unit_diagonal,
                        )
                    end
                end
            end
        end
    else
        rows = size(B, 1)
        tile = max(2 * config.column_tile, 16)
        jobs = cld(rows, tile)
        workers = _workers(config, jobs)
        if workers == 1 || jobs <= 1
            _trsm_right_rows!(
                B, A, 1, rows,
                effective_lower, transposed, unit_diagonal,
            )
        else
            @sync for worker in 1:workers
                Threads.@spawn begin
                    for job in worker:workers:jobs
                        first_row = (job - 1) * tile + 1
                        last_row = min(job * tile, rows)
                        _trsm_right_rows!(
                            B, A, first_row, last_row,
                            effective_lower, transposed, unit_diagonal,
                        )
                    end
                end
            end
        end
    end
    return B
end

# Positional, kwarg-free entry point used by the factor-cache solve hot path to
# avoid the keyword-NamedTuple allocation. Mirrors `trsm!` exactly.
function _trsm!(
    B::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    alpha::MF,
    side::Symbol,
    uplo::Symbol,
    trans::Symbol,
    diag::Symbol,
    config::KernelConfig,
) where {MF<:MultiFloat}
    n, m = size(A)
    n == m || throw(DimensionMismatch("_trsm! requires a square triangular matrix"))
    _check_supported(MF)
    Base.require_one_based_indexing(B, A)
    _require_no_output_alias("_trsm!", B, A)
    _trsm_scale!(B, alpha)
    transposed = trans === :T
    effective_lower = (uplo === :lower) == !transposed
    unit_diagonal = diag === :unit
    if side === :left
        rhs_columns = size(B, 2)
        tile = max(config.column_tile, 4)
        jobs = cld(rhs_columns, tile)
        workers = _workers(config, jobs)
        if workers == 1 || jobs <= 1
            _trsm_left_columns!(B, A, 1, rhs_columns, effective_lower, transposed, unit_diagonal)
        else
            _trsm_left_threaded!(B, A, rhs_columns, tile, workers, jobs, effective_lower, transposed, unit_diagonal)
        end
    else
        rows = size(B, 1)
        tile = max(2 * config.column_tile, 16)
        jobs = cld(rows, tile)
        workers = _workers(config, jobs)
        if workers == 1 || jobs <= 1
            _trsm_right_rows!(B, A, 1, rows, effective_lower, transposed, unit_diagonal)
        else
            _trsm_right_threaded!(B, A, rows, tile, workers, jobs, effective_lower, transposed, unit_diagonal)
        end
    end
    return B
end

# The threaded branches are kept in separate (non-inlined) helpers so the
# single-thread `_trsm!` path contains no `@sync`/`Threads.@spawn` closure to
# hoist, keeping the warm solve allocation-free.
function _trsm_left_threaded!(B, A, rhs_columns, tile, workers, jobs, effective_lower, transposed, unit_diagonal)
    @sync for worker in 1:workers
        Threads.@spawn begin
            for job in worker:workers:jobs
                first_column = (job - 1) * tile + 1
                last_column = min(job * tile, rhs_columns)
                _trsm_left_columns!(B, A, first_column, last_column, effective_lower, transposed, unit_diagonal)
            end
        end
    end
    return B
end

function _trsm_right_threaded!(B, A, rows, tile, workers, jobs, effective_lower, transposed, unit_diagonal)
    @sync for worker in 1:workers
        Threads.@spawn begin
            for job in worker:workers:jobs
                first_row = (job - 1) * tile + 1
                last_row = min(job * tile, rows)
                _trsm_right_rows!(B, A, first_row, last_row, effective_lower, transposed, unit_diagonal)
            end
        end
    end
    return B
end

# View-free, kwarg-free upper-triangular solve on the leading `rank`-square block
# of the parent `A` (offset 0). Solves R[1:rank,1:rank] * B = B in place for a
# matrix destination, matching `trsm!(...; side=:left, uplo=:upper, trans=:N,
# diag=:nonunit)` bit-for-bit but with zero allocation.
function _trsm_leading_upper!(B::AbstractMatrix{MF}, A::AbstractMatrix{MF}, rank::Int) where {MF<:MultiFloat}
    @inbounds for column in axes(B, 2)
        for row in rank:-1:1
            value = B[row, column]
            for k in (row + 1):rank
                value -= A[row, k] * B[k, column]
            end
            B[row, column] = value / A[row, row]
        end
    end
    return B
end

# View-free transposed solve: R'[1:rank,1:rank] * B = B (R' is lower triangular,
# so forward substitution over the columns of R). Matches
# `trsm!(...; side=:left, uplo=:upper, trans=:T, diag=:nonunit)` with zero
# allocation.
function _trsm_leading_upper_trans!(B::AbstractMatrix{MF}, A::AbstractMatrix{MF}, rank::Int) where {MF<:MultiFloat}
    @inbounds for column in axes(B, 2)
        for row in 1:rank
            value = B[row, column]
            for k in 1:(row - 1)
                value -= A[k, row] * B[k, column]
            end
            B[row, column] = value / A[row, row]
        end
    end
    return B
end

# View-free vector solve: R[1:rank,1:rank] * x = x (backward substitution).
function _trsv_leading_upper!(x::AbstractVector{MF}, A::AbstractMatrix{MF}, rank::Int) where {MF<:MultiFloat}
    @inbounds for row in rank:-1:1
        value = x[row]
        for k in (row + 1):rank
            value -= A[row, k] * x[k]
        end
        x[row] = value / A[row, row]
    end
    return x
end

# View-free transposed vector solve: R'[1:rank,1:rank] * x = x (forward
# substitution over the columns of R).
function _trsv_leading_upper_trans!(x::AbstractVector{MF}, A::AbstractMatrix{MF}, rank::Int) where {MF<:MultiFloat}
    @inbounds for row in 1:rank
        value = x[row]
        for k in 1:(row - 1)
            value -= A[k, row] * x[k]
        end
        x[row] = value / A[row, row]
    end
    return x
end
