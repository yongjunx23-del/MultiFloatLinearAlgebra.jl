function _gemv_rows!(
    y::AbstractVector{MF},
    A::AbstractMatrix{MF},
    x::AbstractVector{MF},
    alpha::MF,
    beta::MF,
    first_row::Int,
    last_row::Int,
) where {T,N,MF<:MultiFloat{T,N}}
    V4 = MultiFloatVec{4,T,N}
    row = first_row
    @inbounds while row + 3 <= last_row
        accumulator = zero(V4)
        for column in axes(A, 2)
            values = V4(
                A[row, column],
                A[row + 1, column],
                A[row + 2, column],
                A[row + 3, column],
            )
            accumulator = _structured_mulacc(
                accumulator, values, V4(x[column]),
            )
        end
        result = if beta == zero(MF)
            V4(alpha) * accumulator
        else
            V4(alpha) * accumulator + V4(beta) * V4(
                y[row], y[row + 1], y[row + 2], y[row + 3],
            )
        end
        for lane in 1:4
            y[row + lane - 1] = result[lane]
        end
        row += 4
    end

    tail = last_row - row + 1
    if tail == 3
        _gemv_rows_tail!(y, A, x, alpha, beta, Val(3), row)
    elseif tail == 2
        _gemv_rows_tail!(y, A, x, alpha, beta, Val(2), row)
    elseif tail == 1
        _gemv_rows_tail!(y, A, x, alpha, beta, Val(1), row)
    end
    return nothing
end

# Tail rows (1..3) of a row-oriented gemv.  `MultiFloatVec{L}` lanes run the
# exact scalar accumulation tree per row, so outputs are bit-identical to
# the previous per-row scalar loop while keeping SIMD lanes for row counts
# that are not a multiple of four.
@inline function _gemv_rows_tail!(
    y::AbstractVector{MF},
    A::AbstractMatrix{MF},
    x::AbstractVector{MF},
    alpha::MF,
    beta::MF,
    ::Val{L},
    row::Int,
) where {T,N,L,MF<:MultiFloat{T,N}}
    VL = MultiFloatVec{L,T,N}
    accumulator = zero(VL)
    @inbounds for column in axes(A, 2)
        values = VL(ntuple(l -> A[row + l - 1, column], L))
        coefficient = VL(ntuple(_ -> x[column], L))
        accumulator += values * coefficient
    end
    result = if iszero(beta)
        VL(ntuple(_ -> alpha, L)) * accumulator
    else
        VL(ntuple(_ -> alpha, L)) * accumulator +
        VL(ntuple(_ -> beta, L)) *
        VL(ntuple(l -> y[row + l - 1], L))
    end
    @inbounds for lane in 1:L
        y[row + lane - 1] = result[lane]
    end
    return nothing
end

function _gemv_t_columns!(
    y::AbstractVector{MF},
    A::AbstractMatrix{MF},
    x::AbstractVector{MF},
    alpha::MF,
    beta::MF,
    first_column::Int,
    last_column::Int,
) where {T,N,MF<:MultiFloat{T,N}}
    V4 = MultiFloatVec{4,T,N}
    column = first_column
    @inbounds while column + 3 <= last_column
        accumulator = zero(V4)
        for row in axes(A, 1)
            values = V4(
                A[row, column],
                A[row, column + 1],
                A[row, column + 2],
                A[row, column + 3],
            )
            accumulator = _structured_mulacc(
                accumulator, values, V4(x[row]),
            )
        end
        result = if beta == zero(MF)
            V4(alpha) * accumulator
        else
            V4(alpha) * accumulator + V4(beta) * V4(
                y[column], y[column + 1], y[column + 2], y[column + 3],
            )
        end
        for lane in 1:4
            y[column + lane - 1] = result[lane]
        end
        column += 4
    end

    tail = last_column - column + 1
    if tail == 3
        _gemv_t_columns_tail!(y, A, x, alpha, beta, Val(3), column)
    elseif tail == 2
        _gemv_t_columns_tail!(y, A, x, alpha, beta, Val(2), column)
    elseif tail == 1
        _gemv_t_columns_tail!(y, A, x, alpha, beta, Val(1), column)
    end
    return nothing
end

# Tail columns (1..3) of a transpose-oriented gemv.  `MultiFloatVec{L}`
# lanes run the exact scalar accumulation tree per column, so outputs are
# bit-identical to the previous per-column scalar loop.
@inline function _gemv_t_columns_tail!(
    y::AbstractVector{MF},
    A::AbstractMatrix{MF},
    x::AbstractVector{MF},
    alpha::MF,
    beta::MF,
    ::Val{L},
    column::Int,
) where {T,N,L,MF<:MultiFloat{T,N}}
    VL = MultiFloatVec{L,T,N}
    accumulator = zero(VL)
    @inbounds for row in axes(A, 1)
        values = VL(ntuple(l -> A[row, column + l - 1], L))
        coefficient = VL(ntuple(_ -> x[row], L))
        accumulator += values * coefficient
    end
    result = if iszero(beta)
        VL(ntuple(_ -> alpha, L)) * accumulator
    else
        VL(ntuple(_ -> alpha, L)) * accumulator +
        VL(ntuple(_ -> beta, L)) *
        VL(ntuple(l -> y[column + l - 1], L))
    end
    @inbounds for lane in 1:L
        y[column + lane - 1] = result[lane]
    end
    return nothing
end

"""
    gemv!(y, A, x, alpha=one(eltype(A)), beta=zero(eltype(A));
          trans=:N, config=KernelConfig())

Compute `y = alpha*A*x + beta*y` using four-row MultiFloat SIMD groups.
Rows are task-owned, so the threaded path has no arithmetic synchronization.

With `trans=:T`, compute `y = alpha*transpose(A)*x + beta*y` without forming a
transposed matrix. The transpose kernel maps four output columns into SIMD
lanes while traversing rows in ascending order, preserving the deterministic
reduction order.

When `beta == 0`, `y` is not read, so an uninitialized or stale `y` is safe;
this is consistent between `trans=:N` and `trans=:T`. `y` must not alias `A`
or `x`.
"""
function gemv!(
    y::AbstractVector{MF},
    A::AbstractMatrix{MF},
    x::AbstractVector{MF},
    alpha::MF=one(MF),
    beta::MF=zero(MF);
    trans::Symbol=:N,
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    m, n = size(A)
    trans in (:N, :T) || throw(ArgumentError("trans must be :N or :T"))
    _check_supported(MF)
    Base.require_one_based_indexing(y, A, x)
    _require_no_output_alias("gemv!", y, A)
    _require_no_output_alias("gemv!", y, x)

    if trans === :N
        length(x) == n || throw(DimensionMismatch("gemv! input length differs"))
        length(y) == m || throw(DimensionMismatch("gemv! output length differs"))
        workers = _workers(config, cld(m, 32))
        if workers == 1 || !_vector_thread_work_worthwhile(MF, m, n)
            _gemv_rows!(y, A, x, alpha, beta, 1, m)
            return y
        end

        chunk = cld(m, workers)
        @sync for worker in 1:workers
            first_row = (worker - 1) * chunk + 1
            last_row = min(worker * chunk, m)
            first_row <= last_row || continue
            Threads.@spawn _gemv_rows!(
                y, A, x, alpha, beta, first_row, last_row,
            )
        end
        return y
    end

    # trans === :T
    length(x) == m || throw(DimensionMismatch("gemv! input length differs"))
    length(y) == n || throw(DimensionMismatch("gemv! output length differs"))
    workers = _workers(config, cld(n, 32))
    if workers == 1 || !_vector_thread_work_worthwhile(MF, n, m)
        _gemv_t_columns!(y, A, x, alpha, beta, 1, n)
        return y
    end

    chunk = cld(n, workers)
    @sync for worker in 1:workers
        first_column = (worker - 1) * chunk + 1
        last_column = min(worker * chunk, n)
        first_column <= last_column || continue
        Threads.@spawn _gemv_t_columns!(
            y, A, x, alpha, beta, first_column, last_column,
        )
    end
    return y
end
