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
            accumulator += values * V4(x[column])
        end
        result = V4(alpha) * accumulator + V4(beta) * V4(
            y[row], y[row + 1], y[row + 2], y[row + 3],
        )
        for lane in 1:4
            y[row + lane - 1] = result[lane]
        end
        row += 4
    end

    for scalar_row in row:last_row
        accumulator = zero(MF)
        for column in axes(A, 2)
            accumulator += A[scalar_row, column] * x[column]
        end
        y[scalar_row] = alpha * accumulator + beta * y[scalar_row]
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
            accumulator += values * V4(x[row])
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

    for scalar_column in column:last_column
        accumulator = zero(MF)
        for row in axes(A, 1)
            accumulator += A[row, scalar_column] * x[row]
        end
        y[scalar_column] = beta == zero(MF) ?
            alpha * accumulator :
            alpha * accumulator + beta * y[scalar_column]
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

    if trans === :N
        length(x) == n || throw(DimensionMismatch("gemv! input length differs"))
        length(y) == m || throw(DimensionMismatch("gemv! output length differs"))
        workers = _workers(config, cld(m, 32))
        if workers == 1 || m < 64
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
    if workers == 1 || n < 64
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
