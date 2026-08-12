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

"""
    gemv!(y, A, x, alpha=one(eltype(A)), beta=zero(eltype(A)); config=KernelConfig())

Compute `y = alpha*A*x + beta*y` using four-row MultiFloat SIMD groups.
Rows are task-owned, so the threaded path has no arithmetic synchronization.
"""
function gemv!(
    y::AbstractVector{MF},
    A::AbstractMatrix{MF},
    x::AbstractVector{MF},
    alpha::MF=one(MF),
    beta::MF=zero(MF);
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    m, n = size(A)
    length(x) == n || throw(DimensionMismatch("gemv! input length differs"))
    length(y) == m || throw(DimensionMismatch("gemv! output length differs"))
    _check_supported(MF)
    Base.require_one_based_indexing(y, A, x)

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
