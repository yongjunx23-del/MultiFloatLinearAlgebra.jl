function _symv_rows!(
    y::AbstractVector{MF},
    A::AbstractMatrix{MF},
    x::AbstractVector{MF},
    alpha::MF,
    beta::MF,
    first_row::Int,
    last_row::Int,
    upper::Bool,
) where {MF<:MultiFloat}
    n = size(A, 1)
    @inbounds for row in first_row:last_row
        accumulator = zero(MF)
        if upper
            for column in 1:(row - 1)
                accumulator += A[column, row] * x[column]
            end
            accumulator += A[row, row] * x[row]
            for column in (row + 1):n
                accumulator += A[row, column] * x[column]
            end
        else
            for column in 1:(row - 1)
                accumulator += A[row, column] * x[column]
            end
            accumulator += A[row, row] * x[row]
            for column in (row + 1):n
                accumulator += A[column, row] * x[column]
            end
        end
        y[row] = alpha * accumulator + beta * y[row]
    end
    return nothing
end

"""
    symv!(y, A, x, alpha=one(eltype(A)), beta=zero(eltype(A));
          uplo=:lower, config=KernelConfig())

Compute the symmetric matrix-vector product `y = alpha*A*x + beta*y` using only
the selected triangle of `A`.

Only `uplo` is authoritative: with `uplo=:lower` the upper triangle is never
read (and may contain `NaN`, `Inf`, or stale values), and with `uplo=:upper`
the lower triangle is never read. Each output row is owned by one task, and
each row performs an ascending-column reduction, so the result is deterministic.
"""
function symv!(
    y::AbstractVector{MF},
    A::AbstractMatrix{MF},
    x::AbstractVector{MF},
    alpha::MF=one(MF),
    beta::MF=zero(MF);
    uplo::Symbol=:lower,
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    n, m = size(A)
    n == m || throw(DimensionMismatch("symv! requires a square matrix"))
    length(x) == n || throw(DimensionMismatch("symv! input length differs"))
    length(y) == n || throw(DimensionMismatch("symv! output length differs"))
    uplo in (:lower, :upper) || throw(ArgumentError("uplo must be :lower or :upper"))
    _check_supported(MF)
    Base.require_one_based_indexing(y, A, x)

    upper = uplo === :upper
    workers = _workers(config, cld(n, 32))
    if workers == 1 || n < 64
        _symv_rows!(y, A, x, alpha, beta, 1, n, upper)
        return y
    end

    chunk = cld(n, workers)
    @sync for worker in 1:workers
        first_row = (worker - 1) * chunk + 1
        last_row = min(worker * chunk, n)
        first_row <= last_row || continue
        Threads.@spawn _symv_rows!(
            y, A, x, alpha, beta, first_row, last_row, upper,
        )
    end
    return y
end
