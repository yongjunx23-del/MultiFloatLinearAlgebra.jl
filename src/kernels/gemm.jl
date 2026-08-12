@inline function _gemm_store_pair!(
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    B::AbstractMatrix{MF},
    row::Int,
    column::Int,
    alpha::MF,
    beta::MF,
) where {T,N,MF<:MultiFloat{T,N}}
    V4 = MultiFloatVec{4,T,N}
    first_accumulator = zero(V4)
    second_accumulator = zero(V4)
    @inbounds for k in axes(A, 2)
        values = V4(
            A[row, k],
            A[row + 1, k],
            A[row + 2, k],
            A[row + 3, k],
        )
        first_accumulator += values * V4(B[k, column])
        second_accumulator += values * V4(B[k, column + 1])
    end

    first_result = V4(alpha) * first_accumulator + V4(beta) * V4(
        C[row, column],
        C[row + 1, column],
        C[row + 2, column],
        C[row + 3, column],
    )
    second_result = V4(alpha) * second_accumulator + V4(beta) * V4(
        C[row, column + 1],
        C[row + 1, column + 1],
        C[row + 2, column + 1],
        C[row + 3, column + 1],
    )
    @inbounds for lane in 1:4
        C[row + lane - 1, column] = first_result[lane]
        C[row + lane - 1, column + 1] = second_result[lane]
    end
    return nothing
end

function _gemm_column_range!(
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    B::AbstractMatrix{MF},
    alpha::MF,
    beta::MF,
    first_column::Int,
    last_column::Int,
) where {T,N,MF<:MultiFloat{T,N}}
    m = size(A, 1)
    V4 = MultiFloatVec{4,T,N}
    column = first_column

    @inbounds while column + 1 <= last_column
        row = 1
        while row + 3 <= m
            _gemm_store_pair!(C, A, B, row, column, alpha, beta)
            row += 4
        end
        while row <= m
            first_accumulator = zero(MF)
            second_accumulator = zero(MF)
            for k in axes(A, 2)
                a = A[row, k]
                first_accumulator += a * B[k, column]
                second_accumulator += a * B[k, column + 1]
            end
            C[row, column] =
                alpha * first_accumulator + beta * C[row, column]
            C[row, column + 1] =
                alpha * second_accumulator + beta * C[row, column + 1]
            row += 1
        end
        column += 2
    end

    if column <= last_column
        row = 1
        while row + 3 <= m
            accumulator = zero(V4)
            for k in axes(A, 2)
                values = V4(
                    A[row, k],
                    A[row + 1, k],
                    A[row + 2, k],
                    A[row + 3, k],
                )
                accumulator += values * V4(B[k, column])
            end
            result = V4(alpha) * accumulator + V4(beta) * V4(
                C[row, column],
                C[row + 1, column],
                C[row + 2, column],
                C[row + 3, column],
            )
            for lane in 1:4
                C[row + lane - 1, column] = result[lane]
            end
            row += 4
        end
        while row <= m
            accumulator = zero(MF)
            for k in axes(A, 2)
                accumulator += A[row, k] * B[k, column]
            end
            C[row, column] = alpha * accumulator + beta * C[row, column]
            row += 1
        end
    end
    return nothing
end

"""
    gemm!(C, A, B, alpha=one(eltype(A)), beta=zero(eltype(A)); config=KernelConfig())

CPU matrix multiplication specialized for `MultiFloat{T,N}`. The hot path
maps four independent output rows into `MultiFloatVec{4,T,N}` lanes and pairs
two output columns to reuse every `A` load. Threads own disjoint column tiles.
"""
function gemm!(
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    B::AbstractMatrix{MF},
    alpha::MF=one(MF),
    beta::MF=zero(MF);
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    m, k = size(A)
    size(B, 1) == k || throw(DimensionMismatch("gemm! inner dimensions differ"))
    n = size(B, 2)
    size(C) == (m, n) || throw(DimensionMismatch("gemm! output dimensions differ"))
    _check_supported(MF)

    tile = max(config.column_tile, 1)
    jobs = cld(n, tile)
    workers = _workers(config, jobs)
    if workers == 1 || jobs == 1
        _gemm_column_range!(C, A, B, alpha, beta, 1, n)
        return C
    end

    @sync for worker in 1:workers
        Threads.@spawn begin
            for job in worker:workers:jobs
                first_column = (job - 1) * tile + 1
                last_column = min(job * tile, n)
                _gemm_column_range!(
                    C, A, B, alpha, beta, first_column, last_column,
                )
            end
        end
    end
    return C
end
