function _gemmt_column!(
    output::AbstractMatrix{MF},
    left::AbstractMatrix{MF},
    right::AbstractMatrix{MF},
    column::Int,
    alpha::MF,
    beta::MF,
    ::Val{OVERWRITE},
) where {T,N,MF<:MultiFloat{T,N},OVERWRITE}
    rows = size(left, 1)
    reduction = size(left, 2)
    V4 = MultiFloatVec{4,T,N}
    row = column

    @inbounds while row + 3 <= rows
        accumulator = zero(V4)
        for k in 1:reduction
            values = V4(
                left[row, k],
                left[row + 1, k],
                left[row + 2, k],
                left[row + 3, k],
            )
            accumulator = _structured_mulacc(accumulator, values, V4(right[column, k]))
        end
        result = if OVERWRITE
            V4(alpha) * accumulator
        else
            V4(alpha) * accumulator + V4(beta) * V4(
                output[row, column],
                output[row + 1, column],
                output[row + 2, column],
                output[row + 3, column],
            )
        end
        for lane in 1:4
            output[row + lane - 1, column] = result[lane]
        end
        row += 4
    end

    while row <= rows
        accumulator = zero(MF)
        for k in 1:reduction
            accumulator += left[row, k] * right[column, k]
        end
        output[row, column] = OVERWRITE ? alpha * accumulator :
            alpha * accumulator + beta * output[row, column]
        row += 1
    end
    return nothing
end

"""
    gemmt!(C, A, B, alpha=one(eltype(A)), beta=zero(eltype(A));
           config=KernelConfig())

Update only the lower triangle of

`C = alpha * A * transpose(B) + beta * C`.

This is the triangular-output GEMM primitive used by blocked LDLT. Callers are
responsible for ensuring that the mathematical update is symmetric when only
one triangle is retained. Each task owns complete output columns, and each
SIMD lane preserves the ascending reduction order. `C` must not alias either
input matrix. When `beta == 0`, the authoritative lower destination is not
read; the inactive upper triangle is always preserved.
"""
function gemmt!(
    output::AbstractMatrix{MF},
    left::AbstractMatrix{MF},
    right::AbstractMatrix{MF},
    alpha::MF=one(MF),
    beta::MF=zero(MF);
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    rows, reduction = size(left)
    size(right) == (rows, reduction) ||
        throw(DimensionMismatch("gemmt! input dimensions differ"))
    size(output) == (rows, rows) ||
        throw(DimensionMismatch("gemmt! output dimensions differ"))
    _check_supported(MF)
    Base.require_one_based_indexing(output, left, right)
    _require_no_output_alias("gemmt!", output, left)
    _require_no_output_alias("gemmt!", output, right)

    workers = _workers(config, rows)
    overwrite = Val(iszero(beta))
    if workers == 1 || rows < 24
        @inbounds for column in 1:rows
            _gemmt_column!(
                output, left, right, column, alpha, beta, overwrite,
            )
        end
        return output
    end

    @sync for worker in 1:workers
        Threads.@spawn begin
            @inbounds for column in worker:workers:rows
                _gemmt_column!(
                    output, left, right, column, alpha, beta, overwrite,
                )
            end
        end
    end
    return output
end
