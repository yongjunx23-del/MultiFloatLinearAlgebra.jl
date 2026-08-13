function _syrk_column!(
    output::AbstractMatrix{MF},
    panel::AbstractMatrix{MF},
    column::Int,
    alpha::MF,
    beta::MF,
) where {T,N,MF<:MultiFloat{T,N}}
    columns = size(panel, 2)
    reduction = size(panel, 1)
    V4 = MultiFloatVec{4,T,N}
    row = column

    @inbounds while row + 3 <= columns
        accumulator = zero(V4)
        for k in 1:reduction
            values = V4(
                panel[k, row],
                panel[k, row + 1],
                panel[k, row + 2],
                panel[k, row + 3],
            )
            accumulator = _structured_mulacc(accumulator, values, V4(panel[k, column]))
        end
        result = V4(alpha) * accumulator + V4(beta) * V4(
            output[row, column],
            output[row + 1, column],
            output[row + 2, column],
            output[row + 3, column],
        )
        for lane in 1:4
            output[row + lane - 1, column] = result[lane]
        end
        row += 4
    end

    while row <= columns
        accumulator = zero(MF)
        for k in 1:reduction
            accumulator += panel[k, row] * panel[k, column]
        end
        output[row, column] =
            alpha * accumulator + beta * output[row, column]
        row += 1
    end
    return nothing
end

"""
    syrk!(C, panel, alpha=one(eltype(panel)), beta=zero(eltype(panel));
          config=KernelConfig())

Update only the lower triangle:

`C = alpha * transpose(panel) * panel + beta * C`.

Every output column is independently owned by one task. Within a column,
four lower-triangle entries are evaluated in MultiFloat SIMD lanes while
preserving the ascending reduction order in every lane.
"""
function syrk!(
    output::AbstractMatrix{MF},
    panel::AbstractMatrix{MF},
    alpha::MF=one(MF),
    beta::MF=zero(MF);
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    columns = size(panel, 2)
    size(output) == (columns, columns) ||
        throw(DimensionMismatch("syrk! output dimensions differ"))
    _check_supported(MF)
    Base.require_one_based_indexing(output, panel)

    workers = _workers(config, columns)
    if workers == 1 || columns < 24
        @inbounds for column in 1:columns
            _syrk_column!(output, panel, column, alpha, beta)
        end
        return output
    end

    @sync for worker in 1:workers
        Threads.@spawn begin
            @inbounds for column in worker:workers:columns
                _syrk_column!(output, panel, column, alpha, beta)
            end
        end
    end
    return output
end
