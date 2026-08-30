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
        result = if iszero(beta)
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

    tail = columns - row + 1
    if tail == 3
        _syrk_tail!(output, panel, column, alpha, beta, Val(3), row)
    elseif tail == 2
        _syrk_tail!(output, panel, column, alpha, beta, Val(2), row)
    elseif tail == 1
        _syrk_tail!(output, panel, column, alpha, beta, Val(1), row)
    end
    return nothing
end

# Tail columns (1..3) of a lower-triangle syrk column.  `MultiFloatVec{L}`
# lanes run the exact scalar accumulation tree per column, so the outputs
# are bit-identical to the previous per-column scalar loop while keeping
# SIMD lanes for panels whose column count is not a multiple of four.
@inline function _syrk_tail!(
    output::AbstractMatrix{MF},
    panel::AbstractMatrix{MF},
    column::Int,
    alpha::MF,
    beta::MF,
    ::Val{L},
    row::Int,
) where {T,N,L,MF<:MultiFloat{T,N}}
    VL = MultiFloatVec{L,T,N}
    reduction = size(panel, 1)
    accumulator = zero(VL)
    @inbounds for k in 1:reduction
        values = VL(ntuple(l -> panel[k, row + l - 1], L))
        coefficient = VL(ntuple(_ -> panel[k, column], L))
        accumulator += values * coefficient
    end
    result = if iszero(beta)
        VL(ntuple(_ -> alpha, L)) * accumulator
    else
        VL(ntuple(_ -> alpha, L)) * accumulator +
        VL(ntuple(_ -> beta, L)) *
        VL(ntuple(l -> output[row + l - 1, column], L))
    end
    @inbounds for lane in 1:L
        output[row + lane - 1, column] = result[lane]
    end
    return nothing
end

"""
    syrk!(C, panel, alpha=one(eltype(panel)), beta=zero(eltype(panel));
          uplo=:lower, config=KernelConfig())

Update only the lower triangle:

`C = alpha * transpose(panel) * panel + beta * C`.

Every output column is independently owned by one task. Within a column,
four lower-triangle entries are evaluated in MultiFloat SIMD lanes while
preserving the ascending reduction order in every lane. `uplo=:lower` (or
`:L`) makes this storage contract explicit; other values are rejected. The
upper triangle is neither read nor written. `C` must not alias `panel`.
When `beta == 0`, the authoritative lower destination is not read.
"""
function syrk!(
    output::AbstractMatrix{MF},
    panel::AbstractMatrix{MF},
    alpha::MF=one(MF),
    beta::MF=zero(MF);
    uplo::Symbol=:lower,
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    uplo in (:lower, :L) ||
        throw(ArgumentError("syrk! currently supports only uplo=:lower or :L"))
    columns = size(panel, 2)
    size(output) == (columns, columns) ||
        throw(DimensionMismatch("syrk! output dimensions differ"))
    _check_supported(MF)
    Base.require_one_based_indexing(output, panel)
    _require_no_output_alias("syrk!", output, panel)

    workers = _workers(config, columns)
    if workers == 1 || columns < 24
        @inbounds for column in 1:columns
            _syrk_column!(output, panel, column, alpha, beta)
        end
        return output
    end

    # Dynamic work queue: column c costs (columns - c + 1) reductions, so a
    # static strided split leaves the worker owning column 1 with ~13% more
    # work than the worker owning the last column.  Grabbing the next column
    # from a shared atomic counter self-balances the triangular workload at
    # negligible cost (one atomic add per column).
    next_column = Threads.Atomic{Int}(1)
    @sync for worker in 1:workers
        Threads.@spawn begin
            while true
                column = Threads.atomic_add!(next_column, 1)
                column > columns && break
                _syrk_column!(output, panel, column, alpha, beta)
            end
        end
    end
    return output
end

@inline function _packed_lower_index(
    dimension::Int,
    row::Int,
    column::Int,
)
    return (column - 1) * (2 * dimension - column + 2) ÷ 2 +
           (row - column + 1)
end

function _syrk_packed_column!(
    output::AbstractVector{MF},
    panel::AbstractMatrix{MF},
    column::Int,
    reduction_first::Int,
    reduction_last::Int,
    alpha::MF,
    beta::MF,
) where {T,N,MF<:MultiFloat{T,N}}
    columns = size(panel, 2)
    V4 = MultiFloatVec{4,T,N}
    row = column
    output_index = _packed_lower_index(columns, row, column)

    @inbounds while row + 3 <= columns
        accumulator = zero(V4)
        for k in reduction_first:reduction_last
            values = V4(
                panel[k, row],
                panel[k, row + 1],
                panel[k, row + 2],
                panel[k, row + 3],
            )
            accumulator =
                _structured_mulacc(accumulator, values, V4(panel[k, column]))
        end
        result = if iszero(beta)
            V4(alpha) * accumulator
        else
            V4(alpha) * accumulator + V4(beta) * V4(
                output[output_index],
                output[output_index + 1],
                output[output_index + 2],
                output[output_index + 3],
            )
        end
        for lane in 1:4
            output[output_index + lane - 1] = result[lane]
        end
        row += 4
        output_index += 4
    end

    tail = columns - row + 1
    if tail == 3
        _syrk_packed_tail!(output, panel, column, reduction_first, reduction_last,
            alpha, beta, Val(3), row, output_index)
    elseif tail == 2
        _syrk_packed_tail!(output, panel, column, reduction_first, reduction_last,
            alpha, beta, Val(2), row, output_index)
    elseif tail == 1
        _syrk_packed_tail!(output, panel, column, reduction_first, reduction_last,
            alpha, beta, Val(1), row, output_index)
    end
    return nothing
end

@inline function _syrk_packed_tail!(
    output::AbstractVector{MF}, panel::AbstractMatrix{MF}, column::Int,
    reduction_first::Int, reduction_last::Int, alpha::MF, beta::MF,
    ::Val{L}, row::Int, output_index::Int,
) where {T,N,L,MF<:MultiFloat{T,N}}
    VL = MultiFloatVec{L,T,N}
    accumulator = zero(VL)
    @inbounds for k in reduction_first:reduction_last
        values = VL(ntuple(l -> panel[k, row + l - 1], L))
        coefficient = VL(ntuple(_ -> panel[k, column], L))
        accumulator += values * coefficient
    end
    result = if iszero(beta)
        VL(ntuple(_ -> alpha, L)) * accumulator
    else
        VL(ntuple(_ -> alpha, L)) * accumulator +
        VL(ntuple(_ -> beta, L)) *
        VL(ntuple(l -> output[output_index + l - 1], L))
    end
    @inbounds for lane in 1:L
        output[output_index + lane - 1] = result[lane]
    end
    return nothing
end

"""
    syrk_packed!(packed, panel, alpha=one(eltype(panel)),
                 beta=zero(eltype(panel));
                 reduction_first=1, reduction_last=size(panel, 1),
                 config=KernelConfig())

Update a packed column-major lower triangle with

`packed(C) = alpha * transpose(panel[reduction_first:reduction_last, :]) *
             panel[reduction_first:reduction_last, :] + beta * packed(C)`.

The packed layout stores `(1,1),(2,1),...,(n,1),(2,2),...,(n,n)`. Reduction
bounds are explicit and inclusive so callers can form deterministic partial
Gram matrices without materializing panel slices. `beta == 0` does not read
the destination. Each output column is independently owned when threaded, and
every output preserves ascending reduction order.

This is a numerical storage primitive only. It does not accept global IDs or
perform solver assembly/scatter policy. `packed` must not alias `panel`.
"""
function syrk_packed!(
    output::AbstractVector{MF},
    panel::AbstractMatrix{MF},
    alpha::MF=one(MF),
    beta::MF=zero(MF);
    reduction_first::Int=1,
    reduction_last::Int=size(panel, 1),
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    columns = size(panel, 2)
    required = columns * (columns + 1) ÷ 2
    length(output) == required ||
        throw(DimensionMismatch("syrk_packed! output length differs"))
    1 <= reduction_first <= reduction_last <= size(panel, 1) ||
        throw(BoundsError(panel, reduction_first:reduction_last))
    _check_supported(MF)
    Base.require_one_based_indexing(output, panel)
    _require_no_output_alias("syrk_packed!", output, panel)

    workers = _workers(config, columns)
    if workers == 1 || columns < 24
        @inbounds for column in 1:columns
            _syrk_packed_column!(
                output,
                panel,
                column,
                reduction_first,
                reduction_last,
                alpha,
                beta,
            )
        end
        return output
    end

    # Same dynamic work queue as `syrk!`: the triangular column cost makes a
    # static strided split imbalanced, and an atomic counter self-balances it.
    next_column = Threads.Atomic{Int}(1)
    @sync for worker in 1:workers
        Threads.@spawn begin
            while true
                column = Threads.atomic_add!(next_column, 1)
                column > columns && break
                _syrk_packed_column!(
                    output,
                    panel,
                    column,
                    reduction_first,
                    reduction_last,
                    alpha,
                    beta,
                )
            end
        end
    end
    return output
end
