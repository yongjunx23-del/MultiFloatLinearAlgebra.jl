"""
    mfdot(x, y)

MultiFloat vector or Frobenius inner product with a deterministic ascending
column-major reduction order. Inputs must have identical shapes and one-based
indexing. No conjugation is applied because the supported MultiFloat element
types are real.
"""
function mfdot(
    x::AbstractArray{MF},
    y::AbstractArray{MF},
) where {MF<:MultiFloat}
    size(x) == size(y) ||
        throw(DimensionMismatch("dot product shapes differ"))
    _check_supported(MF)
    Base.require_one_based_indexing(x, y)
    accumulator = zero(MF)
    @inbounds for i in 1:length(x)
        accumulator += x[i] * y[i]
    end
    return accumulator
end
