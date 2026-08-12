"""
    mfdot(x, y)

MultiFloat dot product with a deterministic ascending reduction order.
This is the scalar reference primitive used by the higher-level backend.
"""
function mfdot(
    x::AbstractVector{MF},
    y::AbstractVector{MF},
) where {MF<:MultiFloat}
    length(x) == length(y) ||
        throw(DimensionMismatch("dot product lengths differ"))
    _check_supported(MF)
    accumulator = zero(MF)
    @inbounds for i in eachindex(x, y)
        accumulator += x[i] * y[i]
    end
    return accumulator
end
