"""
    factor_diagnostics(F)

Return a stable package-owned NamedTuple of numerical facts about a
factorization. Diagnostics never choose a fallback, precision, rank threshold,
or acceptance policy. Vector-valued fields are copies and may be mutated by
the caller without changing `F`.

For `MFQR`, `atol` and `rtol` are explicit caller thresholds forwarded to
[`numerical_rank`](@ref); their zero defaults mean exact nonzero rank.
"""
function factor_diagnostics end

@inline function _optional_spread(minimum_value, maximum_value)
    minimum_value === nothing && return nothing
    iszero(minimum_value) && return oftype(maximum_value, Inf)
    return maximum_value / minimum_value
end

function _diagonal_magnitude_range(A::AbstractMatrix{MF}, count::Int) where {MF<:MultiFloat}
    count <= 0 && return (nothing, nothing)
    minimum_value = abs(A[1, 1])
    maximum_value = minimum_value
    @inbounds for index in 2:count
        value = abs(A[index, index])
        minimum_value = min(minimum_value, value)
        maximum_value = max(maximum_value, value)
    end
    return minimum_value, maximum_value
end

function factor_diagnostics(F::MFCholesky{MF}) where {MF<:MultiFloat}
    n = size(F.factors, 1)
    accepted = F.info == 0 ? n : F.info > 0 ? F.info - 1 : 0
    minimum_diagonal, maximum_diagonal =
        _diagonal_magnitude_range(F.factors, accepted)
    return (
        kind=:cholesky,
        status=F.info,
        success=issuccess(F),
        failure_location=F.info > 0 ? F.info : nothing,
        accepted_pivots=accepted,
        minimum_diagonal=minimum_diagonal,
        maximum_diagonal=maximum_diagonal,
        diagonal_spread=_optional_spread(minimum_diagonal, maximum_diagonal),
        finite=_lower_triangle_finite(F.factors),
    )
end

function _lu_maximum_u(F::MFLU{MF}) where {MF<:MultiFloat}
    maximum_value = zero(MF)
    rows, columns = size(F.factors)
    @inbounds for column in 1:columns, row in 1:min(column, rows)
        maximum_value = max(maximum_value, abs(F.factors[row, column]))
    end
    return maximum_value
end

function factor_diagnostics(F::MFLU{MF}) where {MF<:MultiFloat}
    _check_factor_lease(F.lease)
    accepted = F.info > 0 ? F.info - 1 : F.info == 0 ? length(F.ipiv) : 0
    minimum_pivot, maximum_pivot =
        _diagonal_magnitude_range(F.factors, accepted)
    maximum_u = _lu_maximum_u(F)
    pivot_growth = iszero(F.original_maximum) ? nothing :
        maximum_u / F.original_maximum
    return (
        kind=:lu,
        status=F.info,
        success=issuccess(F),
        failure_location=F.info > 0 ? F.info : nothing,
        accepted_pivots=accepted,
        pivots=copy(F.ipiv),
        minimum_pivot=minimum_pivot,
        maximum_pivot=maximum_pivot,
        original_maximum=F.info == -1 ? nothing : F.original_maximum,
        maximum_u=maximum_u,
        pivot_growth=pivot_growth,
        finite=_all_finite(F.factors),
    )
end

@inline function _ldlt_inertia_1x1(value)
    if value > zero(value)
        return (1, 0, 0)
    elseif value < zero(value)
        return (0, 1, 0)
    end
    return (0, 0, 1)
end

function _ldlt_inertia_2x2(d11::MF, d21::MF, d22::MF) where {MF<:MultiFloat}
    scale = max(abs(d11), abs(d21), abs(d22))
    iszero(scale) && return (0, 0, 2)
    a = d11 / scale
    b = d21 / scale
    c = d22 / scale
    determinant = a * c - b * b
    if determinant < zero(MF)
        return (1, 1, 0)
    elseif determinant > zero(MF)
        trace = a + c
        return trace > zero(MF) ? (2, 0, 0) : (0, 2, 0)
    end
    trace = a + c
    return trace > zero(MF) ? (1, 0, 1) :
           trace < zero(MF) ? (0, 1, 1) : (0, 0, 2)
end

function _ldlt_smallest_abs_eigenvalue(
    d11::MF,
    d21::MF,
    d22::MF,
) where {MF<:MultiFloat}
    scale = max(abs(d11), abs(d21), abs(d22))
    iszero(scale) && return zero(MF)
    a = d11 / scale
    b = d21 / scale
    c = d22 / scale
    discriminant = sqrt((a - c) * (a - c) + MF(4) * b * b)
    first = (a + c + discriminant) / MF(2)
    second = (a + c - discriminant) / MF(2)
    largest = max(abs(first), abs(second))
    iszero(largest) && return zero(MF)
    determinant = abs(a * c - b * b)
    return scale * (determinant / largest)
end

function factor_diagnostics(F::MFLDLT{MF}) where {MF<:MultiFloat}
    _check_factor_lease(F.lease)
    one_by_one = 0
    two_by_two = 0
    positive = 0
    negative = 0
    zero_count = 0
    minimum_block = nothing
    maximum_block_entry = zero(MF)
    k = 1
    @inbounds while k <= length(F.blocks)
        block = F.blocks[k]
        if block == UInt8(1)
            one_by_one += 1
            value = F.factors[k, k]
            pos, neg, zer = _ldlt_inertia_1x1(value)
            positive += pos
            negative += neg
            zero_count += zer
            block_value = abs(value)
            minimum_block = minimum_block === nothing ? block_value :
                min(minimum_block, block_value)
            maximum_block_entry = max(maximum_block_entry, block_value)
            k += 1
        elseif block == UInt8(2) && k < length(F.blocks)
            two_by_two += 1
            d11 = F.factors[k, k]
            d21 = F.dsub[k]
            d22 = F.factors[k + 1, k + 1]
            pos, neg, zer = _ldlt_inertia_2x2(d11, d21, d22)
            positive += pos
            negative += neg
            zero_count += zer
            block_value = _ldlt_smallest_abs_eigenvalue(d11, d21, d22)
            minimum_block = minimum_block === nothing ? block_value :
                min(minimum_block, block_value)
            maximum_block_entry = max(
                maximum_block_entry, abs(d11), abs(d21), abs(d22),
            )
            k += 2
        else
            break
        end
    end

    original_maximum = F.info == -1 ? nothing : F.original_maximum
    minimum_scaled_block =
        minimum_block === nothing || iszero(F.original_maximum) ? nothing :
        minimum_block / F.original_maximum
    block_growth = iszero(F.original_maximum) ? nothing :
        maximum_block_entry / F.original_maximum
    return (
        kind=:ldlt,
        status=F.info,
        success=issuccess(F),
        failure_location=F.info > 0 ? F.info : nothing,
        one_by_one_pivots=one_by_one,
        two_by_two_pivots=two_by_two,
        pivots=copy(F.pivots),
        blocks=copy(F.blocks),
        inertia=(positive=positive, negative=negative, zero=zero_count),
        minimum_block_eigenvalue_magnitude=minimum_block,
        minimum_scaled_block=minimum_scaled_block,
        original_maximum=original_maximum,
        maximum_block_entry=maximum_block_entry,
        block_growth=block_growth,
        finite=_all_finite(F.factors) && all(isfinite, F.dsub),
    )
end

function factor_diagnostics(
    F::MFQR{MF};
    atol::Real=zero(MF),
    rtol::Real=zero(MF),
) where {MF<:MultiFloat}
    _check_factor_lease(F.lease)
    diagonal = factor_rdiag(F)
    minimum_diagonal = isempty(diagonal) ? nothing : minimum(abs, diagonal)
    maximum_diagonal = isempty(diagonal) ? nothing : maximum(abs, diagonal)
    rank = issuccess(F) ? numerical_rank(F; atol=atol, rtol=rtol) : 0
    return (
        kind=:qr,
        status=F.info,
        success=issuccess(F),
        failure_location=nothing,
        permutation=copy(F.permutation),
        rdiag=diagonal,
        minimum_rdiag=minimum_diagonal,
        maximum_rdiag=maximum_diagonal,
        rdiag_spread=_optional_spread(minimum_diagonal, maximum_diagonal),
        rank_at_threshold=rank,
        atol=MF(atol),
        rtol=MF(rtol),
        finite=_all_finite(F.factors) && all(isfinite, F.tau),
    )
end
