struct MFLDLT{MF<:MultiFloat,M<:AbstractMatrix{MF}}
    factors::M
    dsub::Vector{MF}
    pivots::Vector{Int}
    blocks::Vector{UInt8}
    info::Int
end

issuccess(F::MFLDLT) = iszero(F.info)

function _mirror_lower_to_upper!(A::AbstractMatrix)
    rows = axes(A, 1)
    columns = axes(A, 2)
    length(rows) == length(columns) ||
        throw(DimensionMismatch("symmetric storage must be square"))
    @inbounds for column in columns
        for row in rows
            row > column || continue
            A[column, row] = A[row, column]
        end
    end
    return A
end

function _ldlt_symmetric_swap!(
    A::AbstractMatrix,
    first::Int,
    second::Int,
    active_first::Int,
)
    first == second && return nothing
    n = size(A, 1)

    # Previously computed L columns move with the active row only.
    @inbounds for column in 1:(active_first - 1)
        A[first, column], A[second, column] =
            A[second, column], A[first, column]
    end

    # The active Schur complement is still stored symmetrically, so apply
    # P*A*P' to that submatrix by one row swap followed by one column swap.
    @inbounds for column in active_first:n
        A[first, column], A[second, column] =
            A[second, column], A[first, column]
    end
    @inbounds for row in active_first:n
        A[row, first], A[row, second] =
            A[row, second], A[row, first]
    end
    return nothing
end

function _select_bk_pivot(
    A::AbstractMatrix{MF},
    k::Int,
    alpha::MF,
) where {MF<:MultiFloat}
    n = size(A, 1)
    absakk = abs(A[k, k])
    k == n && return iszero(absakk) ? (0, k) : (1, k)

    imax = k + 1
    colmax = abs(A[imax, k])
    @inbounds for row in (k + 2):n
        candidate = abs(A[row, k])
        if candidate > colmax
            colmax = candidate
            imax = row
        end
    end

    max(absakk, colmax) == zero(MF) && return (0, k)
    absakk >= alpha * colmax && return (1, k)

    rowmax = zero(MF)
    @inbounds for column in k:(imax - 1)
        rowmax = max(rowmax, abs(A[imax, column]))
    end
    @inbounds for row in (imax + 1):n
        rowmax = max(rowmax, abs(A[row, imax]))
    end

    if absakk >= alpha * colmax * (colmax / rowmax)
        return (1, k)
    elseif abs(A[imax, imax]) >= alpha * rowmax
        return (1, imax)
    end
    return (2, imax)
end

function _ldlt_rank1_update!(
    A::AbstractMatrix{MF},
    k::Int,
    d::MF,
) where {T,N,MF<:MultiFloat{T,N}}
    n = size(A, 1)
    V4 = MultiFloatVec{4,T,N}

    @inbounds for column in (k + 1):n
        coefficient = d * A[column, k]
        row = column
        while row + 3 <= n
            values = V4(
                A[row, column],
                A[row + 1, column],
                A[row + 2, column],
                A[row + 3, column],
            )
            factors = V4(
                A[row, k],
                A[row + 1, k],
                A[row + 2, k],
                A[row + 3, k],
            )
            values -= factors * V4(coefficient)
            for lane in 1:4
                i = row + lane - 1
                value = values[lane]
                A[i, column] = value
                A[column, i] = value
            end
            row += 4
        end
        while row <= n
            value = A[row, column] - A[row, k] * coefficient
            A[row, column] = value
            A[column, row] = value
            row += 1
        end
    end
    return nothing
end

function _ldlt_rank2_update!(
    A::AbstractMatrix{MF},
    k::Int,
    d11::MF,
    d21::MF,
    d22::MF,
) where {T,N,MF<:MultiFloat{T,N}}
    n = size(A, 1)
    V4 = MultiFloatVec{4,T,N}

    @inbounds for column in (k + 2):n
        l1 = A[column, k]
        l2 = A[column, k + 1]
        coefficient1 = d11 * l1 + d21 * l2
        coefficient2 = d21 * l1 + d22 * l2
        row = column
        while row + 3 <= n
            values = V4(
                A[row, column],
                A[row + 1, column],
                A[row + 2, column],
                A[row + 3, column],
            )
            first_factors = V4(
                A[row, k],
                A[row + 1, k],
                A[row + 2, k],
                A[row + 3, k],
            )
            second_factors = V4(
                A[row, k + 1],
                A[row + 1, k + 1],
                A[row + 2, k + 1],
                A[row + 3, k + 1],
            )
            values -= first_factors * V4(coefficient1) +
                      second_factors * V4(coefficient2)
            for lane in 1:4
                i = row + lane - 1
                value = values[lane]
                A[i, column] = value
                A[column, i] = value
            end
            row += 4
        end
        while row <= n
            value = A[row, column] -
                    A[row, k] * coefficient1 -
                    A[row, k + 1] * coefficient2
            A[row, column] = value
            A[column, row] = value
            row += 1
        end
    end
    return nothing
end

"""
    ldlt_plan(T, n, config=KernelConfig())

Resolve whether LDLT uses the robust scalar right-looking path or the lazy
blocked panel path. The blocked panel performs the same Bunch--Kaufman pivot
search on the current Schur complement, but defers its trailing update to one
package-level `gemmt!` triangular-update call.
"""
function ldlt_plan(
    ::Type{MF},
    n::Int,
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    _check_supported(MF)
    strategy = config.ldlt_strategy
    strategy in (:auto, :unblocked, :blocked) ||
        throw(ArgumentError("ldlt_strategy must be :auto, :unblocked, or :blocked"))
    block = max(_resolved_ldlt_block(MF, config), 1)
    if strategy === :unblocked
        return LDLTPlan(:unblocked, :forced_unblocked, 1)
    elseif strategy === :blocked
        return LDLTPlan(:blocked, :forced_blocked, max(block, 2))
    end
    use_blocked = block > 1 && n >= max(config.ldlt_blocked_crossover, 1)
    return LDLTPlan(
        use_blocked ? :blocked : :unblocked,
        use_blocked ? :auto_above_crossover : :auto_below_crossover,
        use_blocked ? block : 1,
    )
end

function _factor_ldlt_unblocked!(
    A::AbstractMatrix{MF},
    dsub::Vector{MF},
    pivots::Vector{Int},
    blocks::Vector{UInt8},
    alpha::MF,
) where {MF<:MultiFloat}
    n = size(A, 1)
    k = 1
    while k <= n
        block_size, pivot = _select_bk_pivot(A, k, alpha)
        if block_size == 0
            return k
        elseif block_size == 1
            pivots[k] = pivot
            blocks[k] = UInt8(1)
            _ldlt_symmetric_swap!(A, k, pivot, k)

            d = A[k, k]
            iszero(d) && return k
            @inbounds for row in (k + 1):n
                A[row, k] /= d
            end
            k < n && _ldlt_rank1_update!(A, k, d)
            k += 1
        else
            k < n || return k
            pivots[k] = pivot
            blocks[k] = UInt8(2)
            blocks[k + 1] = UInt8(0)
            _ldlt_symmetric_swap!(A, k + 1, pivot, k)

            d11 = A[k, k]
            d21 = A[k + 1, k]
            d22 = A[k + 1, k + 1]
            determinant = d11 * d22 - d21 * d21
            iszero(determinant) && return k

            @inbounds for row in (k + 2):n
                first = A[row, k]
                second = A[row, k + 1]
                A[row, k] =
                    (first * d22 - second * d21) / determinant
                A[row, k + 1] =
                    (second * d11 - first * d21) / determinant
            end

            dsub[k] = d21
            k + 1 < n && _ldlt_rank2_update!(A, k, d11, d21, d22)
            A[k + 1, k] = zero(MF)
            A[k, k + 1] = zero(MF)
            k += 2
        end
    end
    return 0
end

@inline function _ldlt_panel_entry(
    A::AbstractMatrix{MF},
    first_index::Int,
    second_index::Int,
    panel_first::Int,
    pivot_first::Int,
    dsub::Vector{MF},
    blocks::Vector{UInt8},
) where {MF<:MultiFloat}
    row = max(first_index, second_index)
    column = min(first_index, second_index)
    value = A[row, column]
    q = panel_first
    @inbounds while q < pivot_first
        block = blocks[q]
        if block == UInt8(1)
            value -= A[row, q] * (A[q, q] * A[column, q])
            q += 1
        elseif block == UInt8(2)
            row_first = A[row, q]
            row_second = A[row, q + 1]
            column_first = A[column, q]
            column_second = A[column, q + 1]
            coefficient_first =
                A[q, q] * column_first + dsub[q] * column_second
            coefficient_second =
                dsub[q] * column_first + A[q + 1, q + 1] * column_second
            value -= row_first * coefficient_first +
                     row_second * coefficient_second
            q += 2
        else
            throw(ArgumentError("invalid LDLT panel block structure"))
        end
    end
    return value
end

function _select_bk_panel_pivot(
    A::AbstractMatrix{MF},
    k::Int,
    panel_first::Int,
    dsub::Vector{MF},
    blocks::Vector{UInt8},
    alpha::MF,
) where {MF<:MultiFloat}
    n = size(A, 1)
    diagonal = _ldlt_panel_entry(
        A, k, k, panel_first, k, dsub, blocks,
    )
    absakk = abs(diagonal)
    k == n && return iszero(absakk) ? (0, k) : (1, k)

    imax = k + 1
    colmax = abs(_ldlt_panel_entry(
        A, imax, k, panel_first, k, dsub, blocks,
    ))
    @inbounds for row in (k + 2):n
        candidate = abs(_ldlt_panel_entry(
            A, row, k, panel_first, k, dsub, blocks,
        ))
        if candidate > colmax
            colmax = candidate
            imax = row
        end
    end

    max(absakk, colmax) == zero(MF) && return (0, k)
    absakk >= alpha * colmax && return (1, k)

    rowmax = zero(MF)
    @inbounds for column in k:(imax - 1)
        rowmax = max(
            rowmax,
            abs(_ldlt_panel_entry(
                A, imax, column, panel_first, k, dsub, blocks,
            )),
        )
    end
    @inbounds for row in (imax + 1):n
        rowmax = max(
            rowmax,
            abs(_ldlt_panel_entry(
                A, row, imax, panel_first, k, dsub, blocks,
            )),
        )
    end

    if absakk >= alpha * colmax * (colmax / rowmax)
        return (1, k)
    end
    candidate_diagonal = abs(_ldlt_panel_entry(
        A, imax, imax, panel_first, k, dsub, blocks,
    ))
    return candidate_diagonal >= alpha * rowmax ? (1, imax) : (2, imax)
end

function _factor_ldlt_panel!(
    A::AbstractMatrix{MF},
    panel_first::Int,
    requested_last::Int,
    dsub::Vector{MF},
    pivots::Vector{Int},
    blocks::Vector{UInt8},
    alpha::MF,
) where {MF<:MultiFloat}
    n = size(A, 1)
    panel_last = requested_last
    k = panel_first
    while k <= panel_last
        block_size, pivot = _select_bk_panel_pivot(
            A, k, panel_first, dsub, blocks, alpha,
        )
        block_size == 0 && return (k, panel_last)
        if block_size == 2 && k == panel_last
            k < n || return (k, panel_last)
            panel_last += 1
        end

        if block_size == 1
            pivots[k] = pivot
            blocks[k] = UInt8(1)
            _ldlt_symmetric_swap!(A, k, pivot, k)

            d = _ldlt_panel_entry(
                A, k, k, panel_first, k, dsub, blocks,
            )
            iszero(d) && return (k, panel_last)
            A[k, k] = d
            @inbounds for row in (k + 1):n
                entry = _ldlt_panel_entry(
                    A, row, k, panel_first, k, dsub, blocks,
                )
                A[row, k] = entry / d
            end
            k += 1
        else
            pivots[k] = pivot
            blocks[k] = UInt8(2)
            blocks[k + 1] = UInt8(0)
            _ldlt_symmetric_swap!(A, k + 1, pivot, k)

            d11 = _ldlt_panel_entry(
                A, k, k, panel_first, k, dsub, blocks,
            )
            d21 = _ldlt_panel_entry(
                A, k + 1, k, panel_first, k, dsub, blocks,
            )
            d22 = _ldlt_panel_entry(
                A, k + 1, k + 1, panel_first, k, dsub, blocks,
            )
            determinant = d11 * d22 - d21 * d21
            iszero(determinant) && return (k, panel_last)
            A[k, k] = d11
            A[k + 1, k + 1] = d22
            dsub[k] = d21

            @inbounds for row in (k + 2):n
                first = _ldlt_panel_entry(
                    A, row, k, panel_first, k, dsub, blocks,
                )
                second = _ldlt_panel_entry(
                    A, row, k + 1, panel_first, k, dsub, blocks,
                )
                A[row, k] =
                    (first * d22 - second * d21) / determinant
                A[row, k + 1] =
                    (second * d11 - first * d21) / determinant
            end
            A[k + 1, k] = zero(MF)
            A[k, k + 1] = zero(MF)
            k += 2
        end
    end
    return (0, panel_last)
end

function _build_ldlt_weighted_panel!(
    weighted::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    panel_first::Int,
    panel_last::Int,
    dsub::Vector{MF},
    blocks::Vector{UInt8},
) where {MF<:MultiFloat}
    trailing_first = panel_last + 1
    trailing_count = size(A, 1) - panel_last
    q = panel_first
    @inbounds while q <= panel_last
        local_column = q - panel_first + 1
        if blocks[q] == UInt8(1)
            d = A[q, q]
            for local_row in 1:trailing_count
                row = trailing_first + local_row - 1
                weighted[local_row, local_column] = A[row, q] * d
            end
            q += 1
        else
            d11 = A[q, q]
            d21 = dsub[q]
            d22 = A[q + 1, q + 1]
            for local_row in 1:trailing_count
                row = trailing_first + local_row - 1
                first = A[row, q]
                second = A[row, q + 1]
                weighted[local_row, local_column] =
                    first * d11 + second * d21
                weighted[local_row, local_column + 1] =
                    first * d21 + second * d22
            end
            q += 2
        end
    end
    return weighted
end

"""
    ldlt!(A; check=true, config=KernelConfig())

Symmetric-indefinite `L*D*L'` factorization for dense MultiFloat matrices.
The lower triangle of `A` is authoritative on input. Pivot selection follows
the Bunch--Kaufman threshold strategy and may use 1x1 or 2x2 diagonal blocks.

The blocked route keeps the panel Schur complement lazy: each pivot column and
candidate pivot row is evaluated against all earlier pivots in that panel,
then one `L21 * (L21*D)'` `gemmt!` updates the trailing matrix. Thus blocking does
not weaken the pivot search or silently fall back to a no-pivot factorization.

On return, the strict lower triangle stores unit-lower `L`, the diagonal stores
the diagonal of `D`, and `F.dsub[k]` stores the off-diagonal of a 2x2 `D`
block starting at `k`.
"""
function ldlt!(
    A::AbstractMatrix{MF};
    check::Bool=true,
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    n, m = size(A)
    n == m || throw(DimensionMismatch("ldlt! requires a square matrix"))
    _check_supported(MF)
    _mirror_lower_to_upper!(A)
    if !_lower_triangle_finite(A)
        check && throw(DomainError(A, "ldlt!: input matrix contains non-finite entries"))
        return MFLDLT{MF,typeof(A)}(A, zeros(MF, n), collect(1:n), zeros(UInt8, n), -1)
    end

    dsub = zeros(MF, n)
    pivots = collect(1:n)
    blocks = zeros(UInt8, n)
    alpha = (one(MF) + sqrt(MF(17))) / MF(8)
    plan = ldlt_plan(MF, n, config)
    info = if plan.strategy === :blocked
        _factor_ldlt_blocked!(
            A, dsub, pivots, blocks, alpha, plan, config,
        )
    else
        _factor_ldlt_unblocked!(A, dsub, pivots, blocks, alpha)
    end

    if !iszero(info) && check
        throw(LinearAlgebra.SingularException(info))
    end
    return MFLDLT{MF,typeof(A)}(A, dsub, pivots, blocks, info)
end
