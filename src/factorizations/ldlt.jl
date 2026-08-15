struct MFLDLT{
    MF<:MultiFloat,
    M<:AbstractMatrix{MF},
    D<:AbstractVector{MF},
    P<:AbstractVector{Int},
    B<:AbstractVector{UInt8},
} <: AbstractMFFactorization{MF}
    factors::M
    dsub::D
    pivots::P
    blocks::B
    info::Int
    original_maximum::MF
end

factor_kind(::MFLDLT) = :ldlt
factor_status(F::MFLDLT) = F.info
factor_matrix(F::MFLDLT) = F.factors

"""
    factor_pivots(F::MFLDLT) -> Vector{Int}

Return a caller-owned copy of the raw Bunch-Kaufman step pivots.
"""
factor_pivots(F::MFLDLT) = copy(F.pivots)

"""
    factor_blocks(F::MFLDLT) -> Vector{UInt8}

Return a caller-owned copy of the LDLT block markers. A block start contains
`1` or `2`; the continuation of a 2x2 block contains `0`.
"""
factor_blocks(F::MFLDLT) = copy(F.blocks)

"""
    factor_permutation(F::MFLDLT) -> Vector{Int}

Return the final symmetric permutation `p` satisfying
`A_original[p, p] = L * D * L'`. The returned vector is caller-owned.
"""
function factor_permutation(F::MFLDLT)
    permutation = collect(1:length(F.blocks))
    k = 1
    @inbounds while k <= length(F.blocks)
        block = F.blocks[k]
        if block == UInt8(1)
            pivot = F.pivots[k]
            permutation[k], permutation[pivot] =
                permutation[pivot], permutation[k]
            k += 1
        elseif block == UInt8(2) && k < length(F.blocks)
            pivot = F.pivots[k]
            permutation[k + 1], permutation[pivot] =
                permutation[pivot], permutation[k + 1]
            k += 2
        elseif block == UInt8(0)
            break
        else
            throw(ArgumentError("invalid LDLT block structure"))
        end
    end
    return permutation
end

@inline function _ldlt_solve_2x2(
    d11::MF,
    d21::MF,
    d22::MF,
    first::MF,
    second::MF,
) where {MF<:MultiFloat}
    scale = max(abs(d11), abs(d21), abs(d22))
    if iszero(scale) || !isfinite(scale)
        return (zero(MF), zero(MF), false)
    end

    a = d11 / scale
    b = d21 / scale
    c = d22 / scale
    r1 = first / scale
    r2 = second / scale
    if abs(a) >= abs(b)
        iszero(a) && return (zero(MF), zero(MF), false)
        t = b / a
        u = c - t * b
        (iszero(u) || !isfinite(u)) &&
            return (zero(MF), zero(MF), false)
        x2 = (r2 - t * r1) / u
        x1 = (r1 - b * x2) / a
        return (x1, x2, true)
    end

    iszero(b) && return (zero(MF), zero(MF), false)
    t = a / b
    u = b - t * c
    (iszero(u) || !isfinite(u)) &&
        return (zero(MF), zero(MF), false)
    x2 = (r1 - t * r2) / u
    x1 = (r2 - c * x2) / b
    return (x1, x2, true)
end

function _prepare_ldlt_metadata!(
    ::Type{MF},
    count::Int,
    block_capacity::Int,
    workspace::Union{Nothing,MFWorkspace{MF}},
) where {MF<:MultiFloat}
    if workspace === nothing
        return (
            zeros(MF, count),
            collect(1:count),
            zeros(UInt8, count),
            block_capacity > 0 ?
                Matrix{MF}(undef, count, block_capacity + 1) : nothing,
        )
    end

    _acquire_factor_workspace!(workspace, count, block_capacity)
    dsub = @view workspace.ldlt_dsub[1:count]
    pivots = @view workspace.ldlt_pivots[1:count]
    blocks = @view workspace.ldlt_blocks[1:count]
    fill!(dsub, zero(MF))
    fill!(blocks, UInt8(0))
    @inbounds for index in 1:count
        pivots[index] = index
    end
    return (
        dsub,
        pivots,
        blocks,
        workspace.ldlt_weighted,
    )
end

@inline _owned_ldlt_metadata(dsub, pivots, blocks, ::Nothing) =
    (dsub, pivots, blocks)
@inline _owned_ldlt_metadata(dsub, pivots, blocks, ::MFWorkspace) =
    (copy(dsub), copy(pivots), copy(blocks))

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
    dsub::AbstractVector{MF},
    pivots::AbstractVector{Int},
    blocks::AbstractVector{UInt8},
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
            dsub[k] = d21
            _, _, nonsingular = _ldlt_solve_2x2(
                d11, d21, d22, zero(MF), zero(MF),
            )
            nonsingular || return k

            @inbounds for row in (k + 2):n
                first = A[row, k]
                second = A[row, k + 1]
                solved_first, solved_second, nonsingular = _ldlt_solve_2x2(
                    d11, d21, d22, first, second,
                )
                nonsingular && isfinite(solved_first) &&
                    isfinite(solved_second) || return k
                A[row, k] = solved_first
                A[row, k + 1] = solved_second
            end

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
    dsub::AbstractVector{MF},
    blocks::AbstractVector{UInt8},
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
    dsub::AbstractVector{MF},
    blocks::AbstractVector{UInt8},
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
    dsub::AbstractVector{MF},
    pivots::AbstractVector{Int},
    blocks::AbstractVector{UInt8},
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
            A[k, k] = d11
            A[k + 1, k + 1] = d22
            dsub[k] = d21
            _, _, nonsingular = _ldlt_solve_2x2(
                d11, d21, d22, zero(MF), zero(MF),
            )
            nonsingular || return (k, panel_last)

            @inbounds for row in (k + 2):n
                first = _ldlt_panel_entry(
                    A, row, k, panel_first, k, dsub, blocks,
                )
                second = _ldlt_panel_entry(
                    A, row, k + 1, panel_first, k, dsub, blocks,
                )
                solved_first, solved_second, nonsingular = _ldlt_solve_2x2(
                    d11, d21, d22, first, second,
                )
                nonsingular && isfinite(solved_first) &&
                    isfinite(solved_second) || return (k, panel_last)
                A[row, k] = solved_first
                A[row, k + 1] = solved_second
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
    dsub::AbstractVector{MF},
    blocks::AbstractVector{UInt8},
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
    ldlt!(A; check=true, config=KernelConfig(), workspace=nothing)

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

With `workspace=MFWorkspace(T)`, metadata and blocked-panel scratch are reused
during factorization. The returned factor owns the metadata required by solve
and diagnostics.
"""
function ldlt!(
    A::AbstractMatrix{MF};
    check::Bool=true,
    config::KernelConfig=KernelConfig(),
    workspace::Union{Nothing,MFWorkspace{MF}}=nothing,
) where {MF<:MultiFloat}
    n, m = size(A)
    n == m || throw(DimensionMismatch("ldlt! requires a square matrix"))
    _check_supported(MF)
    Base.require_one_based_indexing(A)
    if !_lower_triangle_finite(A)
        check && throw(DomainError(A, "ldlt!: input matrix contains non-finite entries"))
        dsub, pivots, blocks, _ = _prepare_ldlt_metadata!(
            MF, n, 0, workspace,
        )
        owned_dsub, owned_pivots, owned_blocks =
            _owned_ldlt_metadata(dsub, pivots, blocks, workspace)
        return MFLDLT{
            MF,typeof(A),typeof(owned_dsub),typeof(owned_pivots),typeof(owned_blocks),
        }(
            A,
            owned_dsub,
            owned_pivots,
            owned_blocks,
            -1,
            zero(MF),
        )
    end
    _mirror_lower_to_upper!(A)

    original_maximum = _lower_maximum_abs(A)
    plan = ldlt_plan(MF, n, config)
    block_capacity = plan.strategy === :blocked ? plan.block_size : 0

    dsub, pivots, blocks, weighted_storage = _prepare_ldlt_metadata!(
        MF, n, block_capacity, workspace,
    )
    alpha = (one(MF) + sqrt(MF(17))) / MF(8)
    info = if plan.strategy === :blocked
        _factor_ldlt_blocked!(
            A,
            dsub,
            pivots,
            blocks,
            weighted_storage::Matrix{MF},
            alpha,
            plan,
            config,
        )
    else
        _factor_ldlt_unblocked!(A, dsub, pivots, blocks, alpha)
    end

    if !iszero(info) && check
        throw(LinearAlgebra.SingularException(info))
    end
    owned_dsub, owned_pivots, owned_blocks =
        _owned_ldlt_metadata(dsub, pivots, blocks, workspace)
    return MFLDLT{
        MF,typeof(A),typeof(owned_dsub),typeof(owned_pivots),typeof(owned_blocks),
    }(
        A, owned_dsub, owned_pivots, owned_blocks, info, original_maximum,
    )
end
