function _ldlt_block_trailing_update!(
    A::AbstractMatrix{MF},
    panel_first::Int,
    panel_last::Int,
    weighted_storage::Matrix{MF},
    dsub::Vector{MF},
    blocks::Vector{UInt8},
    config::KernelConfig,
) where {MF<:MultiFloat}
    n = size(A, 1)
    panel_last < n || return nothing
    trailing_count = n - panel_last
    panel_width = panel_last - panel_first + 1
    weighted = @view weighted_storage[1:trailing_count, 1:panel_width]
    _build_ldlt_weighted_panel!(
        weighted, A, panel_first, panel_last, dsub, blocks,
    )
    L21 = @view A[(panel_last + 1):n, panel_first:panel_last]
    A22 = @view A[(panel_last + 1):n, (panel_last + 1):n]
    gemmt!(
        A22,
        L21,
        weighted,
        -one(MF),
        one(MF);
        config=config,
    )
    # The factorization stores and searches the active Schur complement in a
    # mirrored dense representation so symmetric pivot swaps remain simple.
    # Only the lower triangle performs arithmetic; this copy restores the
    # read-only upper mirror without repeating the matrix product.
    _mirror_lower_to_upper!(A22)
    return nothing
end

function _factor_ldlt_blocked!(
    A::AbstractMatrix{MF},
    dsub::Vector{MF},
    pivots::Vector{Int},
    blocks::Vector{UInt8},
    alpha::MF,
    plan::LDLTPlan,
    config::KernelConfig,
) where {MF<:MultiFloat}
    n = size(A, 1)
    weighted_storage = Matrix{MF}(undef, n, plan.block_size + 1)

    panel_first = 1
    while panel_first <= n
        requested_last = min(panel_first + plan.block_size - 1, n)
        info, panel_last = _factor_ldlt_panel!(
            A,
            panel_first,
            requested_last,
            dsub,
            pivots,
            blocks,
            alpha,
        )
        !iszero(info) && return info
        _ldlt_block_trailing_update!(
            A,
            panel_first,
            panel_last,
            weighted_storage,
            dsub,
            blocks,
            config,
        )
        panel_first = panel_last + 1
    end
    return 0
end
