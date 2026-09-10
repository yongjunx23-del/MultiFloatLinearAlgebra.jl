# =============================================================================
# M02 — SIMD microkernels over packed panels  (CANDIDATE ONLY)
# =============================================================================
#
# The microkernel is the only place in this candidate where arithmetic happens.
# Its shape:
#
#     for k in 1:a_panel.layout.k_block
#         av = W-lane vector of A_panel[k, row0 : row0+W-1]
#         for c in 1:MC
#             acc[c] += av * B_panel[k, col0+c-1]
#
# `W` and `MC` are STATIC parameters (`Val{W}` / `Val{MC}`), so the accumulators
# live in registers as far as the backend puts them there. `MultiFloatVec{W}`
# is the vector vehicle MFLA already uses, so this file adds no new arithmetic
# and no second precision path — the same MultiFloats operations, grouped.
#
# TAILS ARE A DIFFERENT, EXPLICIT CODE PATH
#
# A shape whose extent is not a multiple of `W` (rows) or `MC` (columns) takes
# `microkernel_tail!`, a scalar MF loop over the ragged block. It is deliberately
# not a masked or partial vector: the card asks for the tail to be *visible*, and
# a separate symbol is visible in a way a mask is not. A plan reports the tail
# widths, so "this shape has no tail" and "the tail was not measured" are
# different facts.
#
# NO FAST MATH, NO REASSOCIATION
#
# There is no `@fastmath`, no `@simd ivdep`, and no `muladd`-based contraction in
# this file; the driver measures that claim as a text count over the file rather
# than asserting it in prose. Accumulation into `acc` is ascending in the
# reduction index, which is what makes the fixed-order mode bitwise reproducible
# across the whole variant grid.

"""
    MICROKERNEL_MAX_LANES

The widest static lane count this file instantiates. The driver measures 1, 2
and 4; the grid is bounded by the `MultiFloatVec` widths MFLA's own kernels use,
and a wider vector is not assumed to be faster (register pressure is measured,
not asserted).
"""
const MICROKERNEL_MAX_LANES = 4

"""
    MICROKERNEL_MAX_COLUMNS

The widest static micro-column block this file instantiates.
"""
const MICROKERNEL_MAX_COLUMNS = 4

"""
    microkernel_block!(acc, a_panel::PackedPanel, b_panel::PackedPanel,
                       row0, col0, ::Val{W}, ::Val{MC})

Accumulate the full `W × MC` micro block at `(row0, col0)` of the panel
accumulator `acc`, over the B panel's whole reduction extent:

    acc[row0+w-1, col0+c-1] += sum_k A[k, row0+w-1] * B[k, col0+c-1]

The caller guarantees the block is in range: this routine has no tail handling
on purpose, so the fast path carries no mask and no branch.
"""
@inline function microkernel_block!(
    acc::AbstractMatrix{MF},
    a_panel::PackedPanel,
    b_panel::PackedPanel,
    row0::Int,
    col0::Int,
    ::Val{W},
    ::Val{MC},
) where {N,MF<:MultiFloat{Float64,N},W,MC}
    V = MultiFloatVec{W,Float64,N}
    accumulators = ntuple(_ -> zero(V), Val(MC))
    k_block = a_panel.layout.k_block
    @inbounds for k in 1:k_block
        a_vector = V(ntuple(
            w -> _m02_panel_value(MF, a_panel, k, row0 + w - 1),
            Val(W),
        ))
        accumulators = ntuple(
            c -> accumulators[c] +
                 a_vector * V(_m02_panel_value(MF, b_panel, k, col0 + c - 1)),
            Val(MC),
        )
    end
    @inbounds for c in 1:MC
        accumulator = accumulators[c]
        for w in 1:W
            acc[row0 + w - 1, col0 + c - 1] += accumulator[w]
        end
    end
    return nothing
end

"""
    microkernel_tail!(acc, a_panel::PackedPanel, b_panel::PackedPanel,
                      row0, col0, rows::Int, columns::Int)

Scalar tail path for the (at most `W-1`) leftover rows and (at most `MC-1`)
leftover columns of a panel. `rows` and `columns` are runtime widths and are
recorded by the plan and by the schedule's tail counter; a zero-width tail never
reaches this routine, so a measured zero tail is distinguishable from an
unmeasured one.
"""
@inline function microkernel_tail!(
    acc::AbstractMatrix{MF},
    a_panel::PackedPanel,
    b_panel::PackedPanel,
    row0::Int,
    col0::Int,
    rows::Int,
    columns::Int,
) where {MF<:MultiFloat}
    rows >= 1 && columns >= 1 || throw(ArgumentError(
        "microkernel_tail! requires positive tail widths; got ($rows, $columns)",
    ))
    k_block = a_panel.layout.k_block
    @inbounds for c in 1:columns
        for w in 1:rows
            value = zero(MF)
            for k in 1:k_block
                value += _m02_panel_value(MF, a_panel, k, row0 + w - 1) *
                         _m02_panel_value(MF, b_panel, k, col0 + c - 1)
            end
            acc[row0 + w - 1, col0 + c - 1] += value
        end
    end
    return nothing
end

"""
    _microkernel_full_block!(acc, a_panel, b_panel, row0, col0,
                             ::Val{W}, micro_columns::Int)

Static `W` and a runtime micro-column count resolved by a four-way branch. The
branch is taken once per micro block, never per element, so the inner loop stays
specialized on both `W` and `MC`.
"""
@inline function _microkernel_full_block!(
    acc::AbstractMatrix{MF},
    a_panel::PackedPanel,
    b_panel::PackedPanel,
    row0::Int,
    col0::Int,
    ::Val{W},
    micro_columns::Int,
) where {MF<:MultiFloat,W}
    if micro_columns == 4
        microkernel_block!(acc, a_panel, b_panel, row0, col0, Val(W), Val(4))
    elseif micro_columns == 3
        microkernel_block!(acc, a_panel, b_panel, row0, col0, Val(W), Val(3))
    elseif micro_columns == 2
        microkernel_block!(acc, a_panel, b_panel, row0, col0, Val(W), Val(2))
    else
        microkernel_block!(acc, a_panel, b_panel, row0, col0, Val(W), Val(1))
    end
    return nothing
end

"""
    microkernel_dispatch!(acc, a_panel, b_panel, row0, col0, rows, columns,
                          lanes::Int, micro_columns::Int)

Run the widest full block that fits and fall back to
[`microkernel_tail!`](@ref) for the remainder. Returns the `(rows_handled,
columns_handled)` it covered, so a caller can account for coverage instead of
assuming it.

`lanes` and `micro_columns` are the *plan's* widths; `rows`/`columns` are what is
left in this panel. A block is full only when both remainders reach the plan's
widths, so a plan with `micro_columns = 2` gets its two-column static kernel
rather than being pushed onto the scalar path by a width-4 assumption.
"""
@inline function microkernel_dispatch!(
    acc::AbstractMatrix{MF},
    a_panel::PackedPanel,
    b_panel::PackedPanel,
    row0::Int,
    col0::Int,
    rows::Int,
    columns::Int,
    lanes::Int,
    micro_columns::Int,
) where {MF<:MultiFloat}
    if rows >= lanes && columns >= micro_columns
        if lanes == 4
            _microkernel_full_block!(acc, a_panel, b_panel, row0, col0, Val(4), micro_columns)
        elseif lanes == 3
            _microkernel_full_block!(acc, a_panel, b_panel, row0, col0, Val(3), micro_columns)
        elseif lanes == 2
            _microkernel_full_block!(acc, a_panel, b_panel, row0, col0, Val(2), micro_columns)
        else
            _microkernel_full_block!(acc, a_panel, b_panel, row0, col0, Val(1), micro_columns)
        end
        return (lanes, micro_columns)
    end
    # Ragged block: the scalar path covers it. Keeping this a single fallback
    # (rather than a 16-way Val cascade) is what holds the compiled size of this
    # file down; the measured cost of the ragged path is in the M02 log.
    microkernel_tail!(acc, a_panel, b_panel, row0, col0, rows, columns)
    return (rows, columns)
end

"""
    microkernel_variant_tag(lanes::Int, micro_columns::Int) -> (Val, Val)

Typed tag for one variant, used by the driver to force specialization of a
variant it is about to time or inspect.
"""
@inline microkernel_variant_tag(lanes::Int, micro_columns::Int) =
    (Val(lanes), Val(micro_columns))
