# =============================================================================
# M02 — packed-GEMM schedule with a measured time breakdown  (CANDIDATE ONLY)
# =============================================================================
#
# `packed_gemm_scheduled!` is the only entry point that runs the candidate. It
# is a *packed* GEBP schedule built from the two previous files:
#
#     pack A once into a full (k × m) buffer            -> packing time
#     for each column panel of B:
#         zero the panel accumulator
#         for each k block, in the plan's order:
#             pack that block's B panel  (kb × nc)      -> packing time
#             run the microkernel grid                  -> microkernel time
#         C[:, panel] = alpha*acc + beta*C[:, panel]    -> writeback time
#
# The three phases are timed separately into a `GemmTimingBreakdown`, because the
# card requires packing time to be *counted in the total*, not reported as if a
# packed operand were free. `packing_elements` counts the elements moved, which
# is exact and contention-free, so the packing claim does not rest on a wall
# clock number taken on a busy host.
#
# `beta == 0` NEVER READS C. That is a semantic claim with an oracle in the
# driver: C is prefilled with NaN and must come back finite. `alpha == 0` never
# reads A or B either, matching the reference semantics the driver's oracle
# encodes.
#
# ORDER MODES
#
#   k_block_order = nothing      -> ascending k blocks.
#   k_block_order = :descending  -> the measured-order mode. Same operands, same
#                                   accumulator, different association, so the
#                                   driver can show that its bitwise comparison
#                                   is capable of detecting a difference.
#   k_block_order = ::Vector{Int}-> an explicit order.
#
# WHAT THE FIXED ORDER ACTUALLY GUARANTEES, MEASURED
#
# With one accumulator per output element and ascending block order, the result
# is bitwise IDENTICAL across `lanes`, `micro_columns`, `panel_columns` and
# `layout` — measured on a 33×12×41 full-precision fixture over ten variants
# (`fixed_order_invariance_*` in `test/rebuild/M02.jl`). It is NOT bitwise
# identical across `k_block`: the microkernel forms a block partial sum and the
# panel accumulator adds each block once, so `k = sum of blocks` reassociates the
# sum. That is measured too (`kblock_4_vs_12_bitsame = false` on the same
# fixture, and `kblock_k_vs_scalar_reference_bitsame = true` for a single block),
# which is why `k_block` is part of the variant identity a certified plan pins,
# and why the claim in this comment is bounded to the fixture it was measured on:
# with a well-conditioned sum whose intermediate values are exactly
# representable, the reassociation was measured to return identical bits.
#
# Nothing here mutates a `KernelConfig`, a profile, or a global. The workspace is
# caller-owned, so a warm call allocates nothing (measured by the driver).

"""
    GemmTimingBreakdown

Mutable accumulator for the three schedule phases plus the counters that make
the breakdown auditable. `packed_elements` is a count (exact); the three
`*_seconds` fields are wall clock and are only meaningful together with the
contention flag the driver records next to them.
"""
mutable struct GemmTimingBreakdown
    packing_seconds::Float64
    microkernel_seconds::Float64
    writeback_seconds::Float64
    panels::Int
    k_blocks::Int
    microkernel_calls::Int
    tail_calls::Int
    packed_elements::Int
end

GemmTimingBreakdown() = GemmTimingBreakdown(0.0, 0.0, 0.0, 0, 0, 0, 0, 0)

total_seconds(breakdown::GemmTimingBreakdown) =
    breakdown.packing_seconds + breakdown.microkernel_seconds +
    breakdown.writeback_seconds

function Base.show(io::IO, breakdown::GemmTimingBreakdown)
    print(io, "GemmTimingBreakdown(packing=", breakdown.packing_seconds,
        "s microkernel=", breakdown.microkernel_seconds,
        "s writeback=", breakdown.writeback_seconds,
        "s panels=", breakdown.panels,
        " k_blocks=", breakdown.k_blocks,
        " calls=", breakdown.microkernel_calls,
        " tail_calls=", breakdown.tail_calls,
        " packed_elements=", breakdown.packed_elements, ")")
    return nothing
end

"""
    PackedGemmWorkspace{MF}

Caller-owned scratch for [`packed_gemm_scheduled!`](@ref): the packed A buffer,
the packed B panel buffer, and the panel accumulator. Buffers are typed by the
layout actually in use (`Matrix{MF}` for `:aos`, `Matrix{Float64}` for
`:soa_limbs`), and capacity does not shrink, so a warm repeated call on the same
shape reallocates nothing.
"""
mutable struct PackedGemmWorkspace{MF<:MultiFloat}
    a_panel::Union{Matrix{MF},Matrix{Float64},Nothing}
    b_panel::Union{Matrix{MF},Matrix{Float64},Nothing}
    acc::Matrix{MF}
    a_mode::Symbol
    b_mode::Symbol
    a_capacity::Tuple{Int,Int}
    b_capacity::Tuple{Int,Int}
    reallocations::Int
end

PackedGemmWorkspace{MF}() where {MF<:MultiFloat} = PackedGemmWorkspace{MF}(
    nothing, nothing, Matrix{MF}(undef, 0, 0), :none, :none, (0, 0), (0, 0), 0,
)

"""
    packed_gemm_workspace(MF, plan, m, k, n) -> PackedGemmWorkspace

Build a workspace sized for one `ShapePackingPlan` on an `(m, k, n)` shape. The A
buffer is full-height (`k × m`) because A is packed once per call; the B buffer is
panel-local (`k_block × panel_columns`), which is the whole point of the
shape-oriented layout.
"""
function packed_gemm_workspace(
    ::Type{MF},
    plan::ShapePackingPlan,
    m::Int,
    k::Int,
    n::Int,
) where {MF<:MultiFloat}
    workspace = PackedGemmWorkspace{MF}()
    ensure_packed_gemm_capacity!(workspace, plan, m, k, n)
    return workspace
end

"""
    ensure_packed_gemm_capacity!(workspace, plan, m, k, n) -> Bool

Grow the workspace if the plan's widths or the shape's extents exceed the current
buffers, and record the event in `workspace.reallocations`. Returns `true` when
something was reallocated and `false` when the warm buffers were reused — the
driver measures that distinction instead of assuming it.
"""
function ensure_packed_gemm_capacity!(
    workspace::PackedGemmWorkspace{MF},
    plan::ShapePackingPlan,
    m::Int,
    k::Int,
    n::Int,
) where {MF<:MultiFloat}
    limbs = _m02_limb_count(MF)
    k_block = min(plan.k_block, max(k, 1))
    panel_columns = min(plan.panel_columns, max(n, 1))
    reallocated = false

    a_layout = panel_pack_layout(MF, k, m; mode=plan.layout)
    b_layout = panel_pack_layout(MF, k_block, panel_columns; mode=plan.layout)
    a_needed = panel_buffer_size(a_layout)
    b_needed = panel_buffer_size(b_layout)

    if workspace.a_panel === nothing || workspace.a_mode !== plan.layout ||
       workspace.a_capacity[1] < a_needed[1] || workspace.a_capacity[2] < a_needed[2]
        workspace.a_panel = allocate_panel(MF, a_layout)
        workspace.a_mode = plan.layout
        workspace.a_capacity = a_needed
        reallocated = true
    end
    if workspace.b_panel === nothing || workspace.b_mode !== plan.layout ||
       workspace.b_capacity[1] < b_needed[1] || workspace.b_capacity[2] < b_needed[2]
        workspace.b_panel = allocate_panel(MF, b_layout)
        workspace.b_mode = plan.layout
        workspace.b_capacity = b_needed
        reallocated = true
    end
    if size(workspace.acc, 1) < m || size(workspace.acc, 2) < panel_columns
        workspace.acc = Matrix{MF}(undef, max(m, 1), max(panel_columns, 1))
        reallocated = true
    end
    workspace.reallocations += reallocated ? 1 : 0
    return reallocated
end

"""
    _m02_k_block_order(order, k_blocks::Int) -> Vector{Int}

Normalise the requested k-block order. `nothing` is ascending (the fixed-order
mode); `:descending` reverses it.
"""
function _m02_k_block_order(order, k_blocks::Int)
    if order === nothing
        return collect(1:k_blocks)
    elseif order === :descending
        return collect(k_blocks:-1:1)
    elseif order isa AbstractVector{<:Integer}
        length(order) == k_blocks || throw(ArgumentError(
            "explicit k-block order has $(length(order)) entries; expected $k_blocks",
        ))
        sort(collect(Int, order)) == collect(1:k_blocks) || throw(ArgumentError(
            "explicit k-block order must be a permutation of 1:$k_blocks",
        ))
        return collect(Int, order)
    else
        throw(ArgumentError("unsupported k_block_order $order"))
    end
end

"""
    packed_gemm_scheduled!(C, A, B, alpha, beta, plan, workspace;
                           timings=nothing, k_block_order=nothing)

Compute `C = alpha * A * B + beta * C` for the mathematical `m × k` operand `A`
and `k × n` operand `B`, using the candidate schedule.

`A` and `B` are the *mathematical* operands, so `Transpose`/`Adjoint` wrappers and
strided views are accepted directly: packing uses generic indexing, which is where
transpose and stride support is actually exercised.

Semantics, each with an oracle in the driver:

  * `beta == 0` — C is overwritten without being read.
  * `alpha == 0` — `C = beta * C`; A and B are not read.
  * otherwise   — `alpha * (A*B) + beta * C`.

`timings` may be a `GemmTimingBreakdown`; every phase is added to it. The
`(m, k, n)` of the plan must match the operands: a plan built for one shape is not
silently reused for another.
"""
function packed_gemm_scheduled!(
    C::AbstractMatrix{MF},
    A::AbstractMatrix,
    B::AbstractMatrix,
    alpha::MF,
    beta::MF,
    plan::ShapePackingPlan,
    workspace::PackedGemmWorkspace{MF};
    timings::Union{GemmTimingBreakdown,Nothing}=nothing,
    k_block_order=nothing,
) where {MF<:MultiFloat}
    m, k = size(A)
    k2, n = size(B)
    k == k2 || throw(DimensionMismatch("A is $(size(A)) and B is $(size(B))"))
    size(C) == (m, n) || throw(DimensionMismatch("C is $(size(C)); expected $((m, n))"))
    (plan.shape.m, plan.shape.k, plan.shape.n) == (m, k, n) || throw(ArgumentError(
        "plan was built for $((plan.shape.m, plan.shape.k, plan.shape.n)); " *
        "call got $((m, k, n))",
    ))

    ensure_packed_gemm_capacity!(workspace, plan, m, k, n)

    if iszero(alpha)
        # C = beta * C, and neither operand is read.
        if iszero(beta)
            fill!(C, zero(MF))
        elseif !isone(beta)
            @inbounds for index in eachindex(C)
                C[index] = beta * C[index]
            end
        end
        return C
    end

    a_full_layout = panel_pack_layout(MF, k, m; mode=plan.layout)
    a_panel = workspace.a_panel
    packing_start = time_ns()
    pack_a_panel!(a_panel, A, 1, 1, a_full_layout)
    if timings !== nothing
        timings.packing_seconds += (time_ns() - packing_start) / 1e9
        timings.packed_elements += k * m
    end

    k_block = min(plan.k_block, k)
    k_blocks = cld(k, k_block)
    block_order = _m02_k_block_order(k_block_order, k_blocks)
    panel_columns = min(plan.panel_columns, max(n, 1))

    for panel_start in 1:panel_columns:n
        panel_width = min(panel_columns, n - panel_start + 1)
        @inbounds for row in 1:m, column in 1:panel_width
            workspace.acc[row, column] = zero(MF)
        end
        microkernel_elapsed = 0.0
        packing_elapsed = 0.0
        calls = 0
        tails = 0
        for block in block_order
            k0 = (block - 1) * k_block + 1
            block_width = min(k_block, k - k0 + 1)
            # B panel for this k-block: `block_width` rows, `panel_width` columns.
            b_layout = panel_pack_layout(
                MF, block_width, panel_width;
                mode=plan.layout, k_stride=block_width,
            )
            packing_start = time_ns()
            pack_b_panel!(workspace.b_panel, B, k0, panel_start, b_layout)
            packing_elapsed += (time_ns() - packing_start) / 1e9

            # The A window starts at absolute reduction index `k0`.
            a_layout = panel_pack_layout(
                MF, block_width, m;
                mode=plan.layout, k_stride=k,
            )
            a_window = PackedPanel(a_panel, a_layout, k0 - 1)
            b_window = PackedPanel(workspace.b_panel, b_layout, 0)

            microkernel_start = time_ns()
            for column0 in 1:plan.micro_columns:panel_width
                column_width = min(plan.micro_columns, panel_width - column0 + 1)
                for row0 in 1:plan.lanes:m
                    row_width = min(plan.lanes, m - row0 + 1)
                    microkernel_dispatch!(
                        workspace.acc, a_window, b_window,
                        row0, column0, row_width, column_width,
                        plan.lanes, plan.micro_columns,
                    )
                    calls += 1
                    if row_width < plan.lanes || column_width < plan.micro_columns
                        tails += 1
                    end
                end
            end
            microkernel_elapsed += (time_ns() - microkernel_start) / 1e9
        end

        writeback_start = time_ns()
        @inbounds for column in 1:panel_width
            global_column = panel_start + column - 1
            for row in 1:m
                accumulator = workspace.acc[row, column]
                if iszero(beta)
                    C[row, global_column] = alpha * accumulator
                elseif isone(beta)
                    C[row, global_column] = alpha * accumulator + C[row, global_column]
                else
                    C[row, global_column] =
                        alpha * accumulator + beta * C[row, global_column]
                end
            end
        end
        writeback_elapsed = (time_ns() - writeback_start) / 1e9

        if timings !== nothing
            timings.packing_seconds += packing_elapsed
            timings.packed_elements += k * panel_width
            timings.microkernel_seconds += microkernel_elapsed
            timings.writeback_seconds += writeback_elapsed
            timings.microkernel_calls += calls
            timings.tail_calls += tails
            timings.panels += 1
            timings.k_blocks += length(block_order)
        end
    end
    return C
end

"""
    packed_gemm_reference!(C, A, B, alpha, beta)

The candidate schedule's own scalar reference: one accumulator per output
element, ascending reduction, no packing and no SIMD. It exists so the driver can
separate "the microkernel is wrong" from "the schedule wired the microkernel
wrong"; it is not on any production path.

It obeys the same `alpha`/`beta` semantics as `packed_gemm_scheduled!`, including
not reading C when `beta == 0`.
"""
function packed_gemm_reference!(
    C::AbstractMatrix{MF},
    A::AbstractMatrix,
    B::AbstractMatrix,
    alpha::MF,
    beta::MF,
) where {MF<:MultiFloat}
    m, k = size(A)
    size(B, 1) == k || throw(DimensionMismatch("A*B does not conform to C"))
    n = size(C, 2)
    size(B, 2) == n || throw(DimensionMismatch("A*B does not conform to C"))
    if iszero(alpha)
        if iszero(beta)
            fill!(C, zero(MF))
        elseif !isone(beta)
            @inbounds for index in eachindex(C)
                C[index] = beta * C[index]
            end
        end
        return C
    end
    @inbounds for column in 1:n
        for row in 1:m
            accumulator = zero(MF)
            for reduction in 1:k
                accumulator += A[row, reduction] * B[reduction, column]
            end
            if iszero(beta)
                C[row, column] = alpha * accumulator
            elseif isone(beta)
                C[row, column] = alpha * accumulator + C[row, column]
            else
                C[row, column] = alpha * accumulator + beta * C[row, column]
            end
        end
    end
    return C
end
