# =============================================================================
# M02 — panel-local, shape-oriented packing  (CANDIDATE ONLY)
# =============================================================================
#
# Packing here is *storage reorganisation only*. No arithmetic is performed on
# a limb, so packing is bit-exact by construction: `unpack_panel! ∘ pack_*!`
# returns the original `_limbs` tuple unchanged. The M02 driver asserts that
# bitwise, and separately asserts that the assertion can fail (it corrupts one
# limb and requires the comparison to report a difference), because an oracle
# that cannot fail is not evidence.
#
# TWO LAYOUTS, AND THE DEFAULT IS THE CONSERVATIVE ONE
#
#   :aos        — the panel is `Matrix{MF}`. Limbs of one element stay together.
#                 This is the default: nothing is converted.
#   :soa_limbs  — the panel is `Matrix{Float64}` with one *limb plane* per limb,
#                 stacked along rows: limb `l` of panel reduction index `k` lives
#                 in row `(l-1)*k_stride + k`. Within a plane, consecutive panel
#                 dimensions are contiguous, which is what makes a lane-wide
#                 Float64 load possible. This layout is NEVER selected
#                 implicitly: `shape_packing_plan(...; layout=:soa_limbs)` or an
#                 explicit `panel_pack_layout(...; mode=:soa_limbs)` is required.
#                 It is panel-local — there is no whole-matrix SoA conversion
#                 anywhere in this file, and the buffer is sized by the panel,
#                 not by the operand.
#
# PANEL ORIENTATION AND `k_stride`
#
# Both A and B panels use the same logical orientation, `(k_block, dim)`, where
# `dim` is the panel's row count for A and its column count for B. A fixed
# reduction index then reads a *contiguous* run of panel elements, which is the
# access pattern the SIMD microkernel in `gemm_microkernels.jl` is built on.
#
# A is packed once for the whole call, so the A buffer's limb planes are indexed
# by the full reduction `k`; a k-block of A is then a *window* into that buffer.
# `k_stride` is the plane stride (full `k` for the A buffer, the block width for
# a B panel) and `PackedPanel.k_offset` is the window's first absolute reduction
# index. Keeping those two separate is what lets one A pack serve every k-block
# without re-packing, and it is the defect the first draft of this file had:
# with a single `k_block` field the microkernel read the wrong rows for every
# block after the first.

"""
    PanelPackLayout(k_block, dim, limbs, mode; k_stride=k_block)

Describe one packed `(k_block, dim)` panel. `k_block` is the reduction extent of
the *window* this layout indexes, `dim` the panel's own extent, `limbs` the
MultiFloat limb count, `mode` one of [`PACKING_LAYOUTS`](@ref), and `k_stride`
the number of rows one limb plane occupies in the backing buffer. The layout
carries no data.
"""
struct PanelPackLayout
    k_block::Int
    dim::Int
    limbs::Int
    mode::Symbol
    k_stride::Int
    function PanelPackLayout(
        k_block::Int,
        dim::Int,
        limbs::Int,
        mode::Symbol;
        k_stride::Int=k_block,
    )
        k_block >= 0 || throw(ArgumentError("k_block must be >= 0"))
        dim >= 0 || throw(ArgumentError("dim must be >= 0"))
        limbs >= 1 || throw(ArgumentError("limbs must be >= 1"))
        k_stride >= k_block || throw(ArgumentError(
            "k_stride ($k_stride) must be at least k_block ($k_block)",
        ))
        mode in PACKING_LAYOUTS ||
            throw(ArgumentError("packing mode must be one of $(PACKING_LAYOUTS)"))
        return new(k_block, dim, limbs, mode, k_stride)
    end
end

"""
    panel_pack_layout(MF, k_block, dim; mode=:aos, k_stride=k_block)

Build a panel layout for `MF`. The default mode is `:aos`; `:soa_limbs` must be
named.
"""
function panel_pack_layout(
    ::Type{MF},
    k_block::Int,
    dim::Int;
    mode::Symbol=:aos,
    k_stride::Int=k_block,
) where {MF<:MultiFloat}
    return PanelPackLayout(k_block, dim, _m02_limb_count(MF), mode; k_stride=k_stride)
end

"""
    PackedPanel(data, layout, k_offset)

A window onto a packed buffer. `k_offset` is the absolute reduction index of the
window's first row, so `k_offset = 0` means the buffer starts at reduction 1.
The microkernel takes `PackedPanel`s rather than raw buffers, which is what
keeps the A-window/B-block distinction explicit instead of implicit in an index.
"""
struct PackedPanel{P}
    data::P
    layout::PanelPackLayout
    k_offset::Int
end

PackedPanel(data, layout::PanelPackLayout) = PackedPanel(data, layout, 0)

"""
    panel_buffer_size(layout) -> (rows, cols)

The backing-buffer shape for a layout. `:aos` is `(k_stride, dim)`; `:soa_limbs`
is `(limbs * k_stride, dim)`.
"""
@inline function panel_buffer_size(layout::PanelPackLayout)
    return layout.mode === :aos ?
           (layout.k_stride, layout.dim) :
           (layout.limbs * layout.k_stride, layout.dim)
end

"""
    allocate_panel(MF, layout)

Allocate a panel buffer of the layout's shape: `Matrix{MF}` for `:aos`,
`Matrix{Float64}` for `:soa_limbs`.
"""
function allocate_panel(::Type{MF}, layout::PanelPackLayout) where {MF<:MultiFloat}
    rows, cols = panel_buffer_size(layout)
    return layout.mode === :aos ? Matrix{MF}(undef, rows, cols) :
           Matrix{Float64}(undef, rows, cols)
end

"""
    packing_elements(layout) -> Int

The number of MultiFloat elements a full pack of this window moves. Returned as
a count so a schedule can account for packing work without timing it.
"""
@inline packing_elements(layout::PanelPackLayout) = layout.k_block * layout.dim

"""
    _m02_store_panel!(panel, layout, k::Int, d::Int, value::MF)

Store `value` at panel reduction index `k`, panel index `d`. For `:soa_limbs`
the limbs are written to their planes; no arithmetic touches them.
"""
@inline function _m02_store_panel!(panel, layout::PanelPackLayout, k::Int, d::Int, value::MF) where {MF}
    if layout.mode === :aos
        @inbounds panel[k, d] = value
    else
        stride = layout.k_stride
        limbs = layout.limbs
        @inbounds for limb in 1:limbs
            panel[(limb - 1) * stride + k, d] = value._limbs[limb]
        end
    end
    return nothing
end

"""
    _m02_panel_value(::Type{MF}, panel, layout, k, d) -> MF

Read absolute reduction index `k` (1-based against the buffer) at panel index
`d`, with the limb count statically known from `MF`. For `:soa_limbs` the value
is reassembled with `MultiFloat{T,N}(::NTuple{N,T})`, an exact renormalising
constructor — the round trip is therefore bit-exact rather than merely close.
"""
@inline function _m02_panel_value(
    ::Type{MF},
    panel,
    layout::PanelPackLayout,
    k::Int,
    d::Int,
) where {N,MF<:MultiFloat{Float64,N}}
    if layout.mode === :aos
        return @inbounds panel[k, d]
    else
        stride = layout.k_stride
        limbs = ntuple(limb -> @inbounds(panel[(limb - 1) * stride + k, d]), Val(N))
        return MultiFloat{Float64,N}(limbs)
    end
end

"""
    _m02_panel_value(::Type{MF}, panel::PackedPanel, k, d) -> MF

Read window-relative reduction index `k` (1-based) from a `PackedPanel`.
"""
@inline function _m02_panel_value(::Type{MF}, panel::PackedPanel, k::Int, d::Int) where {MF}
    return _m02_panel_value(MF, panel.data, panel.layout, panel.k_offset + k, d)
end

"""
    panel_element(panel, layout, k, d) -> MF

Dynamic convenience accessor, used by the driver. The microkernel path uses
[`_m02_panel_value`](@ref) instead so the limb count stays a compile-time
constant in the hot loop.
"""
@inline function panel_element(panel, layout::PanelPackLayout, k::Int, d::Int)
    if layout.mode === :aos
        return @inbounds panel[k, d]
    else
        stride = layout.k_stride
        limbs = ntuple(
            limb -> @inbounds(panel[(limb - 1) * stride + k, d]),
            layout.limbs,
        )
        return MultiFloat{eltype(panel),layout.limbs}(limbs)
    end
end

@inline panel_element(panel::PackedPanel, k::Int, d::Int) =
    panel_element(panel.data, panel.layout, panel.k_offset + k, d)

"""
    _m02_pack_panel!(panel, layout, source, row_base, row_dim, row_k,
                     col_base, col_dim, col_k)

Shared packer. Each logical panel axis maps onto a source axis with an explicit
base and stride: the panel dimension `d` moves by `(row_dim, col_dim)` and the
reduction `k` by `(row_k, col_k)`. A and B are the same body under two sign
patterns — A's reduction walks the source columns, B's walks the source rows —
so transposed and strided sources are exercised through one code path instead of
two.
"""
@inline function _m02_pack_panel!(
    panel,
    layout::PanelPackLayout,
    source::AbstractMatrix,
    row_base::Int,
    row_dim::Int,
    row_k::Int,
    col_base::Int,
    col_dim::Int,
    col_k::Int,
)
    k_block = layout.k_block
    dim = layout.dim
    @inbounds for d in 1:dim
        first_row = row_base + (d - 1) * row_dim
        first_col = col_base + (d - 1) * col_dim
        for k in 1:k_block
            value = source[first_row + (k - 1) * row_k, first_col + (k - 1) * col_k]
            _m02_store_panel!(panel, layout, k, d, value)
        end
    end
    return panel
end

"""
    pack_a_panel!(panel, A, row0, k0, layout)

Pack the `layout.dim × layout.k_block` block of the mathematical operand `A`
starting at `(row0, k0)`: `panel[k, i] = A[row0 + i - 1, k0 + k - 1]`.

`A` may be any `AbstractMatrix`, including a `Transpose`/`Adjoint` wrapper or a
strided `view`; indexing is generic on purpose so transposes and non-unit
strides are exercised rather than assumed away.
"""
function pack_a_panel!(panel, A::AbstractMatrix, row0::Int, k0::Int, layout::PanelPackLayout)
    # rows walk with the panel dimension, columns walk with the reduction
    return _m02_pack_panel!(panel, layout, A, row0, 1, 0, k0, 0, 1)
end

"""
    pack_b_panel!(panel, B, k0, col0, layout)

Pack the `layout.k_block × layout.dim` block of the mathematical operand `B`
starting at `(k0, col0)`: `panel[k, j] = B[k0 + k - 1, col0 + j - 1]`.
"""
function pack_b_panel!(panel, B::AbstractMatrix, k0::Int, col0::Int, layout::PanelPackLayout)
    # rows walk with the reduction, columns walk with the panel dimension
    return _m02_pack_panel!(panel, layout, B, k0, 0, 1, col0, 1, 0)
end

"""
    unpack_panel!(destination::AbstractMatrix{MF}, panel, layout)

Write a packed window back into a `Matrix{MF}` of size `(k_block, dim)`. Bit-exact
inverse of the packers for both layouts.
"""
function unpack_panel!(destination::AbstractMatrix{MF}, panel, layout::PanelPackLayout) where {MF}
    size(destination) == (layout.k_block, layout.dim) || throw(DimensionMismatch(
        "destination is $(size(destination)); layout wants " *
        "$((layout.k_block, layout.dim))",
    ))
    for d in 1:layout.dim, k in 1:layout.k_block
        @inbounds destination[k, d] = _m02_panel_value(MF, panel, layout, k, d)
    end
    return destination
end

"""
    panel_limb(panel::AbstractMatrix{Float64}, layout, limb) -> view

The contiguous storage of one limb plane of an `:soa_limbs` panel. There is no
equivalent for `:aos` — its limbs interleave — which is exactly why an
elementwise layout comparison in the driver goes through `panel_element` rather
than through this accessor.
"""
function panel_limb(panel::AbstractMatrix{Float64}, layout::PanelPackLayout, limb::Int)
    layout.mode === :soa_limbs || throw(ArgumentError(
        "a Float64 panel is only meaningful for :soa_limbs",
    ))
    1 <= limb <= layout.limbs || throw(ArgumentError("limb out of range"))
    base = (limb - 1) * layout.k_stride
    return view(panel, base + 1:base + layout.k_block, 1:layout.dim)
end
