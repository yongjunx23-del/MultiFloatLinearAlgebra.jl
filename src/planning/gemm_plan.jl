# =============================================================================
# M02 — shape-oriented GEMM planning  (CANDIDATE ONLY — NEVER A DEFAULT)
# =============================================================================
#
# This file is the planning half of the M02 candidate. It classifies a dense
# GEMM shape by *trace* (near-square / tall / wide / panel, plus the limb
# class), and derives an inspectable packing+microkernel plan from that class.
#
# WHAT THIS FILE DELIBERATELY DOES NOT DO
#
#   * It does not touch, replace, wrap, or shadow `gemm_plan` (declared in
#     `src/kernels/gemm.jl`) nor `_default_gemm_panel_columns`,
#     `_default_gemm_micro_columns` or `_near_square_shape`. The production
#     `:auto` route keeps its existing conservative policy: calibration only
#     measures square GEMM, so `:auto` stays on the direct route for strongly
#     tall or wide shapes. `shape_packing_plan` is a *separate* entry point.
#   * It switches no ISA, sets no default, and enables no route. Every
#     `ShapePackingPlan` carries `default_path = false`, which is asserted by
#     the M02 driver: a plan produced here is never consulted unless a caller
#     asks for it by name.
#   * It exports nothing. Wiring (include + export) is the integration role's
#     authority; the M02 driver includes this file into a sandbox module.
#
# ORDER MODES (M02 card step 3: 固定内核顺序模式与高性能认证模式分开)
#
#   :fixed_order    — reduction visited in ascending k with one accumulator per
#                     output element. The association is then independent of
#                     lanes, micro columns, panel columns and packing layout, so
#                     the result is bitwise reproducible across those four
#                     parameters — MEASURED, on a full-precision 33×12×41 fixture
#                     over ten variants (M02 driver, "fixed order"). It is *not*
#                     independent of `k_block`: the schedule accumulates each
#                     block's partial sum separately, which reassociates the
#                     reduction. That is measured as well, so `k_block` is part of
#                     the order identity a certified variant pins. This is the
#                     mode a correctness claim is made in, and it is the default.
#   :measured_order — a caller-supplied k-block order (a machine-specific,
#                     measured choice). It is only reachable by naming it, and
#                     `select_certified_variant` refuses to return a measured
#                     variant that carries no explicit calibration evidence.
#
# CALIBRATION IS EXPORTED EXPLICITLY
#
#   `shape_calibration_table` builds the record vector and
#   `format_shape_calibration` renders it as text, so a calibration run's raw
#   numbers leave the process in the log instead of living only in a profile
#   object. No calibration data is written to disk by this file.

"""Shape classes produced by [`classify_gemm_shape`](@ref)."""
const GEMM_SHAPE_CLASSES = (:degenerate, :vector, :panel, :near_square, :tall, :wide)

"""Panel-local packing layouts understood by `src/kernels/packing.jl`."""
const PACKING_LAYOUTS = (:aos, :soa_limbs)

"""Reduction-order modes understood by `src/kernels/gemm_schedule.jl`."""
const KERNEL_ORDER_MODES = (:fixed_order, :measured_order)

"""
    GemmShapeClass

The trace-derived classification of one `(m, k, n)` dense GEMM shape for one
MultiFloat limb count. `aspect_*` are plain `Float64` ratios of the dimensions;
`limb_class` is `:x1`..`:x4` for the supported widths and `:unsupported`
otherwise. The classification itself performs no measurement and allocates
nothing.
"""
struct GemmShapeClass
    shape::Symbol
    m::Int
    k::Int
    n::Int
    aspect_mk::Float64
    aspect_kn::Float64
    aspect_mn::Float64
    limbs::Int
    limb_class::Symbol
end

"""
    ShapePackingPlan

An inspectable, caller-owned packing + microkernel plan. `default_path` is
always `false`: this structure describes a candidate route and is never
consulted by `gemm_plan`.

`tail_*` record how many rows / reduction indices / columns fall outside the
full microkernel blocks and therefore take the scalar tail path. A measured
zero tail is a fact (the shape divides exactly); an unmeasured tail is not
representable here, which is why the fields are `Int` and are always filled.

`packing_elements` is the exact number of MultiFloat elements the schedule will
copy for this plan: `k*(m + n)` for the "pack A once, pack B per column panel"
schedule in `src/kernels/gemm_schedule.jl`. It is a count, not a time.
"""
struct ShapePackingPlan
    shape::GemmShapeClass
    panel_columns::Int
    k_block::Int
    lanes::Int
    micro_columns::Int
    layout::Symbol
    order_mode::Symbol
    tail_rows::Int
    tail_reduction::Int
    tail_columns::Int
    packing_elements::Int
    microkernel_calls::Int
    reason::Symbol
    evidence::Symbol
    default_path::Bool
end

@inline function _m02_limb_class(::Type{MF}) where {N,MF<:MultiFloat{Float64,N}}
    return N == 1 ? :x1 : N == 2 ? :x2 : N == 3 ? :x3 : N == 4 ? :x4 : :unsupported
end

@inline _m02_aspect(a::Int, b::Int) = b == 0 ? Inf : Float64(a) / Float64(b)

"""
    classify_gemm_shape(MF, m, k, n; ratio=4, panel_width=8)

Classify a dense GEMM by trace. The grammar is deliberately stated as
comparisons, not as a fitted table, so that a reader can re-derive any row:

  * `:degenerate`  — any dimension is non-positive. No route is offered.
  * `:vector`      — `n == 1 && k > 1`: the shape is a matrix-vector product;
                     `gemv!` owns it and this candidate stands aside.
  * `:panel`       — `n <= panel_width`: a narrow column panel, the shape a
                     blocked factorization generates. `k == 1` is also a panel
                     (a rank-1 update), as is any shape whose shortest side is
                     the reduction `k`.
  * `:near_square` — `maximum(dims) <= ratio * minimum(dims)`.
  * `:tall`        — `m` is the unique maximum and `n` the minimum.
  * `:wide`        — `n` is the unique maximum and `m` the minimum.

`ratio` is exposed rather than hardcoded so that a caller can test the
sensitivity of the classification instead of trusting one threshold.
"""
function classify_gemm_shape(
    ::Type{MF},
    m::Int,
    k::Int,
    n::Int;
    ratio::Int=4,
    panel_width::Int=8,
) where {MF<:MultiFloat}
    ratio >= 1 || throw(ArgumentError("ratio must be >= 1"))
    panel_width >= 1 || throw(ArgumentError("panel_width must be >= 1"))
    limbs = _m02_limb_count(MF)
    shape = if m <= 0 || k <= 0 || n <= 0
        :degenerate
    elseif n <= panel_width || k == 1
        n == 1 && k > 1 ? :vector : :panel
    else
        lo = min(m, k, n)
        hi = max(m, k, n)
        if hi <= ratio * lo
            :near_square
        elseif m == hi && n == lo
            :tall
        elseif n == hi && m == lo
            :wide
        else
            :panel
        end
    end
    return GemmShapeClass(
        shape,
        m,
        k,
        n,
        _m02_aspect(m, k),
        _m02_aspect(k, n),
        _m02_aspect(m, n),
        limbs,
        _m02_limb_class(MF),
    )
end

@inline function _m02_limb_count(::Type{MF}) where {N,MF<:MultiFloat{Float64,N}}
    return N
end

"""
    _m02_default_lanes(MF)

The planner's default SIMD lane count. It mirrors MFLA's existing
`_default_gemm_micro_columns` policy (4 for the narrow limb counts, 2 for x4)
rather than inventing a new threshold: the reason field of a plan built from it
is `:policy_default`, and the M02 driver measures the 1/2/4 lane grid so that a
measured replacement has evidence behind it before anyone changes this.
"""
@inline function _m02_default_lanes(::Type{MF}) where {N,MF<:MultiFloat{Float64,N}}
    return N <= 3 ? 4 : 2
end

"""
    _m02_default_micro_columns(MF)

Mirrors `_default_gemm_micro_columns` so the candidate starts from the
production policy instead of a new claim.
"""
@inline function _m02_default_micro_columns(::Type{MF}) where {N,MF<:MultiFloat{Float64,N}}
    return N <= 3 ? 4 : 2
end

"""
    supported_packing_variants(MF; lane_grid=(1,2,4), micro_grid=(1,2,4),
                               k_block_grid=(0,4,16))

The variant grid the M02 driver measures. `k_block = 0` means "unblocked — the
whole reduction in one k-block". Returned as a tuple of named tuples, with no
measurement attached: it is the *question*, not the answer.
"""
function supported_packing_variants(
    ::Type{MF};
    lane_grid=(1, 2, 4),
    micro_grid=(1, 2, 4),
    k_block_grid=(0, 4, 16),
) where {MF<:MultiFloat}
    variants = NamedTuple{(:lanes, :micro_columns, :k_block),Tuple{Int,Int,Int}}[]
    for lanes in lane_grid, micro_columns in micro_grid, k_block in k_block_grid
        push!(variants, (lanes=lanes, micro_columns=micro_columns, k_block=k_block))
    end
    return Tuple(variants)
end

"""
    shape_packing_plan(MF, m, k, n; ...) -> ShapePackingPlan

Derive the candidate shape-oriented plan. Every policy input is a keyword with
an explicit default, and the returned plan names which policy produced each
number through `evidence` and `reason`:

  * `evidence = :policy_default`  — the value came from the defaults above.
  * `evidence = :caller_forced`   — the caller named `lanes`, `micro_columns`,
                                    `k_block`, `panel_columns` or `layout`.
  * `evidence = :calibrated`      — produced by `select_certified_variant` from
                                    an explicit calibration table entry.

`lanes`, `micro_columns`, `k_block` and `panel_columns` of `0` mean "use the
default". `layout` must be in [`PACKING_LAYOUTS`](@ref) and `order_mode` in
[`KERNEL_ORDER_MODES`](@ref); `:soa_limbs` must be asked for by name, so no
whole-matrix or even whole-panel SoA conversion happens implicitly.
"""
function shape_packing_plan(
    ::Type{MF},
    m::Int,
    k::Int,
    n::Int;
    config::KernelConfig=KernelConfig(),
    ratio::Int=4,
    panel_width::Int=8,
    lanes::Int=0,
    micro_columns::Int=0,
    k_block::Int=0,
    panel_columns::Int=0,
    layout::Symbol=:aos,
    order_mode::Symbol=:fixed_order,
) where {MF<:MultiFloat}
    layout in PACKING_LAYOUTS ||
        throw(ArgumentError("layout must be one of $(PACKING_LAYOUTS)"))
    order_mode in KERNEL_ORDER_MODES ||
        throw(ArgumentError("order_mode must be one of $(KERNEL_ORDER_MODES)"))
    shape = classify_gemm_shape(MF, m, k, n; ratio=ratio, panel_width=panel_width)
    shape.shape === :degenerate && throw(ArgumentError(
        "shape_packing_plan requires positive (m, k, n); got ($m, $k, $n)",
    ))
    _check_supported(MF)

    forced = lanes != 0 || micro_columns != 0 || k_block != 0 ||
             panel_columns != 0 || layout !== :aos || order_mode !== :fixed_order

    resolved_lanes = lanes == 0 ? _m02_default_lanes(MF) : lanes
    resolved_lanes in (1, 2, 3, 4) ||
        throw(ArgumentError("lanes must be 1, 2, 3 or 4"))
    resolved_micro = micro_columns == 0 ? _m02_default_micro_columns(MF) : micro_columns
    resolved_micro in (1, 2, 3, 4) ||
        throw(ArgumentError("micro_columns must be 1, 2, 3 or 4"))
    resolved_panel = if panel_columns == 0
        config.gemm_panel_columns > 0 ? config.gemm_panel_columns :
        _default_gemm_panel_columns(MF, config.thread_count)
    else
        panel_columns
    end
    resolved_panel = max(resolved_panel, 1)
    resolved_k_block = k_block == 0 ? k : k_block
    resolved_k_block = max(1, min(resolved_k_block, k))

    # A plan must be runnable: the microkernel block cannot be wider than the
    # panel it reads.
    resolved_micro = min(resolved_micro, resolved_panel)

    tail_rows = mod(m, resolved_lanes)
    tail_reduction = mod(k, resolved_k_block)
    # The innermost schedule granularity is the micro-column block, so that is
    # the width that decides the column tail.
    tail_columns = mod(n, resolved_micro)

    packing_elements = k * (m + n)
    microkernel_calls = cld(n, resolved_panel) *
                        cld(resolved_panel, resolved_micro) *
                        cld(m, resolved_lanes) *
                        cld(k, resolved_k_block)

    return ShapePackingPlan(
        shape,
        resolved_panel,
        resolved_k_block,
        resolved_lanes,
        resolved_micro,
        layout,
        order_mode,
        tail_rows,
        tail_reduction,
        tail_columns,
        packing_elements,
        microkernel_calls,
        forced ? :caller_forced_parameters : :shape_class_policy,
        forced ? :caller_forced : :policy_default,
        false,
    )
end

"""
    shape_calibration_table(entries) -> Vector{NamedTuple}

Collect calibration records into an explicit, renderable table. Each `entry` is
a named tuple with at least `(:m, :k, :n, :limbs, :lanes, :micro_columns,
:k_block, :layout, :seconds, :packing_seconds, :contended, :bitwise_token)`.
The table is a pure data structure: building it measures nothing and writes
nothing, so a caller cannot mistake analysis for evidence.
"""
function shape_calibration_table(entries)
    table = NamedTuple[]
    for entry in entries
        haskey(entry, :m) && haskey(entry, :seconds) || throw(ArgumentError(
            "calibration entries need at least :m and :seconds",
        ))
        push!(table, entry)
    end
    return table
end

"""
    format_shape_calibration(io::IO, table)

Render a calibration table as one `CALIB` line per record. This is the explicit
export path required by the M02 card: raw samples leave the process as text
that a log can carry, with no threshold applied and nothing dropped.
"""
function format_shape_calibration(io::IO, table)
    for (index, entry) in enumerate(table)
        names = keys(entry)
        print(io, "CALIB[", index, "]")
        for name in names
            print(io, " ", name, "=", entry[name])
        end
        println(io)
    end
    return nothing
end

format_shape_calibration(table) = sprint(format_shape_calibration, table)

"""
    select_certified_variant(MF, m, k, n, table; config=KernelConfig(),
                             criterion=:min_microkernel_seconds)

Return the plan for the fastest *non-contended* calibration entry whose
`bitwise_token` is stable, or a `:policy_default` plan with reason
`:no_calibration_entry` when the table has no such entry.

This is the "certified performance" half of the card's separation: it is only
reachable by passing a table, and it refuses an entry that recorded contention
or an unstable token instead of promoting a contended measurement to a default.
It never mutates `config` and never writes a profile anywhere.
"""
function select_certified_variant(
    ::Type{MF},
    m::Int,
    k::Int,
    n::Int,
    table;
    config::KernelConfig=KernelConfig(),
    criterion::Symbol=:min_microkernel_seconds,
) where {MF<:MultiFloat}
    criterion === :min_microkernel_seconds || throw(ArgumentError(
        "only :min_microkernel_seconds is implemented; got $criterion",
    ))
    eligible = NamedTuple[]
    for entry in table
        entry.m == m && entry.k == k && entry.n == n || continue
        get(entry, :limbs, 0) == _m02_limb_count(MF) || continue
        get(entry, :contended, true) && continue
        get(entry, :bitwise_token, nothing) === nothing && continue
        haskey(entry, :seconds) || continue
        push!(eligible, entry)
    end
    isempty(eligible) && return shape_packing_plan(
        MF,
        m,
        k,
        n;
        config=config,
    ), :no_calibration_entry
    best = eligible[1]
    for entry in eligible
        entry.seconds < best.seconds && (best = entry)
    end
    plan = shape_packing_plan(
        MF,
        m,
        k,
        n;
        config=config,
        lanes=best.lanes,
        micro_columns=best.micro_columns,
        k_block=best.k_block,
        panel_columns=get(best, :panel_columns, 0),
        layout=get(best, :layout, :aos),
        order_mode=get(best, :order_mode, :fixed_order),
    )
    # `:calibrated` evidence, and `default_path` stays false: a certified
    # variant is still a candidate that a caller must name.
    return ShapePackingPlan(
        plan.shape,
        plan.panel_columns,
        plan.k_block,
        plan.lanes,
        plan.micro_columns,
        plan.layout,
        plan.order_mode,
        plan.tail_rows,
        plan.tail_reduction,
        plan.tail_columns,
        plan.packing_elements,
        plan.microkernel_calls,
        :measured_best_variant,
        :calibrated,
        false,
    ), :calibrated
end
