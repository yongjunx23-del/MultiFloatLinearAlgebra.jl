# =============================================================================
# M01 contract layer — O(1) factor summaries recorded at factorize time
# =============================================================================
#
# New, inert contract layer (ownership rules: `src/contracts/factors.jl`;
# snapshot/grammar rules: `src/contracts/workspace.jl`).
#
# Hard requirement (ADR-002, "factor_summary(handle) | O(1) small report |
# allocate a matrix, recompute inertia, or copy the factor"):
#
#   * a summary is RECORDED once, immediately after a factorization completes
#     (`factor_summary(x)`, called by the adapter at the factorize boundary);
#   * every later accessor (`summary_kind`, `summary_size`, `summary_status`,
#     `summary_state`, `summary_pivots`, `summary_inertia`, `summary_grammar`,
#     `summary_lease`, ...) reads the recorded snapshot and nothing else;
#   * no accessor calls back into the factor or the cache to recompute inertia,
#     rebuild a pivot list, or re-read the factor matrix — so no accessor
#     allocates a matrix and no accessor is O(n) in the factor payload.
#
# The one allocation performed *while recording* is the pivot-size copy
# (`n * sizeof(Int)` bytes). No matrix is allocated at any point, and the
# recorded numbers are exactly the ones the factorization produced.

# Self-contained dependency declaration (see `factors.jl`).
import MultiFloatLinearAlgebra
import MultiFloatLinearAlgebra: AbstractMFFactorization, AbstractMFFactorCache,
    factor_kind, factor_status, factor_state, factor_provider, factor_precision,
    factor_pivots, factor_blocks, factor_inertia, factor_diagnostics,
    _FACTOR_CACHE_INVALID, MFLUCache, MFLDLTCache

"""
    FactorSummary

The recorded, immutable small report of one factorization. Field-by-field:

  - `kind`      — `:cholesky` / `:lu` / `:ldlt` / `:rrqr` / `:cholesky_pivoted`
  - `size`      — `(m, n)` of the operator the factor was computed from
  - `status`    — the factor's own integer status at record time, verbatim
                  (the cache sentinel `-2` is preserved as `-2`, *not* mapped)
  - `state`     — the stable symbolic state at record time
  - `storage`   — `:owned` / `:borrowed`
  - `accepted`  — accepted pivot/column count at record time
  - `pivots`    — independent copy of the pivot vector, `nothing` for kinds
                  that have no pivot vector (Cholesky)
  - `inertia`   — captured `(positive, negative, zero)` for indefinite kinds,
                  `nothing` for kinds where inertia is not defined
  - `grammar`   — independent [`BlockGrammar`](@ref) for block-pivot kinds
  - `one_by_one` / `two_by_two` — captured block counts
  - `provider` / `precision` — provider identity and scalar type
  - `generation` — the owner's contract generation at record time
"""
struct FactorSummary
    kind::Symbol
    size::Tuple{Int,Int}
    status::Int
    state::Symbol
    storage::Symbol
    accepted::Int
    pivots::Union{Nothing,Vector{Int}}
    inertia::Union{Nothing,NamedTuple{(:positive, :negative, :zero),Tuple{Int,Int,Int}}}
    grammar::Union{Nothing,BlockGrammar}
    one_by_one::Int
    two_by_two::Int
    provider::Symbol
    precision::Type
    generation::UInt64
end

function Base.show(io::IO, summary::FactorSummary)
    print(io, "FactorSummary(:", summary.kind, ", ", summary.size[1], "x",
          summary.size[2], ", status=", summary.status, ", state=:",
          summary.state, ", storage=:", summary.storage, ")")
    return nothing
end

# ---------------------------------------------------------------------------
# Captured-inertia side table
# ---------------------------------------------------------------------------
#
# A summary must never recompute inertia on read. For kinds whose public API
# exposes inertia through an O(n) accessor (`factor_inertia`), the value is
# computed once when the summary is recorded and kept here so the recorded
# summary itself stays pointer-free of the factor. This table holds small
# integer tuples only: no matrix, no factor storage, no numeric payload.
#
# Identity-keyed (`IdDict`) for the same reason as `_GENERATIONS`: the
# standalone factor types are immutable structs and cannot carry finalizers, so
# a `WeakKeyDict` is not usable here.

const _CAPTURED_INERTIA =
    IdDict{Any,NamedTuple{(:positive, :negative, :zero),Tuple{Int,Int,Int}}}()
const _CAPTURED_INERTIA_LOCK = ReentrantLock()

"""
    capture_inertia!(x, value) -> value

Record `value` as the inertia `x` produced at factorize time. Called by the
adapter exactly once per factorization; never called by an accessor.
"""
function capture_inertia!(x, value::NamedTuple)
    lock(_CAPTURED_INERTIA_LOCK)
    try
        _CAPTURED_INERTIA[x] = value
    finally
        unlock(_CAPTURED_INERTIA_LOCK)
    end
    return value
end

"""
    forget_inertia!(x) -> Nothing

Release the recorded inertia for `x`. Called when the object is no longer
needed; never called by an accessor.
"""
function forget_inertia!(x)
    lock(_CAPTURED_INERTIA_LOCK)
    try
        delete!(_CAPTURED_INERTIA, x)
    finally
        unlock(_CAPTURED_INERTIA_LOCK)
    end
    return nothing
end

"""
    captured_inertia(x) -> Union{Nothing,NamedTuple}

The inertia recorded at factorize time, or `nothing` when none was recorded or
the kind has no inertia. Pure table read.
"""
function captured_inertia(x)
    lock(_CAPTURED_INERTIA_LOCK)
    try
        return get(_CAPTURED_INERTIA, x, nothing)
    finally
        unlock(_CAPTURED_INERTIA_LOCK)
    end
end

# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

# Kinds whose public API exposes an inertia accessor. The captured value is
# computed once, here, and never again by an accessor.
_has_inertia(kind::Symbol) = kind === :ldlt || kind === :sparse_ldlt

# Gap in the existing public API, worked around here rather than patched into
# existing source (M01 may not edit it):
#   `factor_pivots`/`factor_blocks` have methods for the standalone factors
#   (`MFLU`, `MFLDLT`) but NOT for the caches, even though `MFLUCache` /
#   `MFLDLTCache` own exactly the vectors those accessors are meant to expose.
#   Without these two methods a summary of a cache cannot carry its pivots or
#   its block grammar. Reported as an open finding; the durable fix belongs in
#   `src/factor_cache_defs.jl` at integration time.
factor_pivots(cache::MFLUCache) = cache.ipiv
factor_pivots(cache::MFLDLTCache) = cache.pivots
factor_blocks(cache::MFLDLTCache) = cache.blocks

# Third instance of the same gap: `factor_inertia` has a method for `MFLDLT`
# only, while `src/diagnostics.jl` keeps a private `_ldlt_cache_inertia`. The
# contract layer spells the classification out from the cache's own `factors`,
# `dsub`, and `blocks` so it does not depend on a private helper. The 2x2 rule
# is the standard Bunch-Kaufman classification by determinant and trace sign.
function _cache_block_inertia_1x1(value)
    if value > zero(value)
        return (1, 0, 0)
    elseif value < zero(value)
        return (0, 1, 0)
    end
    return (0, 0, 1)
end

function _cache_block_inertia_2x2(d11::MultiFloat, d21::MultiFloat, d22::MultiFloat)
    scale = max(abs(d11), abs(d21), abs(d22))
    iszero(scale) && return (0, 0, 2)
    a = d11 / scale
    b = d21 / scale
    c = d22 / scale
    determinant = a * c - b * b
    determinant < zero(determinant) && return (1, 1, 0)
    trace = a + c
    if determinant > zero(determinant)
        return trace > zero(trace) ? (2, 0, 0) : (0, 2, 0)
    end
    return trace > zero(trace) ? (1, 0, 1) :
           trace < zero(trace) ? (0, 1, 1) : (0, 0, 2)
end

"""
    cache_inertia(cache::MFLDLTCache) -> NamedTuple

The cache equivalent of [`factor_inertia`](@ref). O(n) over the recorded block
grammar; called once when a summary is recorded, never by an accessor.
"""
function cache_inertia(cache::MFLDLTCache)
    positive = 0
    negative = 0
    zero_count = 0
    k = 1
    @inbounds while k <= length(cache.blocks)
        block = cache.blocks[k]
        if block == UInt8(1)
            pos, neg, zer = _cache_block_inertia_1x1(cache.factors[k, k])
            k += 1
        elseif block == UInt8(2) && k < length(cache.blocks)
            pos, neg, zer = _cache_block_inertia_2x2(
                cache.factors[k, k], cache.dsub[k], cache.factors[k + 1, k + 1],
            )
            k += 2
        else
            break
        end
        positive += pos
        negative += neg
        zero_count += zer
    end
    return (positive=positive, negative=negative, zero=zero_count)
end

function _capture_inertia_for(x, kind::Symbol)
    _has_inertia(kind) || return nothing
    existing = captured_inertia(x)
    existing === nothing || return existing
    source = _inertia_source(x)
    # A provider that reports no inertia leaves the summary's field `nothing`
    # rather than a fabricated zero triple.
    source === nothing && return nothing
    return capture_inertia!(x, source)
end

# One dispatch point for "where does this object's inertia come from".
# `_dense_ldlt_inertia` is only called for a real `MFLDLT`, so it needs no
# cache method, and every other route dispatches to a `nothing` default.
_dense_ldlt_inertia(x) = factor_inertia(x)
_inertia_source(::Any) = nothing
_inertia_source(x::AbstractMFFactorization) = _dense_ldlt_inertia(x)
_inertia_source(cache::MFLDLTCache) = cache_inertia(cache)

"""
    provider_inertia(x) -> Union{Nothing,NamedTuple}

The inertia a provider reports for `x` in the contract's uniform shape, or
`nothing` when the provider reports none. Two provider conventions are folded
into one query:

  - providers that report a full `(positive, negative, zero)` triple (MFLA's
    `MFLDLT` and `MFLDLTCache`) are passed through;
  - providers that report only a scalar positive count are widened using the
    matrix order. Where the widening would be a guess, this returns `nothing`
    rather than inventing a number.

Pure and allocation-free. This is a query, not a computation of the factor.

The sparse route is reached through the *public* `factor_diagnostics`, not
through QDLDL. `provider_inertia` deliberately has NO `(::Any)` method: in Julia
`f(::Any)` is MORE specific than `f(x)`, so adding one would silently shadow the
untyped entry point and make every query return the fallback. The untyped method
delegates; the type-specific methods below are the only specializations.
"""
provider_inertia(x) = _provider_inertia(x)

_provider_inertia(x) = nothing

# Providers exposing MFLA's own diagnostics report a scalar positive count.
function _scalar_positive_inertia(x)
    diagnostics = try
        factor_diagnostics(x)
    catch
        return nothing
    end
    diagnostics isa NamedTuple || return nothing
    hasproperty(diagnostics, :positive_inertia) || return nothing
    count = diagnostics.positive_inertia
    count isa Integer && count >= 0 || return nothing
    hasproperty(diagnostics, :pattern_signature) || return nothing
    hasproperty(x, :prepared_shape) || return nothing
    n = x.prepared_shape[1]
    n >= count || return nothing
    return (positive=Int(count), negative=Int(n - count), zero=0)
end

# `MFSparseLDLCache` is the only cache whose diagnostics carry
# `positive_inertia`; the dense caches' diagnostics do not, so this returns
# `nothing` for them without special-casing their types.
_provider_inertia(x::AbstractMFFactorCache) = _scalar_positive_inertia(x)

# The sparse route reports a scalar positive count, which needs the same
# widening the summary capture uses.
function _inertia_source(x::AbstractMFFactorCache)
    kind = factor_kind(x)
    kind === :sparse_ldlt || return nothing
    return provider_inertia(x)
end

function _summary_grammar_for(x, kind::Symbol)
    kind === :ldlt || return nothing
    return block_grammar(factor_blocks(x), size(x, 1))
end

function _summary_pivots_for(x, kind::Symbol)
    kind in (:lu, :ldlt) || return nothing
    pivots = factor_pivots(x)
    pivots === nothing && return nothing
    return copy(pivots)
end

"""
    factor_summary(x) -> FactorSummary

Record the small report for the factorization `x` currently holds. Call this
once at the factorize boundary; the returned value is independent of `x`
afterwards.

This is the *recording* call. It reads the factor's metadata once (pivot count,
block grammar, status) and allocates no matrix. Later queries must go through
the `summary_*` accessors below, which never touch `x` again.
"""
function factor_summary(x::Union{AbstractMFFactorization,AbstractMFFactorCache})
    kind = factor_kind(x)
    size_x = (size(x, 1), size(x, 2))
    status = factor_status(x)
    state = factor_state(x)
    storage = storage_kind(x) === OWNED_STORAGE ? :owned : :borrowed
    grammar = _summary_grammar_for(x, kind)
    inertia = _capture_inertia_for(x, kind)
    pivots = _summary_pivots_for(x, kind)
    one_by_one = grammar === nothing ? 0 : grammar.counts[1]
    two_by_two = grammar === nothing ? 0 : grammar.counts[2]
    accepted = (status == 0 || status == _FACTOR_CACHE_INVALID) ?
               size_x[1] : (status > 0 ? status - 1 : 0)
    return FactorSummary(
        kind, size_x, status, state, storage, accepted, pivots, inertia, grammar,
        one_by_one, two_by_two, factor_provider(x), factor_precision(x),
        generation(x),
    )
end

"""
    factor_summary(x, lease::FactorLease) -> FactorSummary

Record a summary and bind it to `lease`. Throws when `lease` is already stale,
which is the ADR-002 rule that a caller must re-validate before trusting a
provider status it reads.
"""
function factor_summary(x, lease::FactorLease)
    require_lease(x, lease)
    return factor_summary(x)
end

# ---------------------------------------------------------------------------
# Accessors — pure reads of the recorded summary
# ---------------------------------------------------------------------------

"""
    summary_kind(summary) -> Symbol
    summary_size(summary) -> Tuple{Int,Int}
    summary_status(summary) -> Int
    summary_state(summary) -> Symbol
    summary_storage(summary) -> Symbol
    summary_accepted(summary) -> Int
    summary_pivots(summary) -> Union{Nothing,Vector{Int}}
    summary_inertia(summary) -> Union{Nothing,NamedTuple}
    summary_grammar(summary) -> Union{Nothing,BlockGrammar}
    summary_block_counts(summary) -> Tuple{Int,Int}
    summary_generation(summary) -> UInt64
    summary_provider(summary) -> Symbol
    summary_precision(summary) -> Type

Every accessor reads only the recorded `FactorSummary`. None of them touches
the factor, the cache, or any matrix; none of them recomputes inertia or
rebuilds a pivot list.
"""
summary_kind(summary::FactorSummary) = summary.kind
summary_size(summary::FactorSummary) = summary.size
summary_status(summary::FactorSummary) = summary.status
summary_state(summary::FactorSummary) = summary.state
summary_storage(summary::FactorSummary) = summary.storage
summary_accepted(summary::FactorSummary) = summary.accepted
summary_pivots(summary::FactorSummary) = summary.pivots
summary_inertia(summary::FactorSummary) = summary.inertia
summary_grammar(summary::FactorSummary) = summary.grammar
summary_block_counts(summary::FactorSummary) = (summary.one_by_one, summary.two_by_two)
summary_generation(summary::FactorSummary) = summary.generation
summary_provider(summary::FactorSummary) = summary.provider
summary_precision(summary::FactorSummary) = summary.precision
summary_success(summary::FactorSummary) = iszero(summary.status)

"""
    summary_valid_for(summary, x) -> Bool

`true` only while `summary` still describes the factorization `x` currently
holds. For a borrowed cache this is `false` after any later `factorize!`,
`prepare!`, `reconfigure!`, or `invalidate!`, because those transitions move the
owner's generation or commit markers.
"""
summary_valid_for(summary::FactorSummary, x) =
    summary.generation == generation(x) && summary.kind === factor_kind(x)

"""
    stale_reason(summary, x) -> Symbol

Why `summary` no longer describes `x`, or `:current` when it still does.
"""
function stale_reason(summary::FactorSummary, x)
    summary.kind === factor_kind(x) ||
        return :kind_changed
    summary.generation == generation(x) ||
        return :generation_advanced
    return :current
end

"""
    summary_lease(x) -> FactorLease

Take a lease on the recorded state of `x`. Adapters that cache a
`FactorSummary` should also record this lease and re-check it before use.
"""
summary_lease(x) = take_lease(x)

"""
    record_factor_summary!(x) -> FactorSummary

Adapter entry point for the factorize boundary: record the summary, then record
one completed contract transition so every earlier lease on `x` is dead.

Order matters and is part of the contract:

  1. the summary is recorded from the factorization that just finished;
  2. the generation is advanced, which invalidates every previously issued
     lease and every previously recorded summary of `x`.

A caller that takes a summary *before* the bump therefore cannot reuse it
afterwards. No matrix is allocated by this function.
"""
function record_factor_summary!(x)
    summary = factor_summary(x)
    bump_generation!(x)
    return summary
end
