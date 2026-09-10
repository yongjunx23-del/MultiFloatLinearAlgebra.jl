# =============================================================================
# M01 contract layer — ordinary factor / borrowed cache / owned snapshot
# =============================================================================
#
# This file is a NEW, INERT contract layer. It does not modify, redefine, or
# shadow any existing MFLA definition, and it is not `include`d by
# `src/MultiFloatLinearAlgebra.jl` (that include line belongs to the
# integration role at I01/I02/I03). It can be:
#
#   * `include`d into the `MultiFloatLinearAlgebra` module after the existing
#     includes (adds names only, replaces nothing), or
#   * `include`d standalone inside its own module, as `test/rebuild/M01.jl`
#     does.
#
# Binding rules encoded here (ADR-002 §"Provider owns" / M01 task):
#
#   ordinary factor   an independently ownable public-API result of
#                     `lu!`/`ldlt!`/`cholesky!`/`rrqr!`. It owns the metadata
#                     it needs; its recorded summary never changes.
#   borrowed cache    `AbstractMFFactorCache` storage. The factor is owned by
#                     the cache and is invalidated by the next `factorize!`.
#                     A lease taken against an earlier factorization MUST NOT
#                     be reused after a later `factorize!` attempt.
#   owned snapshot    an explicit, expensive copy of the operator. The
#                     operator snapshot is NOT factor storage: mutating it
#                     never changes the factor, and vice versa.
#
# The independent MFLA public API (`factor_status`, `factor_kind`,
# `factor_matrix`, `factor_pivots`, `factor_blocks`, `factor_inertia`,
# `factor_diagnostics`, ...) is left intact and is not redefined here.

# Self-contained dependency declaration: this file works both when included
# into the `MultiFloatLinearAlgebra` module (a self-import is a no-op) and when
# included into a test/adapter module of its own.
import MultiFloatLinearAlgebra
import MultiFloatLinearAlgebra: AbstractMFFactorization, AbstractMFFactorCache,
    factor_kind, factor_status, factor_provider, factor_precision

"""
    FactorStorageKind

How the numeric storage of a factor is owned.

  - `:owned`    — ordinary factor from the standalone public API; the object
                  owns the metadata it needs and stays valid for as long as the
                  caller holds it.
  - `:borrowed` — factor storage owned by an `AbstractMFFactorCache`; the next
                  `factorize!` attempt on that cache ends the lease.
  - `:snapshot` — an explicit copy of the operator, produced by
                  `copy_operator_snapshot`. Not factor storage.
"""
@enum FactorStorageKind::UInt8 begin
    OWNED_STORAGE = 0x01
    BORROWED_STORAGE = 0x02
    SNAPSHOT_STORAGE = 0x03
end

"""
    FactorLease

The (generation, token) pair that binds a request to one exact factorization
result.

A lease is valid only while

  1. `generation == FactorContract.generation(owner)`, and
  2. `token == FactorContract.lease_token(owner)`.

For a borrowed cache, `lease_token` is derived from the cache's own commit
markers (`status`, `config_epoch`, `prepared_epoch`, `prepared_shape`), all of
which are rewritten by every `factorize!`/`reconfigure!`/`prepare!`/
`invalidate!` transition. A lease taken before such a transition therefore
cannot be reused afterwards — including after a *preflight* throw, which leaves
the previous factor numerically intact but still ends the lease for the new
request. This is the M01 requirement "a cache's old lease MUST be invalidated".
"""
struct FactorLease
    generation::UInt64
    token::UInt64
end

"""
    FactorContract

Stable classification of a factor-like object under the M01 contract. This is
deliberately *additive*: `factor_kind`, `factor_status`, `factor_state` and the
other MFLA accessors keep their existing meaning and signatures.
"""
struct FactorContract
    storage::FactorStorageKind
    kind::Symbol
    size::Tuple{Int,Int}
    provider::Symbol
    precision::Type
end

function Base.show(io::IO, contract::FactorContract)
    print(io, "FactorContract(", contract.storage, ", :", contract.kind, ", ",
          contract.size[1], "x", contract.size[2], ")")
    return nothing
end

"""
    storage_kind(x) -> FactorStorageKind
    storage_kind(::Type{T}) -> FactorStorageKind

Classify a live object or a declared type. Ordinary public-API factorizations
are `OWNED_STORAGE`; factor caches are `BORROWED_STORAGE`; explicit operator
snapshots are `SNAPSHOT_STORAGE`.
"""
storage_kind(::AbstractMFFactorization) = OWNED_STORAGE
storage_kind(::AbstractMFFactorCache) = BORROWED_STORAGE
storage_kind(::Type{<:AbstractMFFactorization}) = OWNED_STORAGE
storage_kind(::Type{<:AbstractMFFactorCache}) = BORROWED_STORAGE

# --- generation / lease identity -------------------------------------------
#
# `generation` counts completed contract-level transitions of a live object
# (see `bump_generation!`). It is a *contract* counter: it never influences a
# numeric result, never participates in a kernel, and never touches factor
# storage. Objects with no recorded transition report generation 0.
#
# Keyed by object IDENTITY (`IdDict`), not by `==` and not by weak reference:
# the standalone factor types (`MFLDLT`, `MFLU`, ...) are immutable structs, so
# a `WeakKeyDict` is not usable (its entries attach finalizers, which require a
# mutable object). `IdDict` also makes the table immune to field mutation of a
# mutable cache. Consequence, stated plainly: the table holds a strong
# reference to each object that has a recorded transition, so a long-lived
# process that factorizes unboundedly many distinct objects retains one small
# entry per object. `forget_generation!` releases an entry explicitly.

const _GENERATIONS = IdDict{Any,UInt64}()
const _GENERATION_LOCK = ReentrantLock()

"""
    generation(x) -> UInt64

Return the number of recorded contract transitions for `x`, or 0 if none were
recorded. Never touches numeric storage.
"""
function generation(x)
    lock(_GENERATION_LOCK)
    try
        return get(_GENERATIONS, x, zero(UInt64))
    finally
        unlock(_GENERATION_LOCK)
    end
end

"""
    forget_generation!(x) -> Nothing

Release the contract bookkeeping for `x`. Does not touch numeric storage and
does not change `x`'s validity; it only frees the identity-keyed entry.
"""
function forget_generation!(x)
    lock(_GENERATION_LOCK)
    try
        delete!(_GENERATIONS, x)
    finally
        unlock(_GENERATION_LOCK)
    end
    return nothing
end

"""
    bump_generation!(x) -> UInt64

Record one completed contract transition (a new `factorize!` result, an
`invalidate!`, a `reconfigure!`, a fresh snapshot) and return the new counter.
This is the explicit, adapter-driven half of the lease semantics: it allocates
no matrix, mutates no factor storage, and is safe to call on a hot path (one
dictionary update under a short lock).
"""
function bump_generation!(x)
    lock(_GENERATION_LOCK)
    try
        next = get(_GENERATIONS, x, zero(UInt64)) + one(UInt64)
        _GENERATIONS[x] = next
        return next
    finally
        unlock(_GENERATION_LOCK)
    end
end

@inline _mix64(value::UInt64) = begin
    h = value
    h ⊻= h >> 33
    h *= 0xff51afd7ed558ccd
    h ⊻= h >> 33
    h *= 0xc4ceb9fe1a85ec53
    h ⊻= h >> 33
    h
end

@inline _mix64(value::Integer) = _mix64(reinterpret(UInt64, Int64(value)))

# The cache token reads only scalar commit markers. It never reads `factors`,
# `pivots`, `blocks`, or any other array, so computing it is allocation-free.
#
# The caller's generation is mixed in as well, and that is load-bearing: two
# successive `factorize!` calls at the same size under the same frozen config
# with the same outcome leave every commit marker (`status`, `config_epoch`,
# `prepared_epoch`, `prepared_shape`) numerically identical. Without the
# generation, a lease taken against the first result would still validate
# against the second. `generation` is the only signal that distinguishes them,
# which is why the adapter MUST record the transition at the factorize
# boundary (see `record_factor_summary!`).
function _cache_lease_token(cache::AbstractMFFactorCache, generation_count::UInt64)
    token = _mix64(generation_count) + 0x9e3779b97f4a7c15
    token = _mix64(token ⊻ _mix64(reinterpret(UInt64, Int64(factor_status(cache)))))
    token ⊻= _mix64(reinterpret(UInt64, UInt64(cache.config_epoch))) + 0x9e3779b97f4a7c15
    token = _mix64(token)
    token ⊻= _mix64(reinterpret(UInt64, UInt64(cache.prepared_epoch))) + 0x9e3779b97f4a7c15
    token = _mix64(token)
    token ⊻= _mix64(Int64(cache.prepared_shape[1])) + 0x9e3779b97f4a7c15
    token = _mix64(token)
    token ⊻= _mix64(Int64(cache.prepared_shape[2])) + 0x9e3779b97f4a7c15
    return _mix64(token)
end

# An ordinary factor owns its metadata and is never refactored in place through
# this contract, so its token binds the identity of the object plus the
# generation at which the lease was taken.
function _factor_lease_token(factor::AbstractMFFactorization, generation_count::UInt64)
    token = _mix64(reinterpret(UInt64, UInt64(objectid(factor))))
    token ⊻= _mix64(generation_count) + 0x9e3779b97f4a7c15
    token = _mix64(token)
    token ⊻= _mix64(Int64(factor_status(factor))) + 0x9e3779b97f4a7c15
    return _mix64(token)
end

"""
    lease_token(x) -> UInt64

The generation-independent half of a lease. Pure and allocation-free; reads
scalar commit markers only.
"""
lease_token(cache::AbstractMFFactorCache) =
    _cache_lease_token(cache, generation(cache))
lease_token(factor::AbstractMFFactorization) =
    _factor_lease_token(factor, generation(factor))

@inline _lease_token_at(cache::AbstractMFFactorCache, n::UInt64) =
    _cache_lease_token(cache, n)
@inline _lease_token_at(factor::AbstractMFFactorization, n::UInt64) =
    _factor_lease_token(factor, n)

"""
    take_lease(x) -> FactorLease

Bind a request to the *current* factorization of `x`. The returned lease is
accepted by [`validate_lease`](@ref) only while `x` has not moved on to another
factorization (cache) or another recorded transition (ordinary factor).
"""
take_lease(x) = FactorLease(generation(x), lease_token(x))

"""
    validate_lease(x, lease) -> Bool

`true` only while `lease` still describes the current factorization of `x`.
Never mutates `x` and never reads factor storage.
"""
function validate_lease(x, lease::FactorLease)
    current = generation(x)
    # The token is re-derived at the lease's own generation, so a lease is
    # rejected as soon as the owner advances to any later generation.
    lease.generation == current || return false
    return lease.token == _lease_token_at(x, lease.generation)
end

"""
    require_lease(x, lease) -> Bool

Throwing variant of [`validate_lease`](@ref). A stale lease is a contract
violation: the caller must re-`factorize!` (cache) or re-take a lease.
"""
function require_lease(x, lease::FactorLease)
    validate_lease(x, lease) && return true
    throw(ArgumentError(
        "stale factor lease: the lease was taken against generation " *
        "$(lease.generation) but the current generation is $(generation(x)); " *
        "re-factorize before reusing it",
    ))
    return false
end

"""
    factor_contract(x) -> FactorContract

Stable classification of `x`. Reads only scalar/scalarizable metadata.
"""
function factor_contract(x::AbstractMFFactorization)
    return FactorContract(
        OWNED_STORAGE, factor_kind(x), size(x), factor_provider(x),
        factor_precision(x),
    )
end

function factor_contract(cache::AbstractMFFactorCache)
    return FactorContract(
        BORROWED_STORAGE, factor_kind(cache), size(cache), factor_provider(cache),
        factor_precision(cache),
    )
end

"""
    same_factor_storage(a, b) -> Bool

`true` only when `a` and `b` refer to the same numeric storage. Used by the
contract tests to prove that a summary read did not replace, copy, or rebuild
the factor matrix.
"""
same_factor_storage(a::AbstractMFFactorization, b::AbstractMFFactorization) =
    a === b
same_factor_storage(a::AbstractMFFactorCache, b::AbstractMFFactorCache) =
    a === b
same_factor_storage(a, b) = false
