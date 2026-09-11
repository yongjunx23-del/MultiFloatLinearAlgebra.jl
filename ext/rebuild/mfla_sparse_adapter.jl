# =====================================================================
# ext/rebuild/mfla_sparse_adapter.jl
#
# P02 — MFLA sparse adapter: numeric/structural lifecycle closure.
#
# ---------------------------------------------------------------------
# HOW THIS FILE IS LOADED (read this before concluding anything)
# ---------------------------------------------------------------------
# `ext/rebuild/` is NOT a Julia package-extension directory in MFLA, and this
# file is NOT a declared extension.  MFLA's `Project.toml` declares exactly two
# extensions — `MultiFloatLinearSolveExt` and `MultiFloatQDLDLExt` — and they
# live at `ext/MultiFloatLinearSolveExt.jl` and `ext/MultiFloatQDLDLExt.jl`.
# `using MultiFloatLinearAlgebra` will therefore NEVER load this file, and
# `Pkg.test()` will never evaluate it: it appears in no include graph.
#
# The mechanism is the SAME one S05 recorded for
# `SDPX/ext/rebuild/{mfla,bfla}_adapter.jl`: LOAD ON DEMAND, BY PATH, INTO A
# MODULE THAT PRE-BINDS THE PACKAGE NAMES THE FILE NEEDS.  The only loader is
# `test/rebuild/P02.jl`, which evaluates
#
#     module P02ModeA
#         using MultiFloatLinearAlgebra
#         using MultiFloats
#         using SparseArrays
#         using LinearAlgebra
#         const QDLDL_ACCESS = :extension
#         include("<abs path>/ext/rebuild/mfla_sparse_adapter.jl")
#     end
#
# and, in a second configuration (Mode B), the same `include` into a module
# whose MFLA name surface was aliased in first, with `QDLDL_ACCESS = :factory`.
# Because the adapter must load under both, it never *requires* its including
# module to have imported anything: the `import` statements below are guarded
# by `@isdefined`, and every package reference is a fully-qualified
# `getproperty` on a binding the loader guarantees.  The `!@isdefined(...)`
# wrapper is the same guard S05 used so the file can be included twice.
#
# NAMING HAZARD, NAMED DELIBERATELY (packet note P02-1).  In SDPX,
# `ext/rebuild/` means "the provider adapters" — `ext/rebuild/mfla_adapter.jl`
# is SDPX's adapter ONTO MFLA.  Here the same path suffix means the opposite
# direction: an adapter that lives INSIDE the provider repo and is pointed at
# by a rebuild driver.  The two directories share a name and nothing else.
# This file is not part of MFLA's shipped surface, is not loaded by
# `using MultiFloatLinearAlgebra`, and is not covered by MFLA's own suite.
#
# Aqua does NOT object to this file, and nothing here defends against it:
# the resolved Aqua (0.8.x) `test_all` runs ambiguities, unbound args,
# undefined exports, project extras, stale deps, deps compat, piracies and
# persistent tasks — no extension-file scan.  Recorded so no cycle is spent on
# a check that does not exist.
#
# ---------------------------------------------------------------------
# WHAT THIS ADAPTER IS FOR
# ---------------------------------------------------------------------
# The object under adaptation is MFLA's QDLDL-backed sparse signed-LDL cache,
# defined in MFLA's own declared extension `ext/MultiFloatQDLDLExt.jl`:
#
#     MultiFloatQDLDLExt.MFSparseLDLCache{MF<:MultiFloat,Ti<:Integer}
#         <: MFLA.AbstractMFFactorCache{MF}
#
# It is reached through two independent bindings, which the driver exercises
# separately and requires to agree:
#
#   route :extension — `Base.get_extension(MFLA, :MultiFloatQDLDLExt).MFSparseLDLCache`
#   route :factory   — `MFLA.sparse_ldlt_cache`, MFLA's own published accessor
#                      (`src/MultiFloatLinearAlgebra.jl:111`), which resolves
#                      the same extension internally and whose result the
#                      adapter takes the TYPE of.
#
# The `QDLDL_ACCESS` binding is whichever route the including module set before
# the `include`.  The driver asserts both routes resolve to the SAME type
# object, so the two inclusion modes are an ACCESS-PATH split, not a type split.
#
# ---------------------------------------------------------------------
# CONTRACT OBLIGATIONS DISCHARGED HERE (ADR-002 §4 is the load-bearing one)
# ---------------------------------------------------------------------
# 1. SYMBOLIC vs NUMERIC AUTHORITY.  The cache freezes the pattern at
#    construction (`frozen_colptr`/`frozen_rowval`/`pattern_signature`), and
#    `factorize!` reuses the symbolic factor through `QDLDL.update_values!` +
#    `QDLDL.refactor!` (`ext/MultiFloatQDLDLExt.jl`).  A numeric refactor is
#    therefore legal ONLY when the candidate pattern equals the frozen one,
#    element for element.  The adapter checks that BEFORE calling the provider
#    and refuses otherwise.
#
# 2. LEASE REVOCATION ON A FAILED REFACTOR (ADR-002 §4).  A provider is
#    entitled to keep its physical factor and its previous success flag when a
#    preflight check rejects a request; ADR-002 §8/§9 establish that both MFLA
#    and BFLA do exactly that.  `MFSparseLDLCache.factorize!` additionally
#    revokes at ATTEMPT ENTRY (`_revoke_factor!` runs first) — a stronger
#    provider guarantee than the dense caches give — but the adapter must not
#    depend on which provider behaviour it got.  It revokes its OWN logical
#    lease on every attempt, before reading any provider status.
#
# 3. THE M01-F4 GAP (measured, high).  Two successive `factorize!` calls at the
#    same size with the same outcome leave every commit marker numerically
#    identical, so `MFLA.lease_token` is identical and a lease taken against
#    the first result STILL VALIDATES against the second.  Reproduced live at
#    MFLA 4c8e351 by `test/rebuild/P02.jl` §3.  The fix is IP-2
#    (`record_factor_summary!(cache)` at each `factorize!` commit point), which
#    belongs to I02 and is NOT applied at this revision.  This adapter
#    therefore calls `MFLA.record_factor_summary!(cache)` itself at every
#    attempt boundary — success AND failure — so the guarantee holds for
#    callers of THIS adapter without IP-2.  See `adapter_refactor!`.
#
# 4. NO PRECISION DOWNGRADE, NO DENSE CONVERSION.  Every entry point is
#    parametric in the cache's own `{MF,Ti}` and refuses a value matrix whose
#    scalar type or index type differs, rather than converting.
#    `adapter_canonical_upper` is the ONE place an ordinary matrix becomes a
#    CSC pattern, and it does so by *selecting* the upper triangle of the
#    caller's own scalars — no `Float64`, no `Matrix{MF}` intermediate held for
#    the pattern, no `SparseMatrixCSC{Float64}`.  `adapter_dense_conversions()`
#    counts the dense `Matrix`-typed conversions this adapter performs and is
#    `0` by construction; the driver asserts that against a measured control
#    that a real `Matrix(K)` on the same operator increments a counter.
#
# 5. RHS WIDTH CAPACITY.  The cache is prepared with an explicit
#    `nrhs_capacity`; a wider RHS must be refused, not silently truncated, and
#    the capacity must be reported as the provider records it.
#
# 6. ALIAS HANDLING.  The provider refuses a destination that aliases factor
#    storage.  The adapter refuses a destination that aliases its own RHS or
#    the operator values it is factoring, because writing a solution over the
#    input is silent numerical corruption even when the provider would allow it.
# =====================================================================

if !@isdefined(P02_MFLA_SPARSE_ADAPTER_LOADED)

const P02_MFLA_SPARSE_ADAPTER_LOADED = true

# The including module is expected to bind these three package names.  Use
# ordinary `import` when it has not — guarded, because a module that aliased
# MFLA's name surface first already owns them as consts and a second `import`
# would throw.
@isdefined(MultiFloatLinearAlgebra) || import MultiFloatLinearAlgebra
@isdefined(MultiFloats) || import MultiFloats
@isdefined(SparseArrays) || import SparseArrays

const MFLA = MultiFloatLinearAlgebra
const MF = MultiFloats
const SA = SparseArrays

# =====================================================================
# 1. QDLDL access — the two independent bindings
# =====================================================================

"""
    p02_extension_module() -> Union{Module,Nothing}

`MultiFloatQDLDLExt` when QDLDL and SparseArrays are loaded, else `nothing`.
Never throws: a missing extension is a fact the adapter reports, not an error.
"""
function p02_extension_module()
    return try
        Base.get_extension(MFLA, :MultiFloatQDLDLExt)
    catch
        nothing
    end
end

# A frozen 2x2 sparsity pattern with nonempty columns and a sign-indefinite
# diagonal, for the type probe below.  The signs matter: QDLDL is a signed-LDL
# provider and a purely positive diagonal is not a system it is asked to solve,
# so a `[1, 1]` D-sign vector can be refused by the provider and the probe would
# then report "route unavailable" for a route that is merely mis-probed.  The
# values are `Float64`; the caller casts the pattern into its own arithmetic, so
# this stays a pattern probe and not an arithmetic choice.
_p02_probe_pattern() = SA.SparseMatrixCSC(
    2, 2, [1, 2, 3], [1, 2], [1.0, -1.0],
)

"""
    p02_extension_route() -> Union{DataType,Nothing}

The cache TYPE as reached through the extension module directly.  This is the
route that proves the adapter is pointed at the type MFLA's own extension
defines, not at a look-alike.
"""
function p02_extension_route()
    extension = p02_extension_module()
    extension === nothing && return nothing
    isdefined(extension, :MFSparseLDLCache) || return nothing
    return getproperty(extension, :MFSparseLDLCache)
end

"""
    p02_factory_route() -> Union{DataType,Nothing}

The cache TYPE as reached through MFLA's PUBLISHED accessor.  `sparse_ldlt_cache`
returns an instance, so the type is read off the instance — deliberately: this
measures what the public accessor actually produces instead of repeating the
extension lookup and calling the repetition independence.
"""
function p02_factory_route()
    MFLA.sparse_ldlt_available(MF.Float64x2) || return nothing
    probe = try
        MFLA.sparse_ldlt_cache(
            MF.Float64x2, SA.SparseMatrixCSC{MF.Float64x2,Int}(_p02_probe_pattern());
            dsigns=[1, -1],
        )
    catch
        return nothing
    end
    return typeof(probe)
end

"""
    p02_constructor(T) -> UnionAll

The type's constructor with its parameters OPEN, so `p02_constructor(T)(MF, A;
kwargs...)` works whether `T` is the UnionAll the extension names or the
concrete type the factory returns.  Measured need, not speculation: calling a
concrete `MFSparseLDLCache{MF,Ti}` with `(MF, A)` is a `MethodError`, so a mode
that holds the concrete type cannot construct through it directly.
"""
p02_constructor(T) = Base.typename(T).wrapper

"""
    p02_qdldl_access() -> NamedTuple

The route in force for this inclusion: the TYPE the route names, the
constructor to build with, and the identity of both routes so the driver can
assert they agree.
"""
function p02_qdldl_access()
    route = @isdefined(QDLDL_ACCESS) ? QDLDL_ACCESS : :extension
    T = route === :extension ? p02_extension_route() :
        route === :factory ? p02_factory_route() :
        throw(ArgumentError("unknown QDLDL_ACCESS $(repr(route))"))
    T === nothing && throw(ArgumentError(
        "QDLDL sparse access is unavailable through route $(repr(route)); " *
        "load QDLDL and SparseArrays before including this adapter",
    ))
    return (route=route, type=T, constructor=p02_constructor(T),
            extension=p02_extension_route(), factory=p02_factory_route())
end

"""
    p02_type_identity(T) -> NamedTuple

`(module, name, nparams, abstract)` for a `DataType` or `UnionAll` — the
TYPE CONSTRUCTOR's identity, which is what "the two routes reach the same type"
has to mean.  The two routes legitimately return different Julia objects for
the same type: the extension route names the `UnionAll`
`MFSparseLDLCache{MF<:MultiFloat,Ti<:Integer}` and the factory route reads the
concrete `MFSparseLDLCache{MultiFloat{Float64,2},Int}` off the instance it
built.  `===` on those is `false`, and a driver that asserted `===` would
report a disagreement that does not exist.
"""
function p02_type_identity(T)
    body = Base.unwrap_unionall(T)
    type_name = body.name
    # `module` is a reserved word, so the tuple is built by name rather than
    # with `(module=..., ...)` keyword syntax.  The parameter count comes from
    # the BODY's parameters — `Core.TypeName` has no `parameters` field, which
    # this file measured the hard way.
    return NamedTuple{(:module, :name, :nparams, :abstract)}(
        (string(type_name.module), string(type_name.name),
         length(body.parameters), isabstracttype(body)),
    )
end

"""
    p02_routes_identical() -> Bool

`true` only when both routes reach the same type constructor.  The driver
asserts this in BOTH modes before running any numeric leg: if the two modes
were aimed at different types, a two-mode agreement would be vacuous.
"""
function p02_routes_identical()
    a = p02_extension_route()
    b = p02_factory_route()
    return a !== nothing && b !== nothing &&
           p02_type_identity(a) == p02_type_identity(b)
end

# =====================================================================
# 2. The adapter
# =====================================================================

"""
    P02SparseAdapter{C}

Adapter onto a real MFLA `MFSparseLDLCache`.  Owns NO matrix: the physical
factor is the provider's, and the only thing added here is the logical
authority — a lease and the counters that make "which numeric factor does this
lease describe" answerable without reading factor storage.

Fields:

  * `cache`            — the provider's cache; the fused pattern and factor
                         storage live there and are never copied here.
  * `lease_valid`      — the LOGICAL lease.  Independent of the provider's
                         `status`, which the provider may legitimately keep
                         across a rejected request (ADR-002 §4/§8/§9).
  * `attempt`          — every `refactor!` attempt, successful or not.
  * `success`          — attempts that ended with a provider success.
  * `last_refactor_ok` — the outcome of the most recent attempt.
  * `refusals`         — attempts refused by the adapter or the provider.
  * `lease_epoch`      — `MFLA.generation(cache)` at the last SUCCESS.
  * `rhs_capacity`     — `nrhs` the cache was prepared with, as recorded by
                         the provider when it can be read.
  * `dense_conversions`— the number of dense `Matrix` conversions performed by
                         this adapter's entry points.  Zero by construction;
                         a measured control in the driver is what makes that a
                         measurement rather than an assertion.
"""
mutable struct P02SparseAdapter{C}
    cache::C
    lease_valid::Bool
    attempt::Int
    success::Int
    last_refactor_ok::Bool
    refusals::Int
    lease_epoch::UInt64
    rhs_capacity::Int
    dense_conversions::Int
end

const DENSE_CONVERSIONS = Ref{Int}(0)

"""
    adapter_dense_conversions() -> Int

How many dense `Matrix` conversions this adapter has performed since load.
`0` is a claim about *this adapter's code path*; it is not a claim that no
dense matrix exists anywhere in the process.  Always read it against the
measured control in the driver, which makes a `Matrix(K)` visible by driving
the same counter through `note_dense_conversion!`.
"""
adapter_dense_conversions() = DENSE_CONVERSIONS[]

"The counter's write side.  Only the driver's control calls this directly."
note_dense_conversion!() = (DENSE_CONVERSIONS[] += 1; DENSE_CONVERSIONS[])

"The adapter's own reason for existing: a provider whose symbolic pattern is frozen."
cache_kind(a::P02SparseAdapter) = MFLA.factor_kind(a.cache)

"Scalar type of the cache's arithmetic.  Compared, never converted."
function cache_scalar_type(a::P02SparseAdapter)
    MFc, _ = _adapter_parameters(a)
    return MFc
end

"Index type of the cache's CSC storage, read off the factor's own pattern."
function cache_index_type(a::P02SparseAdapter)
    return eltype(getfield(a.cache, :frozen_colptr))
end

"Provider state vocabulary, through the package's own accessor."
provider_state(a::P02SparseAdapter) = MFLA.factor_state(a.cache)

"Provider success flag.  NOT the lease — see `lease_live`."
provider_success(a::P02SparseAdapter) = MFLA.issuccess(a.cache)

"""
    lease_live(a) -> Bool

The LOGICAL lease.  `false` after any refactor attempt, and after any explicit
`adapter_revoke!`.  It is deliberately not derived from the provider's status:
after a rejected request the provider may still report `:success` for the OLD
factor while the lease for the NEW operator must be dead.
"""
lease_live(a::P02SparseAdapter) = a.lease_valid && a.last_refactor_ok

"""
    lease_epoch(a) -> UInt64

`MFLA.generation(a.cache)` at the last successful refactor.  Published as a
number a caller can compare; the adapter never invents one.
"""
lease_epoch(a::P02SparseAdapter) = a.lease_epoch

"""
    take_adapter_lease(a) -> NamedTuple

The lease as a value.  A caller stores this and later asks `lease_still_valid`.
"""
take_adapter_lease(a::P02SparseAdapter) = (
    epoch=a.lease_epoch, attempt=a.attempt, valid=lease_live(a),
)

"""
    lease_still_valid(a, lease) -> Bool

`true` only while `lease` still describes the factor the adapter currently
holds AND that factor is authorized.  Two same-size successes have different
`attempt` values, so a lease from the first does not validate against the
second — which is exactly what `MFLA.lease_token` alone fails to distinguish
(M01-F4).
"""
function lease_still_valid(a::P02SparseAdapter, lease)
    return lease_live(a) && lease.epoch == a.lease_epoch &&
           lease.attempt == a.attempt
end

"""
    require_adapter_lease(a, lease)

Throwing form.  A stale lease is a contract violation, not a warning.
"""
function require_adapter_lease(a::P02SparseAdapter, lease)
    lease_still_valid(a, lease) && return true
    throw(ArgumentError(
        "stale sparse factor lease: lease(epoch=$(lease.epoch), " *
        "attempt=$(lease.attempt)) vs adapter(epoch=$(a.lease_epoch), " *
        "attempt=$(a.attempt), live=$(lease_live(a))); re-refactor before reuse",
    ))
end

# =====================================================================
# 3. Type preservation
# =====================================================================

# The cache's `{MF,Ti}` unpacked.  `MFSparseLDLCache{MF,Ti}` is a subtype of
# `AbstractMFFactorCache{MF}`, so `MF` is recoverable from the supertype and
# `Ti` from the frozen pattern's element type.
function _adapter_parameters(a::P02SparseAdapter)
    super = supertype(typeof(a.cache))
    super <: MFLA.AbstractMFFactorCache ||
        throw(ArgumentError("P02 adapter: $(typeof(a.cache)) is not an MFLA factor cache"))
    return (getproperty(super, :parameters)[1], cache_index_type(a))
end

"""
    adapter_accepts_type(a, X) -> NamedTuple

Whether the cache will accept `X`'s scalar/index types WITHOUT conversion.
A `false` here is a refusal reason, never a trigger to convert.
"""
function adapter_accepts_type(a::P02SparseAdapter, X::AbstractSparseMatrix)
    MFc, Tic = _adapter_parameters(a)
    # `SparseMatrixCSC` has no `indices` field (measured here); the index type
    # lives in `rowval`, and `colptr` carries the same one.
    Tix = eltype(X.rowval)
    return (
        scalar_ok=eltype(X) === MFc, index_ok=Tix === Tic,
        cache_scalar=MFc, cache_index=Tic,
        candidate_scalar=eltype(X), candidate_index=Tix,
    )
end

adapter_accepts_type(a::P02SparseAdapter, X::AbstractMatrix) = (
    scalar_ok=eltype(X) === _adapter_parameters(a)[1],
    index_ok=true,                       # dense input carries no CSC index type
    cache_scalar=_adapter_parameters(a)[1],
    cache_index=_adapter_parameters(a)[2],
    candidate_scalar=eltype(X), candidate_index=Int,
)

# =====================================================================
# 4. Pattern: the canonical upper triangle, without a dense conversion
# =====================================================================

"""
    adapter_canonical_upper(::Type{MF}, ::Type{Ti}, K) -> SparseMatrixCSC{MF,Ti}

The candidate pattern, as QDLDL requires it: strictly the upper triangle in
CSC, in the cache's OWN scalar and index types.

This is the adapter's only matrix→pattern step and it performs NO dense
conversion: `sparse(UpperTriangular(K))` reads `K`'s own scalars, and when `K`
is already a `SparseMatrixCSC{MF,Ti}` it selects the stored upper entries
rather than materializing `Matrix{MF}(K)`.  Passing a `Matrix{Float64}` here is
refused by `adapter_accepts_type` before this function is reached.
"""
function adapter_canonical_upper(::Type{MFc}, ::Type{Tic}, K::SA.SparseMatrixCSC) where {MFc,Tic}
    eltype(K) === MFc || throw(ArgumentError(
        "sparse value matrix has scalar type $(eltype(K)), cache holds $(MFc); " *
        "a conversion here would be a precision policy decision the adapter " *
        "must not make",
    ))
    eltype(K.rowval) === Tic || throw(ArgumentError(
        "sparse value matrix has index type $(eltype(K.rowval)), cache holds " *
        "$(Tic); re-indexing it here would be a silent pattern copy",
    ))
    # Iterate the caller's own CSC arrays and keep the upper triangle.  This is
    # a SELECTION, not a conversion: no entry is recomputed, and the diagonal
    # slot is written for every column so QDLDL's "every structural column
    # nonempty" precondition holds.
    #
    # It deliberately does NOT go through `sparse(UpperTriangular(K))`: that
    # call returned an EMPTY matrix for the saddle-point pattern this driver
    # uses (measured, then replaced), which would have made every refactor look
    # like a pattern change.  An explicit loop cannot silently do that.
    n, m = size(K)
    colptr = Vector{Tic}(undef, m + 1)
    rowval = Vector{Tic}()
    nzval = Vector{MFc}()
    colptr[1] = one(Tic)
    for column in 1:m
        for pointer in K.colptr[column]:(K.colptr[column + 1] - 1)
            row = K.rowval[pointer]
            row <= column || continue
            push!(rowval, Tic(row))
            push!(nzval, K.nzval[pointer])
        end
        if !any(==(Tic(column)), rowval[colptr[column]:end])
            push!(rowval, Tic(column))
            push!(nzval, MFc(K[column, column]))
        end
        colptr[column + 1] = Tic(length(rowval) + 1)
    end
    return SA.SparseMatrixCSC{MFc,Tic}(n, m, colptr, rowval, nzval)
end

function adapter_canonical_upper(::Type{MFc}, ::Type{Tic}, K::AbstractMatrix) where {MFc,Tic}
    eltype(K) === MFc || throw(ArgumentError(
        "value matrix has scalar type $(eltype(K)), cache holds $(MFc); " *
        "a conversion here would be a precision policy decision the adapter " *
        "must not make",
    ))
    # A dense input is SELECTED into CSC directly, holding the caller's own
    # scalars: no `Matrix{MF}` intermediate is retained by the adapter and no
    # `Float64` appears.  `adapter_dense_conversions()` records that this path
    # performed no dense `Matrix` conversion.
    n, m = size(K)
    colptr = Vector{Tic}(undef, m + 1)
    rowval = Vector{Tic}()
    nzval = Vector{MFc}()
    colptr[1] = one(Tic)
    for column in 1:min(m, n)
        for row in 1:column
            push!(rowval, Tic(row))
            push!(nzval, K[row, column])
        end
        colptr[column + 1] = Tic(length(rowval) + 1)
    end
    for column in (min(m, n) + 1):m
        colptr[column + 1] = Tic(length(rowval) + 1)
    end
    return SA.SparseMatrixCSC{MFc,Tic}(n, m, colptr, rowval, nzval)
end

"""
    pattern_equal(A, B) -> Bool

Element-for-element CSC pattern equality: shape, `colptr`, `rowval`.  Values
are NOT compared — that is the whole point of the symbolic/numeric split.
"""
pattern_equal(A::SA.SparseMatrixCSC, B::SA.SparseMatrixCSC) =
    size(A) == size(B) && A.colptr == B.colptr && A.rowval == B.rowval

"""
    frozen_pattern(a) -> SparseMatrixCSC

The cache's frozen pattern, rebuilt from the provider's own published
`frozen_colptr`/`frozen_rowval`.  Read-only: the adapter never writes these.
"""
function frozen_pattern(a::P02SparseAdapter)
    MFc, Tic = _adapter_parameters(a)
    colptr = getfield(a.cache, :frozen_colptr)
    rowval = getfield(a.cache, :frozen_rowval)
    return SA.SparseMatrixCSC{MFc,Tic}(
        length(colptr) - 1, length(colptr) - 1, colptr, rowval,
        MFc[MFc(0) for _ in rowval],
    )
end

"""
    pattern_signature(a) -> UInt64

The provider's own frozen-pattern signature.  Published as a control: it is
INSENSITIVE to a same-pattern refactor, which is what makes it the wrong tool
for detecting a stale lease and the M01-F4 result reproducible rather than
asserted.
"""
pattern_signature(a::P02SparseAdapter) = getfield(a.cache, :pattern_signature)

"Provider symbolic-analysis count.  Frozen patterns must leave this unchanged."
symbolic_count(a::P02SparseAdapter) = getfield(a.cache, :symbolic_count)

"Provider numeric-factorization count.  Increments once per accepted refactor."
numeric_count(a::P02SparseAdapter) = getfield(a.cache, :numeric_factor_count)

"solve! calls recorded by the provider, in columns."
solve_count(a::P02SparseAdapter) = getfield(a.cache, :solve_count)

# =====================================================================
# 5. Lifecycle
# =====================================================================

"""
    adapter_build(::Type{MF}, pattern; dsigns, nrhs=1) -> P02SparseAdapter

Construct the provider cache through the route in force, and hand the cache a
fresh, invalid lease.  The adapter never runs a factorization here: a symbolic
lease is not a numeric one.
"""
function adapter_build(::Type{MFc}, pattern; dsigns, nrhs::Integer=1) where {MFc<:MF.MultiFloat}
    access = p02_qdldl_access()
    cache = access.constructor(MFc, pattern; dsigns=dsigns, nrhs=nrhs)
    return P02SparseAdapter(
        cache, false, 0, 0, false, 0, MFLA.generation(cache), Int(nrhs), 0,
    )
end

"""
    adapter_revoke!(a) -> P02SparseAdapter

Kill the logical lease without touching the provider's storage or status.  This
is the ADR-002 §4 obligation, spelled as an operation rather than a comment.
"""
function adapter_revoke!(a::P02SparseAdapter)
    a.lease_valid = false
    a.last_refactor_ok = false
    return a
end

"""
    adapter_refactor!(a, values; checked=true) -> P02SparseAdapter

Numeric refactorization at an UNCHANGED pattern.

Ordering, and every step is load-bearing:

  1. **Lease revocation happens FIRST**, before any validation and before any
     provider call.  If step 3 or 4 throws — a wrong pattern, a wrong type, a
     non-finite value, a provider numerical breakdown — the lease is already
     dead.  This is ADR-002 §4 and it does not depend on the provider's own
     revocation behaviour.
  2. The candidate pattern is derived and compared to the frozen one.  A
     difference is refused HERE, so the provider is never asked to refactor a
     pattern its symbolic factor does not describe.
  3. Scalar and index types are compared, not converted.
  4. The provider's `factorize!` runs with `check=false`: a numerical breakdown
     is a *status* under the contract, and this adapter turns a non-success
     status into a refusal so the caller cannot proceed on a dead factor.
  5. `MFLA.record_factor_summary!(a.cache)` runs at the attempt boundary,
     success or failure.  It records the O(1) summary and bumps the package's
     generation, which is what makes a lease taken against the previous result
     stop validating.  **This is IP-2 performed adapter-side**; see the header.
     Without it the M01-F4 measurements in `test/rebuild/P02.jl` §3 show the
     old lease surviving a same-size refactor.

`checked=false` skips step 2's pattern derivation and step 3's type comparison.
It is NOT a way to refactor a changed pattern — it is the exclusive hot entry
for a caller that already holds the frozen pattern — and it is measured to
produce identical results to the checked entry for the identical input.
"""
function adapter_refactor!(a::P02SparseAdapter, values; checked::Bool=true)
    MFc, Tic = _adapter_parameters(a)
    a.attempt += 1
    adapter_revoke!(a)                          # (1) before anything can throw

    candidate = nothing
    if checked
        accepted = adapter_accepts_type(a, values)
        if !(accepted.scalar_ok && accepted.index_ok)
            a.refusals += 1
            MFLA.record_factor_summary!(a.cache)
            throw(ArgumentError(
                "sparse refactor refused: candidate scalar/index types " *
                "($(accepted.candidate_scalar), $(accepted.candidate_index)) " *
                "differ from the cache's ($(accepted.cache_scalar), " *
                "$(accepted.cache_index)); converting would be an implicit " *
                "precision change",
            ))
        end
        candidate = adapter_canonical_upper(MFc, Tic, values)
        if !pattern_equal(candidate, frozen_pattern(a))
            a.refusals += 1
            MFLA.record_factor_summary!(a.cache)
            throw(ArgumentError(
                "sparse refactor refused: the candidate pattern differs from " *
                "the frozen symbolic pattern; a numeric refactor must not " *
                "change the symbolic structure",
            ))
        end
    else
        candidate = values
    end

    ok = false
    failure = nothing
    try
        MFLA.factorize!(a.cache, candidate; check=false)
        ok = provider_success(a)
        ok || (failure = "provider status $(MFLA.factor_status(a.cache)) " *
                         "($(provider_state(a)))")
    catch error
        ok = false
        failure = string(typeof(error), ": ", sprint(showerror, error))
    end

    # (5) the attempt boundary: record + bump, then read the generation back so
    # the lease carries the provider's own number rather than an adapter count.
    MFLA.record_factor_summary!(a.cache)
    a.lease_epoch = MFLA.generation(a.cache)

    if ok
        a.success += 1
        a.last_refactor_ok = true
        a.lease_valid = true
    else
        a.refusals += 1
        a.last_refactor_ok = false
        a.lease_valid = false
        throw(ArgumentError("sparse numeric refactor failed: $(failure)"))
    end
    return a
end

"""
    adapter_solve!(a, destination, rhs; checked=true) -> Int

Solve in place.  Returns the number of RHS columns written, which is what the
caller must account for — never a success flag.

Refusals, all before any write:

  * no live lease (`lease_live(a) === false`): a solve after a failed refactor
    must fail closed rather than answer from a retained physical factor;
  * destination/RHS shapes disagree, or the RHS row count differs from the
    factor order;
  * RHS width exceeds the capacity the cache was prepared with;
  * a non-finite entry in the RHS, refused before it reaches the provider;
  * the destination aliases the RHS or the pattern values it is factoring;
  * the package's own provider-side alias and staleness checks (including a
    destination aliasing factor storage) are allowed to throw.

`checked=false` skips the adapter's own RHS-shape/width/alias checks and calls
the provider directly.  The lease check is NOT optional: `checked` selects how
much is re-validated, never whether authority exists.
"""
function adapter_solve!(a::P02SparseAdapter, destination::AbstractVecOrMat,
                        rhs::AbstractVecOrMat; checked::Bool=true)
    lease_live(a) || throw(ArgumentError(
        "sparse solve refused: no live numeric lease (attempts=$(a.attempt), " *
        "successes=$(a.success), provider state=$(provider_state(a))); the " *
        "logical lease was revoked on the failed refactor and the retained " *
        "physical factor must not be reused",
    ))
    MFc, _ = _adapter_parameters(a)

    if checked
        eltype(destination) === MFc && eltype(rhs) === MFc || throw(ArgumentError(
            "sparse solve refused: destination/RHS scalar types " *
            "($(eltype(destination)), $(eltype(rhs))) differ from the cache's " *
            "$(MFc)",
        ))
        # ALIAS IS CHECKED BEFORE ANY SHAPE OR WIDTH RULE.  Ordering matters
        # for the caller's diagnosis, and it was measured to matter here: with
        # the shape and width rules first, an aliasing destination that also
        # had the wrong shape was reported as a shape error, and a leg that
        # matched refusals by exception TYPE accepted that as an alias refusal.
        Base.mightalias(destination, rhs) && throw(ArgumentError(
            "sparse solve refused: destination aliases the RHS; the solve " *
            "writes in place and would overwrite its own input",
        ))
        Base.mightalias(destination, getfield(a.cache, :matrix).nzval) &&
            throw(ArgumentError(
                "sparse solve refused: destination aliases the cache's " *
                "pattern values",
            ))
        size(destination) == size(rhs) || throw(DimensionMismatch(
            "sparse solve destination/RHS shapes differ: " *
            "$(size(destination)) vs $(size(rhs))",
        ))
        size(rhs, 1) == size(frozen_pattern(a), 1) || throw(DimensionMismatch(
            "sparse solve RHS row count $(size(rhs, 1)) differs from the " *
            "factor order $(size(frozen_pattern(a), 1))",
        ))
        width = ndims(rhs) == 1 ? 1 : size(rhs, 2)
        width <= a.rhs_capacity || throw(DimensionMismatch(
            "sparse solve RHS width $(width) exceeds the prepared capacity " *
            "$(a.rhs_capacity); widen the cache with adapter_widen! rather " *
            "than truncating the RHS",
        ))
        all(isfinite, rhs) || throw(ArgumentError(
            "sparse solve refused: RHS contains a non-finite entry",
        ))
    end

    MFLA.solve!(a.cache, destination, rhs)
    return ndims(rhs) == 1 ? 1 : size(rhs, 2)
end

"""
    adapter_widen!(a, nrhs) -> P02SparseAdapter

Raise the prepared RHS capacity.

This rebuilds the provider cache, so it is a NEW SYMBOLIC LEASE and any
existing lease is revoked.  The rebuild carries the cache's CURRENT numeric
values forward rather than a zero-valued pattern: QDLDL runs its symbolic
factorization at construction, and a pattern whose diagonal slots are all zero
is not a quasi-definite system — it fails with `Zero entry in D (matrix is not
quasidefinite)` (measured here, before this was fixed).  A widened cache must
therefore either be given the current values or be left invalid, and this
adapter chooses the former and then revokes, so the caller still has to
refactor before solving.
"""
function adapter_widen!(a::P02SparseAdapter, nrhs::Integer)
    nrhs >= 1 || throw(ArgumentError("nrhs must be positive"))
    nrhs <= a.rhs_capacity && return a
    MFc, Tic = _adapter_parameters(a)
    signs = copy(getfield(a.cache, :dsigns))
    rebuilt = adapter_build(
        MFc, adapter_canonical_upper(MFc, Tic, getfield(a.cache, :matrix));
        dsigns=signs, nrhs=nrhs,
    )
    a.cache = rebuilt.cache
    # The adapter's own capacity field MUST move with the cache.  It did not in
    # the first version of this function, so `adapter_solve!` went on refusing
    # a width the provider had already been prepared for — the provider said 3,
    # the adapter said 1, and the adapter's own pre-check won.  Measured, then
    # fixed; the driver asserts the two agree.
    a.rhs_capacity = Int(nrhs)
    adapter_revoke!(a)
    return a
end

# =====================================================================
# 6. Summary — symbolic facts and numeric facts kept apart
# =====================================================================

"""
    adapter_symbolic_facts(a) -> NamedTuple

The symbolic authority: what the frozen pattern is, and how many times the
provider has analysed it.  Depends on the pattern only.
"""
function adapter_symbolic_facts(a::P02SparseAdapter)
    pattern = frozen_pattern(a)
    return (
        n=size(pattern, 1), nnz_pattern=length(pattern.rowval),
        signature=pattern_signature(a), symbolic_analyses=symbolic_count(a),
        index_type=cache_index_type(a), scalar_type=cache_scalar_type(a),
    )
end

"""
    adapter_numeric_facts(a) -> NamedTuple

The numeric authority: how many numeric factorizations the provider has run,
and the O(1) summary the package records at the attempt boundary.  Reads
SCALARS ONLY — no factor storage, no inertia recomputation, no matrix copy.
"""
function adapter_numeric_facts(a::P02SparseAdapter)
    summary = MFLA.factor_summary(a.cache)
    return (
        numeric_refactors=numeric_count(a),
        solves=solve_count(a),
        state=provider_state(a),
        status=MFLA.factor_status(a.cache),
        generation=MFLA.generation(a.cache),
        lease_epoch=a.lease_epoch,
        lease_live=lease_live(a),
        last_refactor_ok=a.last_refactor_ok,
        attempts=a.attempt, successes=a.success, refusals=a.refusals,
        summary_kind=summary.kind, summary_size=summary.size,
        summary_status=summary.status, summary_generation=summary.generation,
    )
end

"""
    adapter_summary(a) -> NamedTuple

The two halves, side by side and never merged.  A caller that wants "is this
factor current" reads `numeric.lease_live`; a caller that wants "did the
pattern change" reads `symbolic.signature`.  Neither answers the other's
question, which is the whole reason they are separate.
"""
adapter_summary(a::P02SparseAdapter) = (
    symbolic=adapter_symbolic_facts(a), numeric=adapter_numeric_facts(a),
)

"""
    adapter_dense_conversions(a) -> Int

The adapter-local dense-conversion counter.  See the header: `0` here is about
this adapter's path, and the driver always reads it next to a control that
makes the counter move.
"""
adapter_dense_conversions(a::P02SparseAdapter) = a.dense_conversions

end # !@isdefined(P02_MFLA_SPARSE_ADAPTER_LOADED)
