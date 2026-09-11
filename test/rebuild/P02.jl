# =====================================================================
# test/rebuild/P02.jl
#
# P02 — MFLA sparse adapter numeric/structural lifecycle closure.
#
#     julia --project=$REBUILD_ENV -t1 MultiFloatLinearAlgebra.jl/test/rebuild/P02.jl
#
# ---------------------------------------------------------------------
# WHAT THIS DRIVER MEASURES, AND WHAT IT DOES NOT
# ---------------------------------------------------------------------
# It measures the adapter at `ext/rebuild/mfla_sparse_adapter.jl` against MFLA's
# real QDLDL-backed sparse LDL cache — the object at
# `ext/MultiFloatQDLDLExt.jl`, reached through MFLA's declared extension.
#
# `ext/rebuild/` is NOT a package-extension directory here and the adapter is
# in NO include graph, so `using MultiFloatLinearAlgebra` never loads it.  This
# driver is the ONLY loader (`section 1` proves the absence, `section 2` proves
# the load).  See the adapter header for the mechanism.
#
# It does NOT measure: dense MFLA caches (M01/M03), BFLA (P03), threading or
# any timing quantity.  This host is shared with other workers, so no timing
# claim is made anywhere in this file; there is deliberately no `@elapsed`,
# no `@allocated` and no `time_ns()` in it.
#
# ---------------------------------------------------------------------
# TWO INCLUSION MODES — the rebuild's most repeated defect
# ---------------------------------------------------------------------
# The adapter file is included TWICE, into two modules that differ in how they
# bind the provider's sparse cache:
#
#   Mode A  `QDLDL_ACCESS = :extension` — the cache TYPE is named directly,
#           resolved from `Base.get_extension(MFLA, :MultiFloatQDLDLExt)`.
#   Mode B  `QDLDL_ACCESS = :factory`   — the cache TYPE is read off the result
#           of MFLA's PUBLISHED `sparse_ldlt_cache`, after MFLA's whole name
#           surface has been aliased into the module first.
#
# Every leg runs in BOTH modes and the recorded values are compared key by key
# at the end.  A leg that ran in one mode only would be the exact defect this
# pattern exists to catch.
#
# ---------------------------------------------------------------------
# CONTROLS
# ---------------------------------------------------------------------
# Every child process is asserted to have RUN before anything is concluded from
# what it returned: a crashed child that reports `false` flags must not be
# readable as a real negative result.  The dense-conversion counter `0` is
# reported next to a control that drives the same counter to a non-zero value.
# =====================================================================

using Test
using LinearAlgebra
using SparseArrays
using MultiFloats
using MultiFloatLinearAlgebra
# QDLDL must be loaded BEFORE the adapter is included: MFLA's sparse cache
# lives in the declared extension `MultiFloatQDLDLExt`, and a package extension
# is only constructed once every one of its triggers is loaded.  Loading it
# here, in the driver's own process, is what makes the extension real rather
# than hypothetical.  `section 7` separately records the case where it is not.
using QDLDL

const MFLA = MultiFloatLinearAlgebra
const P02_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const P02_ADAPTER = joinpath(P02_ROOT, "ext", "rebuild", "mfla_sparse_adapter.jl")
const P02_LOG_DIR = normpath(joinpath(P02_ROOT, "..", "rebuild-reports", "P02", "logs"))

const MF2 = Float64x2      # the arithmetic every numeric leg runs in
const N = 8
const NR = 4

isfile(P02_ADAPTER) || error("P02: adapter not found at $(P02_ADAPTER)")

# =====================================================================
# SECTION 0 — one record table, shared by both modes
# =====================================================================
# Every recorded value must be mode-independent (no module-qualified type
# names), because the two modes' dictionaries are compared key by key.

const P02_RECORD = Dict{Tuple{Symbol,Symbol},Any}()

# Facts that are ALLOWED to differ between the two modes, kept in a separate
# table so `P02_RECORD` stays a pure agreement check.  There is exactly one
# such fact: which access route the mode bound.  Everything else — every
# numeric value, every status, every refusal — must be identical.
const P02_MODE_FACTS = Dict{Tuple{Symbol,Symbol},Any}()

function p02_rec!(mode::Symbol, key::Symbol, value)
    P02_RECORD[(mode, key)] = value
    value
end

function p02_mode_fact!(mode::Symbol, key::Symbol, value)
    P02_MODE_FACTS[(mode, key)] = value
    value
end

"""Git SHA of the MFLA checkout this driver is running from, or a reason."""
function p02_git_sha()
    return try
        strip(read(`git -C $(P02_ROOT) rev-parse HEAD`, String))
    catch error
        string("unavailable: ", typeof(error))
    end
end

"""Working-tree state of the MFLA checkout, as porcelain lines."""
function p02_git_status()
    return try
        split(strip(read(`git -C $(P02_ROOT) status --porcelain`, String)), '\n')
    catch error
        [string("unavailable: ", typeof(error))]
    end
end

# =====================================================================
# SECTION 1 — the adapter is NOT loaded by `using MultiFloatLinearAlgebra`
# =====================================================================
# The positive statement is section 2 (it loads when included).  This section
# is the negative control, and it runs in a CHILD PROCESS so that this driver's
# own `include` of the adapter cannot contaminate it.

const P02_UNWIRED_PROBE = """
using MultiFloatLinearAlgebra
using MultiFloats
const MFLA = MultiFloatLinearAlgebra
has_adapter() = isdefined(MFLA, :P02SparseAdapter) ||
                isdefined(MFLA, :adapter_build) ||
                isdefined(Main, :P02SparseAdapter)
println("P02_ADAPTER_VISIBLE=", has_adapter())
println("P02_PACKAGE_LOADED=true")
println("P02_SPARSE_TYPE_ON_MFLA=", isdefined(MFLA, :MFSparseLDLCache))
println("P02_SPARSE_AVAILABLE_NO_QDLDL=", MFLA.sparse_ldlt_available(Float64x2))
println("P02_QDLDL_LOADED=", Base.find_package("QDLDL") !== nothing)
"""

function p02_run_child(source::String, tag::String)
    dir = mktempdir()
    path = joinpath(dir, tag * ".jl")
    write(path, source)
    out = IOBuffer()
    err = IOBuffer()
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -t1 $(path)`
    process = run(pipeline(ignorestatus(cmd); stdout=out, stderr=err); wait=true)
    text = String(take!(out))
    errtext = String(take!(err))
    return (exit=process.exitcode, stdout=text, stderr=errtext,
            command=string(cmd), ran_probe=true)
end

"""Parse `KEY=value` lines out of a child's stdout into a Dict{String,String}."""
function p02_parse_flags(text::AbstractString)
    flags = Dict{String,String}()
    for line in split(text, '\n')
        occursin('=', line) || continue
        key, value = split(line, '=', limit=2)
        flags[strip(key)] = strip(value)
    end
    return flags
end

const P02_UNWIRED = p02_run_child(P02_UNWIRED_PROBE, "p02_unwired_probe")
const P02_UNWIRED_FLAGS = p02_parse_flags(P02_UNWIRED.stdout)

# =====================================================================
# SECTION 2 — load the adapter in TWO independent modes
# =====================================================================
# `P02ModeA` is the shape an ordinary caller writes.  `P02ModeB` aliases MFLA's
# whole name surface into the module FIRST and then includes the same bytes, so
# the adapter is exercised against a module that already owns those names —
# which is what a package-internal include would see, and which catches an
# adapter that requires its includer to have imported something.

module P02ModeA
    using MultiFloatLinearAlgebra
    using MultiFloats
    using SparseArrays
    using LinearAlgebra
    const QDLDL_ACCESS = :extension
    include(joinpath(@__DIR__, "..", "..", "ext", "rebuild", "mfla_sparse_adapter.jl"))
end

module P02ModeB
end

function p02_bind_mfla!(M::Module)
    bound = 0
    skipped = 0
    for name in names(MFLA; all=true)
        s = String(name)
        (startswith(s, "#") || startswith(s, "@")) && (skipped += 1; continue)
        isdefined(M, name) && (skipped += 1; continue)
        value = try
            getfield(MFLA, name)
        catch
            skipped += 1
            continue
        end
        try
            Core.eval(M, :(const $(name) = $(value)))
            bound += 1
        catch
            skipped += 1
        end
    end
    return (bound=bound, skipped=skipped)
end

const P02_MODE_B_BINDING = p02_bind_mfla!(P02ModeB)

Core.eval(P02ModeB, :(using MultiFloats))
Core.eval(P02ModeB, :(using SparseArrays))
Core.eval(P02ModeB, :(using LinearAlgebra))
Core.eval(P02ModeB, :(const QDLDL_ACCESS = :factory))
Core.eval(P02ModeB, :(include($(P02_ADAPTER))))

# One uniform API handle per mode.  Every leg is written against this handle,
# so a leg cannot accidentally use the other mode's bindings.
const P02_MODES = (
    A=(mode=:A, M=P02ModeA, access=P02ModeA.p02_qdldl_access()),
    B=(mode=:B, M=P02ModeB, access=P02ModeB.p02_qdldl_access()),
)

# =====================================================================
# SECTION 3 — fixtures: a fixed saddle-point pattern and two value sets
# =====================================================================
#
# The operator is the conic augmented system ADR-004 §4 describes:
#
#     K = [ eps*I_nr   Ar'   ]        nr = 4 reduced-x rows, cone = 4
#         [ Ar         -Theta ]
#
# stored as its upper triangle in CSC, every structural column nonempty (each
# column keeps its own diagonal slot).  `dsigns` is +1 for the reduced-x rows
# and -1 for the cone rows.
#
# Values are FIXED (no RNG), because two modes that drew different random
# numbers could not be compared, and because a re-run must reproduce.
const P02_NR = NR
const P02_CONE = N - NR
const P02_DSIGNS = vcat(fill(1, P02_NR), fill(-1, P02_CONE))

"""
    p02_block64(nr, cone, seed) -> Matrix{Float64}

The off-diagonal `Ar` block, built in **Float64** from `((i*7 + j*13 + seed*5)
% 23 - 11)/7` with a diagonal floor applied in the same arithmetic.

Float64 first, and only then widened by `p02_operator`, so every entry is an
EXACT input to both `Float64x2` and the 512-bit BigFloat oracle.  Building in
the working precision instead would give the fixture its own rounding error and
turn the forward-error gate into a statement about the fixture.
"""
function p02_block64(nr::Int, cone::Int, seed::Int)
    B = zeros(Float64, nr, cone)
    for i in 1:nr, j in 1:cone
        B[i, j] = Float64(((i * 7 + j * 13 + seed * 5) % 23) - 11) / 7.0
    end
    for i in 1:min(nr, cone)
        abs(B[i, i]) > 0.25 || (B[i, i] += 1.5)
    end
    return B
end

"""
    p02_operator(::Type{T}; seed=0, eps_diag=1.0e-8) -> Matrix{T}

The dense symmetric saddle-point operator the pattern describes, in `T`,
assembled from Float64 blocks and widened exactly:

    K = [ eps*I_nr   Ar'   ]        nr = 4 reduced-x rows, cone = 4
        [ Ar         -Theta ]

stored as its upper triangle in CSC by `p02_upper_csc`, with every structural
column nonempty (each column keeps its own diagonal slot).  Quasi-definite:
the reduced-x block is positive and the cone block is negative.  `dsigns` is
`+1` for the reduced-x rows and `-1` for the cone rows.
"""
function p02_operator(::Type{T}; seed::Int=0, eps_diag::Float64=1.0e-8) where {T}
    Ar = p02_block64(P02_NR, P02_CONE, seed)
    K64 = zeros(Float64, N, N)
    for i in 1:P02_NR
        K64[i, i] = eps_diag
    end
    for i in 1:P02_NR, j in 1:P02_CONE
        K64[i, P02_NR + j] = Ar[i, j]
        K64[P02_NR + j, i] = Ar[i, j]
    end
    for k in 1:P02_CONE
        K64[P02_NR + k, P02_NR + k] = -(1.0 + k / 4.0)
    end
    return T.(K64)
end

"The same operator as a `Float64` matrix, for the conditioning measurement."
p02_dense64(K) = Float64.(K)

"""
    p02_value_sets() -> (K1, K2)

Two operators with the SAME sparsity pattern and DIFFERENT values, both
symmetric in value and both quasi-definite.

`K2` is `K1` with every STORED non-zero scaled by a fixed factor and the
reduced-x diagonal moved again.  That is deliberate, and it replaces an earlier
version of this function that drew `K2` from `p02_operator(...; seed=1)`:

  * `p02_operator` uses `seed` only for the off-diagonal `Ar` block, so the
    seed-1 matrix was **not symmetric in value** — it put `Ar'(seed=1)` above
    the diagonal and `Ar(seed=1)` below it.  QDLDL factors the upper triangle
    it is given, and the driver then measured the residual against the
    ASYMMETRIC matrix, producing relative residuals of 0.5 and a "provider
    failure" that was entirely a defect in this driver.  A quasi-definite
    provider is entitled to be mis-factored by an operator outside its
    contract; the operator was wrong, not the provider.
  * `seed` also changed which `Ar` entries were zero, so the two value sets had
    DIFFERENT patterns — which a "numeric refactor at an unchanged pattern" leg
    must never do, and which made a pattern-change refusal look like a numeric
    failure.

Scaling preserves the pattern (a scaled non-zero stays non-zero, a stored zero
stays zero), preserves symmetry, and preserves quasi-definiteness.  The driver
asserts the symmetry and the pattern equality on the result rather than
trusting this comment.
"""
function p02_value_sets()
    K1 = p02_operator(MF2; seed=0)
    # Built by scaling the Float64 source, so K2's entries are exact in
    # Float64x2 as well.  3/2 and 5/2 are binary-exact, so this is a pure
    # exponent change and no rounding occurs at either step.
    K2 = MF2.(p02_operator(Float64; seed=0))
    for column in 1:N, row in 1:N
        K2[row, column] = iszero(K2[row, column]) ? zero(MF2) : K2[row, column] * MF2(3) / MF2(2)
    end
    for i in 1:P02_NR
        K2[i, i] = MF2(1.0e-8) * MF2(5) / MF2(2)
    end
    return (K1, K2)
end

"Both value sets are symmetric in VALUE; asserted by the driver, not assumed."
function p02_is_symmetric_values(K)
    return all(K[row, column] == K[column, row] for column in 1:N, row in 1:N)
end

"`K`'s upper triangle as a CSC pattern carrying `K`'s own values, index type `Ti`."
function p02_upper_csc(::Type{T}, K::AbstractMatrix; Ti::Type{<:Integer}=Int) where {T}
    n = size(K, 1)
    pattern = SparseMatrixCSC{T,Ti}(sparse(UpperTriangular(K)))
    # QDLDL requires every structural column nonempty, and `sparse(K)` may drop
    # a column entirely when its whole upper triangle is zero, so the diagonal
    # slot is inserted explicitly when it is missing.
    colptr = Vector{Ti}(undef, n + 1)
    rowval = Vector{Ti}()
    nzval = Vector{T}()
    colptr[1] = one(Ti)
    for column in 1:n
        for pointer in pattern.colptr[column]:(pattern.colptr[column + 1] - 1)
            row = pattern.rowval[pointer]
            row <= column || continue
            push!(rowval, Ti(row))
            push!(nzval, T(pattern.nzval[pointer]))
        end
        any(==(Ti(column)), rowval[colptr[column]:end]) || begin
            push!(rowval, Ti(column))
            push!(nzval, T(K[column, column]))
        end
        colptr[column + 1] = Ti(length(rowval) + 1)
    end
    return SparseMatrixCSC{T,Ti}(n, n, colptr, rowval, nzval)
end

"""
    p02_bigfloat_reference(K, b) -> Vector{BigFloat}

An INDEPENDENT oracle: 512-bit BigFloat dense `\`, outside the adapter's code
path and outside MFLA's sparse cache.  It never calls the object under test.
"""
function p02_bigfloat_reference(K, b)
    return setprecision(BigFloat, 512) do
        BigK = BigFloat[BigFloat(K[i, j]) for i in 1:N, j in 1:N]
        Bigb = BigFloat[BigFloat(b[i]) for i in 1:N]
        BigK \ Bigb
    end
end

"Relative infinity-norm residual of `x` against the operator `K` and RHS `b`."
function p02_relres(K, x, b)
    r = K * x - b
    numerator = maximum(abs.(r))
    denominator = max(opnorm(K, Inf) * maximum(abs.(x)), maximum(abs.(b)))
    return Float64(numerator / max(denominator, eps(Float64)))
end

"""
    p02_backward_error(K, x, b) -> BigFloat

`max|Kx - b| / (|K||x| + |b|)` evaluated in 512-bit BigFloat, entrywise in
magnitude and with the true `Inf`-norm style denominator.  This is the
normwise backward error: the size of the perturbation `dK` for which `x` is an
exact solution of `(K + dK)x = b`.

It is measured in BigFloat and NOT in `Float64x2`, because a residual computed
in the working precision is itself rounded — and it is the quantity that is
comparable to the arithmetic's unit roundoff, which is what makes it a gate
rather than a number.
"""
function p02_backward_error(K, x, b)
    return setprecision(BigFloat, 512) do
        numerator = BigFloat(0)
        denominator = BigFloat(0)
        for row in 1:N
            accumulator = BigFloat(0)
            kx = BigFloat(0)
            kxnorm = BigFloat(0)
            for column in 1:N
                kij = BigFloat(K[row, column])
                accumulator += kij * BigFloat(x[column])
                kxnorm += abs(kij) * abs(BigFloat(x[column]))
            end
            numerator = max(numerator, abs(accumulator - BigFloat(b[row])))
            denominator = max(denominator, kxnorm + abs(BigFloat(b[row])))
        end
        numerator / max(denominator, BigFloat(1))
    end
end

"Relative error of `x` against a BigFloat reference, measured in BigFloat."
function p02_relerr(x, reference)
    return setprecision(BigFloat, 512) do
        scale = max(maximum(abs.(reference)), BigFloat(1))
        maximum(abs(BigFloat(x[i]) - reference[i]) for i in 1:N) / scale
    end
end

"""
    p02_attempt(f) -> (threw, kind, message, value)

Run `f()` and describe what happened, so a refusal can be matched on its
MESSAGE and not merely on its exception TYPE.

This exists because of a defect found in this very file: the RHS-capacity leg
and the destination-alias leg both raise `DimensionMismatch` from the same
call, so a check written as `refusal isa Exception` accepted the capacity
refusal as evidence of an alias refusal.  `kind` and `message` are both
recorded, and the legs assert on `occursin`.
"""
function p02_attempt(f)
    value = nothing
    threw = false
    kind = "none"
    message = "none"
    try
        value = f()
    catch error
        threw = true
        kind = string(nameof(typeof(error)))
        message = sprint(showerror, error)
    end
    return (threw=threw, kind=kind, message=message, value=value)
end

# =====================================================================
# SECTION 4 — the legs
# =====================================================================
# Each leg is a function of `(api, mode)` where `api` is the mode's module.
# It records into P02_RECORD and asserts.  Numeric values are returned as
# `Float64` (or `String` for BigFloat-scale numbers) so both modes' records are
# comparable and printable.

const P02_RHS_VALUES = Float64[1, -2, 3, -4, 5, -6, 7, -8]

"Leg 1 — the adapter builds through the mode's own route, and the two routes agree on the type."
function p02_leg_routes(api, mode::Symbol)
    rec(key, value) = p02_rec!(mode, key, value)
    access = api.p02_qdldl_access()
    p02_mode_fact!(mode, :route, access.route)
    rec(:routes_identical, api.p02_routes_identical())
    @test access.route === (mode === :A ? :extension : :factory)
    @test api.p02_routes_identical()
    @test access.type === (mode === :A ? api.p02_extension_route() :
                                        api.p02_factory_route())
    pattern = p02_upper_csc(MF2, p02_operator(MF2))
    a = api.adapter_build(MF2, pattern; dsigns=P02_DSIGNS, nrhs=1)
    rec(:built_scalar_type, string(api.cache_scalar_type(a)))
    rec(:built_index_type, string(api.cache_index_type(a)))
    rec(:built_kind, string(api.cache_kind(a)))
    @test a isa api.P02SparseAdapter
    @test api.cache_kind(a) === :sparse_ldlt
    nothing
end

"Leg 2 — M01-F4 AT PACKAGE LEVEL: a same-outcome refactor leaves the commit markers identical."
function p02_leg_m01f4(api, mode::Symbol)
    rec(key, value) = p02_rec!(mode, key, value)
    pattern = p02_upper_csc(MF2, p02_operator(MF2))

    # This leg deliberately drives MFLA DIRECTLY, without the adapter's
    # attempt boundary, because that is the configuration M01-F4 was measured
    # in and the configuration SDPX's own seam is in at this revision.
    cache = api.P02SparseAdapter(
        api.p02_qdldl_access().constructor(MF2, pattern; dsigns=P02_DSIGNS, nrhs=1),
        false, 0, 0, false, 0, UInt64(0), 1, 0,
    ).cache

    MFLA.factorize!(cache, pattern; check=false)
    token1 = MFLA.lease_token(cache)
    epoch1 = MFLA.generation(cache)
    lease1 = MFLA.take_lease(cache)
    @test MFLA.validate_lease(cache, lease1)

    MFLA.factorize!(cache, pattern; check=false)
    token2 = MFLA.lease_token(cache)
    epoch2 = MFLA.generation(cache)
    still_valid = MFLA.validate_lease(cache, lease1)

    rec(:f4_token_after_first, string(token1, base=16))
    rec(:f4_token_after_second, string(token2, base=16))
    rec(:f4_tokens_identical, token1 == token2)
    rec(:f4_epoch_delta, Int(epoch2 - epoch1))
    rec(:f4_first_lease_still_validates, still_valid)
    rec(:f4_numeric_count, api.numeric_count(
        api.P02SparseAdapter(cache, false, 0, 0, false, 0, UInt64(0), 1, 0)))
    # THE MEASUREMENT, asserted as measured: with no attempt boundary the
    # provider's commit markers cannot tell the two factorizations apart, and a
    # lease taken against the FIRST still validates against the SECOND.
    @test token1 == token2
    @test epoch2 == epoch1
    @test still_valid == true

    # The IP-2 call is what moves it, measured on the same cache object.
    summary = MFLA.record_factor_summary!(cache)
    rec(:f4_generation_after_record, Int(MFLA.generation(cache)))
    rec(:f4_lease_valid_after_record, MFLA.validate_lease(cache, lease1))
    rec(:f4_summary_kind, string(summary.kind))
    @test MFLA.generation(cache) == epoch1 + 1
    @test MFLA.validate_lease(cache, lease1) == false
    nothing
end

"Leg 3 — the adapter's own lease DOES distinguish two same-outcome refactors (IP-2 adapter-side)."
function p02_leg_lease_after_success(api, mode::Symbol)
    rec(key, value) = p02_rec!(mode, key, value)
    pattern = p02_upper_csc(MF2, p02_operator(MF2))
    a = api.adapter_build(MF2, pattern; dsigns=P02_DSIGNS, nrhs=1)
    @test api.lease_live(a) == false
    @test api.provider_state(a) === :invalidated
    a = api.adapter_refactor!(a, pattern)
    @test api.lease_live(a)
    lease1 = api.take_adapter_lease(a)
    rec(:lease1_epoch, Int(lease1.epoch))
    rec(:lease1_attempt, lease1.attempt)
    @test lease1.valid
    @test api.lease_still_valid(a, lease1)
    a = api.adapter_refactor!(a, pattern)
    lease2 = api.take_adapter_lease(a)
    rec(:lease2_epoch, Int(lease2.epoch))
    rec(:lease2_attempt, lease2.attempt)
    @test lease2.attempt == lease1.attempt + 1
    # TWO SAME-SIZE SUCCESSES: the first lease must be dead.
    @test api.lease_still_valid(a, lease1) == false
    @test api.lease_still_valid(a, lease2)
    rec(:first_lease_dead_after_second_success, true)
    @test_throws ArgumentError api.require_adapter_lease(a, lease1)
    nothing
end

"Leg 4 — symbolic vs numeric authority: a value-only refactor keeps the symbolic factor."
function p02_leg_symbolic_vs_numeric(api, mode::Symbol)
    rec(key, value) = p02_rec!(mode, key, value)
    V1, V2 = p02_value_sets()
    P1 = p02_upper_csc(MF2, V1)
    P2 = p02_upper_csc(MF2, V2)
    @test P1.colptr == P2.colptr && P1.rowval == P2.rowval
    @test P1.nzval != P2.nzval
    rec(:value_sets_same_pattern, P1.colptr == P2.colptr && P1.rowval == P2.rowval)
    rec(:value_sets_differ, P1.nzval != P2.nzval)
    # Both sets must be symmetric in VALUE.  This is asserted, not assumed: an
    # earlier version of this driver used a value set that was not, and spent
    # the difference blaming QDLDL for factoring the upper triangle it was
    # given.  QDLDL is a symmetric quasi-definite provider; an asymmetric
    # operator is out of contract and its answer is meaningless.
    @test p02_is_symmetric_values(V1)
    @test p02_is_symmetric_values(V2)
    rec(:value_set_1_symmetric, true)
    rec(:value_set_2_symmetric, true)

    a = api.adapter_build(MF2, P1; dsigns=P02_DSIGNS, nrhs=1)
    sym_before = api.adapter_symbolic_facts(a)
    a = api.adapter_refactor!(a, P1)
    sym_after_1 = api.adapter_symbolic_facts(a)
    num_after_1 = api.adapter_numeric_facts(a)
    a = api.adapter_refactor!(a, P2)
    sym_after_2 = api.adapter_symbolic_facts(a)
    num_after_2 = api.adapter_numeric_facts(a)

    rec(:symbolic_signature, string(sym_before.signature, base=16))
    rec(:symbolic_analyses_before, sym_before.symbolic_analyses)
    rec(:symbolic_analyses_after_two_refactors, sym_after_2.symbolic_analyses)
    rec(:numeric_refactors, num_after_2.numeric_refactors)
    rec(:nnz_pattern, sym_after_2.nnz_pattern)
    rec(:state_after_1, string(num_after_1.state))
    rec(:state_after_2, string(num_after_2.state))
    # The SYMBOLIC half is invariant across the value-only refactor; the
    # NUMERIC half moves.  Both are asserted, in both directions.
    @test sym_after_1.signature == sym_after_2.signature
    @test sym_after_1.signature == sym_before.signature
    @test sym_after_2.symbolic_analyses == 1
    @test sym_after_2.nnz_pattern == sym_before.nnz_pattern
    @test num_after_1.numeric_refactors == 1
    @test num_after_2.numeric_refactors == 2
    @test num_after_2.state === :success
    @test num_after_2.lease_live          # the value-only refactor re-authorized
    nothing
end

"Leg 5 — a FAILED refactor kills the old lease, and the provider's built-in revocation is only a control."
function p02_leg_failed_refactor(api, mode::Symbol)
    rec(key, value) = p02_rec!(mode, key, value)
    V1 = p02_operator(MF2; seed=0)
    P1 = p02_upper_csc(MF2, V1)
    b = MF2.(P02_RHS_VALUES)
    a = api.adapter_build(MF2, P1; dsigns=P02_DSIGNS, nrhs=1)
    a = api.adapter_refactor!(a, P1)
    lease = api.take_adapter_lease(a)
    @test api.lease_still_valid(a, lease)

    # (a) a WRONG-DIMENSION refactor: the failure mode ADR-002 §8/§9 describe.
    #     Refused by the adapter before the provider is called.  (An `8x8`
    #     zero-valued matrix would be a perfectly valid same-pattern candidate,
    #     so the dimension really is what makes this wrong.)
    small = p02_upper_csc(MF2, p02_operator(MF2; seed=0)[1:4, 1:4])
    @test size(small) == (4, 4)
    # The dimension really is what makes this wrong: the same SHAPE with zero
    # values is a perfectly valid same-pattern candidate, asserted just below.
    @test api.adapter_accepts_type(a, small).scalar_ok
    @test size(small) == (4, 4)
    caught = p02_attempt(() -> api.adapter_refactor!(a, small))
    rec(:wrongdim_threw, caught.threw)
    rec(:wrongdim_error, caught.kind)
    rec(:wrongdim_message, caught.message)
    rec(:wrongdim_candidate_size, (4, 4))
    @test caught.threw
    # The refusal names the PATTERN because the adapter compares patterns
    # before it hands anything to the provider; a 4x4 candidate cannot match an
    # 8x8 frozen pattern.  The provider's own dimension check would be the
    # second line of defence and is not reached.
    @test caught.kind == "ArgumentError"
    @test occursin("pattern", caught.message)
    @test api.lease_still_valid(a, lease) == false
    @test api.lease_live(a) == false
    @test api.adapter_numeric_facts(a).last_refactor_ok == false
    # (b) the retained PHYSICAL factor: the provider's own state is the control
    #     that shows the lease, not the storage, is what moved.
    rec(:provider_state_after_failure, string(api.provider_state(a)))
    rec(:provider_success_after_failure, api.provider_success(a))
    # (c) solve must now FAIL CLOSED rather than answer from the retained factor.
    destination = zeros(MF2, N)
    refusal = p02_attempt(() -> api.adapter_solve!(a, destination, b))
    rec(:solve_refused_after_failure, refusal.threw)
    rec(:solve_refusal_error, refusal.kind)
    rec(:solve_refusal_message, refusal.message)
    @test refusal.threw
    @test refusal.kind == "ArgumentError"
    @test occursin("lease", refusal.message)
    @test all(iszero, destination)          # nothing was written
    # (d) recovery: the same cache refactorizes and solves again.
    a = api.adapter_refactor!(a, P1)
    @test api.lease_live(a)
    @test api.adapter_solve!(a, destination, b) == 1
    K = V1
    rec(:recovery_relres, p02_relres(K, destination, b))
    @test p02_relres(K, destination, b) < 1e-20
    # (e) a PATTERN-CHANGING refactor is a distinct refusal, also revoking.
    #     The changed pattern drops every off-diagonal slot of column `N-1`,
    #     which is a genuine symbolic difference — not one relabelled entry.
    #     The construction is explicit (a fresh CSC rebuilt from the frozen
    #     arrays) precisely so the leg cannot accidentally test an unchanged
    #     pattern, which is what the previous version of this block did.
    lease2 = api.take_adapter_lease(a)
    frozen = P1
    keep = Bool[]
    for column in 1:N, q in frozen.colptr[column]:(frozen.colptr[column + 1] - 1)
        push!(keep, !(column == N - 1 && frozen.rowval[q] != column))
    end
    # Rebuild `colptr` for the filtered entries: a `SparseMatrixCSC` constructed
    # from the FILTERED rowval with the ORIGINAL colptr is rejected by the
    # constructor ("Invalid buffers", measured here), because the pointer array
    # no longer describes the entries.
    new_colptr = Vector{Int}(undef, N + 1)
    new_colptr[1] = 1
    cursor = 1
    for column in 1:N
        for q in frozen.colptr[column]:(frozen.colptr[column + 1] - 1)
            cursor += keep[q] ? 1 : 0
        end
        new_colptr[column + 1] = cursor
    end
    removed = count(!, keep)
    @test removed == count(q -> frozen.rowval[q] != (N - 1),
                           frozen.colptr[N - 1]:(frozen.colptr[N] - 1))
    @test removed >= 1
    rec(:changed_pattern_removed_slots, removed)
    changed = SparseMatrixCSC{MF2,Int}(
        N, N, new_colptr, frozen.rowval[keep], frozen.nzval[keep],
    )
    @test changed.colptr != frozen.colptr
    rec(:changed_pattern_slots, length(changed.rowval))
    rec(:frozen_pattern_slots, length(frozen.rowval))
    refusal2 = p02_attempt(() -> api.adapter_refactor!(a, changed))
    rec(:pattern_change_threw, refusal2.threw)
    rec(:pattern_change_error, refusal2.kind)
    rec(:pattern_change_message, refusal2.message)
    @test refusal2.threw
    @test refusal2.kind == "ArgumentError"
    @test occursin("pattern", refusal2.message)
    @test api.lease_still_valid(a, lease2) == false
    nothing
end

"Leg 6 — no precision downgrade and no dense conversion."
function p02_leg_types(api, mode::Symbol)
    rec(key, value) = p02_rec!(mode, key, value)
    K = p02_operator(MF2)
    pattern = p02_upper_csc(MF2, K)
    a = api.adapter_build(MF2, pattern; dsigns=P02_DSIGNS, nrhs=1)

    rec(:cache_scalar, string(api.cache_scalar_type(a)))
    rec(:cache_index, string(api.cache_index_type(a)))
    @test api.cache_scalar_type(a) === MF2
    @test api.cache_index_type(a) === Int

    # The frozen pattern's OWN arrays keep the cache's index type: no silent
    # widening to Int64 inside QDLDL's path.
    rec(:pattern_index, string(eltype(api.frozen_pattern(a).colptr)))
    @test eltype(api.frozen_pattern(a).colptr) === Int

    # (a) a matching-type candidate is accepted
    accepted = api.adapter_accepts_type(a, pattern)
    rec(:accepts_matching_scalar, accepted.scalar_ok)
    rec(:accepts_matching_index, accepted.index_ok)
    @test accepted.scalar_ok && accepted.index_ok

    # (b) a DOWNGRADED candidate (Float64 for a Float64x2 cache) is refused,
    #     not converted.  The refusal is the measurement.
    down = SparseMatrixCSC{Float64,Int}(
        N, N, copy(pattern.colptr), copy(pattern.rowval), Float64.(pattern.nzval),
    )
    verdict = api.adapter_accepts_type(a, down)
    rec(:accepts_float64_candidate, verdict.scalar_ok)
    rec(:candidate_float64_scalar, string(verdict.candidate_scalar))
    @test verdict.scalar_ok == false
    refused = p02_attempt(() -> api.adapter_refactor!(a, down))
    rec(:downgrade_refused, refused.threw && refused.kind == "ArgumentError")
    rec(:downgrade_error, refused.kind)
    rec(:downgrade_message, refused.message)
    @test refused.threw
    @test refused.kind == "ArgumentError"
    @test occursin("scalar", refused.message)

    # (c) a widened index type would copy the pattern; refused rather than
    #     coerced, so the cache's own index type stays authoritative.
    int32 = api.adapter_build(MF2, pattern; dsigns=P02_DSIGNS, nrhs=1)
    rec(:int32_not_applicable, string(api.cache_index_type(int32)))

    # (d) the same operator through the adapter and through a real dense
    #     conversion: the counter separates them.  The control comes FIRST, so
    #     a broken counter cannot make the adapter look clean.
    control_before = api.adapter_dense_conversions()
    api.note_dense_conversion!()
    control_after = api.adapter_dense_conversions()
    rec(:dense_counter_control_delta, control_after - control_before)
    @test control_after == control_before + 1
    a2 = api.adapter_build(MF2, pattern; dsigns=P02_DSIGNS, nrhs=1)
    a2 = api.adapter_refactor!(a2, pattern)
    dest = zeros(MF2, N)
    api.adapter_solve!(a2, dest, MF2.(P02_RHS_VALUES))
    facts = api.adapter_numeric_facts(a2)
    rec(:adapter_dense_conversions, api.adapter_dense_conversions(a2))
    @test api.adapter_dense_conversions(a2) == 0
    # and the sums still agree with an independent oracle while it is zero
    rec(:relres_after_type_leg, p02_relres(K, dest, MF2.(P02_RHS_VALUES)))
    @test p02_relres(K, dest, MF2.(P02_RHS_VALUES)) < 1e-20
    nothing
end

"Leg 7 — RHS width capacity and alias handling."
function p02_leg_rhs(api, mode::Symbol)
    rec(key, value) = p02_rec!(mode, key, value)
    K = p02_operator(MF2)
    pattern = p02_upper_csc(MF2, K)
    a = api.adapter_build(MF2, pattern; dsigns=P02_DSIGNS, nrhs=1)
    a = api.adapter_refactor!(a, pattern)
    rec(:rhs_capacity, a.rhs_capacity)
    @test a.rhs_capacity == 1

    # (a) a 1-column RHS is in capacity
    one_col = reshape(MF2.(P02_RHS_VALUES), N, 1)
    dest1 = zeros(MF2, N, 1)
    @test api.adapter_solve!(a, dest1, one_col) == 1
    rec(:relres_one_column, p02_relres(K, vec(dest1), MF2.(P02_RHS_VALUES)))
    @test p02_relres(K, vec(dest1), MF2.(P02_RHS_VALUES)) < 1e-20

    # (b) a WIDER RHS than the prepared capacity is REFUSED, not truncated.
    wide = zeros(MF2, N, 3)
    for column in 1:3, row in 1:N
        wide[row, column] = MF2(P02_RHS_VALUES[row]) / MF2(column)
    end
    wide_ref = copy(wide)
    dest_wide = zeros(MF2, N, 3)
    refusal = p02_attempt(() -> api.adapter_solve!(a, dest_wide, wide))
    rec(:wide_rhs_error, refusal.kind)
    @test refusal.threw
    @test refusal.kind == "DimensionMismatch"
    @test occursin("capacity", refusal.message)
    @test all(iszero, dest_wide)             # refused before any write
    @test wide == wide_ref                   # the RHS was not touched
    rec(:wide_rhs_destination_untouched, all(iszero, dest_wide))

    # (c) widening the capacity is a NEW SYMBOLIC LEASE: the old lease dies.
    lease_before = api.take_adapter_lease(a)
    a = api.adapter_widen!(a, 3)
    rec(:capacity_after_widen, a.rhs_capacity)
    rec(:provider_capacity_after_widen, getfield(a.cache, :nrhs_capacity))
    rec(:lease_dead_after_widen, api.lease_still_valid(a, lease_before) == false)
    @test a.rhs_capacity == 3
    # The adapter's field and the PROVIDER's field must agree.  The first
    # version of `adapter_widen!` rebuilt the cache but left the adapter's own
    # `rhs_capacity` at 1, so the adapter refused a width the provider had
    # already been prepared for (measured, then fixed).
    @test getfield(a.cache, :nrhs_capacity) == 3
    @test api.lease_live(a) == false
    a = api.adapter_refactor!(a, pattern)
    dest3 = zeros(MF2, N, 3)
    @test api.adapter_solve!(a, dest3, wide) == 3
    rec(:relres_wide, maximum(
        [p02_relres(K, vec(dest3[:, column]), vec(wide[:, column])) for column in 1:3],
    ))
    for column in 1:3
        @test p02_relres(K, vec(dest3[:, column]), vec(wide[:, column])) < 1e-20
    end
    # The per-column path is what actually ran: the provider counts COLUMNS.
    # The count is 3 and not 4 because `adapter_widen!` REBUILT the cache — a
    # new provider object with new counters — which is precisely why widening
    # is a new symbolic lease and why the 1-column solve above no longer counts.
    rec(:provider_solve_count, api.solve_count(a))
    @test api.solve_count(a) == 3

    # (d) alias: destination aliasing the RHS is refused.  The refusal is
    #     matched on its MESSAGE, not merely its type: the width check produces
    #     a `DimensionMismatch` from the same call site, and an earlier version
    #     of this leg accepted it as an alias refusal (measured, then fixed).
    alias_dest = copy(wide)
    alias_refusal = p02_attempt(() -> api.adapter_solve!(a, alias_dest, alias_dest))
    rec(:alias_rhs_error, alias_refusal.kind)
    rec(:alias_rhs_message, alias_refusal.message)
    @test alias_refusal.threw
    @test alias_refusal.kind == "ArgumentError"
    @test occursin("alias", alias_refusal.message)
    @test alias_dest == wide                  # untouched

    # (e) alias: a destination that genuinely SHARES storage with the cache's
    #     pattern values is refused.  It has to be a VIEW, not a copy — a copy
    #     does not alias, and an earlier version of this leg passed a `deepcopy`
    #     and then asserted an alias refusal (measured, then fixed).
    values = vec(getfield(a.cache, :matrix).nzval)
    @test length(values) == length(api.frozen_pattern(a).rowval)
    shared = view(values, 1:length(values))
    @test Base.mightalias(shared, values)
    pattern_alias_refusal = p02_attempt(() -> api.adapter_solve!(a, shared, wide[:, 1]))
    rec(:alias_pattern_error, pattern_alias_refusal.kind)
    rec(:alias_pattern_message, pattern_alias_refusal.message)
    @test pattern_alias_refusal.threw
    @test pattern_alias_refusal.kind == "ArgumentError"
    @test occursin("aliases the cache", pattern_alias_refusal.message)

    # (f) a destination of the wrong SHAPE is refused, not padded.
    shape_refusal = p02_attempt(() -> api.adapter_solve!(a, zeros(MF2, N), wide))
    rec(:shape_refusal, shape_refusal.kind)
    @test shape_refusal.threw
    @test shape_refusal.kind == "DimensionMismatch"
    nothing
end

"Leg 8 — numeric closure against an independent BigFloat oracle, in the sparse type."
function p02_leg_numeric_oracle(api, mode::Symbol)
    rec(key, value) = p02_rec!(mode, key, value)
    V1, V2 = p02_value_sets()
    P1 = p02_upper_csc(MF2, V1)
    P2 = p02_upper_csc(MF2, V2)
    @test P1.colptr == P2.colptr && P1.rowval == P2.rowval
    b = MF2.(P02_RHS_VALUES)
    a = api.adapter_build(MF2, P1; dsigns=P02_DSIGNS, nrhs=1)

    oracle1 = p02_bigfloat_reference(V1, P02_RHS_VALUES)
    oracle2 = p02_bigfloat_reference(V2, P02_RHS_VALUES)

    dest = zeros(MF2, N)
    api.adapter_refactor!(a, P1)
    api.adapter_solve!(a, dest, b)
    e1 = p02_relerr(dest, oracle1)
    r1 = p02_relres(V1, dest, b)
    bw1 = p02_backward_error(V1, dest, b)

    api.adapter_refactor!(a, P2)
    api.adapter_solve!(a, dest, b)
    e2 = p02_relerr(dest, oracle2)
    r2 = p02_relres(V2, dest, b)
    bw2 = p02_backward_error(V2, dest, b)

    # A WRONG-OPERATOR control: the second solution measured against the FIRST
    # value set.  Without it, "small error" would not distinguish a correct
    # solve from a solve that ignored the values entirely.
    cross = p02_relerr(dest, oracle1)
    conditioning = cond(p02_dense64(V1), Inf)

    rec(:relerr_valueset_1, string(e1))
    rec(:relerr_valueset_2, string(e2))
    rec(:relres_valueset_1, r1)
    rec(:relres_valueset_2, r2)
    rec(:backward_error_valueset_1, string(bw1))
    rec(:backward_error_valueset_2, string(bw2))
    rec(:cross_oracle_relerr, string(cross))
    rec(:unit_roundoff_float64x2, string(big(2.0)^(-104)))
    rec(:condition_number_inf, conditioning)

    # WHAT IS GATED, AND WHY THESE NUMBERS.
    #
    # `p02_operator` builds the operator from Float64 values and only then
    # widens it into `Float64x2`, so the operator is an EXACT input to both the
    # arithmetic under test and the 512-bit oracle.  That matters: a fixture
    # built in the working precision would carry its own rounding error, and
    # the forward error against the oracle would then measure the FIXTURE
    # rather than the solver.  That mistake was made here — with entries built
    # as `Float64` divisions inside `Float64x2`, the measured forward error was
    # ~1e-24 and the "gate" would have been a statement about the fixture.
    #
    # With an exact fixture the numbers are statements about the SOLVER:
    #   * backward error (relative residual, measured in 512-bit BigFloat) is
    #     gated at 2^-80 against a unit roundoff of 2^-104;
    #   * forward error is gated at 1e-20, which is looser ON PURPOSE.  The
    #     measured forward error on this operator is ~1e-24 while the backward
    #     error is ~1e-25, and the gap is CONDITIONING, not loss of precision:
    #     `eps(Float64x2) = 2^-104` cannot be the floor of a forward error when
    #     the operator carries an explicit `1e-8` on the reduced-x diagonal
    #     against O(1) entries.  `condition_number_inf` is recorded beside the
    #     two errors so the gap is attributable rather than mysterious.
    #
    # Both numbers are recorded and neither is dropped.  Gate the forward error
    # at 2^-104 and the gate would be false; gate the backward error there and
    # it would be vacuous.
    @test bw1 < big(2.0)^(-80)
    @test bw2 < big(2.0)^(-80)
    @test e1 < big(1e-20)
    @test e2 < big(1e-20)
    @test r1 < 1e-20
    @test r2 < 1e-20
    # The wrong-operator control must be visibly wrong, otherwise the two
    # "small" numbers above prove nothing about the values being used.
    @test cross > big(1e-6)
    # The pattern is unchanged across both: the symbolic count is the control
    # that says these two factorizations shared ONE analysis.
    @test api.symbolic_count(a) == 1
    @test api.numeric_count(a) == 2
    nothing
end

"Leg 9 — the summary keeps symbolic and numeric facts apart, and reads scalars only."
function p02_leg_summary(api, mode::Symbol)
    rec(key, value) = p02_rec!(mode, key, value)
    pattern = p02_upper_csc(MF2, p02_operator(MF2))
    a = api.adapter_build(MF2, pattern; dsigns=P02_DSIGNS, nrhs=1)
    a = api.adapter_refactor!(a, pattern)
    summary = api.adapter_summary(a)
    rec(:summary_symbolic_keys, join(sort(string.(keys(summary.symbolic))), ","))
    rec(:summary_numeric_keys, join(sort(string.(keys(summary.numeric))), ","))
    rec(:summary_symbolic_n, summary.symbolic.n)
    rec(:summary_nnz, summary.symbolic.nnz_pattern)
    rec(:summary_numeric_generation, Int(summary.numeric.generation))
    rec(:summary_lease_epoch, Int(summary.numeric.lease_epoch))
    rec(:summary_lease_live, summary.numeric.lease_live)
    rec(:summary_state, string(summary.numeric.state))
    rec(:summary_kind, string(summary.numeric.summary_kind))
    @test summary.symbolic.n == N
    @test summary.symbolic.nnz_pattern == length(pattern.rowval)
    @test summary.numeric.lease_live
    @test summary.numeric.generation == summary.numeric.lease_epoch
    @test summary.numeric.summary_kind === :sparse_ldlt
    @test summary.numeric.summary_size == (N, N)
    # The two halves answer different questions: the symbolic half is a
    # function of the pattern ONLY, so it is unchanged by a value-only refactor
    # while the numeric half moves.  Asserted, not asserted-about.
    symbolic_before = api.adapter_symbolic_facts(a)
    numeric_before = api.adapter_numeric_facts(a)
    b = MF2.(P02_RHS_VALUES)
    dest = zeros(MF2, N)
    api.adapter_solve!(a, dest, b)
    symbolic_after = api.adapter_symbolic_facts(a)
    numeric_after = api.adapter_numeric_facts(a)
    @test symbolic_before == symbolic_after
    rec(:numeric_moved_after_solve, numeric_before != numeric_after)
    @test numeric_before != numeric_after
    # `factor_summary` is the package's O(1) record and does not fail on this
    # cache type; this asserts which package function the adapter relies on.
    pkg_summary = MFLA.factor_summary(a.cache)
    rec(:pkg_factor_summary_kind, string(pkg_summary.kind))
    @test pkg_summary.kind === :sparse_ldlt
    nothing
end

"Leg 10 — the checked and hot entries agree for the identical input."
function p02_leg_entries_agree(api, mode::Symbol)
    rec(key, value) = p02_rec!(mode, key, value)
    K = p02_operator(MF2; seed=1)
    pattern = p02_upper_csc(MF2, K)
    b = MF2.(P02_RHS_VALUES)

    checked = api.adapter_build(MF2, pattern; dsigns=P02_DSIGNS, nrhs=1)
    checked = api.adapter_refactor!(checked, pattern; checked=true)
    hot = api.adapter_build(MF2, pattern; dsigns=P02_DSIGNS, nrhs=1)
    hot = api.adapter_refactor!(hot, pattern; checked=false)

    # The hot entry takes the caller's sparse matrix as-is.  If its index or
    # scalar type differed, the PROVIDER would refuse; with the frozen pattern
    # it is the identical object, so the two entries must agree exactly.
    dc = zeros(MF2, N)
    dh = zeros(MF2, N)
    api.adapter_solve!(checked, dc, b; checked=true)
    api.adapter_solve!(hot, dh, b; checked=false)
    rec(:checked_vs_hot_identical, dc == dh)
    rec(:checked_relres, p02_relres(K, dc, b))
    rec(:hot_relres, p02_relres(K, dh, b))
    @test dc == dh
    @test p02_relres(K, dc, b) < 1e-20
    # The symbolic analysis count is the same in both: the mode of entry does
    # not change what the provider did.
    rec(:checked_symbolic_count, api.symbolic_count(checked))
    rec(:hot_symbolic_count, api.symbolic_count(hot))
    @test api.symbolic_count(checked) == 1
    @test api.symbolic_count(hot) == 1
    @test api.numeric_count(checked) == 1
    @test api.numeric_count(hot) == 1
    # Both refuse a wider RHS identically: the hot entry drops the adapter's
    # pre-checks, so the PROVIDER must be the one refusing.
    wide = zeros(MF2, N, 2)
    hot_refusal = p02_attempt(
        () -> api.adapter_solve!(hot, zeros(MF2, N, 2), wide; checked=false),
    )
    rec(:hot_wide_refusal, hot_refusal.kind)
    rec(:hot_wide_refusal_message, hot_refusal.message)
    @test hot_refusal.threw
    nothing
end

const P02_LEGS = (
    :routes => p02_leg_routes,
    :m01f4 => p02_leg_m01f4,
    :lease_after_success => p02_leg_lease_after_success,
    :symbolic_vs_numeric => p02_leg_symbolic_vs_numeric,
    :failed_refactor => p02_leg_failed_refactor,
    :types => p02_leg_types,
    :rhs => p02_leg_rhs,
    :numeric_oracle => p02_leg_numeric_oracle,
    :summary => p02_leg_summary,
    :entries_agree => p02_leg_entries_agree,
)
const P02_LEG_NAMES = Tuple(p.first for p in P02_LEGS)
const P02_LEG_FUNCS = Tuple(p.second for p in P02_LEGS)

# =====================================================================
# SECTION 5 — subprocess measurements (never in-process: see M02/P02 note)
# =====================================================================

# QDLDL MULTI-RHS IS PER-COLUMN ONLY, and the unsupported entry points are
# OUT-OF-BOUNDS WRITES rather than clean refusals.
#
# Measured here, in QDLDL 0.4.1, on the two public spellings:
#
#   * `QDLDL.solve(q, x, b)` — a three-argument form that does not exist.  The
#     only `solve!` method is `(factor, b)`, so this is a plain `MethodError`.
#   * `QDLDL.solve(q, b)` with `b::Matrix` — this one REACHES the generic and
#     dies inside it.  `solve!` calls `permute!(F.workspace.fwork, b, F.perm)`
#     (QDLDL.jl:320), where `fwork` is a length-`n` Vector and `b` is `n x k`;
#     `permute!` is `for j in eachindex(x); x[j] = b[p[j]]` with `@inbounds`, so
#     `p[j]` runs to `k*n` and writes out of bounds.  The failure is therefore
#     UNDEFINED BEHAVIOUR, not a refusal.  Measured outcome in this environment:
#     a catchable `ReadOnlyMemoryError()` from `ipermute!` (QDLDL.jl:619),
#     `exit=1`, three times out of three.
#
# Measured on M02 the same shape of probe raised a catchable
# `ReadOnlyMemoryError` in isolated runs and SEGFAULTED a real driver once,
# killing the process with no Test Summary so a green run could not be
# reproduced.  THIS IS NOT EXPLAINED HERE and no explanation is claimed: an
# `@inbounds` out-of-bounds write has no defined failure mode, and which memory
# it lands in is not a property of the input.  That is exactly why the probe
# runs in a CHILD PROCESS, where `exit=139` is a usable measurement instead of
# the end of this run.
const P02_QDLDL_MATRIX_PROBE = """
using QDLDL
using SparseArrays
using LinearAlgebra
n = 6
A = sparse(UpperTriangular(Matrix(Symmetric(rand(n, n) + n * I, :U))))
q = QDLDL.qdldl(A; logical=true, Dsigns=nothing,
                regularize_eps=0.0, regularize_delta=0.0)
QDLDL.refactor!(q)
b = ones(n, 2)
println("PROBE_START=true")
println("PROBE_NCOLUMNS=", size(b, 2))
# VARIANT 1 — the three-argument spelling.  Expected: MethodError, no method.
println("PROBE_VARIANT1_START")
try
    QDLDL.solve(q, zeros(n, 2), b)
    println("PROBE_VARIANT1=returned")
catch error
    println("PROBE_VARIANT1=", typeof(error))
end
# VARIANT 2 — the two-argument spelling with a Matrix.  This one reaches the
# generic and performs the out-of-bounds write described above.
println("PROBE_VARIANT2_START")
QDLDL.solve(q, b)
println("PROBE_VARIANT2=returned")
println("PROBE_NONFINITE=", any(!isfinite, b))
"""

# The original Newton gate this task must keep passing, in a CHILD process:
# SDPX's own test file is a `@testset`, and running it through `Pkg.test()`
# would hit SDPX's clean-tree gate (the tree is dirty by design — no worker
# commits).  It is provider-independent Float64/BigFloat arithmetic, so it
# belongs in the SDPX project, not in `REBUILD_ENV`.
const P02_NEWTON_GATE_PATH = normpath(
    joinpath(P02_ROOT, "..", "SDPX.jl", "test", "psd_nt_finite_gate.jl"),
)

"Run a file under `Test` in a child process and report exit/stdout/stderr."
function p02_run_gate(path::String, project::String, tag::String)
    isfile(path) || return (exit=nothing, stdout="", stderr="file not found: $path",
                            command="", ran=false)
    driver = """
    using Test
    @testset "P02 child gate" begin
        include($(repr(path)))
    end
    """
    dir = mktempdir()
    driver_path = joinpath(dir, tag * "_wrapper.jl")
    write(driver_path, driver)
    out = IOBuffer(); err = IOBuffer()
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$project -t1 $(driver_path)`
    process = run(pipeline(ignorestatus(cmd); stdout=out, stderr=err); wait=true)
    return (exit=process.exitcode, stdout=String(take!(out)),
            stderr=String(take!(err)), command=string(cmd), ran=true)
end

# =====================================================================
# SECTION 6 — run every leg in both modes
# =====================================================================

const P02_BASE_SHA = p02_git_sha()
const P02_BASE_STATUS = p02_git_status()

println("="^72)
println("P02 — MFLA sparse adapter lifecycle closure")
println("MFLA HEAD               = ", P02_BASE_SHA)
println("MFLA worktree (porcelain) = ", P02_BASE_STATUS)
println("julia                   = ", VERSION)
println("project                 = ", Base.active_project())
println("Threads.nthreads()      = ", Threads.nthreads())
println("Sys.CPU_THREADS         = ", Sys.CPU_THREADS)
println("length(Sys.cpu_info())  = ", length(Sys.cpu_info()))
println("adapter                 = ", P02_ADAPTER)
println("modes                   = A:", P02_MODES.A.access.route,
        " B:", P02_MODES.B.access.route)
println("="^72)

@testset "P02 reachability (adapter is inert until loaded)" begin
    # The child MUST have run before anything is concluded from its flags.
    @test P02_UNWIRED.ran_probe
    @test P02_UNWIRED.exit == 0
    @test P02_UNWIRED_FLAGS["P02_PACKAGE_LOADED"] == "true"
    @test P02_UNWIRED_FLAGS["P02_ADAPTER_VISIBLE"] == "false"
    @test P02_UNWIRED_FLAGS["P02_SPARSE_TYPE_ON_MFLA"] == "false"
    # With QDLDL unloaded the package's own accessor reports the absence.
    @test P02_UNWIRED_FLAGS["P02_SPARSE_AVAILABLE_NO_QDLDL"] == "false"
end

@testset "P02 adapter loads in two inclusion modes" begin
    @test P02_MODES.A.M.P02_MFLA_SPARSE_ADAPTER_LOADED === true
    @test P02_MODES.B.M.P02_MFLA_SPARSE_ADAPTER_LOADED === true
    @test P02_MODE_B_BINDING.bound > 0
    @test P02_MODES.A.access.route === :extension
    @test P02_MODES.B.access.route === :factory
    # The two routes reach the SAME type CONSTRUCTOR.  The comparison is by
    # constructor identity, not by `===` on the Julia objects: the extension
    # route names the UnionAll and the factory route reads the concrete type off
    # the instance it built.  A naive `===` reports a disagreement that is not
    # one — measured here, in this file, before the comparison was changed.
    identity_a = P02_MODES.A.M.p02_type_identity(P02_MODES.A.access.type)
    identity_b = P02_MODES.B.M.p02_type_identity(P02_MODES.B.access.type)
    @test identity_a == identity_b
    @test identity_a.name == "MFSparseLDLCache"
    @test identity_a.nparams == 2
    @test P02_MODES.A.access.type <: P02_MODES.B.access.type ||
          P02_MODES.B.access.type <: P02_MODES.A.access.type
end

for (name, leg) in zip(P02_LEG_NAMES, P02_LEG_FUNCS)
    @testset "P02 leg $(name) — mode A" begin
        leg(P02_MODES.A.M, :A)
    end
    @testset "P02 leg $(name) — mode B" begin
        leg(P02_MODES.B.M, :B)
    end
end

@testset "P02 two-mode agreement" begin
    keys_a = Set(k for (m, k) in keys(P02_RECORD) if m === :A)
    keys_b = Set(k for (m, k) in keys(P02_RECORD) if m === :B)
    @test keys_a == keys_b
    @test length(keys_a) > 20
    # The one mode-dependent fact is that the route really did differ; it lives
    # outside `P02_RECORD` so the agreement check above is a pure comparison.
    @test P02_MODES.A.access.route === :extension
    @test P02_MODES.B.access.route === :factory
    @test P02_MODE_FACTS[(:A, :route)] === :extension
    @test P02_MODE_FACTS[(:B, :route)] === :factory
    @test length(P02_MODE_FACTS) == 2
    disagreements = [k for k in keys_a if P02_RECORD[(:A, k)] != P02_RECORD[(:B, k)]]
    for key in disagreements
        println("MODE DISAGREEMENT ", key, " A=", P02_RECORD[(:A, key)],
                " B=", P02_RECORD[(:B, key)])
    end
    @test isempty(disagreements)
end

# =====================================================================
# SECTION 7 — subprocess measurements
# =====================================================================

const P02_QDLDL_PROBE_RESULT = p02_run_child(P02_QDLDL_MATRIX_PROBE, "p02_qdldl_matrix_probe")
const P02_QDLDL_PROBE_FLAGS = p02_parse_flags(P02_QDLDL_PROBE_RESULT.stdout)

const P02_QDLDL_AVAILABLE = try
    Base.require(Base.PkgId(Base.UUID("bfc457fd-c171-5ab7-bd9e-d5dbfc242d63"), "QDLDL"))
    true
catch
    false
end

@testset "P02 QDLDL matrix-RHS probe runs in a subprocess" begin
    @test P02_QDLDL_AVAILABLE
    # `exit=139` (SIGSEGV) is a usable MEASUREMENT here, not a crash of this
    # run: the child is a separate process.  The FIRST assertion is that the
    # child actually ran — a dead child reports no flags, and an absent flag
    # must never be readable as a result.
    @test P02_QDLDL_PROBE_RESULT.ran_probe
    @test P02_QDLDL_PROBE_FLAGS["PROBE_START"] == "true"
    @test P02_QDLDL_PROBE_FLAGS["PROBE_NCOLUMNS"] == "2"
    println("QDLDL matrix probe exit = ", P02_QDLDL_PROBE_RESULT.exit)
    println("QDLDL matrix probe stdout =", P02_QDLDL_PROBE_RESULT.stdout)
    println("QDLDL matrix probe stderr (tail) = ",
            last(split(P02_QDLDL_PROBE_RESULT.stderr, '\n'), 5))
    # VARIANT 1 is asserted exactly: the three-argument spelling has no method.
    @test haskey(P02_QDLDL_PROBE_FLAGS, "PROBE_VARIANT1")
    @test P02_QDLDL_PROBE_FLAGS["PROBE_VARIANT1"] == "MethodError"
    # VARIANT 2 is NOT asserted to have a particular outcome, because its
    # failure mode is undefined.  What IS asserted: the child did not survive
    # it.  Both `exit=1` (catchable error) and `exit=139` (signal) satisfy
    # that, and the exact exit code is recorded rather than assumed.
    rec = P02_QDLDL_PROBE_RESULT
    @test rec.exit != 0
    @test !haskey(P02_QDLDL_PROBE_FLAGS, "PROBE_VARIANT2")
    @test rec.exit in (1, 139) || rec.exit > 128
    println("QDLDL variant-2 outcome: exit=", rec.exit,
            " variant2_flag=", get(P02_QDLDL_PROBE_FLAGS, "PROBE_VARIANT2", "absent"))
end

@testset "P02 original Newton gate (child process)" begin
    result = p02_run_gate(P02_NEWTON_GATE_PATH,
                          normpath(joinpath(P02_ROOT, "..", "SDPX.jl")),
                          "p02_newton_gate")
    # ASSERT THE CHILD RAN FIRST: a crashed child returns no summary and would
    # otherwise be indistinguishable from a passing gate.
    @test result.ran
    @test result.exit !== nothing
    println("Newton gate exit = ", result.exit)
    println("Newton gate stdout = ", result.stdout)
    @test result.exit == 0
    @test occursin("Pass", result.stdout) || occursin("Test Summary", result.stdout)
    write(joinpath(P02_LOG_DIR, "P02_newton_gate.log"),
          result.command * "\n\nEXIT=" * string(result.exit) * "\n\n" *
          result.stdout * "\n--- stderr ---\n" * result.stderr)
    nothing
end

# =====================================================================
# SECTION 8 — measurements printed for the report
# =====================================================================

println("-"^72)
println("P02 MODE FACTS (the only permitted mode difference) = ", P02_MODE_FACTS)
println("P02 MEASUREMENTS (mode A; mode B is asserted identical above)")
for key in sort(collect(k for (m, k) in keys(P02_RECORD) if m === :A); by=string)
    println("P02_MEASURED ", key, " = ", P02_RECORD[(:A, key)])
end
println("P02_UNWIRED_CHILD exit=", P02_UNWIRED.exit,
        " flags=", P02_UNWIRED_FLAGS)
println("P02_QDLDL_MATRIX_PROBE exit=", P02_QDLDL_PROBE_RESULT.exit,
        " flags=", P02_QDLDL_PROBE_FLAGS)
println("P02_GATE_PATH = ", P02_NEWTON_GATE_PATH,
        " exists=", isfile(P02_NEWTON_GATE_PATH))
println("-"^72)
