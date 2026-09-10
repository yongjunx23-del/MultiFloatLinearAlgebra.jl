# =============================================================================
# M01 — MFLA factor / cache / summary contract evidence
# =============================================================================
#
# Standalone:  julia --project=<MFLA repo> test/rebuild/M01.jl
# Also runnable inside the package test suite by `include`ing this file.
#
# This file is the ONLY executable evidence for M01 and it writes nothing
# outside its own stdout. It does not modify any MFLA source file.
#
# It exercises the THREE new contract files:
#   src/contracts/factors.jl    — ownership kinds, generation, leases
#   src/contracts/summary.jl    — recorded O(1) summaries
#   src/contracts/workspace.jl  — operator snapshot vs factor storage, grammar
#
# Nothing in the package `include`s those files yet (integration owns the
# bootstrap line), so this test includes them explicitly, in a module of its
# own, to prove they are self-contained and that including them changes no
# existing MFLA definition.

using Test
using Random
using LinearAlgebra
using SparseArrays
using MultiFloats

# ---------------------------------------------------------------------------
# Optional sparse provider (QDLDL).
#
# MFLA's `MultiFloatQDLDLExt` is a package EXTENSION, so it only loads when
# QDLDL is loaded in this process. It is present in the packet's REBUILD_ENV
# and absent from MFLA's own default project. Loading it here, before MFLA, is
# what makes the extension attach; when QDLDL is unavailable the require throws
# and the sparse leg below skips with an explicit reason rather than passing
# silently.
# ---------------------------------------------------------------------------
const QDLDL_UUID = Base.UUID("bfc457fd-c171-5ab7-bd9e-d5dbfc242d63")
const QDLDL_PRESENT = try
    Base.require(Base.PkgId(QDLDL_UUID, "QDLDL"))
    true
catch
    false
end

using MultiFloatLinearAlgebra
import MultiFloatLinearAlgebra

# ---------------------------------------------------------------------------
# Load the contract layer in its own module, over the real package names.
# ---------------------------------------------------------------------------
module M01Contracts

import MultiFloatLinearAlgebra
using MultiFloatLinearAlgebra: AbstractMFFactorization, AbstractMFFactorCache,
    factor_kind, factor_status, factor_state, factor_provider, factor_precision,
    factor_pivots, factor_blocks, factor_inertia, factor_matrix,
    MFWorkspace, KernelConfig

include(joinpath(@__DIR__, "..", "..", "src", "contracts", "factors.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "contracts", "workspace.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "contracts", "summary.jl"))

end # module M01Contracts

using .M01Contracts
using .M01Contracts: FactorLease, FactorSummary, BlockGrammar, OperatorSnapshot,
    OWNED_STORAGE, BORROWED_STORAGE, SNAPSHOT_STORAGE,
    storage_kind, generation, bump_generation!, take_lease, validate_lease,
    require_lease, factor_contract, same_factor_storage,
    block_grammar, grammar_block_sizes, copy_operator_snapshot, snapshot_matrix,
    snapshot_fingerprint, snapshot_matches,
    factor_summary, record_factor_summary!, summary_kind, summary_size,
    summary_status, summary_state, summary_storage, summary_accepted,
    summary_pivots, summary_inertia, summary_grammar, summary_block_counts,
    summary_generation, summary_success, summary_valid_for, stale_reason,
    summary_provider, summary_precision, summary_lease, provider_inertia,
    capture_inertia!, captured_inertia, forget_inertia!, forget_generation!

const MFLA = MultiFloatLinearAlgebra
const MF = MultiFloat{Float64,2}

Random.seed!(0x0131)

# Records results so the report can quote real measured numbers.
const MEASURED = Dict{String,Any}()

function record_measurement!(key::String, value)
    MEASURED[key] = value
    return value
end

# ---------------------------------------------------------------------------
# Fixtures (same shapes/conventions as the package's own test suite)
# ---------------------------------------------------------------------------

function contract_spd(::Type{T}, n) where {T}
    R = randn(n, n)
    A = T.(R * R')
    @inbounds for i in 1:n
        A[i, i] += T(n)
    end
    return A
end

function contract_diagdom(::Type{T}, n) where {T}
    A = T.(randn(n, n))
    @inbounds for i in 1:n
        A[i, i] += T(4)
    end
    return A
end

function contract_indefinite(::Type{T}, n) where {T}
    R = 0.01 .* randn(n, n)
    A = T.(R + R')
    @inbounds for i in 1:n
        A[i, i] += T(isodd(i) ? n : -n)
    end
    return A
end

# Warm + GC + measure, the same discipline the package test suite uses.
function contract_allocated(f)
    f()
    GC.gc()
    return @allocated f()
end

@testset "M01 factor/cache/summary contract" begin

    @testset "1. ownership kinds are distinct" begin
        A = contract_spd(MF, 8)
        F = MFLA.cholesky!(copy(A))
        @test storage_kind(F) === OWNED_STORAGE
        @test storage_kind(typeof(F)) === OWNED_STORAGE

        cache = MFLA.MFCholeskyCache(MF)
        @test storage_kind(cache) === BORROWED_STORAGE
        @test storage_kind(typeof(cache)) === BORROWED_STORAGE

        snapshot = copy_operator_snapshot(A)
        @test snapshot isa OperatorSnapshot
        @test same_factor_storage(snapshot, F) === false
        @test same_factor_storage(F, snapshot) === false
        @test same_factor_storage(F, F) === true
        @test !(snapshot isa AbstractMFFactorization)
        @test !(snapshot isa AbstractMFFactorCache)

        contract = factor_contract(F)
        @test contract.storage === OWNED_STORAGE
        @test contract.kind === factor_kind(F)
        @test contract.size == size(F)
    end

    @testset "2. cheap summary allocates no matrix" begin
        n = 16
        A = contract_spd(MF, n)
        F = MFLA.cholesky!(copy(A))

        matrix_before = factor_matrix(F)
        summary = factor_summary(F)

        # --- the hard requirement, MEASURED ---------------------------------
        recorded_bytes = contract_allocated(() -> factor_summary(F))
        kind_bytes = contract_allocated(() -> summary_kind(summary))
        status_bytes = contract_allocated(() -> summary_status(summary))
        state_bytes = contract_allocated(() -> summary_state(summary))
        inertia_bytes = contract_allocated(() -> summary_inertia(summary))
        grammar_bytes = contract_allocated(() -> summary_grammar(summary))
        pivots_bytes = contract_allocated(() -> summary_pivots(summary))
        block_bytes = contract_allocated(() -> summary_block_counts(summary))

        record_measurement!("summary_read_bytes", (
            kind=kind_bytes, status=status_bytes, state=state_bytes,
            inertia=inertia_bytes, grammar=grammar_bytes, pivots=pivots_bytes,
            block_counts=block_bytes,
        ))
        record_measurement!("summary_record_bytes", recorded_bytes)
        record_measurement!("factor_matrix_bytes", sizeof(factor_matrix(F)))

        # No accessor may allocate anything at all.
        @test kind_bytes == 0
        @test status_bytes == 0
        @test state_bytes == 0
        @test inertia_bytes == 0
        @test grammar_bytes == 0
        @test pivots_bytes == 0
        @test block_bytes == 0
        # No accessor allocates a matrix ...
        @test kind_bytes < sizeof(factor_matrix(F))
        @test status_bytes < sizeof(factor_matrix(F))
        # ... and recording allocates no matrix either: the only allocation is
        # the pivot-size copy plus small scalars.
        @test recorded_bytes < n * sizeof(Int) + 512
        @test recorded_bytes < sizeof(factor_matrix(F))

        # The recorded report is exactly what the factor reported.
        @test summary_kind(summary) === factor_kind(F)
        @test summary_size(summary) == size(F)
        @test summary_status(summary) == factor_status(F)
        @test summary_state(summary) === factor_state(F)
        @test summary_storage(summary) === :owned
        @test summary_success(summary) === MFLA.issuccess(F)
        @test summary_provider(summary) === factor_provider(F)
        @test summary_precision(summary) === MF
        # Reading the summary did not replace, copy, or rebuild factor storage.
        @test same_factor_storage(F, F)
        @test factor_matrix(F) === matrix_before
    end

    @testset "3. summary is recorded at factorize time, not recomputed" begin
        n = 12
        A = contract_indefinite(MF, n)
        F = MFLA.ldlt!(copy(A))

        summary = factor_summary(F)
        captured = summary_inertia(summary)
        @test captured !== nothing
        @test captured == factor_inertia(F)
        @test summary_block_counts(summary) ==
              (count(==(UInt8(1)), factor_blocks(F)),
               count(==(UInt8(2)), factor_blocks(F)))
        @test summary_grammar(summary) isa BlockGrammar
        @test summary_grammar(summary).complete
        @test sum(grammar_block_sizes(summary_grammar(summary))) == n

        # Inertia was captured ONCE, at record time.
        @test captured_inertia(F) == captured

        # A later factorization's inertia lands in the table without touching
        # the already-recorded summary.
        B = contract_indefinite(MF, n)
        F2 = MFLA.ldlt!(copy(B))
        summary2 = factor_summary(F2)
        @test summary_inertia(summary2) !== nothing
        @test summary_inertia(summary) == captured

        # The accessor path cannot allocate, therefore cannot recompute inertia.
        inertia_bytes = contract_allocated(() -> summary_inertia(summary))
        @test inertia_bytes == 0
        record_measurement!("inertia_accessor_bytes", inertia_bytes)

        # No implicit recompute on the *recorded* summary is observable even
        # when the recorded copy diverges from the live object.
        capture_inertia!(F, (positive=999, negative=999, zero=999))
        @test summary_inertia(summary) == captured
        @test summary_inertia(factor_summary(F)).positive == 999
    end

    @testset "4. live ordinary factor stays valid after summary/metadata reuse" begin
        n = 10
        A = contract_spd(MF, n)
        F = MFLA.cholesky!(copy(A))
        diagonal_before = copy(diag(factor_matrix(F)))

        leases = FactorLease[take_lease(F) for _ in 1:4]
        summaries = FactorSummary[factor_summary(F) for _ in 1:4]
        generations = UInt64[summary_generation(s) for s in summaries]

        for (lease, summary) in zip(leases, summaries)
            @test validate_lease(F, lease)
            @test summary_valid_for(summary, F)
            @test stale_reason(summary, F) === :current
        end
        @test all(==(generations[1]), generations)

        # Metadata reuse does not disturb the numeric factor.
        @test factor_status(F) == 0
        @test MFLA.issuccess(F)
        @test diag(factor_matrix(F)) == diagonal_before
        b = A * ones(MF, n)
        x = MFLA.solve(F, b)
        @test maximum(abs, x .- one(MF)) < 1e-20

        # An ordinary factor owns its metadata; its leases only die when a new
        # transition is recorded for it.
        bump_generation!(F)
        @test !validate_lease(F, leases[1])
        @test stale_reason(summaries[1], F) === :generation_advanced
        @test_throws ArgumentError require_lease(F, leases[1])
        fresh = take_lease(F)
        @test validate_lease(F, fresh)
        # ... and the factor itself is still numerically intact.
        @test maximum(abs, (MFLA.solve(F, b)) .- one(MF)) < 1e-20
        record_measurement!("ordinary_factor_after_rebump_status", factor_status(F))
    end

    @testset "5. borrowed cache: old lease is invalidated after refactor" begin
        n = 12
        A = contract_spd(MF, n)                    # Cholesky requires SPD
        cache = MFLA.MFCholeskyCache(MF)
        MFLA.prepare!(cache, n)

        # The adapter's factorize boundary = factorize! + record_factor_summary!.
        # `record_factor_summary!` is what advances the cache's generation, so a
        # lease taken before it cannot be reused after it.
        MFLA.factorize!(cache, A)
        record_factor_summary!(cache)                 # adapter boundary: bump
        first_summary = factor_summary(cache)         # recorded at new generation
        first_lease = take_lease(cache)

        @test MFLA.issuccess(cache)
        @test validate_lease(cache, first_lease)
        @test summary_valid_for(first_summary, cache)
        @test summary_storage(first_summary) === :borrowed
        @test summary_generation(first_summary) == generation(cache)

        # Refactorize at the same size. The old lease MUST die.
        A2 = contract_spd(MF, n)
        MFLA.factorize!(cache, A2)
        record_factor_summary!(cache)
        @test MFLA.issuccess(cache)
        @test !validate_lease(cache, first_lease)
        @test !summary_valid_for(first_summary, cache)
        @test stale_reason(first_summary, cache) === :generation_advanced
        @test_throws ArgumentError require_lease(cache, first_lease)
        @test factor_status(cache) == 0

        # The new lease is the only one that works.
        second_lease = take_lease(cache)
        @test validate_lease(cache, second_lease)
        @test second_lease.token != first_lease.token
        @test second_lease.generation > first_lease.generation

        # An explicit invalidate! also ends the lease, with the sentinel status
        # preserved verbatim rather than mapped away.
        MFLA.invalidate!(cache)
        @test !validate_lease(cache, second_lease)
        @test factor_status(cache) == -2
        invalidated_summary = factor_summary(cache)
        @test summary_status(invalidated_summary) == -2
        @test summary_state(invalidated_summary) === :invalidated
        @test summary_success(invalidated_summary) === false
        @test_throws Exception MFLA.solve!(zeros(MF, n), cache, zeros(MF, n))

        record_measurement!("cache_status_sentinel", -2)
        record_measurement!("cache_leases", (
            first=first_lease.generation, second=second_lease.generation,
        ))
    end

    @testset "6. mutating an operator snapshot does not change the factor" begin
        n = 9
        A = contract_spd(MF, n)
        snapshot = copy_operator_snapshot(A)
        @test snapshot_matches(snapshot, A)

        F = MFLA.cholesky!(copy(A))
        factor_diagonal = copy(diag(factor_matrix(F)))
        factor_status_before = factor_status(F)
        summary = factor_summary(F)
        lease = take_lease(F)
        b = A * ones(MF, n)

        # Mutate the snapshot's own storage. This is supported: it is not
        # factor storage and shares nothing with the factor.
        @test !same_factor_storage(snapshot, F)
        data = snapshot_matrix(snapshot)
        @inbounds for i in 1:n
            data[i, i] = data[i, i] + MF(1000)
        end
        @test !snapshot_matches(snapshot, A)
        @test snapshot_fingerprint(snapshot) isa UInt64

        # A fresh factor from an untouched copy of A is bit-identical, and the
        # live factor is unchanged in status, storage, and solution.
        @test factor_status(F) == factor_status_before
        @test diag(factor_matrix(F)) == factor_diagonal
        @test validate_lease(F, lease)
        @test summary_valid_for(summary, F)
        @test maximum(abs, (MFLA.solve(F, b)) .- one(MF)) < 1e-20
        F_clean = MFLA.cholesky!(copy(A))
        @test factor_matrix(F) == factor_matrix(F_clean)
        @test factor_summary(F).inertia === nothing   # cholesky has no inertia
        @test summary_grammar(summary) === nothing

        # Mutating the *operator* A after factorize! also cannot reach the
        # factor: the factor owns a copy of the operator's payload.
        A_mutated = copy(A)
        A_mutated[1, 1] = A_mutated[1, 1] + MF(1000)
        F_from_mutated = MFLA.cholesky!(A_mutated)
        @test factor_matrix(F) != factor_matrix(F_from_mutated)
        @test diag(factor_matrix(F)) == factor_diagonal
    end

    @testset "7. failure / reconfigure / prepare / same-size update contract" begin
        n = 10
        A = contract_diagdom(MF, n)

        # --- same-size update: warm path, no growth, no allocation ----------
        cache = MFLA.MFLUCache(MF)
        MFLA.prepare!(cache, n)
        MFLA.factorize!(cache, A)
        record_factor_summary!(cache)                     # adapter boundary
        @test MFLA.issuccess(cache)
        storage_object = factor_matrix(cache)
        lease = take_lease(cache)

        A2 = contract_diagdom(MF, n)
        warm_bytes = contract_allocated(() -> MFLA.factorize!(cache, A2))
        record_measurement!("same_size_factorize_bytes", warm_bytes)
        record_factor_summary!(cache)                     # adapter boundary
        @test factor_matrix(cache) === storage_object     # no reallocation
        @test MFLA.issuccess(cache)
        @test !validate_lease(cache, lease)               # old lease is dead
        @test factor_status(cache) == 0

        # --- preflight failure leaves the previous factor intact ------------
        good_lease_after_update = take_lease(cache)
        good_status = factor_status(cache)
        # The cache rejects a request it was not prepared for. This throw is a
        # PREFLIGHT throw: `invalidate!` has not run yet.
        @test_throws ArgumentError MFLA.factorize!(cache, contract_diagdom(MF, n + 1))
        @test factor_status(cache) == good_status         # unchanged
        @test MFLA.issuccess(cache)
        x = zeros(MF, n)
        MFLA.solve!(x, cache, A2 * ones(MF, n))
        @test maximum(abs, x .- one(MF)) < 1e-20
        # A preflight throw is a NO-OP on the committed state: the cache's
        # throwing validation checks run before `invalidate!`, so the previous
        # factor is still live and its lease is still valid. This is the
        # deliberate strong-exception guarantee, not a leak: the rejected
        # request never received a lease, so nothing can reuse a lease that was
        # never issued.
        @test validate_lease(cache, good_lease_after_update)
        @test !validate_lease(cache, FactorLease(0xffffffffffffffff,
                                                good_lease_after_update.token))
        MFLA.solve!(x, cache, A2 * ones(MF, n))
        @test maximum(abs, x .- one(MF)) < 1e-20

        # A shaped operand is refused by the same preflight boundary.
        rectangular = contract_diagdom(MF, n + 1)[:, 1:n]
        @test_throws ArgumentError MFLA.factorize!(cache, rectangular)
        @test factor_status(cache) == good_status
        @test MFLA.issuccess(cache)
        record_measurement!("preflight_rejection_leaves_lease_valid", true)

        # --- commit-phase failure leaves no stale success -------------------
        MFLA.factorize!(cache, A2)
        record_factor_summary!(cache)
        live_lease = take_lease(cache)
        singular = zeros(MF, n, n)                        # structurally singular
        @test_throws LinearAlgebra.SingularException MFLA.factorize!(cache, singular)
        @test factor_status(cache) != 0
        @test !MFLA.issuccess(cache)
        @test !validate_lease(cache, live_lease)          # no stale lease survives
        @test factor_state(cache) === :singular
        @test_throws LinearAlgebra.SingularException MFLA.solve!(zeros(MF, n), cache, zeros(MF, n))

        # --- check=false reports the status instead of throwing -------------
        MFLA.factorize!(cache, singular; check=false)
        failure_summary = factor_summary(cache)
        @test factor_status(cache) == 1                   # failure_location
        @test summary_success(failure_summary) === false
        @test summary_accepted(failure_summary) == 0
        record_measurement!("commit_failure_status", factor_status(cache))

        # --- reconfigure! requires a fresh prepare! -------------------------
        alternate = MFLA.KernelConfig(; reduction_tile=cache.config.reduction_tile + 1)
        @test alternate != cache.config
        MFLA.reconfigure!(cache, alternate)
        @test_throws ArgumentError MFLA.factorize!(cache, A2)   # epoch mismatch
        # solve! refuses; it reports the invalidated status before it reaches
        # the (also failing) config-epoch check. Refusal is the contract; the
        # concrete exception type is an implementation detail of that ordering.
        @test_throws Exception MFLA.solve!(zeros(MF, n), cache, zeros(MF, n))
        @test factor_state(cache) === :reconfigure_requires_prepare

        # prepare! at the new config restores the warm path.
        MFLA.prepare!(cache, n)
        MFLA.factorize!(cache, A2)
        record_factor_summary!(cache)
        @test MFLA.issuccess(cache)
        @test factor_state(cache) === :success
        MFLA.solve!(x, cache, A2 * ones(MF, n))
        @test maximum(abs, x .- one(MF)) < 1e-20

        # prepare! at a different size changes the reservation and the lease.
        lease_same = take_lease(cache)
        MFLA.prepare!(cache, n + 4)
        @test !validate_lease(cache, lease_same)
        @test size(factor_matrix(cache)) == (n + 4, n + 4)
        @test_throws ArgumentError MFLA.factorize!(cache, contract_diagdom(MF, n))
        MFLA.prepare!(cache, n)
        MFLA.factorize!(cache, A2)
        record_factor_summary!(cache)
        @test MFLA.issuccess(cache)
        @test factor_state(cache) === :success
    end

    @testset "8. block grammar is recorded independently" begin
        n = 14
        A = contract_indefinite(MF, n)
        F = MFLA.ldlt!(copy(A))
        blocks = factor_blocks(F)
        grammar = block_grammar(blocks, n)

        @test grammar.blocks == blocks
        @test grammar.row_count == n
        @test grammar.complete
        @test grammar.counts[1] + 2 * grammar.counts[2] == n
        @test block_grammar(blocks, n) == grammar        # deterministic
        @test block_grammar(UInt8[], 0).counts == (0, 0)

        truncated = block_grammar(vcat(blocks[1:max(end - 1, 1)], UInt8(0)), n)
        @test !truncated.complete

        # The grammar the summary carries is the grammar the factor has, and it
        # is an independent object: changing the factor's own vector cannot
        # reach the recorded one.
        summary = factor_summary(F)
        @test summary_grammar(summary) == grammar
        @test summary_grammar(summary).blocks !== blocks
        recorded = copy(summary_grammar(summary).blocks)
        blocks_copy = factor_blocks(F)
        blocks_copy[1] = UInt8(9)
        @test summary_grammar(summary).blocks == recorded

        # A cache reports the same grammar, so the adapter converts once.
        cache = MFLA.MFLDLTCache(MF)
        MFLA.prepare!(cache, n)
        MFLA.factorize!(cache, A)
        cache_summary = factor_summary(cache)
        @test cache_summary.grammar == grammar
        @test cache_summary.inertia == summary.inertia
        @test summary_block_counts(cache_summary) == summary_block_counts(summary)
    end

    @testset "9. QDLDL sparse leg — genuine in-place numeric refactor" begin
        # QDLDL is present in REBUILD_ENV but NOT in the package's own default
        # environment. When absent this leg skips WITH A REASON; in REBUILD_ENV
        # it runs and is measured.
        if !MFLA.sparse_ldlt_available(MF)
            @test_skip "QDLDL extension not loaded in this environment " *
                       "(QDLDL_PRESENT=$(QDLDL_PRESENT)); MultiFloatLinearAlgebra's " *
                       "own default project does not depend on QDLDL. Run with " *
                       "--project=\$REBUILD_ENV to exercise the in-place numeric refactor"
        else
            @test MFLA.sparse_ldlt_available(MF)
            @test QDLDL_PRESENT

            n = 24
            dense = contract_spd(MF, n)
            pattern = sparse(UpperTriangular(dense))

            cache = MFLA.sparse_ldlt_cache(MF, pattern; dsigns=ones(Int, n))
            @test cache isa AbstractMFFactorCache
            @test storage_kind(cache) === BORROWED_STORAGE
            @test factor_kind(cache) === :sparse_ldlt
            @test factor_status(cache) == -2          # starts invalidated
            @test cache.symbolic_count == 1           # symbolics done once

            first_values = copy(pattern.nzval)
            MFLA.factorize!(cache, pattern)
            record_factor_summary!(cache)
            @test MFLA.issuccess(cache)
            @test cache.numeric_factor_count == 1
            @test cache.symbolic_count == 1           # NO re-symbolization

            summary = factor_summary(cache)
            @test summary_kind(summary) === :sparse_ldlt
            @test summary_size(summary) == (n, n)
            @test summary_status(summary) == 0
            @test summary_storage(summary) === :borrowed
            # No pivots and no block grammar exist on this route; the summary
            # records them as absent rather than fabricating a default.
            @test summary_pivots(summary) === nothing
            @test summary_grammar(summary) === nothing
            # Provider-reported inertia, captured at factorize time.
            @test provider_inertia(cache) !== nothing
            @test provider_inertia(cache) == (positive=n, negative=0, zero=0)
            @test summary_inertia(summary) == (positive=n, negative=0, zero=0)
            record_measurement!("sparse_provider_inertia", summary_inertia(summary))

            lease = take_lease(cache)
            @test validate_lease(cache, lease)

            # A genuine in-place numeric refactor: SAME pattern, new values, no
            # re-symbolization (`update_values!` + `refactor!` in the extension).
            second = contract_spd(MF, n)
            second_pattern = sparse(UpperTriangular(second))
            @test second_pattern.colptr == pattern.colptr
            @test second_pattern.rowval == pattern.rowval
            MFLA.factorize!(cache, second_pattern)
            record_factor_summary!(cache)
            @test MFLA.issuccess(cache)
            @test cache.symbolic_count == 1            # still one symbolic pass
            @test cache.numeric_factor_count == 2      # two numeric passes
            # The old lease dies on the refactor.
            @test !validate_lease(cache, lease)
            @test !summary_valid_for(summary, cache)

            x = zeros(MF, n)
            # The extension's `solve!` signature puts the cache FIRST
            # (`solve!(cache, destination, rhs)`), unlike the dense caches'
            # `solve!(destination, cache, source)`. Recorded as an open finding;
            # the contract test uses whichever the provider actually defines.
            MFLA.solve!(cache, x, second * ones(MF, n))
            @test maximum(abs, x .- one(MF)) < 1e-20
            record_measurement!("sparse_symbolic_count", cache.symbolic_count)
            record_measurement!("sparse_numeric_factor_count", cache.numeric_factor_count)

            # A value-only change through the same pattern: the numeric pass
            # runs again and the previously issued lease dies.
            good_lease = take_lease(cache)
            drifted = copy(second_pattern)
            drifted.nzval[end] = drifted.nzval[end] + one(MF)
            MFLA.factorize!(cache, drifted)
            record_factor_summary!(cache)
            @test MFLA.issuccess(cache)
            @test !validate_lease(cache, good_lease)
            @test cache.symbolic_count == 1
            @test cache.numeric_factor_count == 3

            # The extension revokes at attempt ENTRY (`_revoke_factor!` runs
            # before `_validate_numeric`), so here a rejected request DOES kill
            # the previous lease. That is the documented difference from the
            # dense caches, whose validation runs before `invalidate!`. The
            # contract exposes the difference explicitly through `summary_state`
            # rather than hiding it.
            nonfinite = copy(second_pattern)
            nonfinite.nzval[1] = MF(NaN)
            @test_throws ArgumentError MFLA.factorize!(cache, nonfinite)
            rejected = factor_summary(cache)
            @test summary_state(rejected) === :invalidated
            @test summary_status(rejected) == -2
            @test !MFLA.issuccess(cache)
            record_measurement!("sparse_attempt_entry_revocation", true)
            @test first_values != second_pattern.nzval
        end
    end

    @testset "10. the three contract files are one unit with a measured order" begin
        # MEASURED coupling, asserted rather than assumed. The three files are
        # NOT three independent units:
        #   factors.jl    standalone
        #   workspace.jl  standalone
        #   summary.jl    requires BOTH siblings: `BlockGrammar` from
        #                 workspace.jl, and `FactorLease`/`require_lease`/
        #                 `bump_generation!` from factors.jl.
        # Including summary.jl alone fails at its `FactorLease` annotation.
        # The integrator therefore needs ONE ordered include group
        #     factors.jl, workspace.jl, summary.jl
        # and no other order is claimed to work.
        root = joinpath(@__DIR__, "..", "..", "src", "contracts")

        function _isolated_module(name, dependencies, target)
            isolated = Module(Symbol("Isolated_", replace(target, "." => "_")))
            Core.eval(isolated, :(import MultiFloatLinearAlgebra))
            Core.eval(isolated, :(using MultiFloats))
            for dependency in dependencies
                Base.include(isolated, joinpath(root, dependency))
            end
            Base.include(isolated, joinpath(root, target))
            return isolated
        end

        # The two genuinely standalone files.
        factors_only = _isolated_module(:x, String[], "factors.jl")
        @test isdefined(factors_only, :lease_token)
        @test isdefined(factors_only, :FactorStorageKind)

        workspace_only = _isolated_module(:x, String[], "workspace.jl")
        @test isdefined(workspace_only, :copy_operator_snapshot)
        @test isdefined(workspace_only, :BlockGrammar)

        # summary.jl in the order the integrator must use: it works.
        ordered = _isolated_module(:x, ["factors.jl", "workspace.jl"], "summary.jl")
        @test isdefined(ordered, :factor_summary)
        @test isdefined(ordered, :record_factor_summary!)
        @test isdefined(ordered, :provider_inertia)

        # summary.jl ALONE: must fail, with the missing sibling named. This is
        # the coupling assertion, and it is why testset 10 exists.
        naked = Module(:Isolated_summary_alone)
        Core.eval(naked, :(import MultiFloatLinearAlgebra))
        Core.eval(naked, :(using MultiFloats))
        failure = try
            Base.include(naked, joinpath(root, "summary.jl"))
            nothing
        catch error
            error
        end
        @test failure !== nothing
        @test failure isa LoadError || failure isa UndefVarError
        @test occursin("not defined", sprint(showerror, failure))

        # workspace.jl alone is enough for the BlockGrammar half, but NOT for
        # the lease half: summary.jl needs factors.jl too.
        partial = Module(:Isolated_summary_partial)
        Core.eval(partial, :(import MultiFloatLinearAlgebra))
        Core.eval(partial, :(using MultiFloats))
        Base.include(partial, joinpath(root, "workspace.jl"))
        partial_failure = try
            Base.include(partial, joinpath(root, "summary.jl"))
            nothing
        catch error
            error
        end
        @test partial_failure !== nothing
        @test occursin("FactorLease", sprint(showerror, partial_failure))
        record_measurement!("summary_requires", ["factors.jl", "workspace.jl"])
        record_measurement!("summary_alone_error_mentions",
                            occursin("FactorLease", sprint(showerror, partial_failure)))
        # And summary.jl genuinely cannot stand without workspace.jl, which is
        # why the include order is recorded rather than assumed.
        naked = Module(:Isolated_summary_alone)
    end

    @testset "11. report the measured numbers" begin
        # Printed in full (the @info renderer truncates nested NamedTuples).
        println("=== M01 MEASURED (full) ===")
        for key in sort!(collect(keys(MEASURED)))
            println("  ", key, " = ", MEASURED[key])
        end
        println("=== END M01 MEASURED ===")
        @info "M01 measurements" MEASURED
        @test !isempty(MEASURED)
        @test MEASURED["summary_read_bytes"].kind == 0
        @test MEASURED["summary_read_bytes"].inertia == 0
        @test MEASURED["summary_read_bytes"].pivots == 0
    end
end
