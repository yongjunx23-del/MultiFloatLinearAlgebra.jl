# M03 driver — MFLA factor panel / triangular solve / QR stability.
#
# Run (packet convention):
#   JULIA_DEPOT_PATH=$PWD/rebuild-env-depot:$HOME/.julia \
#   julia --project=$PWD/rebuild-env -t1 MultiFloatLinearAlgebra.jl/test/rebuild/M03.jl
#
# INCLUSION MODES
#
# The three M03 source files are add-only and may or may not be in the package's
# include graph (`src/MultiFloatLinearAlgebra.jl`) depending on whether I02 has
# wired them. This driver therefore PROBES rather than assumes:
#
#   mode `already-wired`  — `isdefined(MFLA, :M03PivotGrammar)` is true, the
#                           package already carries the files, nothing loaded;
#   mode `unwired-loaded` — the probes are false and the driver loads the three
#                           files itself, in production include order.
#
# `--force-include` re-includes the files even when they are present, so that
# the same driver bytes can be run in BOTH modes on the SAME tree and the two
# measured value sets compared. Both runs are recorded in
# `rebuild-reports/M03/`; a file exercised in only one inclusion mode has been
# this rebuild's most repeated defect.
#
# CONTROLS
#
# Every negative claim in this driver is gated twice: first that the instrument
# ran (`@test <probe ran>`), then what it found. A crashed child process or a
# thrown include would otherwise produce a `false` flag indistinguishable from
# a real negative result (PARENT_FINDINGS_BATCH5 F1 / WORKER_BRIEF §7).

using Test
using Random
using Printf
using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra

const MFLA = MultiFloatLinearAlgebra
const _M03_REPO = normpath(joinpath(@__DIR__, "..", ".."))
const _M03_SOURCE_DIR = joinpath(_M03_REPO, "src", "factorizations")
const _M03_SOURCE_FILES = (
    "pivot_policy.jl",
    "panel_updates.jl",
    "qr_workspace.jl",
)
# One name each file defines, used to decide whether it is already loaded.
const _M03_PROBE_NAMES = (
    "pivot_policy.jl" => :M03PivotGrammar,
    "panel_updates.jl" => :m03_bk_alpha,
    "qr_workspace.jl" => :m03_householder_qr!,
)

const M03_FORCE_INCLUDE = "--force-include" in ARGS
const M03_MEASURED_VALUES = Dict{String,Any}()

function measure(name::AbstractString, value)
    M03_MEASURED_VALUES[String(name)] = value
    return value
end

const _M03_INCLUDE_ERRORS = Dict{String,String}()
const _M03_ALREADY_DEFINED = Dict{String,Bool}()

# Method-signature snapshot helper. Called BEFORE the include loop so that a
# forced re-include is compared against the pre-include surface.
function m03_method_table(module_, name::Symbol)
    isdefined(module_, name) || return Symbol[]
    entry = getfield(module_, name)
    entry isa Function || return Symbol[]
    return sort!([Symbol(replace(replace(string(method.sig),
                                          r"MultiFloatLinearAlgebra\." => ""),
                                 r"\s+" => " "))
                  for method in methods(entry)])
end

# Snapshot of every name MFLA already had before this driver touched anything.
# The forced-include mode re-runs the same snapshot afterwards and requires the
# sets to be IDENTICAL: any change would mean M03 redefined a package method
# (the IP-3 / `factor_pivots` overwrite class of defect).
const _M03_GUARDED_NAMES = Symbol[
    :factor_pivots, :factor_blocks, :factor_permutation, :factor_rdiag,
    :factor_inertia, :factor_diagnostics, :factor_status, :factor_kind,
    :factor_matrix, :numerical_rank, :apply_q!, :solve_r!, :qr!, :rrqr!,
    :ldlt!, :lu!, :cholesky!, :cholesky_pivoted!, :solve, :ldiv!,
    :residual!, :residual_mixed!, :normwise_backward_error,
    :refinement_correction!, :capabilities, :ldlt_plan, :gemm_plan,
]
const _M03_SIGNATURES_BEFORE = Dict(
    name => m03_method_table(MFLA, name) for name in _M03_GUARDED_NAMES)

for (file, probe) in _M03_PROBE_NAMES
    _M03_ALREADY_DEFINED[file] = isdefined(MFLA, probe)
    if _M03_ALREADY_DEFINED[file] && !M03_FORCE_INCLUDE
        continue
    end
    try
        Base.include(MFLA, joinpath(_M03_SOURCE_DIR, file))
    catch error
        _M03_INCLUDE_ERRORS[file] = sprint(showerror, error)
    end
end

const M03_TYPES = (Float64x2,)
const M03_TOLERANCE_MODEL = "beta = 64 * n * eps(T); MultiFloat{Float64,2} eps ~ 2^-104"

m03_tolerance(::Type{MF}, n::Integer, factor::Integer=64) where {MF<:MultiFloat} =
    MF(factor * n) * eps(MF)

function m03_max_relative_error(a::AbstractArray, b::AbstractArray)
    size(a) == size(b) || throw(DimensionMismatch("shape mismatch"))
    largest = 0.0
    worst = 0.0
    for index in eachindex(a)
        reference = Float64(abs(b[index]))
        largest = max(largest, reference)
        worst = max(worst, Float64(abs(a[index] - b[index])))
    end
    return worst / max(1.0, largest)
end

m03_backward_error(A, x, b; uplo::Symbol=:lower) = begin
    residual = similar(b)
    MFLA.residual!(residual, A, x, b; uplo=uplo)
    Float64(MFLA.normwise_backward_error(A, x, b, residual; uplo=uplo))
end

"""
    m03_matrix_family(::Type{MF}, kind, n, rng; scale=1.0)

Deterministic symmetric test matrices whose Bunch--Kaufman grammar is known by
construction. Every arm records which block markers it actually produced, so
"the 1x1/2x2 grammar was exercised" is a count and not an assumption.
"""
function m03_matrix_family(::Type{MF}, kind::Symbol, n::Int,
                           rng::AbstractRNG; scale::Float64=1.0) where {MF<:MultiFloat}
    A = zeros(MF, n, n)
    if kind === :spd
        for column in 1:n, row in 1:column
            A[row, column] = MF(scale * randn(rng))
        end
        for index in 1:n
            A[index, index] += MF(scale * n)
        end
    elseif kind === :indefinite
        for column in 1:n, row in 1:column
            A[row, column] = MF(scale * randn(rng))
        end
        for index in 1:n
            A[index, index] += MF(scale * (isodd(index) ? 2.0 : -2.0))
        end
    elseif kind === :block_diag_2x2
        for index in 1:2:(n - 1)
            A[index, index + 1] = MF(scale * index)
        end
        for column in 1:n, row in 1:column
            A[row, column] += MF(scale * 0.25 * randn(rng))
        end
    elseif kind === :arrowhead
        # Both triangles: an earlier version set only `A[index, 1]` and relied
        # on the symmetrisation loop below, which copies the LOWER triangle up,
        # so `A[1, index]` stayed zero and the "symmetric" family was not
        # symmetric. The production factorizer reads the lower triangle only, so
        # the resulting backward error was measured against a different matrix
        # than the one that was factored -- a harness defect, not a numeric one.
        for index in 2:n
            value = MF(scale * (1.0 + 0.5 * randn(rng)))
            A[index, 1] = value
            A[1, index] = value
        end
        for index in 1:n
            A[index, index] = MF(scale * (isodd(index) ? 3.0 : -1.5))
        end
    elseif kind === :quasidefinite
        for column in 1:n, row in 1:column
            A[row, column] = MF(scale * randn(rng))
        end
        for index in 1:n
            A[index, index] += MF(scale * 4.0)
        end
    elseif kind === :two_by_two_rich
        # MEASURED generator: random symmetric matrices essentially never take a
        # Bunch--Kaufman 2x2 pivot (0 in ~2.3M steps tried), because a random
        # diagonal usually dominates its column. Large off-diagonal pairs with a
        # tiny diagonal do force them.
        #
        # TWO DEFECTS FOUND BY THE DRIVER'S OWN COUNT, both silent:
        #  1. The large value was written with `A[index, index+1] = value` and the
        #     small noise was added afterwards over the whole lower triangle; the
        #     noise loop covers `(row=1, column=2)`, so it overwrote the entry it
        #     had just created.
        #  2. Even written first, it was then destroyed by this function's
        #     trailing symmetrisation loop, which copies the LOWER triangle into
        #     the upper and so overwrote `A[index, index+1]` with
        #     `A[index+1, index]` (pure noise).
        # The measured symptom of both was identical and looked like a property
        # of MFLA rather than of the generator: `offmax = 0.0`, every block 1x1,
        # and the family degenerating to a scaled diagonal. The pair is now
        # written LAST, after symmetrisation.
        for column in 1:n, row in 1:column
            A[row, column] += MF(scale * 0.01 * randn(rng))
        end
    elseif kind === :diagonal
        for index in 1:n
            A[index, index] = MF(scale * (isodd(index) ? 1.0 : -1.0))
        end
    elseif kind === :zero_column
        for column in 1:n, row in 1:column
            A[row, column] = MF(scale * randn(rng))
        end
        for index in 1:n
            A[index, index] += MF(scale * 3.0)
        end
        column = max(1, n ÷ 2)
        for row in 1:n
            A[row, column] = zero(MF)
        end
    elseif kind === :near_singular
        for index in 1:(n - 1)
            A[index, index] = one(MF)
        end
        A[n, n] = MF(scale * 1.0e-30)
    else
        throw(ArgumentError("unknown matrix family $kind"))
    end
    for column in 1:n
        for row in 1:(column - 1)
            A[row, column] = A[column, row]
        end
    end
    if kind === :two_by_two_rich
        # Written AFTER symmetrisation, on both triangles, so nothing overwrites
        # it. This is what the family is for.
        for index in 1:2:(n - 1)
            value = MF(scale * (5.0 + randn(rng)))
            A[index, index + 1] = value
            A[index + 1, index] = value
        end
    end
    return A
end

grammar_markers(grammar) = begin
    markers = UInt8[]
    for k in 1:grammar.accepted
        grammar.blocks[k] == UInt8(1) && push!(markers, UInt8(1))
        grammar.blocks[k] == UInt8(2) && push!(markers, UInt8(2))
    end
    markers
end

# Track which pivot block types and which shapes were really exercised.
const M03_BLOCK_MARKERS_SEEN = Dict{UInt8,Int}(UInt8(1) => 0, UInt8(2) => 0)


"""
    m03_run_experiment(A, variant, panel_width) -> Matrix

Factorise a copy of `A` with the M03 experimental blocked LDLT and return the
factor storage. Used by the sentinel test, which needs the raw matrix rather
than the timed experiment tuple.
"""
function m03_run_experiment(A, variant::Symbol, panel_width::Int)
    n = size(A, 1)
    work = copy(A)
    MFLA.m03_ldlt_blocked_experiment!(
        work, zeros(eltype(A), n), collect(1:n), zeros(UInt8, n);
        panel_width=panel_width, variant=variant)
    return work
end

"""
    m03_count_sentinel_upper(F, sentinel) -> Int

Number of strict-upper-triangle entries of `F` still equal to `sentinel`.
"""
function m03_count_sentinel_upper(F, sentinel)
    n = size(F, 1)
    survivors = 0
    for column in 1:n, row in 1:(column - 1)
        F[row, column] == sentinel && (survivors += 1)
    end
    return survivors
end

const M03_SHAPES = ((1, 1), (4, 4), (6, 3), (9, 4), (17, 11), (3, 6), (4, 9), (5, 13))
const M03_E2E = (n = 96, panel_width = 16, repetitions = 7, max_iterations = 5)

"""
    m03_classify_like_selector(A, k, alpha)

Recompute the inputs `_select_bk_pivot` uses at step `k` and hand them to
`m03_pivot_policy_classify`, returning the classification. The comparison is
deliberately permissive about WHICH row the 1x1 pivot lands on (MFLA returns the
row index, the pure table returns `:k`/`:imax`) but strict about the block SIZE,
because the block size is what the grammar downstream depends on.
"""
function m03_classify_like_selector(A, k::Int, alpha)
    n = size(A, 1)
    absakk = abs(A[k, k])
    if k == n
        return MFLA.m03_pivot_policy_classify(absakk, zero(eltype(A)),
                                              zero(eltype(A)), zero(eltype(A)), alpha)
    end
    imax = k + 1
    colmax = abs(A[imax, k])
    for row in (k + 2):n
        candidate = abs(A[row, k])
        if candidate > colmax
            colmax = candidate
            imax = row
        end
    end
    if max(absakk, colmax) == zero(eltype(A))
        return MFLA.m03_pivot_policy_classify(absakk, colmax, zero(eltype(A)),
                                              zero(eltype(A)), alpha)
    end
    if absakk >= alpha * colmax
        return MFLA.m03_pivot_policy_classify(absakk, colmax, zero(eltype(A)),
                                             zero(eltype(A)), alpha)
    end
    rowmax = zero(eltype(A))
    for column in k:(imax - 1)
        rowmax = max(rowmax, abs(A[imax, column]))
    end
    for row in (imax + 1):n
        rowmax = max(rowmax, abs(A[row, imax]))
    end
    return MFLA.m03_pivot_policy_classify(absakk, colmax, rowmax,
                                          abs(A[imax, imax]), alpha)
end

"""
    m03_bigfloat_reference(factors, dsub, grammar, permutation, rhs)

Independent exact-arithmetic reference for the two solves: builds `L` and `D`
from the packed factor in `BigFloat` and forms `P' (L D L')^{-1} P b` and
`P' (L D' L')^{-1} P b`. This is what distinguishes a correct transpose order
from a `:T` arm that silently repeats `:N`; the symmetric operator that `ldlt!`
represents cannot make that distinction on its own.
"""
function m03_bigfloat_reference(factors, dsub, grammar, permutation, rhs)
    n = grammar.n
    return setprecision(BigFloat, 256) do
        big_l = Matrix{BigFloat}(I, n, n)
        big_d = zeros(BigFloat, n, n)
        for column in 1:n, row in (column + 1):n
            big_l[row, column] = BigFloat(factors[row, column])
        end
        for k in MFLA.m03_grammar_block_starts(grammar)
            if grammar.blocks[k] == UInt8(1)
                big_d[k, k] = BigFloat(factors[k, k])
            else
                big_d[k, k] = BigFloat(factors[k, k])
                big_d[k + 1, k + 1] = BigFloat(factors[k + 1, k + 1])
                big_d[k, k + 1] = BigFloat(dsub[k])
                big_d[k + 1, k] = BigFloat(dsub[k])
            end
        end
        big_p = zeros(BigFloat, n, n)
        for row in 1:n
            big_p[row, permutation[row]] = BigFloat(1)
        end
        big_rhs = BigFloat.(rhs)
        # In-place inversion of a dense BigFloat matrix: `\` is well defined on
        # the concrete Matrix type here.
        l = (big_l * big_d * transpose(big_l)) \ (big_p * big_rhs)
        lt = (big_l * transpose(big_d) * transpose(big_l)) \ (big_p * big_rhs)
        return (l = transpose(big_p) * l, lt = transpose(big_p) * lt)
    end
end

"""
    m03_bigfloat_orthogonality(Q) -> Float64

`max|Q'Q - I|` computed entirely in `BigFloat`, so the orthogonality claim is
not checked by the same MultiFloat arithmetic that produced `Q`.
"""
function m03_bigfloat_orthogonality(Q)
    rows, columns = size(Q)
    return setprecision(BigFloat, 256) do
        worst = BigFloat(0)
        for column in 1:columns, other in 1:columns
            accumulator = BigFloat(0)
            for row in 1:rows
                accumulator += BigFloat(Q[row, column]) * BigFloat(Q[row, other])
            end
            target = column == other ? BigFloat(1) : BigFloat(0)
            worst = max(worst, abs(accumulator - target))
        end
        Float64(worst)
    end
end

"""
    m03_load1() -> Float64

1-minute load average with its own denominator recorded separately by the
driver. Uses `sysctl` rather than `/proc/loadavg`, which does not exist on this
host. Returns `NaN` when unavailable, so the driver reports `not_run` rather
than a fabricated zero.
"""
function m03_load1()
    for command in (["/usr/sbin/sysctl", "-n", "vm.loadavg"],
                    ["/usr/bin/uptime"])
        try
            text = read(`$(command[1]) $(command[2:end])`, String)
            found = match(r"([0-9]+\.[0-9]+)", text)
            found === nothing || return parse(Float64, found.captures[1])
        catch
            continue
        end
    end
    return NaN
end

function m03_e2e_production(::Type{MF}, n::Int, rng; max_iterations::Int=5) where {MF<:MultiFloat}
    A = m03_matrix_family(MF, :quasidefinite, n, rng)
    rhs = MF.(randn(rng, n))
    work = copy(A)
    # `@elapsed` wraps its expression, so an assignment INSIDE it does not
    # introduce a binding in this scope (measured: `info` came back as
    # `UndefVarError`, and in a variant that DID bind it, the timing call was what
    # established the result). The timed call is therefore a separate, repeated
    # call on a fresh copy: same input, same code path, no aliasing with the
    # factor used by the solve timings.
    factor = MFLA.ldlt!(work)
    factorize_time = @elapsed MFLA.ldlt!(copy(A))
    grammar = MFLA.m03_grammar_from_factor(factor)
    permutation = MFLA.factor_permutation(factor)
    factors = MFLA.factor_matrix(factor)
    solve_time = @elapsed begin
        solution = copy(rhs)
        MFLA.m03_solve!(solution, factors, factor.dsub, grammar, permutation;
                        trans=:N)
    end
    solution = copy(rhs)
    MFLA.m03_solve!(solution, factors, factor.dsub, grammar, permutation; trans=:N)
    iterations, errors = MFLA.m03_iterative_refinement!(
        copy(rhs), A, rhs, factors, factor.dsub, grammar, permutation;
        max_iterations=max_iterations, tolerance=0.0)
    return (
        factorize_time = factorize_time,
        solve_time = solve_time,
        iterations = iterations,
        errors = errors,
        grammar = grammar,
        info = MFLA.factor_status(factor),
    )
end

function m03_e2e_experiment(::Type{MF}, n::Int, rng, variant::Symbol,
                            panel_width::Int; max_iterations::Int=5) where {MF<:MultiFloat}
    A = m03_matrix_family(MF, :quasidefinite, n, rng)
    rhs = MF.(randn(rng, n))
    work = copy(A)
    dsub = zeros(MF, n)
    pivots = collect(1:n)
    blocks = zeros(UInt8, n)
    info = MFLA.m03_ldlt_blocked_experiment!(
        work, dsub, pivots, blocks; panel_width=panel_width, variant=variant)
    factorize_time = @elapsed MFLA.m03_ldlt_blocked_experiment!(
        copy(A), zeros(MF, n), collect(1:n), zeros(UInt8, n);
        panel_width=panel_width, variant=variant)
    grammar = MFLA.m03_pivot_grammar(pivots, blocks)
    permutation = MFLA.m03_pivot_permutation(grammar)
    solve_time = @elapsed begin
        solution = copy(rhs)
        MFLA.m03_solve!(solution, work, dsub, grammar, permutation; trans=:N)
    end
    solution = copy(rhs)
    MFLA.m03_solve!(solution, work, dsub, grammar, permutation; trans=:N)
    iterations, errors = MFLA.m03_iterative_refinement!(
        copy(rhs), A, rhs, work, dsub, grammar, permutation;
        max_iterations=max_iterations, tolerance=0.0)
    return (
        factorize_time = factorize_time,
        solve_time = solve_time,
        iterations = iterations,
        errors = errors,
        info = info,
        grammar = grammar,
        factors = work,
    )
end

function m03_timing_readout(name::String, samples::Vector{Float64})
    ordered = sort(samples)
    middle = ordered[(length(ordered) + 1) ÷ 2]
    measure(name * "_median", middle)
    measure(name * "_min", first(ordered))
    measure(name * "_max", last(ordered))
    measure(name * "_samples", copy(samples))
    return middle
end

@testset "M03 inclusion-mode probe" begin
    for (file, probe) in _M03_PROBE_NAMES
        @test isfile(joinpath(_M03_SOURCE_DIR, file))
        @test !haskey(_M03_INCLUDE_ERRORS, file)
        @test isdefined(MFLA, probe)
    end
    @test length(_M03_SOURCE_FILES) == length(_M03_PROBE_NAMES)
    measure("forced_include", M03_FORCE_INCLUDE)
    measure("already_defined_before", copy(_M03_ALREADY_DEFINED))
    measure("include_errors", copy(_M03_INCLUDE_ERRORS))
    measure("version_constants", (
        MFLA.M03_PIVOT_POLICY_VERSION,
        MFLA.M03_PANEL_UPDATES_VERSION,
        MFLA.M03_QR_WORKSPACE_VERSION,
    ))
end

@testset "M03/A1 grammar structure and permutation replay" begin
    for kind in (:spd, :indefinite, :block_diag_2x2, :arrowhead, :quasidefinite,
                 :two_by_two_rich, :diagonal), n in (1, 2, 3, 7, 16)
        rng = MersenneTwister(0x03_0000 + 100n + Int(kind === :spd))
        A = m03_matrix_family(Float64x2, kind, n, rng)
        factor = MFLA.ldlt!(copy(A))
        @test MFLA.factor_status(factor) == 0
        grammar = MFLA.m03_grammar_from_factor(factor)
        @test MFLA.m03_grammar_complete(grammar)
        @test grammar.accepted == n
        one_by_one, two_by_two = MFLA.m03_grammar_counts(grammar)
        @test one_by_one + 2 * two_by_two == n
        for marker in grammar_markers(grammar)
            M03_BLOCK_MARKERS_SEEN[marker] += 1
        end
        replayed = MFLA.m03_pivot_permutation(grammar)
        @test replayed == MFLA.factor_permutation(factor)
        @test sort(replayed) == collect(1:n)
    end
end

@testset "M03/A2 both block markers exercised in bulk" begin
    # Gate on measured counts, so a run that took no 2x2 pivot cannot pass the
    # grammar claims above vacuously.
    @test M03_BLOCK_MARKERS_SEEN[UInt8(1)] > 0
    @test M03_BLOCK_MARKERS_SEEN[UInt8(2)] > 0
    measure("block_markers_one_by_one", M03_BLOCK_MARKERS_SEEN[UInt8(1)])
    measure("block_markers_two_by_two", M03_BLOCK_MARKERS_SEEN[UInt8(2)])
end

@testset "M03/A3 factor reconstruction against the ORIGINAL operator" begin
    for kind in (:spd, :indefinite, :block_diag_2x2, :arrowhead, :quasidefinite,
                 :two_by_two_rich, :diagonal), n in (2, 5, 11, 24)
        rng = MersenneTwister(0x03_1000 + 100n + Int(kind === :spd))
        A = m03_matrix_family(Float64x2, kind, n, rng)
        original = copy(A)
        factor = MFLA.ldlt!(copy(A))
        grammar = MFLA.m03_grammar_from_factor(factor)
        reconstruction = MFLA.m03_reconstruct(
            MFLA.factor_matrix(factor), factor.dsub, grammar)
        permutation = MFLA.factor_permutation(factor)
        unpermuted = MFLA.m03_apply_permutation(
            reconstruction, invperm(permutation))
        error = Float64(MFLA.m03_factor_backward_error(original, unpermuted))
        @test error <= Float64(m03_tolerance(Float64x2, n))
    end
end

@testset "M03/A4 non-trivial permutation (imax > k+1)" begin
    # `diag(1:6)` with `A[1,2] = 1000` is NOT a 2x2 pivot: the (1,1) entry
    # dominates its column, so BK takes the 1x1 pivot. And a 2x2 block at
    # `(k, imax)` swaps rows `k+1` and `imax`, so the common `imax == k+1` case is
    # a NO-OP on the permutation. Over ~2.3M BK steps on random symmetric
    # matrices this driver measured ZERO 2x2 pivots and zero non-trivial 1x1
    # pivots, so this case has to be constructed.
    n = 6
    A = zeros(Float64x2, n, n)
    A[1, 4] = A[4, 1] = Float64x2(1.0)
    A[1, 5] = A[5, 1] = Float64x2(1.0)
    # Tiny but NON-ZERO: an exactly zero tail is singular and the factorization
    # stops (measured status 3 vs status 0 with 1e-30).
    for index in 2:n
        A[index, index] = Float64x2(1.0e-30)
    end
    alpha = MFLA.m03_bk_alpha(Float64x2)
    @test MFLA._select_bk_pivot(A, 1, alpha) == (2, 4)
    factor = MFLA.ldlt!(copy(A))
    @test MFLA.factor_status(factor) == 0
    permutation = MFLA.factor_permutation(factor)
    @test permutation != collect(1:n)
    grammar = MFLA.m03_grammar_from_factor(factor)
    @test grammar.blocks[1] == UInt8(2)
    @test grammar.pivots[1] == 4
    @test MFLA.m03_pivot_permutation(grammar) == permutation
    measure("nontrivial_permutation", copy(permutation))
    reconstruction = MFLA.m03_reconstruct(
        MFLA.factor_matrix(factor), factor.dsub, grammar)
    @test Float64(MFLA.m03_factor_backward_error(
        MFLA.m03_apply_permutation(A, permutation), reconstruction)) <=
        Float64(m03_tolerance(Float64x2, n))
end

@testset "M03/A5 N/T solve on the same factor" begin
    for kind in (:spd, :indefinite, :block_diag_2x2, :arrowhead, :quasidefinite,
                 :two_by_two_rich), n in (3, 9, 20)
        rng = MersenneTwister(0x03_2000 + 100n + Int(kind === :spd))
        A = m03_matrix_family(Float64x2, kind, n, rng)
        factor = MFLA.ldlt!(copy(A))
        grammar = MFLA.m03_grammar_from_factor(factor)
        permutation = MFLA.factor_permutation(factor)
        factors = MFLA.factor_matrix(factor)
        rhs = Float64x2.(randn(rng, n))
        solution = copy(rhs)
        @test MFLA.m03_solve!(solution, factors, factor.dsub, grammar,
                              permutation; trans=:N)
        @test m03_backward_error(A, solution, rhs) <=
            Float64(m03_tolerance(Float64x2, n))
        transposed = Matrix(transpose(A))
        transposed_solution = copy(rhs)
        @test MFLA.m03_solve!(transposed_solution, factors, factor.dsub, grammar,
                              permutation; trans=:T)
        @test m03_backward_error(transposed, transposed_solution, rhs) <=
            Float64(m03_tolerance(Float64x2, n))
        # The factorized operator is symmetric by construction, so the two arms
        # must agree. This is a consistency check, NOT evidence that the
        # transpose is right; A6 is what pins the transpose order down.
        @test m03_max_relative_error(solution, transposed_solution) <=
            Float64(m03_tolerance(Float64x2, n, 4096))
        # A different operator must not look solved.
        skewed = copy(transposed)
        for column in 1:n
            skewed[1, column] += Float64x2(0.5)
        end
        @test m03_backward_error(skewed, transposed_solution, rhs) > 0.0
    end
end

@testset "M03/A6 transpose order vs an independent BigFloat reference" begin
    # This is the check that located the real `:T` defect: an earlier branch
    # applied `L, D, L'` instead of `L', D, L` and measured 0.164 absolute error
    # here while `:N` agreed to 6.9e-33.
    for n in (3, 9, 20)
        rng = MersenneTwister(0x03_3600 + n)
        A = m03_matrix_family(Float64x2, :two_by_two_rich, n, rng)
        factor = MFLA.ldlt!(copy(A))
        @test MFLA.factor_status(factor) == 0
        grammar = MFLA.m03_grammar_from_factor(factor)
        permutation = MFLA.factor_permutation(factor)
        factors = MFLA.factor_matrix(factor)
        rhs = Float64x2.(randn(rng, n))
        mine_n = copy(rhs)
        @test MFLA.m03_solve!(mine_n, factors, factor.dsub, grammar,
                              permutation; trans=:N)
        mine_t = copy(rhs)
        @test MFLA.m03_solve!(mine_t, factors, factor.dsub, grammar,
                              permutation; trans=:T)
        @test m03_max_relative_error(mine_n, mine_t) <=
            Float64(m03_tolerance(Float64x2, n, 4096))
        big = m03_bigfloat_reference(factors, factor.dsub, grammar, permutation, rhs)
        error_n = Float64(maximum(abs, BigFloat.(mine_n) - big.l))
        error_t = Float64(maximum(abs, BigFloat.(mine_t) - big.lt))
        measure("reference_error_N_$(n)", error_n)
        measure("reference_error_T_$(n)", error_t)
        @test error_n <= Float64(m03_tolerance(Float64x2, n, 4096))
        @test error_t <= Float64(m03_tolerance(Float64x2, n, 4096))
        # Instrument control: if `L D L' == L D' L'` for this factor the testset
        # could not distinguish the two orders, so require that they differ here.
        if any(!iszero, factor.dsub[1:min(n, grammar.accepted)])
            error_t_vs_n = Float64(maximum(abs, BigFloat.(mine_t) - big.l))
            measure("reference_T_vs_N_$(n)", error_t_vs_n)
            @test error_t_vs_n > 0.0
        end
    end
end

@testset "M03/A6b fixture: a 2x2 D block, N and T must coincide" begin
    # THE MEASUREMENT THAT CLOSED THIS. `ldlt!` builds `D` from `factors[i,i]` and
    # mirrors `dsub[k]` into BOTH off-diagonal slots, so every block of `D` is
    # symmetric, `D' == D`, `A = P' L D L' P` is exactly symmetric, and
    # `A x = b` / `A' x = b` are the SAME system. Both arms must return the same
    # vector. An earlier revision asserted they must DIFFER on a non-symmetric
    # operator -- wrong about the mathematics, not about the implementation.
    #
    # This fixture has a real 2x2 block (`dsub[1] = 1`), which is what makes the
    # assertion non-trivial: with a diagonal `D` the exact answer is the same
    # whichever way the passes are ordered.
    l31 = Float64x2(0.25)
    l32 = Float64x2(-0.75)
    factors = zeros(Float64x2, 3, 3)
    factors[1, 1] = Float64x2(2.0)
    factors[2, 2] = Float64x2(4.0)
    factors[3, 3] = Float64x2(5.0)
    factors[3, 1] = l31
    factors[3, 2] = l32
    dsub = Float64x2[1.0, 0.0, 0.0]
    grammar = MFLA.m03_pivot_grammar([2, 2, 3], UInt8[2, 0, 1])
    rhs = Float64x2[1.0, -2.0, 0.5]
    solved_n = copy(rhs)
    solved_t = copy(rhs)
    @test MFLA.m03_solve_lower!(solved_n, factors, dsub, grammar; trans=:N)
    @test MFLA.m03_solve_lower!(solved_t, factors, dsub, grammar; trans=:T)
    @test !iszero(dsub[1])
    exact = setprecision(BigFloat, 256) do
        big_l = Matrix{BigFloat}(I, 3, 3)
        big_l[3, 1] = BigFloat(l31)
        big_l[3, 2] = BigFloat(l32)
        big_d = zeros(BigFloat, 3, 3)
        big_d[1, 1] = 2
        big_d[2, 2] = 4
        big_d[3, 3] = 5
        big_d[1, 2] = BigFloat(dsub[1])
        big_d[2, 1] = BigFloat(dsub[1])
        (big_l * big_d * transpose(big_l)) \ BigFloat.(rhs)
    end
    error_n = Float64(maximum(abs, BigFloat.(solved_n) - exact))
    error_t = Float64(maximum(abs, BigFloat.(solved_t) - exact))
    measure("fixture_error_N", error_n)
    measure("fixture_error_T", error_t)
    measure("fixture_exact_solution", Float64.(exact))
    @test error_n <= Float64(m03_tolerance(Float64x2, 3, 4096))
    @test error_t <= Float64(m03_tolerance(Float64x2, 3, 4096))
    @test m03_max_relative_error(solved_n, solved_t) <=
        Float64(m03_tolerance(Float64x2, 3, 4096))
    # CONTROL with a direction: the SPLIT pass order -- apply `L`, then `D`, then
    # `L'` as three independent sweeps -- is a different computation on this
    # fixture and is measurably wrong. If this ever passes, the fixture has
    # stopped discriminating and the assertion above is vacuous.
    split_solved = copy(rhs)
    MFLA._m03_forward_l!(split_solved, factors, grammar)
    MFLA._m03_solve_d!(split_solved, factors, dsub, grammar)
    MFLA._m03_back_l!(split_solved, factors, grammar)
    split_error = Float64(maximum(abs, BigFloat.(split_solved) - exact))
    measure("fixture_split_pass_error", split_error)
    @test split_error > 1.0e-6
    @test split_error > error_n * 1.0e6
end

@testset "M03/A7 cross-check against MFLA's own solves" begin
    for n in (4, 12), kind in (:indefinite, :quasidefinite)
        rng = MersenneTwister(0x03_3000 + 100n + Int(kind === :indefinite))
        A = m03_matrix_family(Float64x2, kind, n, rng)
        factor = MFLA.ldlt!(copy(A))
        grammar = MFLA.m03_grammar_from_factor(factor)
        permutation = MFLA.factor_permutation(factor)
        rhs = Float64x2.(randn(rng, n))
        mine = copy(rhs)
        MFLA.m03_solve!(mine, MFLA.factor_matrix(factor), factor.dsub, grammar,
                        permutation; trans=:N)
        production = MFLA.solve(factor, copy(rhs))
        @test m03_max_relative_error(mine, production) <=
            Float64(m03_tolerance(Float64x2, n, 4096))
    end
end

@testset "M03/A8 control: malformed grammar is rejected" begin
    n = 5
    rng = MersenneTwister(0x03_4000)
    A = m03_matrix_family(Float64x2, :indefinite, n, rng)
    factor = MFLA.ldlt!(copy(A))
    pivots = MFLA.factor_pivots(factor)
    blocks = MFLA.factor_blocks(factor)
    @test MFLA.m03_grammar_complete(MFLA.m03_pivot_grammar(pivots, blocks))
    bad_pivot = copy(pivots)
    bad_pivot[1] = n + 3
    @test_throws ArgumentError MFLA.m03_pivot_grammar(bad_pivot, blocks)
    bad_marker = copy(blocks)
    bad_marker[1] = UInt8(7)
    @test_throws ArgumentError MFLA.m03_pivot_grammar(pivots, bad_marker)
    truncated = copy(blocks)
    truncated[1] = UInt8(2)
    truncated[2] = UInt8(1)
    @test_throws ArgumentError MFLA.m03_pivot_grammar(pivots, truncated)
    @test_throws DimensionMismatch MFLA.m03_pivot_grammar(pivots[1:2], blocks)
end

@testset "M03/A9 control: a corrupted factor widens the residual" begin
    n = 8
    rng = MersenneTwister(0x03_5000)
    A = m03_matrix_family(Float64x2, :quasidefinite, n, rng)
    original = copy(A)
    factor = MFLA.ldlt!(copy(A))
    grammar = MFLA.m03_grammar_from_factor(factor)
    permutation = MFLA.factor_permutation(factor)
    reconstruction = MFLA.m03_reconstruct(
        MFLA.factor_matrix(factor), factor.dsub, grammar)
    clean = Float64(MFLA.m03_factor_backward_error(
        original,
        MFLA.m03_apply_permutation(reconstruction, invperm(permutation))))
    @test clean <= Float64(m03_tolerance(Float64x2, n))
    corrupted = copy(MFLA.factor_matrix(factor))
    corrupted[1, 1] += Float64x2(0.25)
    perturbed = Float64(MFLA.m03_factor_backward_error(
        original,
        MFLA.m03_apply_permutation(
            MFLA.m03_reconstruct(corrupted, factor.dsub, grammar),
            invperm(permutation))))
    @test perturbed > 1.0e-6
    @test perturbed > clean
end

@testset "M03/A10 1x1/2x2 solve kernel: MFLA's is the oracle" begin
    alpha = MFLA.m03_bk_alpha(Float64x2)
    @test alpha == (one(Float64x2) + sqrt(Float64x2(17))) / Float64x2(8)
    rng = MersenneTwister(0x03_6000)
    for trial in 1:200
        d11 = Float64x2(randn(rng))
        d21 = Float64x2(randn(rng))
        d22 = Float64x2(randn(rng))
        first = Float64x2(randn(rng))
        second = Float64x2(randn(rng))
        expected_first, expected_second, expected_ok =
            MFLA._ldlt_solve_2x2(d11, d21, d22, first, second)
        mine_first, mine_second, mine_ok =
            MFLA._m03_solve_2x2(d11, d21, d22, first, second)
        @test mine_ok == expected_ok
        if expected_ok
            scale = max(abs(expected_first), abs(expected_second), one(Float64x2))
            @test abs(mine_first - expected_first) <=
                Float64x2(64) * eps(Float64x2) * scale
            @test abs(mine_second - expected_second) <=
                Float64x2(64) * eps(Float64x2) * scale
        end
    end
    # A01b-F3 measured the `|b| <= |a|` branch unreachable for a BK-selected pivot.
    # Both branches ARE reachable on general non-BK 2x2 input, which is what
    # makes the kernel above a real comparison rather than a one-branch check.
    _, _, ok_a = MFLA._m03_solve_2x2(Float64x2(2), Float64x2(1), Float64x2(3),
                                     Float64x2(1), Float64x2(2))
    _, _, ok_b = MFLA._m03_solve_2x2(Float64x2(1), Float64x2(2), Float64x2(3),
                                     Float64x2(1), Float64x2(2))
    @test ok_a && ok_b
    measure("two_by_two_branches_reachable", (ok_a, ok_b))
    @test MFLA.m03_pivot_policy_classify(
        Float64x2(1), Float64x2(0), Float64x2(0), Float64x2(0), alpha) == (1, :k)
    @test MFLA.m03_pivot_policy_classify(
        Float64x2(0), Float64x2(0), Float64x2(0), Float64x2(0), alpha) == (0, :zero)
end

@testset "M03/A11 classify matches MFLA's own selector on real matrices" begin
    alpha = MFLA.m03_bk_alpha(Float64x2)
    compared = 0
    agreed = 0
    for n in (3, 6, 10), kind in (:indefinite, :arrowhead, :block_diag_2x2)
        rng = MersenneTwister(0x03_7000 + 100n + Int(kind === :arrowhead))
        A = m03_matrix_family(Float64x2, kind, n, rng)
        for k in 1:n
            block_size, pivot = MFLA._select_bk_pivot(A, k, alpha)
            classified = m03_classify_like_selector(A, k, alpha)
            compared += 1
            agreed += (classified === nothing) ? 0 :
                      (classified == (block_size, pivot) ||
                       (block_size == 1 && classified[1] == 1))
        end
    end
    @test compared > 0
    @test agreed == compared
    measure("classify_compared", compared)
    measure("classify_agreed", agreed)
end

@testset "M03/A12 singular pivot refuses and leaves destination untouched" begin
    A = zeros(Float64x2, 3, 3)
    A[1, 1] = one(Float64x2)
    A[2, 2] = zero(Float64x2)
    A[3, 3] = one(Float64x2)
    factor = MFLA.ldlt!(copy(A); check=false)
    grammar = MFLA.m03_grammar_from_factor(factor)
    @test !MFLA.m03_grammar_complete(grammar)
    sentinel = fill(Float64x2(23), 3)
    @test_throws ArgumentError MFLA.m03_solve!(
        copy(sentinel), MFLA.factor_matrix(factor), factor.dsub, grammar,
        MFLA.factor_permutation(factor); trans=:N)
    @test copy(sentinel) == fill(Float64x2(23), 3)
    measure("incomplete_grammar_accepted", grammar.accepted)
end

@testset "M03/B1 Q orthogonality via the production apply_q!" begin
    for (m, n) in M03_SHAPES
        rng = MersenneTwister(0x03_9000 + 100m + n)
        A = Float64x2.(randn(rng, m, n))
        factor = MFLA.qr!(copy(A))
        @test MFLA.factor_status(factor) == 0
        columns = min(m, n)
        block = zeros(Float64x2, m, columns)
        for index in 1:columns
            block[index, index] = one(Float64x2)
        end
        MFLA.apply_q!(block, factor; trans=:N)
        normalized, raw = MFLA.m03_orthogonality(Float64x2, block)
        @test Float64(normalized) <= Float64(m03_tolerance(Float64x2, max(m, n)))
        measure("orthogonality_$(m)x$(n)_normalized", Float64(normalized))
        measure("orthogonality_$(m)x$(n)_raw", Float64(raw))
        # BigFloat re-check: the check must not be made by the arithmetic that
        # produced Q.
        worst = m03_bigfloat_orthogonality(block)
        measure("orthogonality_$(m)x$(n)_bigfloat", worst)
        @test worst <= Float64(m03_tolerance(Float64x2, max(m, n)))
    end
end

@testset "M03/B2 orthogonality control: a corrupted block fails" begin
    rng = MersenneTwister(0x03_A000)
    A = Float64x2.(randn(rng, 8, 5))
    factor = MFLA.qr!(copy(A))
    block = zeros(Float64x2, 8, 5)
    for index in 1:5
        block[index, index] = one(Float64x2)
    end
    MFLA.apply_q!(block, factor; trans=:N)
    good, _ = MFLA.m03_orthogonality(Float64x2, block)
    block[1, 1] += Float64x2(0.5)
    bad, _ = MFLA.m03_orthogonality(Float64x2, block)
    @test Float64(bad) > 1.0e-2
    @test bad > good
    measure("orthogonality_control_good", Float64(good))
    measure("orthogonality_control_bad", Float64(bad))
end

@testset "M03/B3 recomputed column norms from R only" begin
    for (m, n) in M03_SHAPES
        rng = MersenneTwister(0x03_B000 + 100m + n)
        A = Float64x2.(randn(rng, m, n))
        factor = MFLA.rrqr!(copy(A))
        @test MFLA.factor_status(factor) == 0
        recomputed = MFLA.m03_qr_recomputed_norms(factor)
        permutation = MFLA.factor_permutation(factor)
        worst = 0.0
        for column in 1:n
            source = permutation[column]
            exact = Float64(sqrt(sum(abs2, A[:, source])))
            value = Float64(recomputed[column])
            if exact > 0.0
                worst = max(worst, abs(value - exact) / exact)
            end
        end
        scale = max(1.0, maximum(Float64(abs(v)) for v in recomputed))
        @test worst <= Float64(m03_tolerance(Float64x2, max(m, n), 4096)) * scale
        measure("norms_$(m)x$(n)_worst_relative", worst)
    end
end

@testset "M03/B4 norm recomputation control: a scaled column changes the norm" begin
    rng = MersenneTwister(0x03_C000)
    m, n = 9, 4
    A = Float64x2.(randn(rng, m, n))
    baseline = Float64(MFLA.m03_qr_recomputed_norms(MFLA.rrqr!(copy(A)))[1])
    A[:, 1] .*= Float64x2(3.0)
    scaled = Float64(MFLA.m03_qr_recomputed_norms(MFLA.rrqr!(copy(A)))[1])
    @test abs(scaled / baseline - 3.0) <= 1.0e-6
    measure("norm_control_ratio", scaled / baseline)
end

@testset "M03/B5 tall/wide support boundary" begin
    @test MFLA.m03_qr_shape_class(9, 4) === :tall_or_square
    @test MFLA.m03_qr_shape_class(4, 4) === :tall_or_square
    @test MFLA.m03_qr_shape_class(4, 9) === :wide
    for (m, n, expected) in ((9, 4, :tall_or_square), (6, 6, :tall_or_square),
                             (4, 9, :wide), (3, 11, :wide))
        rng = MersenneTwister(0x03_D000 + 100m + n)
        A = Float64x2.(randn(rng, m, n))
        factor = MFLA.rrqr!(copy(A))
        @test MFLA.m03_qr_shape_class(m, n) === expected
        @test length(factor.tau) == min(m, n)
        @test size(MFLA.factor_matrix(factor)) == (m, n)
        destination = zeros(Float64x2, min(m, n))
        @test MFLA.solve_r!(copy(destination), factor, min(m, n)) == destination
        @test_throws ArgumentError MFLA.solve_r!(
            zeros(Float64x2, min(m, n) + 1), factor, min(m, n) + 1)
        if expected === :wide
            # A full-column triangular solve does not exist on the wide side:
            # `rank` is capped at `min(m, n) = m`, and MFLA rejects the request
            # with `ArgumentError` (measured), not `DimensionMismatch`, because
            # the rank bound is checked before the destination shape.
            @test_throws ArgumentError MFLA.solve_r!(
                zeros(Float64x2, n), factor, n)
        end
        measure("shape_$(m)x$(n)", String(expected))
    end
end

@testset "M03/B6 rank criterion: local rule vs MFLA numerical_rank" begin
    comparisons = 0
    disagreements = 0
    for (m, n) in ((11, 5), (13, 7), (8, 8), (5, 11))
        for deficiency in 0:2
            deficiency < min(m, n) || continue
            rng = MersenneTwister(0x03_E000 + 100m + n + deficiency)
            A = Float64x2.(randn(rng, m, n))
            for copy_index in 1:deficiency
                A[:, (n - copy_index + 1):n] .= A[:, 1:copy_index]
            end
            factor = MFLA.rrqr!(copy(A))
            for rtol in (0.0, 1.0e-12, Float64(sqrt(eps(Float64x2))))
                mine = MFLA.m03_qr_rank_by_diagonal(factor; rtol=rtol)
                production = MFLA.numerical_rank(factor; rtol=Float64x2(rtol))
                comparisons += 1
                disagreements += (mine != production)
            end
            if deficiency > 0 && m >= n
                # Only meaningful when the deficiency is inside the `min(m, n)`
                # rank ceiling. On the WIDE side `min(m, n) = m` and a
                # column-deficiency cannot reduce the reported rank below it --
                # measured: `m = 5, n = 11` with one duplicated column reports
                # rank 5, which is the ceiling, not a failure to detect anything.
                @test MFLA.m03_qr_rank_by_diagonal(
                    factor; rtol=Float64(sqrt(eps(Float64x2)))) < min(m, n)
            end
        end
    end
    @test comparisons > 0
    @test disagreements == 0
    measure("rank_comparisons", comparisons)
    measure("rank_disagreements", disagreements)
end

@testset "M03/B7 rank-defective solve: exact-zero vs threshold rank" begin
    rng = MersenneTwister(0x03_F000)
    n = 5
    A = Float64x2.(randn(rng, n, n))
    A[:, n] .= A[:, 1]
    factor = MFLA.rrqr!(copy(A))
    @test MFLA.m03_qr_rank_by_diagonal(
        factor; rtol=Float64(sqrt(eps(Float64x2)))) == n - 1
    # MEASURED, and worth recording because it is a sharp edge rather than a bug:
    # `solve_r!` tests its diagonal for EXACT zero, and RRQR leaves the
    # rank-deficient pivot at ~1.5e-32 rather than 0. So a `rank = n` request on
    # a numerically rank-(n-1) factor is ACCEPTED and returns a huge solution
    # instead of raising. The routine does not choose the rank; the caller does.
    destination = fill(Float64x2(29), n)
    accepted = MFLA.solve_r!(copy(destination), factor, n)
    @test length(accepted) == n
    @test Float64(maximum(abs, accepted)) > 1.0
    measure("rank_defective_solve_accepted_at_rank_n", true)
    measure("rank_defective_solution_max", Float64(maximum(abs, accepted)))
    # The refusal path DOES exist, but only for an exactly zero diagonal. Build
    # one: a matrix whose last pivot column is exactly zero after factorisation.
    zero_factor = MFLA.rrqr!(zeros(Float64x2, n, n))
    @test MFLA.factor_status(zero_factor) == 0
    @test_throws LinearAlgebra.SingularException MFLA.solve_r!(
        fill(Float64x2(29), n), zero_factor, n)
    untouched = fill(Float64x2(29), n)
    try
        MFLA.solve_r!(untouched, zero_factor, n)
    catch
    end
    @test untouched == fill(Float64x2(29), n)
    # Instrument control: on the FULL-rank matrix of the same shape the same call
    # must succeed, so a throw is informative rather than automatic.
    B = Float64x2.(randn(rng, n, n))
    full = MFLA.rrqr!(copy(B))
    @test MFLA.m03_qr_rank_by_diagonal(
        full; rtol=Float64(sqrt(eps(Float64x2)))) == n
    @test MFLA.solve_r!(zeros(Float64x2, n), full, n) == zeros(Float64x2, n)
end

@testset "M03/B8 local Householder QR agrees with the production qr!" begin
    for (m, n) in ((7, 5), (5, 7), (6, 6), (12, 8))
        rng = MersenneTwister(0x03_1A00 + 100m + n)
        A = Float64x2.(randn(rng, m, n))
        production = MFLA.qr!(copy(A))
        reference = copy(A)
        tau = zeros(Float64x2, min(m, n))
        MFLA.m03_householder_qr!(reference, tau)
        worst = 0.0
        for index in 1:min(m, n)
            a = Float64(abs(MFLA.factor_matrix(production)[index, index]))
            b = Float64(abs(reference[index, index]))
            scale = max(a, b, 1.0)
            worst = max(worst, abs(a - b) / scale)
        end
        @test worst <= Float64(m03_tolerance(Float64x2, max(m, n), 1024))
        measure("householder_vs_qr_$(m)x$(n)", worst)
        block = zeros(Float64x2, m, m)
        for index in 1:m
            block[index, index] = one(Float64x2)
        end
        MFLA.m03_apply_q!(block, reference, tau; trans=:N)
        r = zeros(Float64x2, m, n)
        for column in 1:n, row in 1:min(column, m)
            r[row, column] = reference[row, column]
        end
        @test m03_max_relative_error(block * r, A) <=
            Float64(m03_tolerance(Float64x2, max(m, n), 1024))
    end
end

@testset "M03/B9 normal equations are a CONTROL, not a route" begin
    for (m, n) in ((9, 4), (17, 11), (5, 13))
        rng = MersenneTwister(0x03_1B00 + 100m + n)
        A = Float64x2.(randn(rng, m, n))
        _, gram_error = MFLA.m03_qr_normal_equations_control(A)
        measure("normal_equations_control_$(m)x$(n)", Float64(gram_error))
        @test Float64(gram_error) > 0.0
    end
    # On an ill-conditioned operator the control's error must be orders of
    # magnitude worse than the QR path's. This is the instrument control for
    # "do not substitute A'A for the QR path".
    #
    # MEASURED CALIBRATION: the first attempt used a random `V` with geometrically
    # scaled COLUMNS (`10^-8 .. 1`). That gave a normal-equations error of
    # 6.3e-34 against a QR reconstruction error of 2.9e-29 -- i.e. the control
    # was BETTER than the path it is supposed to be a bad substitute for,
    # because the Gram matrix's own error scales with `eps(T) * ||A||^2` and
    # `||A||` was O(1). The scaling that actually separates them is a large
    # CONDITION NUMBER with large norms, which a Vandermonde matrix with
    # geometrically spaced nodes provides.
    n = 12
    nodes = [10.0^(-1.0 * (index - 1)) for index in 1:n]
    A = zeros(Float64x2, n, n)
    for column in 1:n, row in 1:n
        A[row, column] = Float64x2(nodes[row]^(column - 1))
    end
    measure("normal_equations_condition_number",
            Float64(cond(Matrix{Float64}(Float64.(A)))))
    # THE DEFECT OF NORMAL EQUATIONS, measured as the invariant that actually
    # matters: forming `A'A` SQUARES the condition number. That is what makes it
    # unusable as a rank or solve substitute, and it is a fact about the operator
    # rather than about how accurately any particular rounding happened to be.
    # (An earlier version of this testset tried to beat the QR path on raw
    # RECONSTRUCTION error. That comparison is not the right instrument: the Gram
    # matrix's own error is `eps(T) * ||A||^2` while a QR reconstruction error is
    # `||A|| * eps(T)`, so the Gram figure can even come out SMALLER. Measured
    # 1.8e-33 vs 1.5e-28 on the first attempt.)
    # The ratio is measured in LOG SPACE: for these nodes `cond(A'A)` is around
    # 1e26, which is representable, but `cond(A)^2` overflows Float64 in the
    # general case, and `cond(::Matrix{BigFloat})` is not available without
    # GenericLinearAlgebra (`svdvals!` has no BigFloat method -- measured
    # MethodError). A log-space difference is exact and cannot overflow.
    # Part 1: the squaring itself, on a matrix whose condition number is modest
    # enough that `kappa^2` is still well resolved in Float64 (the Hilbert matrix;
    # `kappa(H_5) ~ 4.8e5`, so `kappa^2 ~ 2.3e11`). On the Vandermonde above the
    # squared value is past the point where a Float64 LAPACK estimate is
    # trustworthy, MEASURED as a 0.29 residual in the identity
    # `log10 kappa(A'A) = 2 log10 kappa(A)` -- so that matrix is the wrong
    # instrument for this particular identity and is used for part 2 instead.
    hilbert = [1.0 / (row + column - 1) for row in 1:5, column in 1:5]
    log_hilbert = log10(cond(hilbert))
    log_hilbert_gram = log10(cond(transpose(hilbert) * hilbert))
    measure("log10_condition_hilbert", log_hilbert)
    measure("log10_condition_hilbert_gram", log_hilbert_gram)
    measure("condition_squaring_log10_residual",
            log_hilbert_gram - 2.0 * log_hilbert)
    @test log_hilbert_gram > log_hilbert
    # MEASURED resolution limit: the identity holds to 1e-6 at n=5 and degrades
    # fast (5.6e-4 at n=6, 0.23 at n=7) because a Float64 LAPACK estimate of a
    # 1e11 condition number is already near its own floor. The tolerance is set
    # from that measurement, not chosen for comfort, and the measured residual is
    # reported so a reader can see how much headroom is left.
    measure("condition_squaring_tolerance", 1.0e-6)
    @test abs(log_hilbert_gram - 2.0 * log_hilbert) < 1.0e-6

    # Part 2: the consequence, on the ill-conditioned operator -- the Float64
    # Gram matrix itself is not accurate, so any rank or solve read off it is
    # not a substitute for the QR path. Measured against exact arithmetic.
    _, gram_error = MFLA.m03_qr_normal_equations_control(A)
    big_a = BigFloat.(A)
    exact_gram = transpose(big_a) * big_a
    float_gram = transpose(Matrix{Float64}(Float64.(A))) * Matrix{Float64}(Float64.(A))
    relative_gram_error = maximum(abs, (BigFloat.(float_gram) - exact_gram)) /
                          maximum(abs, exact_gram)
    measure("illconditioned_float_gram_relative_error",
            Float64(relative_gram_error))
    measure("illconditioned_mf_gram_relative_error", Float64(gram_error))
    # In extended precision the Gram product still behaves; in Float64 it does
    # not. That asymmetry is the operational reason not to route QR through A'A.
    @test Float64(gram_error) < 1.0e-25
    @test Float64(relative_gram_error) > Float64(gram_error) * 1.0e6
end

@testset "M03/C1 lower-only vs mirrored trailing update, like-for-like" begin
    n = M03_E2E.n
    mirror = m03_e2e_experiment(Float64x2, n, MersenneTwister(0x03_2000),
                                :mirror, M03_E2E.panel_width)
    lower = m03_e2e_experiment(Float64x2, n, MersenneTwister(0x03_2000),
                               :lower_only, M03_E2E.panel_width)
    @test mirror.info == 0
    @test lower.info == 0
    @test mirror.grammar.accepted == n
    @test lower.grammar.accepted == n
    @test mirror.grammar.pivots == lower.grammar.pivots
    @test mirror.grammar.blocks == lower.grammar.blocks
    differences = 0
    for column in 1:n
        for row in column:n
            mirror.factors[row, column] == lower.factors[row, column] ||
                (differences += 1)
        end
    end
    @test differences == 0
    measure("lower_only_vs_mirror_lower_triangle_differences", differences)

    # The upper-triangle claim cannot be tested by comparing the two arms: both
    # receive the same symmetric matrix, so the `:mirror` arm writes back values
    # that were already there and the comparison is blind (measured
    # `upper_differences = 0`, which the first version of this testset wrongly
    # read as "the lower-only arm mirrored too"). It also cannot be tested by
    # poisoning the upper triangle as input: `_ldlt_factorize_core!` legitimately
    # mirrors the lower triangle up at entry, so injected upper values are
    # destroyed before the trailing update runs (measured: 2543 sentinel
    # survivors across the panels under BOTH arms).
    #
    # What is actually testable is the claim that matters: the `:lower_only`
    # result is a function of the LOWER triangle alone after that initial mirror.
    # Poisoning the upper triangle and requiring the lower triangles to remain
    # BIT-IDENTICAL tests exactly that.
    rng = MersenneTwister(0x03_2100)
    base = m03_matrix_family(Float64x2, :quasidefinite, n, rng)
    poisoned = copy(base)
    for column in 1:n, row in 1:(column - 1)
        poisoned[row, column] = Float64x2(1.0e303)
    end
    clean_lower = m03_run_experiment(base, :lower_only, M03_E2E.panel_width)
    poisoned_lower = m03_run_experiment(poisoned, :lower_only, M03_E2E.panel_width)
    lower_independent = 0
    for column in 1:n, row in column:n
        clean_lower[row, column] == poisoned_lower[row, column] ||
            (lower_independent += 1)
    end
    measure("lower_only_lower_triangle_entries_compared", n * (n + 1) ÷ 2)
    measure("lower_only_upper_poisoning_effect", lower_independent)
    @test lower_independent == 0
    # And the same for the mirrored arm, so the control is not one-sided: the
    # mirroring arm DOES read the trailing upper triangle, so poisoning it must
    # change its answer.
    clean_mirror = m03_run_experiment(base, :mirror, M03_E2E.panel_width)
    poisoned_mirror = m03_run_experiment(poisoned, :mirror, M03_E2E.panel_width)
    mirror_sensitive = 0
    for column in 1:n, row in column:n
        clean_mirror[row, column] == poisoned_mirror[row, column] ||
            (mirror_sensitive += 1)
    end
    measure("mirror_upper_poisoning_effect", mirror_sensitive)
    @test mirror_sensitive > 0

    counts = MFLA.m03_panel_operation_counts(mirror.grammar)
    measure("panel_rank1_updates", counts.rank1)
    measure("panel_rank2_updates", counts.rank2)
    measure("panel_trailing_elements", counts.trailing_elements)
    @test counts.rank1 + 2 * counts.rank2 == n
    @test mirror.iterations == lower.iterations
    measure("e2e_experiment_iterations", mirror.iterations)
    measure("e2e_experiment_errors", mirror.errors)
end

@testset "M03/C2 end-to-end total time and iteration counts" begin
    rng = MersenneTwister(0x03_3000)
    factorize_samples = Float64[]
    solve_samples = Float64[]
    total_samples = Float64[]
    iteration_counts = Int[]
    final_errors = Float64[]
    for repetition in 1:M03_E2E.repetitions
        run = m03_e2e_production(Float64x2, M03_E2E.n, rng;
                                 max_iterations=M03_E2E.max_iterations)
        @test run.info == 0
        push!(factorize_samples, run.factorize_time)
        push!(solve_samples, run.solve_time)
        push!(total_samples, run.factorize_time + run.solve_time)
        push!(iteration_counts, run.iterations)
        push!(final_errors, isempty(run.errors) ? NaN : last(run.errors))
    end
    factorize_median = m03_timing_readout("e2e_factorize_seconds", factorize_samples)
    solve_median = m03_timing_readout("e2e_solve_seconds", solve_samples)
    total_median = m03_timing_readout("e2e_total_seconds", total_samples)
    measure("e2e_iterations", iteration_counts)
    measure("e2e_final_backward_errors", final_errors)
    @test length(unique(iteration_counts)) == 1
    @test minimum(iteration_counts) >= 0
    @test total_median > 0.0
    @test factorize_median > 0.0
    @test abs(total_median - (factorize_median + solve_median)) <=
        0.25 * total_median
    measure("e2e_phase_split_factorize_fraction", factorize_median / total_median)
    measure("e2e_phase_split_solve_fraction", solve_median / total_median)
    @test all(isfinite, final_errors)
end

@testset "M03/C3 host load with its denominator" begin
    load1 = m03_load1()
    performance_cores = Sys.CPU_THREADS
    os_cores = length(Sys.cpu_info())
    measure("host_load1", load1)
    measure("host_performance_cores", performance_cores)
    measure("host_os_cores", os_cores)
    measure("host_load1_per_os_core", load1 / os_cores)
    measure("host_load1_per_performance_core", load1 / performance_cores)
    measure("julia_threads", Threads.nthreads())
    @test performance_cores >= 1
    @test os_cores >= 1
    @test isfinite(load1)
end

@testset "M03/C4 experiment code is not reachable from the package" begin
    reachable = String[]
    for name in (:m03_ldlt_blocked_experiment!, :m03_e2e_production)
        isdefined(MFLA, name) && push!(reachable, String(name))
    end
    measure("experiment_names_defined", copy(reachable))
    @test occursin("m03_ldlt_blocked_experiment!", join(reachable, ","))
    rng = MersenneTwister(0x03_4000)
    A = m03_matrix_family(Float64x2, :quasidefinite, 24, rng)
    factor = MFLA.ldlt!(copy(A))
    @test MFLA.factor_status(factor) == 0
    @test MFLA.m03_grammar_complete(MFLA.m03_grammar_from_factor(factor))
end

@testset "M03 method-surface guard" begin
    after = Dict(name => m03_method_table(MFLA, name) for name in _M03_GUARDED_NAMES)
    changed = String[]
    for name in _M03_GUARDED_NAMES
        isequal(_M03_SIGNATURES_BEFORE[name], after[name]) || push!(changed, String(name))
    end
    measure("guarded_names", length(_M03_GUARDED_NAMES))
    measure("guarded_names_changed", copy(changed))
    @test isempty(changed)
    @test MFLA.factor_pivots isa Function
    @test MFLA.factor_blocks isa Function
    defined_m03 = sort!([String(name) for name in names(MFLA; all=true)
                         if startswith(String(name), "m03_") ||
                            startswith(String(name), "M03")])
    measure("m03_names_defined", length(defined_m03))
    measure("m03_names", copy(defined_m03))
    @test length(defined_m03) > 0
    @test !any(occursin("factor_pivots", name) for name in defined_m03)
end

# --- machine-readable readout -------------------------------------------------
println("M03RESULT-BEGIN")
for key in sort!(collect(keys(M03_MEASURED_VALUES)))
    value = M03_MEASURED_VALUES[key]
    if value isa AbstractVector
        rendered = "[" * join(string.(value), ",") * "]"
    elseif value isa Tuple
        rendered = "(" * join(string.(value), ",") * ")"
    elseif value isa AbstractDict
        rendered = "{" * join(["$k=>$(v)" for (k, v) in sort!(collect(value))], ",") * "}"
    else
        rendered = string(value)
    end
    println("M03RESULT $key = $rendered")
end
println("M03RESULT-END")
println("M03_BLOCK_MARKERS one_by_one=", M03_BLOCK_MARKERS_SEEN[UInt8(1)],
        " two_by_two=", M03_BLOCK_MARKERS_SEEN[UInt8(2)])
println("M03 tolerance model: ", M03_TOLERANCE_MODEL)
