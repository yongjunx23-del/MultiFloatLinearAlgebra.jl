# Allocation regression gate for the MultiFloatLinearAlgebra factor-cache layer.
#
# Usage:
#     julia -t 1 --project=. benchmark/allocation_gate.jl            # report only
#     julia -t 1 --project=. benchmark/allocation_gate.jl --check   # CI hard gate
#
# The `--check` mode exits nonzero (exit(1)) as soon as any measured value
# EXCEEDS its per-operation limit, so CI fails on an allocation regression.
# All limits are strict regression locks baked from the current baseline (see
# the BASELINE table). Zero-allocation hot paths keep a hard limit of 0 bytes;
# any path that is NOT yet zero reports its real baseline and a limit of
# `baseline + TOLERANCE`, and prints a WARNING so nobody mistakes it for 0.
#
# Measurements are single-thread (BLAS and Julia threads = 1), warm: each
# closure runs twice to force full compilation, then `@allocated` is sampled
# SAMPLES times after a GC and the minimum is kept (robust to one-off compile /
# GC spikes).
#
# Deliberately excluded: MFLDLTCache *matrix* solve — the `solve!` matrix route
# for LDLT currently calls `trsv!` with a Matrix destination, which has no method
# on this branch (see src/factor_caches.jl `_ldlt_cache_solve!`), so it throws.
# Vector LDLT solve is gated; matrix solve is gated for the Cholesky, LU, and
# RRQR caches where it works.

using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra
using Printf
using Random

const MFLA = MultiFloatLinearAlgebra
Random.seed!(0xa110ca7e)
BLAS.set_num_threads(1)

const TYPES = (Float64x2, Float64x3, Float64x4)
limb(::Type{MultiFloat{Float64,N}}) where {N} = N

# Sampling: warmup twice, then min over SAMPLES @allocated measurements.
const SAMPLES = 6

# Regression tolerance (bytes) added to any NONZERO baseline so the gate fails
# only when the measured value rises above the current baseline. Genuinely zero
# paths keep a hard limit of 0 (tolerance is not applied).
const TOLERANCE = 16

# -----------------------------------------------------------------------------
# Baselines (bytes), baked on Julia 1.12 single-thread with this branch.
# Key = (operation::Symbol, limb_count::Int). 0 = genuinely zero-allocation hot
# path (limit stays 0). Any nonzero value is reported as-is and warns.
# -----------------------------------------------------------------------------
const BASELINE = Dict{Tuple{Symbol,Int},Int}(
    # --- core cached factorize / solve / invalidate hot paths ---
    (:cholesky_factorize, 2) => 0,  (:cholesky_factorize, 3) => 0,  (:cholesky_factorize, 4) => 0,
    (:cholesky_solve_vec, 2) => 0,  (:cholesky_solve_vec, 3) => 0,  (:cholesky_solve_vec, 4) => 0,
    (:cholesky_solve_mat, 2) => 96, (:cholesky_solve_mat, 3) => 96, (:cholesky_solve_mat, 4) => 96,
    (:invalidate, 2) => 0,          (:invalidate, 3) => 0,          (:invalidate, 4) => 0,
    (:lu_factorize, 2) => 912,      (:lu_factorize, 3) => 912,      (:lu_factorize, 4) => 1008,
    (:lu_vector_solve, 2) => 0,     (:lu_vector_solve, 3) => 0,     (:lu_vector_solve, 4) => 0,
    (:lu_matrix_solve, 2) => 96,    (:lu_matrix_solve, 3) => 96,    (:lu_matrix_solve, 4) => 96,
    (:ldlt_factorize, 2) => 0,      (:ldlt_factorize, 3) => 0,      (:ldlt_factorize, 4) => 0,
    (:ldlt_vector_solve, 2) => 0,   (:ldlt_vector_solve, 3) => 0,   (:ldlt_vector_solve, 4) => 0,
    (:rrqr_factorize, 2) => 0,      (:rrqr_factorize, 3) => 0,      (:rrqr_factorize, 4) => 0,
    (:rrqr_vector_solve, 2) => 0,   (:rrqr_vector_solve, 3) => 0,   (:rrqr_vector_solve, 4) => 0,
    (:rrqr_matrix_solve, 2) => 48,  (:rrqr_matrix_solve, 3) => 48,  (:rrqr_matrix_solve, 4) => 48,
    # GEMM direct / forced-packed (with a prepared GemmWorkspace)
    (:gemm_direct, 2) => 64,        (:gemm_direct, 3) => 64,        (:gemm_direct, 4) => 96,
    (:gemm_packed, 2) => 128,       (:gemm_packed, 3) => 128,       (:gemm_packed, 4) => 160,
    # blocked LDLT (ldlt_strategy=:blocked) and blocked RRQR (large n).
    # NOTE: blocked LDLT factorize is not allocation-free on this branch; it
    # allocates per call, and the amount depends on the pivot pattern of the
    # fixed seeded matrix (deterministic within this script). The value below is
    # the gate's own stable baseline, not a zero target.
    (:ldlt_blocked_factorize, 2) => 23296, (:ldlt_blocked_factorize, 3) => 35840, (:ldlt_blocked_factorize, 4) => 63360,
    (:ldlt_blocked_solve_vec, 2) => 0,     (:ldlt_blocked_solve_vec, 3) => 0,     (:ldlt_blocked_solve_vec, 4) => 0,
    (:rrqr_blocked_factorize, 2) => 1792,  (:rrqr_blocked_factorize, 3) => 1792,  (:rrqr_blocked_factorize, 4) => 2016,
    (:rrqr_blocked_solve_vec, 2) => 0,     (:rrqr_blocked_solve_vec, 3) => 0,     (:rrqr_blocked_solve_vec, 4) => 0,
    # repeated same-size A refactor, and success/failure/recovery refactor
    (:lu_repeated_refactor, 2) => 912,     (:lu_repeated_refactor, 3) => 912,     (:lu_repeated_refactor, 4) => 1008,
    (:lu_recovery_refactor, 2) => 912,     (:lu_recovery_refactor, 3) => 912,     (:lu_recovery_refactor, 4) => 1008,
    (:chol_recovery_refactor, 2) => 0,     (:chol_recovery_refactor, 3) => 0,     (:chol_recovery_refactor, 4) => 0,
    (:ldlt_recovery_refactor, 2) => 0,     (:ldlt_recovery_refactor, 3) => 0,     (:ldlt_recovery_refactor, 4) => 0,
    (:rrqr_recovery_refactor, 2) => 0,     (:rrqr_recovery_refactor, 3) => 0,     (:rrqr_recovery_refactor, 4) => 0,
)

# Hard limit: 0 for genuinely-zero paths, otherwise baseline + tolerance.
function gate_limit(op::Symbol, ::Type{T}) where {T}
    base = BASELINE[(op, limb(T))]
    return iszero(base) ? 0 : base + TOLERANCE
end

# -----------------------------------------------------------------------------
# Measurement helpers
# -----------------------------------------------------------------------------

function measure(f; samples::Int=SAMPLES)
    f()                      # warmup 1 (compiles the hot path)
    GC.gc()
    f()                      # warmup 2
    GC.gc()
    return minimum(@allocated(f()) for _ in 1:samples)
end

# -----------------------------------------------------------------------------
# Matrix builders
# -----------------------------------------------------------------------------

function make_spd(::Type{T}, n) where {T}
    R = randn(n, n)
    A = T.(R * R')
    @inbounds for i in 1:n
        A[i, i] += T(n)
    end
    return A
end

function make_diagdom(::Type{T}, n) where {T}
    A = T.(randn(n, n))
    @inbounds for i in 1:n
        A[i, i] += T(4)
    end
    return A
end

function make_indefinite(::Type{T}, n) where {T}
    R = T.(randn(n, n))
    A = R + R'
    @inbounds for i in 1:n
        A[i, i] += T(isodd(i) ? n : -n)
    end
    return A
end

# rank-deficient (zero out the last column) -> singular factorizations
function make_singular(::Type{T}, n) where {T}
    A = make_diagdom(T, n)
    @inbounds for i in 1:n
        A[i, n] = zero(T)
    end
    return A
end

# -----------------------------------------------------------------------------
# Per-type measurements. Returns a Vector of (operation, T, bytes).
# -----------------------------------------------------------------------------
function measure_type(::Type{T}, n, nrhs) where {T}
    results = Tuple{Symbol,DataType,Int}[]
    config = KernelConfig(thread_count=1)
    L = limb(T)

    # ------------------------- Cholesky cache -------------------------
    Ac = make_spd(T, n)
    cc = MFCholeskyCache(T; config=config)
    prepare!(cc, n)
    factorize!(cc, Ac)
    b = T.(randn(n)); x = zeros(T, n)
    Bm = T.(randn(n, nrhs)); Xm = zeros(T, n, nrhs)
    solve!(x, cc, b); solve!(Xm, cc, Bm)
    push!(results, (:cholesky_factorize, T, measure(() -> factorize!(cc, Ac))))
    push!(results, (:cholesky_solve_vec, T, measure(() -> solve!(x, cc, b))))
    push!(results, (:cholesky_solve_mat, T, measure(() -> solve!(Xm, cc, Bm))))
    push!(results, (:invalidate, T, measure(() -> invalidate!(cc))))

    # ------------------------------ LU -------------------------
    Al = make_diagdom(T, n)
    lc = MFLUCache(T; config=config)
    prepare!(lc, n)
    factorize!(lc, Al)
    solve!(x, lc, b); solve!(Xm, lc, Bm)
    push!(results, (:lu_factorize, T, measure(() -> factorize!(lc, Al))))
    push!(results, (:lu_vector_solve, T, measure(() -> solve!(x, lc, b))))
    push!(results, (:lu_matrix_solve, T, measure(() -> solve!(Xm, lc, Bm))))
    # repeated same-size refactor of a different matrix
    A2 = make_diagdom(T, n)
    push!(results, (:lu_repeated_refactor, T, measure(() -> factorize!(lc, A2))))

    # ------------------------- LDLT -------------------------
    Ai = make_indefinite(T, n)
    dc = MFLDLTCache(T; config=config)
    prepare!(dc, n)
    factorize!(dc, Ai)
    solve!(x, dc, b)
    push!(results, (:ldlt_factorize, T, measure(() -> factorize!(dc, Ai))))
    push!(results, (:ldlt_vector_solve, T, measure(() -> solve!(x, dc, b))))

    # ------------------------- RRQR -------------------------
    Ar = T.(randn(n, n))
    rc = MFRRQRCache(T; config=config)
    prepare!(rc, n, n)
    factorize!(rc, Ar)
    solve!(x, rc, b); solve!(Xm, rc, Bm)
    push!(results, (:rrqr_factorize, T, measure(() -> factorize!(rc, Ar))))
    push!(results, (:rrqr_vector_solve, T, measure(() -> solve!(x, rc, b))))
    push!(results, (:rrqr_matrix_solve, T, measure(() -> solve!(Xm, rc, Bm))))

    # ------------------------- direct GEMM -------------------------
    C = zeros(T, n, n); G1 = T.(randn(n, n)); G2 = T.(randn(n, n))
    dcfg = KernelConfig(thread_count=1, gemm_strategy=:direct)
    push!(results, (:gemm_direct, T, measure(() -> gemm!(C, G1, G2; config=dcfg))))

    # ------------------------- packed GEMM (prepared workspace) ------
    pcfg = KernelConfig(thread_count=1, gemm_strategy=:packed,
                        gemm_panel_columns=16, gemm_micro_columns=4)
    ws = GemmWorkspace(T; thread_count=1, capacity=n * 16)
    push!(results, (:gemm_packed, T, measure(() -> gemm!(C, G1, G2; config=pcfg, workspace=ws))))

    # ------------------------- blocked LDLT -------------------------
    blkcfg = KernelConfig(thread_count=1, ldlt_strategy=:blocked, ldlt_blocked_crossover=1)
    bdc = MFLDLTCache(T; config=blkcfg)
    prepare!(bdc, n)
    factorize!(bdc, Al)
    solve!(x, bdc, b)
    push!(results, (:ldlt_blocked_factorize, T, measure(() -> factorize!(bdc, Al))))
    push!(results, (:ldlt_blocked_solve_vec, T, measure(() -> solve!(x, bdc, b))))

    # ------------------------- blocked RRQR (large n) ----------------
    nb = 128
    Arb = T.(randn(nb, nb))
    rcb = MFRRQRCache(T; config=config)
    prepare!(rcb, nb, nb)
    factorize!(rcb, Arb)
    xb = zeros(T, nb); bb = T.(randn(nb))
    solve!(xb, rcb, bb)
    push!(results, (:rrqr_blocked_factorize, T, measure(() -> factorize!(rcb, Arb))))
    push!(results, (:rrqr_blocked_solve_vec, T, measure(() -> solve!(xb, rcb, bb))))

    # ------------------------- success / failure / recovery ----------
    # LU: success -> singular failure -> recovery
    lca = MFLUCache(T; config=config); prepare!(lca, n)
    factorize!(lca, Al)
    factorize!(lca, make_singular(T, n); check=false)
    push!(results, (:lu_recovery_refactor, T, measure(() -> factorize!(lca, Al))))
    # Cholesky: success -> non-PD failure -> recovery
    cf = MFCholeskyCache(T; config=config); prepare!(cf, n)
    factorize!(cf, Ac)
    factorize!(cf, -Ac; check=false)
    push!(results, (:chol_recovery_refactor, T, measure(() -> factorize!(cf, Ac))))
    # LDLT: success -> singular failure -> recovery
    dr = MFLDLTCache(T; config=config); prepare!(dr, n)
    factorize!(dr, Ai)
    factorize!(dr, make_singular(T, n); check=false)
    push!(results, (:ldlt_recovery_refactor, T, measure(() -> factorize!(dr, Ai))))
    # RRQR: success -> NaN failure -> recovery
    nan_mat = T.(randn(n, n)); nan_mat[1, 1] = T(NaN)
    rr = MFRRQRCache(T; config=config); prepare!(rr, n, n)
    factorize!(rr, Ar)
    factorize!(rr, nan_mat; check=false)
    push!(results, (:rrqr_recovery_refactor, T, measure(() -> factorize!(rr, Ar))))

    return results
end

# -----------------------------------------------------------------------------
# Threaded-task allocation (report only, NOT gated).
# -----------------------------------------------------------------------------
function threaded_task_allocation()
    function body()
        t = Threads.@spawn begin
            s = 0.0
            for i in 1:64
                s += i
            end
            s
        end
        fetch(t)
    end
    return measure(body)
end

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

function main()
    check = "--check" in ARGS
    n = 64
    nrhs = 4
    for a in ARGS
        if !startswith(a, "--") && all(isdigit, a)
            n = parse(Int, a)
        end
    end

    println("MFLA allocation regression gate (julia $(VERSION), threads=$(Threads.nthreads()), n=$n)")
    println("All measurements single-thread warm; min over $SAMPLES @allocated samples; tolerance=$TOLERANCE bytes")
    println()

    all_results = Tuple{Symbol,Any,Int}[]
    for T in TYPES
        append!(all_results, measure_type(T, n, nrhs))
    end

    # threaded task allocation is informational only
    thread_bytes = threaded_task_allocation()
    println("# threaded task spawn/fetch allocation (report only, not gated): $thread_bytes bytes")
    println()
    println("operation,type,bytes,limit,pass/fail")

    failures = 0
    nonzero_limits = 0
    for (op, T, bytes) in all_results
        limit = gate_limit(op, T)
        iszero(limit) || (nonzero_limits += 1)
        pass = bytes <= limit
        pass || (failures += 1)
        @printf("%s,%s,%d,%d,%s\n",
            op, "x$(limb(T))", bytes, limit, pass ? "pass" : "FAIL")
    end

    println()
    if nonzero_limits > 0
        println("WARNING: $nonzero_limits operation/type limits are nonzero (baseline not yet 0);")
        println("         the gate is a strict regression lock at baseline+tolerance=$TOLERANCE bytes.")
    else
        println("All limits are 0: every gated hot path is genuinely zero-allocation.")
    end

    if failures > 0
        println("GATE FAILED: $failures measurement(s) exceeded their limit.")
        check && exit(1)
    else
        println("GATE PASSED: no measurement exceeded its limit.")
    end
    return nothing
end

main()
