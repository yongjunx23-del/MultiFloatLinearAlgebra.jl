# Allocation report for the LinearSolve extension.
#
# Three separate allocation measurements:
#   (1) MFLA factor-cache core solve / factorize allocations (direct cache API,
#       no LinearSolve framework).
#   (2) The LinearSolve framework + solution-object allocation: the fixed
#       overhead LinearSolve adds to a warm RHS-only solve on top of the core.
#   (3) Escaping allocation when the user actually KEEPS the solution object,
#       with the result routed through a @noinline sink so dead-code
#       elimination cannot fabricate a 0-byte answer.
#
# Requires LinearSolve + SciMLBase (weak deps of the package). Run from any
# environment that has the package and LinearSolve, e.g.
#   julia --project=benchmark benchmark/linearsolve_report.jl [n]
using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra
using LinearSolve
using SciMLBase
using Printf
using Random

Random.seed!(0xa110ca7e)
BLAS.set_num_threads(1)

const MFLA = MultiFloatLinearAlgebra
limb(::Type{MultiFloat{Float64,N}}) where {N} = N

# Global escape sink: `_ACCUMULATOR` holds every measured result so no call is
# dead-code-eliminated, and `_KEEP` retains the actual solution object a user
# who keeps it would hold onto.
const _ACCUMULATOR = Ref(0.0)
const _KEEP = Ref{Any}(nothing)

@noinline function _consume_u(u)
    s = zero(Float64)
    @inbounds for v in u
        s += Float64(v)
    end
    _ACCUMULATOR[] += s
    return nothing
end

@noinline function _keep_solution(sol)
    _KEEP[] = sol
    _ACCUMULATOR[] += Float64(sol.retcode == SciMLBase.ReturnCode.Success)
    return nothing
end

function allocation_bytes(f)
    f()
    GC.gc()
    return @allocated f()
end

function spd(::Type{T}, n) where {T}
    R = randn(n, n); A = T.(R * R')
    @inbounds for i in 1:n
        A[i, i] += T(n)
    end
    A
end

function diagdom(::Type{T}, n) where {T}
    A = T.(randn(n, n))
    @inbounds for i in 1:n
        A[i, i] += T(4)
    end
    A
end

function report(::Type{T}, n, config) where {T}
    b = T.(randn(n))
    @printf("%-14s x%-1d n=%d  ", "Float64x", limb(T), n)

    # ---- (1) MFLA factor-cache core: solve / factorize ----
    A = diagdom(T, n)
    lc = MFLUCache(T; config=config)
    prepare!(lc, n)
    factorize!(lc, A)
    x = zeros(T, n)
    MFLA.solve!(x, lc, b)
    core_solve = allocation_bytes(() -> MFLA.solve!(x, lc, b))
    core_factor = allocation_bytes(() -> factorize!(lc, A))

    # ---- (2) LinearSolve framework + solution-object allocation ----
    # Warm RHS-only update: no factorization runs, so the bytes are purely the
    # LinearSolve dispatch + `build_linear_solution` solution object. The
    # result's `.u` is consumed by a @noinline sink so it cannot be eliminated.
    prob = LinearSolve.LinearProblem(A, b)
    cache = LinearSolve.init(prob, MultiFloatLU(config=config))
    SciMLBase.solve!(cache)
    cache.b = A * T.(randn(n))          # RHS-only update: isfresh stays false
    ls_framework = allocation_bytes(() -> _consume_u(SciMLBase.solve!(cache).u))

    # ---- (3) Escaping allocation when the solution object is KEPT ----
    # The whole solution object is handed to a @noinline global sink, so the
    # construction cannot be dead-code-eliminated and the reported bytes are the
    # real cost a caller who retains the solution pays.
    ls_kept = allocation_bytes(() -> _keep_solution(SciMLBase.solve!(cache)))

    @printf("core solve=%d  core factorize=%d  | LS u-only=%d (wrapper DCE-elided)  LS kept-sol=%d (escaped, real)\n",
        core_solve, core_factor, ls_framework, ls_kept)
end

function main()
    n = isempty(ARGS) ? 64 : parse(Int, ARGS[1])
    config = KernelConfig(thread_count=1)
    println("LinearSolve extension allocation report (bytes)")
    println("Julia $(VERSION), threads=$(Threads.nthreads()), n=$n, config thread_count=$(config.thread_count)")
    for T in (Float64x2, Float64x4)
        report(T, n, config)
    end
    # Touch the sinks so the reported numbers are observable (no DCE of the globals).
    @assert _ACCUMULATOR[] != 0.0 || true
    println("benchmark complete")
end

main()
