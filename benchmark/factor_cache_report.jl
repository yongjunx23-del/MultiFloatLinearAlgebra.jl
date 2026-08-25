using MultiFloatLinearAlgebra
using MultiFloats, LinearAlgebra, Random, Printf
import LinearSolve, SciMLBase

const MFLA = MultiFloatLinearAlgebra
Random.seed!(0xa110ca7e)
BLAS.set_num_threads(1)
limb(::Type{MultiFloat{Float64,N}}) where {N} = N

function alloc(f)
    f(); GC.gc()
    @allocated f()
end

function spd(::Type{T}, n) where {T}
    R = randn(n, n); A = T.(R * R')
    for i in 1:n; A[i,i] += T(n); end
    A
end
function diagdom(::Type{T}, n) where {T}
    A = T.(randn(n, n)); for i in 1:n; A[i,i] += T(4); end
    A
end

function run(::Type{T}, n) where {T}
    cfg = KernelConfig(thread_count=1)
    b = T.(randn(n))

    # ---- Factor-cache solve/factorize allocation (the kernel gate) ----
    A = diagdom(T, n)
    c = MFLUCache(T; config=cfg); prepare!(c, n)
    factorize!(c, A)
    x = zeros(T, n)
    solve!(x, c, b)
    lu_solve = alloc(() -> solve!(x, c, b))
    lu_factorize = (f=alloc(() -> factorize!(c, A)); f)
    lu_factorize2 = alloc(() -> factorize!(c, A))
    @printf("%-10s x%-1d  LU cache factorize=%d (stable=%s)  solve=%d\n",
        "MFLA", limb(T), lu_factorize, lu_factorize == lu_factorize2, lu_solve)

    sp = spd(T, n)
    ch = MFLA.MFCholeskyCache(T; config=cfg); prepare!(ch, n)
    factorize!(ch, sp)
    ch_solve = alloc(() -> solve!(x, ch, b))
    @printf("%-10s x%-1d  cholesky cache solve=%d\n", "MFLA", limb(T), ch_solve)

    # ---- LinearSolve full round-trip (includes framework solution object) ----
    alg = MFLA.MultiFloatLU(config=cfg)
    prob = LinearSolve.LinearProblem(A, b)
    cache = LinearSolve.init(prob, alg)
    sol = SciMLBase.solve!(cache)
    @assert sol.retcode == SciMLBase.ReturnCode.Success
    ls_first = alloc(() -> begin
        c2 = LinearSolve.init(prob, alg); SciMLBase.solve!(c2)
    end)
    # RHS-only reuse through the cache
    b2 = A * (T.(1:n))
    cache.b = b2
    ls_rhs = alloc(() -> SciMLBase.solve!(cache))
    @printf("%-10s x%-1d  LinearSolve init+factor+solve=%d  rhs-reuse=%d\n",
        "LinearSolve", limb(T), ls_first, ls_rhs)
end

run(Float64x2, 48)
