# Allocation audit for the factor-cache layer. Self-contained (no LinearSolve):
# measures that repeated cached solves are 0 bytes and repeated cached
# factorize! never grows owned storage (allocation is bounded and stable).
using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra
using Printf
using Random

Random.seed!(0xa110ca7e)
BLAS.set_num_threads(1)

const TYPES = (Float64x2, Float64x3, Float64x4)
const MFLA = MultiFloatLinearAlgebra
limb(::Type{MultiFloat{Float64,N}}) where {N} = N

function allocation_bytes(f)
    f()
    GC.gc()
    @allocated f()
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

function audit(::Type{T}, n, threads) where {T}
    config = KernelConfig(thread_count=threads)
    b = T.(randn(n))

    # Cholesky
    A = spd(T, n)
    c = MFCholeskyCache(T; config=config)
    prepare!(c, n)
    factorize!(c, A)
    x = zeros(T, n)
    solve!(x, c, b)
    c_solve = allocation_bytes(() -> solve!(x, c, b))
    c_f1 = allocation_bytes(() -> factorize!(c, A))
    c_f2 = allocation_bytes(() -> factorize!(c, A))
    @printf("%-16s x%-1d threads=%d cholesky solve=%d factorize=%d (stable=%s)\n",
        "MFLA-cache", limb(T), threads, c_solve, c_f1, c_f1 == c_f2)

    # LU
    A = diagdom(T, n)
    lc = MFLUCache(T; config=config)
    prepare!(lc, n)
    factorize!(lc, A)
    solve!(x, lc, b)
    l_solve = allocation_bytes(() -> solve!(x, lc, b))
    l_f1 = allocation_bytes(() -> factorize!(lc, A))
    l_f2 = allocation_bytes(() -> factorize!(lc, A))
    @printf("%-16s x%-1d threads=%d lu solve=%d factorize=%d/%d (stable=%s)\n",
        "MFLA-cache", limb(T), threads, l_solve, l_f1, l_f2, l_f1 == l_f2)

    # LDLT
    ind = T.(randn(n, n)); ind = ind + transpose(ind)
    @inbounds for i in 1:n
        ind[i, i] += T(isodd(i) ? n : -n)
    end
    ldc = MFLDLTCache(T; config=config)
    prepare!(ldc, n)
    factorize!(ldc, ind)
    solve!(x, ldc, b)
    ld_solve = allocation_bytes(() -> solve!(x, ldc, b))
    @printf("%-16s x%-1d threads=%d ldlt solve=%d\n", "MFLA-cache", limb(T), threads, ld_solve)
end

function main()
    n = isempty(ARGS) ? 64 : parse(Int, ARGS[1])
    thread_counts = unique((1, min(4, Threads.nthreads())))
    println("MFLA factor-cache allocation audit")
    println("Julia $(VERSION), threads=$(Threads.nthreads()), n=$n")
    for threads in thread_counts, T in TYPES
        audit(T, n, threads)
        println()
    end
end

main()
