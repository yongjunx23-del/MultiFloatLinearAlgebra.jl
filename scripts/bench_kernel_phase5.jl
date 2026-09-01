#!/usr/bin/env julia
# Phase-5 kernel microbenchmarks. Run with:
#   JULIA_NUM_THREADS=4 julia --gcthreads=1 --project=. scripts/bench_kernel_phase5.jl
#
# Large LDLT sizes are intentionally opt-in because Float64x4 dense storage is
# 2.0 GiB at n=8000 before factor workspace. Set PHASE5_LDLT_SIZES=2000,4000,8000
# to request the full table on a sufficiently large host.
using MultiFloatLinearAlgebra, MultiFloats, LinearAlgebra, Random, Statistics

const T = Float64x4
const repeats = 3
const sizes = parse.(Int, split(get(ENV, "PHASE5_LDLT_SIZES", "2000"), ','))
const seed = 0x5a17
Random.seed!(seed)

function median_time(f)
    f() # warmup
    samples = Float64[]
    for _ in 1:repeats
        push!(samples, (@timed f()).time)
    end
    return median(samples), samples
end

function make_symmetric(n)
    A = Matrix{T}(undef, n, n)
    @inbounds for j in 1:n, i in j:n
        v = T(0.0001 * sin(i + 3j))
        v += i == j ? T(4.0 + 0.001 * i) : zero(T)
        A[i, j] = v
        A[j, i] = v
    end
    return A
end

println("Phase-5 MFLA kernel benchmark T=$(T) threads=$(Threads.nthreads())")
println("ldlt sizes=$(sizes) repeats=$(repeats)")
for n in sizes
    if n > 2000 && get(ENV, "PHASE5_ALLOW_LARGE", "0") != "1"
        println("ldlt n=$n skipped (set PHASE5_ALLOW_LARGE=1; dense Float64x4 storage=$(round(32n*n/2^30,digits=2)) GiB)")
        continue
    end
    A = make_symmetric(n)
    cfg = KernelConfig(thread_count=Threads.nthreads(), ldlt_strategy=:blocked,
        ldlt_block=32, ldlt_blocked_crossover=1)
    elapsed, samples = median_time(() -> MultiFloatLinearAlgebra.ldlt!(copy(A); config=cfg, check=false))
    println("ldlt n=$n median_s=$elapsed samples=$(join(samples, ','))")
end

m, n = 8400, 42
panel = Matrix{T}(undef, m, n)
@inbounds for j in 1:n, i in 1:m
    panel[i, j] = T(sin(0.013i + 0.017j))
end
x = T.(range(-0.5, 0.5; length=n))
y = zeros(T, m)
x_transpose = T.(range(-0.5, 0.5; length=m))
y_transpose = zeros(T, n)
for tc in (1, Threads.nthreads())
    cfg = KernelConfig(thread_count=tc)
    elapsed, samples = median_time(() -> MultiFloatLinearAlgebra.gemv!(y, panel, x; config=cfg))
    println("gemv m=$m n=$n workers=$tc median_s=$elapsed samples=$(join(samples, ','))")
    elapsed_t, samples_t = median_time(() -> MultiFloatLinearAlgebra.gemv!(y_transpose, panel, x_transpose; trans=:T, config=cfg))
    println("tgemv m=$m n=$n workers=$tc median_s=$elapsed_t samples=$(join(samples_t, ','))")
end

# Triangular solve uses a compact diagonal-dominant triangular matrix to keep
# this probe tractable; operation order is unchanged by the kernel.
nt = parse(Int, get(ENV, "PHASE5_TRSM_N", "8400"))
nrhs = parse(Int, get(ENV, "PHASE5_TRSM_NRHS", "4"))
tri = zeros(T, nt, nt)
@inbounds for i in 1:nt
    tri[i, i] = T(2.0 + 0.0001i)
end
rhs0 = Matrix{T}(undef, nt, nrhs)
@inbounds for j in 1:nrhs, i in 1:nt
    rhs0[i, j] = T(sin(0.003i + 0.11j))
end
for tc in (1, Threads.nthreads())
    cfg = KernelConfig(thread_count=tc)
    elapsed, samples = median_time(() -> MultiFloatLinearAlgebra.trsm!(copy(rhs0), tri; config=cfg))
    println("trsm n=$nt nrhs=$nrhs workers=$tc median_s=$elapsed samples=$(join(samples, ','))")
end
# A vector RHS follows the inherently serial substitution dependency. Keep it
# separate from TRSM so a multi-RHS speedup cannot be mistaken for a TRSV one.
rhs_vec = copy(view(rhs0, :, 1))
for tc in (1, Threads.nthreads())
    cfg = KernelConfig(thread_count=tc)
    elapsed, samples = median_time(() -> MultiFloatLinearAlgebra.trsv!(copy(rhs_vec), tri; config=cfg))
    println("trsv n=$nt workers=$tc median_s=$elapsed samples=$(join(samples, ','))")
end
