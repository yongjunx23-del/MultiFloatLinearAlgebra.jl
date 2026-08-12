using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra
using Printf
using Random

Random.seed!(20260812)
BLAS.set_num_threads(1)
const TYPES = (Float64x2, Float64x3, Float64x4)

function median_seconds(function_call, samples)
    count = max(Int(samples), 1)
    function_call()
    elapsed = Vector{Float64}(undef, count)
    for sample in 1:count
        GC.gc()
        start = time_ns()
        function_call()
        elapsed[sample] = (time_ns() - start) / 1.0e9
    end
    sort!(elapsed)
    return elapsed[cld(count, 2)]
end

function make_kkt(::Type{T}, total_dimension::Int) where {T}
    constraints = max(8, total_dimension ÷ 4)
    primal = total_dimension - constraints

    R = randn(primal, primal)
    H = (R * transpose(R)) / primal
    @inbounds for i in 1:primal
        H[i, i] += 0.25
    end
    A = randn(constraints, primal) / sqrt(primal)
    regularization = 1.0e-8

    K = zeros(Float64, total_dimension, total_dimension)
    K[1:primal, 1:primal] .= H
    K[1:primal, (primal + 1):end] .= transpose(A)
    K[(primal + 1):end, 1:primal] .= A
    @inbounds for i in 1:constraints
        K[primal + i, primal + i] = -regularization
    end
    return T.(K), primal, constraints
end

function max_relative_residual(A, x, b)
    residual = A * x - b
    numerator = maximum(abs, residual)
    scale = max(maximum(abs, b), maximum(abs, A) * maximum(abs, x), one(eltype(A)))
    return Float64(numerator / scale)
end

function report(T, route, n, block, seconds, residual, reference=nothing)
    @printf(
        "%-12s %-11s n=%4d block=%3d %10.6f s residual=%9.2e",
        string(T), route, n, block, seconds, residual,
    )
    reference === nothing || @printf(" speedup=%6.2fx", reference / seconds)
    println()
end

function benchmark_type(::Type{T}, n, samples) where {T}
    K, primal, constraints = make_kkt(T, n)
    rhs = T.(randn(n))
    threads = Threads.nthreads()

    unblocked_config = KernelConfig(
        thread_count=threads,
        ldlt_strategy=:unblocked,
        gemm_strategy=:direct,
    )
    unblocked_seconds = median_seconds(samples) do
        X = copy(K)
        MultiFloatLinearAlgebra.ldlt!(X; config=unblocked_config)
    end
    Fu = MultiFloatLinearAlgebra.ldlt!(copy(K); config=unblocked_config)
    xu = MultiFloatLinearAlgebra.solve(Fu, rhs)
    unblocked_residual = max_relative_residual(K, xu, rhs)
    report(T, "unblocked", n, 1, unblocked_seconds, unblocked_residual)

    auto_config = KernelConfig(
        thread_count=threads,
        ldlt_strategy=:auto,
        ldlt_blocked_crossover=1,
        gemm_strategy=:direct,
    )
    plan = ldlt_plan(T, n, auto_config)
    auto_seconds = median_seconds(samples) do
        X = copy(K)
        MultiFloatLinearAlgebra.ldlt!(X; config=auto_config)
    end
    Fa = MultiFloatLinearAlgebra.ldlt!(copy(K); config=auto_config)
    xa = MultiFloatLinearAlgebra.solve(Fa, rhs)
    auto_residual = max_relative_residual(K, xa, rhs)
    report(
        T,
        "auto",
        n,
        plan.block_size,
        auto_seconds,
        auto_residual,
        unblocked_seconds,
    )
    println(
        "  KKT split: primal=$primal constraints=$constraints " *
        "plan=$(plan.strategy)/$(plan.reason)",
    )
    println()
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 512
    samples = n >= 1024 ? 1 : 2
    println("Dense saddle-point KKT LDLT benchmark")
    println(
        "Julia threads=$(Threads.nthreads()), BLAS threads=1, " *
        "n=$n, samples=$samples",
    )
    println()
    for T in TYPES
        benchmark_type(T, n, samples)
    end
end

main()
