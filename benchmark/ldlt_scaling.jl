using MultiFloats
using MultiFloatLinearAlgebra
using Printf
using Random

Random.seed!(20260812)
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

function make_indefinite(::Type{T}, n) where {T}
    R = randn(n, n)
    A = 0.005 .* (R + transpose(R))
    @inbounds for index in 1:n
        A[index, index] += isodd(index) ? n : -n
    end
    return T.(Matrix(A))
end

function candidates(::Type{Float64x2})
    return (16, 24, 32, 48, 64)
end
function candidates(::Type{Float64x3})
    return (12, 16, 24, 32, 48)
end
function candidates(::Type{Float64x4})
    return (8, 12, 16, 24, 32)
end

function report(arithmetic, route, n, block, seconds, reference=nothing)
    @printf(
        "%-12s %-13s n=%4d block=%3d %10.6f s",
        arithmetic,
        route,
        n,
        block,
        seconds,
    )
    reference === nothing || @printf(" speedup=%6.2fx", reference / seconds)
    println()
end

function benchmark_type(::Type{T}, n, samples) where {T}
    A = make_indefinite(T, n)
    threads = Threads.nthreads()
    unblocked_config = KernelConfig(
        thread_count=threads,
        ldlt_strategy=:unblocked,
        gemm_strategy=:direct,
    )
    unblocked_seconds = median_seconds(samples) do
        X = copy(A)
        MultiFloatLinearAlgebra.ldlt!(
            X;
            check=true,
            config=unblocked_config,
        )
    end
    report(string(T), "unblocked", n, 1, unblocked_seconds)

    best_seconds = Inf
    best_block = 0
    for block in candidates(T)
        blocked_config = KernelConfig(
            thread_count=threads,
            ldlt_strategy=:blocked,
            ldlt_block=block,
            ldlt_blocked_crossover=1,
            gemm_strategy=:direct,
        )
        seconds = median_seconds(samples) do
            X = copy(A)
            MultiFloatLinearAlgebra.ldlt!(
                X;
                check=true,
                config=blocked_config,
            )
        end
        report(
            string(T),
            "blocked",
            n,
            block,
            seconds,
            unblocked_seconds,
        )
        if seconds < best_seconds
            best_seconds = seconds
            best_block = block
        end
    end
    @printf(
        "best %-12s n=%4d block=%3d %10.6f s speedup=%6.2fx\n\n",
        string(T),
        n,
        best_block,
        best_seconds,
        unblocked_seconds / best_seconds,
    )
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 512
    samples = n >= 1024 ? 1 : 2
    println("Blocked LDLT scaling benchmark")
    println("Julia threads=$(Threads.nthreads()), n=$n, samples=$samples")
    println()
    for T in TYPES
        benchmark_type(T, n, samples)
    end
end

main()
