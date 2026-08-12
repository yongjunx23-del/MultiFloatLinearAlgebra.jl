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

function report(arithmetic, backend, n, seconds, reference=nothing)
    equivalent_gflops = 2.0 * n^3 / seconds / 1.0e9
    @printf(
        "%-12s %-18s n=%4d  %10.6f s  %8.3f matrix-GF/s",
        arithmetic,
        backend,
        n,
        seconds,
        equivalent_gflops,
    )
    reference === nothing || @printf("  speedup=%6.2fx", reference / seconds)
    println()
end

function benchmark_type(::Type{T}, n, samples, include_generic) where {T}
    A = T.(randn(n, n))
    B = T.(randn(n, n))
    Cdirect = zeros(T, n, n)
    Cpacked = zeros(T, n, n)
    Cauto = zeros(T, n, n)
    threads = Threads.nthreads()
    calibration_sizes = (256, min(n, 512))

    calibration = calibrate_gemm(
        T;
        sizes=calibration_sizes,
        samples=1,
        thread_count=threads,
        minimum_speedup=1.05,
    )
    profile = calibration.profile
    direct_config = KernelConfig(
        thread_count=threads,
        gemm_strategy=:direct,
        gemm_panel_columns=profile.panel_columns,
        gemm_micro_columns=profile.micro_columns,
    )
    packed_config = KernelConfig(
        thread_count=threads,
        gemm_strategy=:packed,
        gemm_panel_columns=profile.panel_columns,
        gemm_micro_columns=profile.micro_columns,
    )
    auto_config = with_gemm_profile(
        KernelConfig(thread_count=threads),
        profile,
    )
    workspace = GemmWorkspace(
        T;
        thread_count=threads,
        capacity=n * profile.panel_columns,
    )

    println(
        "profile $T: source=$(profile.source), strategy=$(profile.strategy), " *
        "panel=$(profile.panel_columns), micro=$(profile.micro_columns), " *
        "crossover=$(profile.packed_crossover), " *
        "minimum_speedup=$(calibration.minimum_speedup), " *
        "calibration_sizes=$calibration_sizes, fingerprint=$(profile.fingerprint)",
    )

    direct_seconds = median_seconds(samples) do
        MultiFloatLinearAlgebra.gemm!(Cdirect, A, B; config=direct_config)
    end
    packed_seconds = median_seconds(samples) do
        MultiFloatLinearAlgebra.gemm!(
            Cpacked,
            A,
            B;
            config=packed_config,
            workspace=workspace,
        )
    end
    auto_seconds = median_seconds(samples) do
        MultiFloatLinearAlgebra.gemm!(
            Cauto,
            A,
            B;
            config=auto_config,
            workspace=workspace,
        )
    end

    Cpacked == Cdirect || error("packed GEMM changed the per-output reduction result")
    Cauto == Cdirect || error("auto GEMM changed the per-output reduction result")

    if include_generic
        Cgeneric = zeros(T, n, n)
        generic_seconds = median_seconds(1) do
            LinearAlgebra.mul!(Cgeneric, A, B)
        end
        report(string(T), "LinearAlgebra", n, generic_seconds)
        report(string(T), "MFLA/direct", n, direct_seconds, generic_seconds)
        report(string(T), "MFLA/packed", n, packed_seconds, generic_seconds)
        report(string(T), "MFLA/auto", n, auto_seconds, generic_seconds)
    else
        report(string(T), "MFLA/direct", n, direct_seconds)
        report(string(T), "MFLA/packed", n, packed_seconds, direct_seconds)
        report(string(T), "MFLA/auto", n, auto_seconds, direct_seconds)
    end
    println()
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 512
    include_generic = "--generic" in ARGS
    samples = n >= 1024 ? 1 : 3
    println("Packed-panel MultiFloat GEMM benchmark")
    println(
        "Julia threads=$(Threads.nthreads()), BLAS threads=1, n=$n, " *
        "samples=$samples, generic=$include_generic",
    )
    println()

    A64 = randn(n, n)
    B64 = randn(n, n)
    C64 = zeros(n, n)
    float_seconds = median_seconds(samples) do
        mul!(C64, A64, B64)
    end
    report("Float64", "BLAS(1 thread)", n, float_seconds)
    println()

    for T in TYPES
        benchmark_type(T, n, samples, include_generic)
    end
end

main()
