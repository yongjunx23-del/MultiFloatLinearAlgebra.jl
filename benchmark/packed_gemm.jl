using BenchmarkTools
using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra
using Printf
using Random

Random.seed!(20260812)
BLAS.set_num_threads(1)
const TYPES = (Float64x2, Float64x3, Float64x4)

median_seconds(trial) = BenchmarkTools.median(trial).time / 1e9

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

    calibration = calibrate_gemm(
        T;
        sizes=(128, 256),
        samples=1,
        thread_count=threads,
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
        "crossover=$(profile.packed_crossover), fingerprint=$(profile.fingerprint)",
    )

    direct_trial = @benchmark MultiFloatLinearAlgebra.gemm!(
        $Cdirect, $A, $B; config=$direct_config,
    ) samples=$samples evals=1
    packed_trial = @benchmark MultiFloatLinearAlgebra.gemm!(
        $Cpacked, $A, $B; config=$packed_config, workspace=$workspace,
    ) samples=$samples evals=1
    auto_trial = @benchmark MultiFloatLinearAlgebra.gemm!(
        $Cauto, $A, $B; config=$auto_config, workspace=$workspace,
    ) samples=$samples evals=1

    direct_seconds = median_seconds(direct_trial)
    packed_seconds = median_seconds(packed_trial)
    auto_seconds = median_seconds(auto_trial)
    Cpacked == Cdirect || error("packed GEMM changed the per-output reduction result")
    Cauto == Cdirect || error("auto GEMM changed the per-output reduction result")

    if include_generic
        Cgeneric = zeros(T, n, n)
        generic_trial = @benchmark LinearAlgebra.mul!(
            $Cgeneric, $A, $B,
        ) samples=1 evals=1
        generic_seconds = median_seconds(generic_trial)
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
    float_trial = @benchmark mul!($C64, $A64, $B64) samples=$samples evals=1
    report("Float64", "BLAS(1 thread)", n, median_seconds(float_trial))
    println()

    for T in TYPES
        benchmark_type(T, n, samples, include_generic)
    end
end

main()
