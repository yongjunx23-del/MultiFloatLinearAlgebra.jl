using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra
using Printf
using Random

Random.seed!(0x7368617065)
BLAS.set_num_threads(1)

const TYPES = (Float64x2, Float64x3, Float64x4)
const SHAPES = (
    ("square", 256, 256, 256),
    ("tall", 1024, 128, 64),
    ("wide", 64, 128, 1024),
    ("moderate", 384, 192, 256),
)

limb_count(::Type{MultiFloat{Float64,N}}) where {N} = N
baseline_strategy(::Type{Float64x3}) = :fused
baseline_strategy(::Type{<:MultiFloat}) = :direct
panel_columns(::Type{Float64x2}) = 32
panel_columns(::Type{Float64x3}) = 24
panel_columns(::Type{Float64x4}) = 16
micro_columns(::Type{Float64x4}) = 2
micro_columns(::Type{<:MultiFloat}) = 4

function median_seconds(f, samples)
    f()
    elapsed = Vector{Float64}(undef, samples)
    for sample in 1:samples
        GC.gc()
        start = time_ns()
        f()
        elapsed[sample] = (time_ns() - start) / 1.0e9
    end
    sort!(elapsed)
    return elapsed[cld(samples, 2)]
end

function allocation_bytes(f)
    f()
    GC.gc()
    return @allocated f()
end

function benchmark_shape(::Type{T}, label, m, k, n, threads, samples) where {T}
    A = T.(randn(m, k))
    B = T.(randn(k, n))
    direct = zeros(T, m, n)
    packed = zeros(T, m, n)
    panel = panel_columns(T)
    direct_config = KernelConfig(
        thread_count=threads,
        gemm_strategy=baseline_strategy(T),
        gemm_panel_columns=panel,
    )
    packed_config = KernelConfig(
        thread_count=threads,
        gemm_strategy=:packed,
        gemm_panel_columns=panel,
        gemm_micro_columns=micro_columns(T),
    )
    workspace = GemmWorkspace(
        T;
        thread_count=threads,
        capacity=k * panel,
    )

    direct_call = () -> gemm!(direct, A, B; config=direct_config)
    packed_call = () -> gemm!(
        packed, A, B; config=packed_config, workspace=workspace,
    )
    direct_call()
    packed_call()
    direct == packed || error(
        "packed GEMM changed the direct-family result for $T/$label",
    )

    direct_seconds = median_seconds(direct_call, samples)
    packed_seconds = median_seconds(packed_call, samples)
    direct_allocations = allocation_bytes(direct_call)
    packed_allocations = allocation_bytes(packed_call)
    calibrated_config = KernelConfig(
        thread_count=threads,
        gemm_packed_crossover=128,
        gemm_panel_columns=panel,
        gemm_micro_columns=micro_columns(T),
    )
    plan = gemm_plan(T, m, k, n, calibrated_config)
    @printf(
        "x%d %-9s %4dx%4dx%4d threads=%d direct=%9.6fs packed=%9.6fs speedup=%6.3fx alloc=%d/%dB plan=%s/%s\n",
        limb_count(T),
        label,
        m,
        k,
        n,
        threads,
        direct_seconds,
        packed_seconds,
        direct_seconds / packed_seconds,
        direct_allocations,
        packed_allocations,
        plan.strategy,
        plan.reason,
    )
end

function main()
    samples = isempty(ARGS) ? 5 : parse(Int, ARGS[1])
    samples > 0 || throw(ArgumentError("sample count must be positive"))
    thread_counts = unique((1, min(4, Threads.nthreads())))
    println("Solver-shape direct-family versus packed GEMM")
    println(
        "Julia $(VERSION), Julia threads=$(Threads.nthreads()), " *
        "BLAS threads=1, samples=$samples",
    )
    println("Speedup > 1 favors packed; outputs must be exactly equal.")
    println()
    for threads in thread_counts, T in TYPES
        for (label, m, k, n) in SHAPES
            benchmark_shape(T, label, m, k, n, threads, samples)
        end
        println()
    end
end

main()
