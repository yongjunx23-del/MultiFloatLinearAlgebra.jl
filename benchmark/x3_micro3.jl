using MultiFloats
using MultiFloatLinearAlgebra
using Printf
using Random

Random.seed!(20260812)
const T = Float64x3

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

function packed3!(C, A, B, panel_columns)
    n = size(B, 2)
    reduction = size(A, 2)
    jobs = cld(n, panel_columns)
    workers = max(1, min(Threads.nthreads(), jobs))
    buffers = [Vector{T}(undef, reduction * panel_columns) for _ in 1:workers]

    function run_worker(worker)
        buffer = buffers[worker]
        for job in worker:workers:jobs
            first_column = (job - 1) * panel_columns + 1
            last_column = min(job * panel_columns, n)
            width = MultiFloatLinearAlgebra._pack_b_panel!(
                buffer, B, first_column, last_column,
            )
            MultiFloatLinearAlgebra._packed_gemm_panel_nr!(
                C,
                A,
                buffer,
                first_column,
                width,
                one(T),
                zero(T),
                Val(3),
            )
        end
        return nothing
    end

    if workers == 1
        run_worker(1)
    else
        @sync for worker in 1:workers
            Threads.@spawn run_worker(worker)
        end
    end
    return C
end

function report(label, n, seconds, reference)
    gflops = 2.0 * n^3 / seconds / 1.0e9
    @printf(
        "%-18s n=%4d %10.6f s %8.3f matrix-GF/s speedup=%6.3fx\n",
        label,
        n,
        seconds,
        gflops,
        reference / seconds,
    )
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 512
    samples = n >= 1024 ? 1 : 3
    A = T.(randn(n, n))
    B = T.(randn(n, n))

    direct = zeros(T, n, n)
    direct_config = KernelConfig(
        thread_count=Threads.nthreads(),
        gemm_strategy=:direct,
        gemm_panel_columns=32,
        gemm_micro_columns=2,
    )
    direct_seconds = median_seconds(samples) do
        MultiFloatLinearAlgebra.gemm!(direct, A, B; config=direct_config)
    end

    baseline = zeros(T, n, n)
    baseline_workspace = GemmWorkspace(
        T;
        thread_count=Threads.nthreads(),
        capacity=n * 32,
    )
    baseline_config = KernelConfig(
        thread_count=Threads.nthreads(),
        gemm_strategy=:packed,
        gemm_panel_columns=32,
        gemm_micro_columns=2,
    )
    baseline_seconds = median_seconds(samples) do
        MultiFloatLinearAlgebra.gemm!(
            baseline,
            A,
            B;
            config=baseline_config,
            workspace=baseline_workspace,
        )
    end
    baseline == direct || error("32x2 packed baseline changed the direct result")

    println("Float64x3 three-column GEMM microkernel experiment")
    println("threads=$(Threads.nthreads()), n=$n, samples=$samples")
    report("direct-32x2", n, direct_seconds, direct_seconds)
    report("packed-32x2", n, baseline_seconds, direct_seconds)

    for panel_columns in (24, 30, 32)
        output = zeros(T, n, n)
        seconds = median_seconds(samples) do
            packed3!(output, A, B, panel_columns)
        end
        output == direct || error("$(panel_columns)x3 changed the direct result")
        report("packed-$(panel_columns)x3", n, seconds, direct_seconds)
    end
end

main()
