using MultiFloats
using MultiFloatLinearAlgebra
using Printf
using Random

Random.seed!(20260812)
const TYPES = (Float64x2, Float64x3, Float64x4)

panel_columns(::Type{Float64x2}) = 32
panel_columns(::Type{Float64x3}) = 24
panel_columns(::Type{Float64x4}) = 16

function median_seconds(f, samples)
    f()
    elapsed = Vector{Float64}(undef, samples)
    for sample in eachindex(elapsed)
        GC.gc()
        start = time_ns()
        f()
        elapsed[sample] = (time_ns() - start) / 1e9
    end
    sort!(elapsed)
    return elapsed[cld(samples, 2)]
end

@inline function direct4_vecblock!(C, A, B, row, column)
    T = eltype(A)
    V4 = MultiFloatVec{4,Float64,T.parameters[2]}
    acc1 = zero(V4)
    acc2 = zero(V4)
    acc3 = zero(V4)
    acc4 = zero(V4)
    @inbounds for k in axes(A, 2)
        values = V4(
            A[row, k],
            A[row + 1, k],
            A[row + 2, k],
            A[row + 3, k],
        )
        acc1 += values * V4(B[k, column])
        acc2 += values * V4(B[k, column + 1])
        acc3 += values * V4(B[k, column + 2])
        acc4 += values * V4(B[k, column + 3])
    end
    @inbounds for lane in 1:4
        r = row + lane - 1
        C[r, column] = acc1[lane]
        C[r, column + 1] = acc2[lane]
        C[r, column + 2] = acc3[lane]
        C[r, column + 3] = acc4[lane]
    end
    return nothing
end

@inline function direct4_scalarblock!(C, A, B, row, column)
    T = eltype(A)
    acc1 = zero(T)
    acc2 = zero(T)
    acc3 = zero(T)
    acc4 = zero(T)
    @inbounds for k in axes(A, 2)
        value = A[row, k]
        acc1 += value * B[k, column]
        acc2 += value * B[k, column + 1]
        acc3 += value * B[k, column + 2]
        acc4 += value * B[k, column + 3]
    end
    C[row, column] = acc1
    C[row, column + 1] = acc2
    C[row, column + 2] = acc3
    C[row, column + 3] = acc4
    return nothing
end

function direct4_range!(C, A, B, first_column, last_column)
    m = size(A, 1)
    column = first_column
    @inbounds while column + 3 <= last_column
        row = 1
        while row + 3 <= m
            direct4_vecblock!(C, A, B, row, column)
            row += 4
        end
        while row <= m
            direct4_scalarblock!(C, A, B, row, column)
            row += 1
        end
        column += 4
    end
    if column <= last_column
        MultiFloatLinearAlgebra._gemm_direct_column_range!(
            C,
            A,
            B,
            one(eltype(A)),
            zero(eltype(A)),
            column,
            last_column,
        )
    end
    return C
end

function direct4!(C, A, B, panel)
    n = size(B, 2)
    jobs = cld(n, panel)
    workers = max(1, min(Threads.nthreads(), jobs))
    if workers == 1
        for job in 1:jobs
            first_column = (job - 1) * panel + 1
            last_column = min(job * panel, n)
            direct4_range!(C, A, B, first_column, last_column)
        end
        return C
    end
    @sync for worker in 1:workers
        Threads.@spawn begin
            for job in worker:workers:jobs
                first_column = (job - 1) * panel + 1
                last_column = min(job * panel, n)
                direct4_range!(C, A, B, first_column, last_column)
            end
        end
    end
    return C
end

function report(T, n, label, seconds, reference)
    @printf(
        "%-12s n=%4d %-13s %10.6f s %8.3f matrix-GF/s speedup=%6.3fx\n",
        string(T),
        n,
        label,
        seconds,
        2.0 * n^3 / seconds / 1e9,
        reference / seconds,
    )
end

function benchmark_type(::Type{T}, n, samples) where {T}
    A = T.(randn(n, n))
    B = T.(randn(n, n))
    baseline = zeros(T, n, n)
    candidate = zeros(T, n, n)
    panel = panel_columns(T)
    config = KernelConfig(
        thread_count=Threads.nthreads(),
        gemm_strategy=:direct,
        gemm_panel_columns=panel,
    )
    baseline_seconds = median_seconds(samples) do
        MultiFloatLinearAlgebra.gemm!(baseline, A, B; config=config)
    end
    candidate_seconds = median_seconds(samples) do
        direct4!(candidate, A, B, panel)
    end
    candidate == baseline || error("direct4 changed the direct GEMM result for $T")
    report(T, n, "direct-pair", baseline_seconds, baseline_seconds)
    report(T, n, "direct-four", candidate_seconds, baseline_seconds)
    println()
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 512
    samples = n >= 1024 ? 1 : 3
    println("Four-column direct MultiFloat GEMM scheduling experiment")
    println("Julia=$(VERSION) threads=$(Threads.nthreads()) n=$n samples=$samples")
    println()
    for T in TYPES
        benchmark_type(T, n, samples)
    end
end

main()
