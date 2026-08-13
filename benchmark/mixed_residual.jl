using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra
using Printf
using Random

Random.seed!(0x6d69786564)
BLAS.set_num_threads(1)

const PAIRS = (
    (Float64x2, Float64x3),
    (Float64x2, Float64x4),
    (Float64x3, Float64x4),
)

limb_count(::Type{MultiFloat{Float64,N}}) where {N} = N

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

function report(
    operation,
    ::Type{Source},
    ::Type{Residual},
    threads,
    shape,
    seconds,
    bytes,
) where {Source,Residual}
    @printf(
        "%-8s x%d->x%d threads=%d  %-16s %10.6f s  %8d bytes\n",
        operation,
        limb_count(Source),
        limb_count(Residual),
        threads,
        shape,
        seconds,
        bytes,
    )
end

function benchmark_pair(
    ::Type{Source},
    ::Type{Residual},
    rows,
    columns,
    right_hand_sides,
    samples,
    threads,
) where {Source,Residual}
    config = KernelConfig(thread_count=threads)
    A = Source.(randn(rows, columns))
    x = Source.(randn(columns))
    b = Source.(randn(rows))
    low = zeros(Source, rows)
    high = zeros(Residual, rows)

    ordinary_call = () -> residual!(low, A, x, b; config=config)
    mixed_call = () -> residual_mixed!(high, A, x, b; config=config)
    report(
        "ordinary",
        Source,
        Source,
        threads,
        "$(rows)x$(columns)x1",
        median_seconds(ordinary_call, samples),
        allocation_bytes(ordinary_call),
    )
    report(
        "mixed",
        Source,
        Residual,
        threads,
        "$(rows)x$(columns)x1",
        median_seconds(mixed_call, samples),
        allocation_bytes(mixed_call),
    )

    X = Source.(randn(columns, right_hand_sides))
    B = Source.(randn(rows, right_hand_sides))
    Low = zeros(Source, rows, right_hand_sides)
    High = zeros(Residual, rows, right_hand_sides)
    ordinary_matrix_call = () -> residual!(Low, A, X, B; config=config)
    mixed_matrix_call = () -> residual_mixed!(High, A, X, B; config=config)
    report(
        "ordinary",
        Source,
        Source,
        threads,
        "$(rows)x$(columns)x$(right_hand_sides)",
        median_seconds(ordinary_matrix_call, samples),
        allocation_bytes(ordinary_matrix_call),
    )
    report(
        "mixed",
        Source,
        Residual,
        threads,
        "$(rows)x$(columns)x$(right_hand_sides)",
        median_seconds(mixed_matrix_call, samples),
        allocation_bytes(mixed_matrix_call),
    )
end

function main()
    rows = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 257
    columns = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 129
    right_hand_sides = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 4
    samples = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 5
    minimum((rows, columns, right_hand_sides, samples)) > 0 ||
        throw(ArgumentError("all benchmark dimensions and sample count must be positive"))
    thread_counts = unique((1, min(4, Threads.nthreads())))

    println("Explicit mixed-MultiFloat residual benchmark")
    println(
        "Julia $(VERSION), Julia threads=$(Threads.nthreads()), " *
        "BLAS threads=1, samples=$samples",
    )
    println("All inputs and outputs are preallocated; reductions ascend by source column.")
    println()
    for threads in thread_counts, (Source, Residual) in PAIRS
        benchmark_pair(
            Source,
            Residual,
            rows,
            columns,
            right_hand_sides,
            samples,
            threads,
        )
        println()
    end
end

main()
