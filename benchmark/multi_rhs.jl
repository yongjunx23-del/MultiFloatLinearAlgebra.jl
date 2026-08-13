using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra
using Printf
using Random

Random.seed!(0x6d727873)
BLAS.set_num_threads(1)

const TYPES = (Float64x2, Float64x3, Float64x4)

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

function report(kind, ::Type{T}, n, nrhs, threads, seconds, bytes, error) where {T}
    @printf(
        "%-8s x%d n=%4d nrhs=%2d threads=%d %9.6fs %6dB backward=%9.2e\n",
        kind,
        limb_count(T),
        n,
        nrhs,
        threads,
        seconds,
        bytes,
        error,
    )
end

function make_systems(::Type{T}, n) where {T}
    R = randn(n, n)
    spd = R * transpose(R)
    @inbounds for index in 1:n
        spd[index, index] += n
    end
    general = randn(n, n)
    @inbounds for index in 1:n
        general[index, index] += 5
    end
    indefinite = zeros(Float64, n, n)
    @inbounds for index in 1:2:(n - 1)
        indefinite[index, index + 1] = 1 + index / n
        indefinite[index + 1, index] = indefinite[index, index + 1]
    end
    isodd(n) && (indefinite[n, n] = 1)
    return T.(spd), T.(general), T.(indefinite)
end

function benchmark_factor(kind, A, factor, ::Type{T}, n, nrhs, threads, samples) where {T}
    config = KernelConfig(thread_count=threads)
    truth = T.(randn(n, nrhs))
    B = zeros(T, n, nrhs)
    gemm!(
        B,
        A,
        truth;
        config=KernelConfig(thread_count=1, gemm_strategy=:direct),
    )
    destination = similar(B)
    solve_call = () -> begin
        copyto!(destination, B)
        MultiFloatLinearAlgebra.ldiv!(destination, factor; config=config)
    end
    solve_call()
    residual_storage = similar(B)
    residual!(residual_storage, A, destination, B; config=config)
    errors = normwise_backward_error(A, destination, B, residual_storage)
    report(
        kind,
        T,
        n,
        nrhs,
        threads,
        median_seconds(solve_call, samples),
        allocation_bytes(solve_call),
        Float64(maximum(errors)),
    )
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 256
    samples = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 7
    n > 1 || throw(ArgumentError("matrix size must exceed one"))
    samples > 0 || throw(ArgumentError("sample count must be positive"))
    thread_counts = unique((1, min(4, Threads.nthreads())))
    println("Dense factor multi-RHS solve benchmark")
    println(
        "Julia $(VERSION), Julia threads=$(Threads.nthreads()), " *
        "BLAS threads=1, n=$n, samples=$samples",
    )
    println()
    for T in TYPES
        spd, general, indefinite = make_systems(T, n)
        factor_config = KernelConfig(
            thread_count=min(4, Threads.nthreads()),
            ldlt_strategy=:blocked,
            ldlt_block=16,
        )
        factors = (
            ("chol", spd, MultiFloatLinearAlgebra.cholesky!(copy(spd); config=factor_config)),
            ("lu", general, MultiFloatLinearAlgebra.lu!(copy(general); config=factor_config)),
            ("ldlt", indefinite, MultiFloatLinearAlgebra.ldlt!(copy(indefinite); config=factor_config)),
        )
        for threads in thread_counts, nrhs in (1, 2, 4, 8)
            for (kind, A, factor) in factors
                benchmark_factor(
                    kind, A, factor, T, n, nrhs, threads, samples,
                )
            end
        end
        println()
    end
end

main()
