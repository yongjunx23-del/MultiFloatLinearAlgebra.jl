using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra
using Printf
using Random

Random.seed!(0x1a2b)
BLAS.set_num_threads(1)

const TYPES = (Float64x2, Float64x3, Float64x4)

arithmetic_label(::Type{MultiFloat{Float64,N}}) where {N} = "x$N"

function median_seconds(f, samples)
    f()
    elapsed = Vector{Float64}(undef, samples)
    for i in 1:samples
        GC.gc()
        t0 = time_ns()
        f()
        elapsed[i] = (time_ns() - t0) / 1.0e9
    end
    sort!(elapsed)
    return elapsed[cld(samples, 2)]
end

function report(operation, arithmetic, shape, backend, seconds, reference=nothing)
    ratio = reference === nothing ? NaN : reference / seconds
    @printf("%-16s %-10s %-12s %-18s %10.6f s", operation, arithmetic, shape, backend, seconds)
    if reference !== nothing
        @printf("  %6.2fx", ratio)
    end
    println()
end

println("=== transpose-GEMV (1 thread, alpha=1, beta=0) ===")
for T in TYPES
    for (m, n) in ((64, 4096), (4096, 64), (1024, 1024))
        A = T.(randn(m, n))
        x = T.(randn(m))
        y = zeros(T, n)
        s = median_seconds(() -> gemv!(y, A, x; trans=:T, config=KernelConfig(thread_count=1)), 5)
        report("gemvT", arithmetic_label(T), "$(m)x$(n)", "transpose", s)
    end
end

println("\n=== TRSV vs one-column TRSM (1 thread) ===")
for T in TYPES
    for n in (64, 256, 1024)
        A = zeros(T, n, n)
        for c in 1:n, r in c:n
            A[r, c] = T(randn())
        end
        for i in 1:n
            A[i, i] += T(3)
        end
        x = T.(randn(n))
        trsv_t = median_seconds(
            () -> trsv!(copy(x), A; uplo=:lower, trans=:N, diag=:nonunit, config=KernelConfig(thread_count=1)),
            5,
        )
        trsm_t = median_seconds(
            () -> trsm!(reshape(copy(x), n, 1), A; side=:left, uplo=:lower, trans=:N, diag=:nonunit, config=KernelConfig(thread_count=1)),
            5,
        )
        report("trsv", arithmetic_label(T), "n=$n", "vector", trsv_t)
        report("trsm", arithmetic_label(T), "n=$n", "n×1", trsm_t, trsv_t)
    end
end

println("\n=== SYMV (1 thread, lower authoritative) ===")
for T in TYPES
    for n in (64, 256, 1024)
        R = randn(n, n)
        S = T.(Matrix(R + transpose(R)))
        x = T.(randn(n))
        y = zeros(T, n)
        s = median_seconds(
            () -> symv!(y, S, x; uplo=:lower, config=KernelConfig(thread_count=1)),
            5,
        )
        report("symv", arithmetic_label(T), "n=$n", "lower", s)
    end
end
