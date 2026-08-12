using BenchmarkTools
using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra
using Printf
using Random

Random.seed!(20260812)
BLAS.set_num_threads(1)

const TYPES = (Float64x2, Float64x3, Float64x4)

function median_seconds(trial)
    BenchmarkTools.median(trial).time / 1e9
end

function report_row(operation, arithmetic, backend, n, seconds, reference=nothing)
    ratio = reference === nothing ? NaN : reference / seconds
    @printf(
        "%-10s %-12s %-18s n=%4d  %10.6f s",
        operation, arithmetic, backend, n, seconds,
    )
    if reference !== nothing
        @printf("  speedup=%6.2fx", ratio)
    end
    println()
end

function make_spd(::Type{T}, n) where {T}
    R = randn(n, n)
    A = R * transpose(R)
    @inbounds for i in 1:n
        A[i, i] += n
    end
    return T.(A)
end

function make_lower(::Type{T}, n) where {T}
    L = zeros(T, n, n)
    @inbounds for column in 1:n
        for row in column:n
            L[row, column] = T(randn())
        end
        L[column, column] += T(4)
    end
    return L
end

function benchmark_gemm(::Type{T}, n, config) where {T}
    A = T.(randn(n, n))
    B = T.(randn(n, n))
    Cgeneric = zeros(T, n, n)
    Cbackend = zeros(T, n, n)

    generic = @benchmark LinearAlgebra.mul!(
        $Cgeneric, $A, $B,
    ) samples=5 evals=1
    backend = @benchmark MultiFloatLinearAlgebra.gemm!(
        $Cbackend, $A, $B; config=$config,
    ) samples=5 evals=1

    tg = median_seconds(generic)
    tb = median_seconds(backend)
    report_row("GEMM", string(T), "LinearAlgebra", n, tg)
    report_row("GEMM", string(T), "MFLA", n, tb, tg)
end

function benchmark_trsm(::Type{T}, n, config) where {T}
    L = make_lower(T, n)
    rhs_columns = min(32, n)
    rhs = T.(randn(n, rhs_columns))
    Ltri = LowerTriangular(L)

    generic = @benchmark LinearAlgebra.ldiv!(
        $Ltri, X,
    ) setup=(X=copy($rhs)) samples=5 evals=1
    backend = @benchmark MultiFloatLinearAlgebra.trsm!(
        X, $L; side=:left, uplo=:lower, trans=:N,
        diag=:nonunit, config=$config,
    ) setup=(X=copy($rhs)) samples=5 evals=1

    tg = median_seconds(generic)
    tb = median_seconds(backend)
    report_row("TRSM", string(T), "LinearAlgebra", n, tg)
    report_row("TRSM", string(T), "MFLA", n, tb, tg)
end

function benchmark_cholesky(::Type{T}, n, config) where {T}
    A = make_spd(T, n)

    generic = @benchmark LinearAlgebra.cholesky!(
        Hermitian(X, :L); check=false,
    ) setup=(X=copy($A)) samples=5 evals=1

    backend = @benchmark MultiFloatLinearAlgebra.cholesky!(
        X; check=false, config=$config,
    ) setup=(X=copy($A)) samples=5 evals=1

    tg = median_seconds(generic)
    tb = median_seconds(backend)
    report_row("CHOLESKY", string(T), "LinearAlgebra", n, tg)
    report_row("CHOLESKY", string(T), "MFLA", n, tb, tg)
end

function benchmark_lu(::Type{T}, n, config) where {T}
    A = T.(randn(n, n))
    @inbounds for i in 1:n
        A[i, i] += T(4)
    end

    generic = @benchmark LinearAlgebra.lu!(
        X; check=false,
    ) setup=(X=copy($A)) samples=5 evals=1

    backend = @benchmark MultiFloatLinearAlgebra.lu!(
        X; check=false, config=$config,
    ) setup=(X=copy($A)) samples=5 evals=1

    tg = median_seconds(generic)
    tb = median_seconds(backend)
    report_row("LU", string(T), "LinearAlgebra", n, tg)
    report_row("LU", string(T), "MFLA", n, tb, tg)
end

function benchmark_ldlt(::Type{T}, n) where {T}
    R = randn(n, n)
    A64 = 0.02 .* (R + transpose(R))
    @inbounds for i in 1:n
        A64[i, i] += isodd(i) ? n : -n
    end
    A = T.(Matrix(A64))
    trial = @benchmark MultiFloatLinearAlgebra.ldlt!(
        X; check=false,
    ) setup=(X=copy($A)) samples=5 evals=1
    report_row("LDLT", string(T), "MFLA", n, median_seconds(trial))
end

function benchmark_float64(n)
    A = randn(n, n)
    B = randn(n, n)
    C = zeros(n, n)
    trial = @benchmark LinearAlgebra.mul!($C, $A, $B) samples=5 evals=1
    report_row("GEMM", "Float64", "BLAS(1 thread)", n, median_seconds(trial))
end

function benchmark_bigfloat(n, bits)
    setprecision(BigFloat, bits) do
        A = BigFloat.(randn(n, n))
        B = BigFloat.(randn(n, n))
        C = zeros(BigFloat, n, n)
        trial = @benchmark LinearAlgebra.mul!($C, $A, $B) samples=3 evals=1
        report_row(
            "GEMM", "BigFloat/$bits", "LinearAlgebra", n, median_seconds(trial),
        )
    end
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 96
    threads = Threads.nthreads()
    config = KernelConfig(thread_count=threads)
    println("MultiFloatLinearAlgebra benchmark")
    println("Julia threads: $threads; BLAS threads: 1; matrix size: $n")
    println()

    benchmark_float64(n)
    for T in TYPES
        benchmark_gemm(T, n, config)
    end
    println()

    factor_n = min(n, 256)
    for T in TYPES
        benchmark_trsm(T, factor_n, config)
        benchmark_cholesky(T, factor_n, config)
        benchmark_lu(T, factor_n, config)
        benchmark_ldlt(T, min(factor_n, 128))
    end

    if "--bigfloat" in ARGS
        println()
        benchmark_bigfloat(min(n, 64), 256)
    end
end

main()
