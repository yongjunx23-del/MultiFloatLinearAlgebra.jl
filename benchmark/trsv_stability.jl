using BenchmarkTools
using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra
using Printf

const TYPES = (Float64x2, Float64x3, Float64x4)
const SIZES = (64, 128, 256, 512, 1024)

limbcount(::Type{MultiFloat{Float64,N}}) where {N} = N

function make_lower(::Type{T}, n) where {T}
    A = zeros(T, n, n)
    for c in 1:n, r in c:n
        A[r, c] = T(randn())
    end
    for i in 1:n
        A[i, i] += T(3)
    end
    return A
end

function bench_stats(trial)
    estimate = BenchmarkTools.median(trial)
    return (seconds=estimate.time / 1e9, bytes=estimate.memory)
end

println("=== TRSV vs one-column TRSM (stable, BenchmarkTools) ===")
println("lower / nonunit, trans=N; kernel and end-to-end (copy reset) measured separately")
for T in TYPES
    for n in SIZES
        A = make_lower(T, n)
        x = T.(randn(n))

        trsv_kernel_trial = @benchmarkable begin
            trsv!(b, $A; uplo=:lower, trans=:N, diag=:nonunit,
                  config=KernelConfig(thread_count=1))
        end setup=(b=copy($x))
        trsm_kernel_trial = @benchmarkable begin
            trsm!(B, $A; side=:left, uplo=:lower, trans=:N, diag=:nonunit,
                  config=KernelConfig(thread_count=1))
        end setup=(B=reshape(copy($x), $n, 1))
        trsv_end_to_end_trial = @benchmarkable begin
            b = copy($x)
            trsv!(b, $A; uplo=:lower, trans=:N, diag=:nonunit,
                  config=KernelConfig(thread_count=1))
        end
        trsm_end_to_end_trial = @benchmarkable begin
            B = reshape(copy($x), $n, 1)
            trsm!(B, $A; side=:left, uplo=:lower, trans=:N, diag=:nonunit,
                  config=KernelConfig(thread_count=1))
        end

        trsv_kernel = bench_stats(run(trsv_kernel_trial, samples=30, evals=1))
        trsm_kernel = bench_stats(run(trsm_kernel_trial, samples=30, evals=1))
        trsv_end_to_end =
            bench_stats(run(trsv_end_to_end_trial, samples=30, evals=5))
        trsm_end_to_end =
            bench_stats(run(trsm_end_to_end_trial, samples=30, evals=5))

        @printf(
            "x%d n=%4d  kernel trsv=%9.6fs trsm=%9.6fs ratio=%5.2fx  end-to-end trsv=%9.6fs trsm=%9.6fs ratio=%5.2fx  alloc kernel=%d/%dB end-to-end=%d/%dB\n",
            limbcount(T), n,
            trsv_kernel.seconds, trsm_kernel.seconds,
            trsv_kernel.seconds / trsm_kernel.seconds,
            trsv_end_to_end.seconds, trsm_end_to_end.seconds,
            trsv_end_to_end.seconds / trsm_end_to_end.seconds,
            trsv_kernel.bytes, trsm_kernel.bytes,
            trsv_end_to_end.bytes, trsm_end_to_end.bytes,
        )
    end
end
