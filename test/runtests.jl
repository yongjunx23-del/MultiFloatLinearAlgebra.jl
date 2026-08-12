using Test
using Random
using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra

Random.seed!(0x5eed)

function max_relative_error(A, B)
    @assert size(A) == size(B)
    return setprecision(BigFloat, 512) do
        numerator = BigFloat(0)
        denominator = BigFloat(0)
        for i in eachindex(A, B)
            a = BigFloat(A[i])
            b = BigFloat(B[i])
            numerator = max(numerator, abs(a - b))
            denominator = max(denominator, abs(b))
        end
        numerator / max(denominator, BigFloat(1))
    end
end

function tolerance(::Type{T}, work::Int=1) where {T}
    setprecision(BigFloat, 512) do
        BigFloat(4096 * max(work, 1)) * BigFloat(eps(T))
    end
end

@testset "MultiFloatLinearAlgebra" begin
    for T in (Float64x2, Float64x3, Float64x4)
        @testset "$T" begin
            n = 12
            A = T.(randn(n, n))
            B = T.(randn(n, n))
            x = T.(randn(n))

            @testset "dot" begin
                reference = zero(T)
                for i in eachindex(x)
                    reference += x[i] * x[i]
                end
                @test mfdot(x, x) == reference
            end

            @testset "gemv!" begin
                reference = A * x
                output = zeros(T, n)
                gemv!(output, A, x; config=KernelConfig(thread_count=2))
                @test max_relative_error(output, reference) <= tolerance(T, n)
            end

            @testset "gemm!" begin
                reference = A * B
                output = zeros(T, n, n)
                gemm!(output, A, B; config=KernelConfig(thread_count=2))
                @test max_relative_error(output, reference) <= tolerance(T, n)
            end

            @testset "syrk!" begin
                panel = T.(randn(9, n))
                reference = transpose(panel) * panel
                output = zeros(T, n, n)
                syrk!(output, panel; config=KernelConfig(thread_count=2))
                lower_error = max_relative_error(
                    LowerTriangular(output),
                    LowerTriangular(reference),
                )
                @test lower_error <= tolerance(T, size(panel, 1))
            end

            @testset "cholesky!" begin
                R = randn(n, n)
                A64 = R * transpose(R) + n * I
                Aspd = T.(Matrix(A64))
                original = copy(Aspd)
                F = cholesky!(
                    Aspd;
                    config=KernelConfig(
                        cholesky_block=4,
                        thread_count=2,
                    ),
                )
                @test issuccess(F)
                L = Matrix(LowerTriangular(F.factors))
                reconstructed = L * transpose(L)
                @test max_relative_error(reconstructed, original) <=
                    tolerance(T, 8n)
            end

            @testset "LU solve" begin
                Alu = T.(randn(n, n))
                for i in 1:n
                    Alu[i, i] += T(4)
                end
                original = copy(Alu)
                rhs = T.(randn(n))
                rhs_original = copy(rhs)
                F = lu!(Alu)
                @test issuccess(F)
                solution = solve(F, rhs)
                residual = original * solution - rhs_original
                @test max_relative_error(residual, zeros(T, n)) <=
                    tolerance(T, 64n)
            end
        end
    end
end
