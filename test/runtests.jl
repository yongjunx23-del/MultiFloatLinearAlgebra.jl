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

function triangular_matrix(::Type{T}, n, uplo; unit=false) where {T}
    A = zeros(T, n, n)
    if uplo === :lower
        for column in 1:n, row in column:n
            A[row, column] = T(randn())
        end
    else
        for column in 1:n, row in 1:column
            A[row, column] = T(randn())
        end
    end
    for i in 1:n
        A[i, i] = unit ? one(T) : A[i, i] + T(3)
    end
    return A
end

@testset "MultiFloatLinearAlgebra" begin
    for T in (Float64x2, Float64x3, Float64x4)
        @testset "$T" begin
            n = 12
            config = KernelConfig(
                cholesky_block=4,
                lu_block=4,
                column_tile=4,
                thread_count=2,
            )
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
                gemv!(output, A, x; config=config)
                @test max_relative_error(output, reference) <= tolerance(T, n)
            end

            @testset "direct and packed gemm!" begin
                reference = A * B
                direct = zeros(T, n, n)
                direct_config = KernelConfig(
                    thread_count=2,
                    gemm_strategy=:direct,
                    gemm_panel_columns=5,
                )
                gemm!(direct, A, B; config=direct_config)
                @test max_relative_error(direct, reference) <= tolerance(T, n)

                packed = zeros(T, n, n)
                packed_config = KernelConfig(
                    thread_count=2,
                    gemm_strategy=:packed,
                    gemm_panel_columns=5,
                    gemm_micro_columns=4,
                )
                workspace = GemmWorkspace(
                    T;
                    thread_count=2,
                    capacity=n * 5,
                )
                gemm!(
                    packed,
                    A,
                    B;
                    config=packed_config,
                    workspace=workspace,
                )
                @test packed == direct

                direct_plan = gemm_plan(T, n, n, n, direct_config)
                packed_plan = gemm_plan(T, n, n, n, packed_config)
                @test direct_plan.strategy === :direct
                @test packed_plan.strategy === :packed
                @test packed_plan.panel_columns == 5
                @test packed_plan.micro_columns == 4
            end

            @testset "GEMM calibration contract" begin
                builtin = default_gemm_profile(T; thread_count=2)
                resolved = with_gemm_profile(config, builtin)
                plan = gemm_plan(T, 256, 256, 256, resolved)
                @test plan.strategy in (:direct, :packed)
                @test builtin.source === :builtin
                @test builtin.fingerprint.julia_threads == 2

                calibration = calibrate_gemm(
                    T;
                    sizes=(8,),
                    samples=1,
                    thread_count=1,
                    candidates=((4, 2), (4, 4)),
                )
                @test calibration.profile.source === :calibrated
                @test length(calibration.measurements) == 3
                @test calibration.profile.panel_columns == 4
                @test calibration.profile.micro_columns in (2, 4)
            end

            @testset "syrk!" begin
                panel = T.(randn(9, n))
                reference = transpose(panel) * panel
                output = zeros(T, n, n)
                syrk!(output, panel; config=config)
                lower_error = max_relative_error(
                    LowerTriangular(output),
                    LowerTriangular(reference),
                )
                @test lower_error <= tolerance(T, size(panel, 1))
            end

            @testset "trsm!" begin
                Xleft = T.(randn(n, 6))
                Xright = T.(randn(6, n))
                for uplo in (:lower, :upper), trans in (:N, :T)
                    triangular = triangular_matrix(T, n, uplo)
                    opA = trans === :N ? triangular : transpose(triangular)

                    left_rhs = Matrix(opA * Xleft)
                    trsm!(
                        left_rhs,
                        triangular;
                        side=:left,
                        uplo=uplo,
                        trans=trans,
                        diag=:nonunit,
                        config=config,
                    )
                    @test max_relative_error(left_rhs, Xleft) <= tolerance(T, 8n)

                    right_rhs = Matrix(Xright * opA)
                    trsm!(
                        right_rhs,
                        triangular;
                        side=:right,
                        uplo=uplo,
                        trans=trans,
                        diag=:nonunit,
                        config=config,
                    )
                    @test max_relative_error(right_rhs, Xright) <= tolerance(T, 8n)
                end

                unit_lower = triangular_matrix(T, n, :lower; unit=true)
                unit_rhs = unit_lower * Xleft
                trsm!(
                    unit_rhs,
                    unit_lower;
                    side=:left,
                    uplo=:lower,
                    trans=:N,
                    diag=:unit,
                    config=config,
                )
                @test max_relative_error(unit_rhs, Xleft) <= tolerance(T, 8n)
            end

            @testset "cholesky!" begin
                R = randn(n, n)
                A64 = R * transpose(R) + n * I
                Aspd = T.(Matrix(A64))
                original = copy(Aspd)
                F = MultiFloatLinearAlgebra.cholesky!(Aspd; config=config)
                @test MultiFloatLinearAlgebra.issuccess(F)
                L = Matrix(LowerTriangular(F.factors))
                reconstructed = L * transpose(L)
                @test max_relative_error(reconstructed, original) <=
                    tolerance(T, 8n)

                rhs = T.(randn(n, 5))
                rhs_original = copy(rhs)
                solution = MultiFloatLinearAlgebra.solve(F, rhs; config=config)
                residual = original * solution - rhs_original
                @test max_relative_error(residual, zeros(T, size(residual))) <=
                    tolerance(T, 96n)
            end

            @testset "blocked LU solve" begin
                Alu = T.(randn(n, n))
                for i in 1:n
                    Alu[i, i] += T(4)
                end
                original = copy(Alu)
                F = MultiFloatLinearAlgebra.lu!(Alu; config=config)
                @test MultiFloatLinearAlgebra.issuccess(F)

                rhs = T.(randn(n))
                rhs_original = copy(rhs)
                solution = MultiFloatLinearAlgebra.solve(F, rhs; config=config)
                residual = original * solution - rhs_original
                @test max_relative_error(residual, zeros(T, n)) <=
                    tolerance(T, 96n)

                rhs_matrix = T.(randn(n, 5))
                rhs_matrix_original = copy(rhs_matrix)
                matrix_solution = MultiFloatLinearAlgebra.solve(
                    F, rhs_matrix; config=config,
                )
                matrix_residual = original * matrix_solution - rhs_matrix_original
                @test max_relative_error(
                    matrix_residual,
                    zeros(T, size(matrix_residual)),
                ) <= tolerance(T, 128n)
            end

            @testset "LDLT 2x2 pivots" begin
                nld = 8
                Aind = zeros(T, nld, nld)
                for k in 1:2:nld
                    coupling = T(1 + k / 10)
                    Aind[k, k + 1] = coupling
                    Aind[k + 1, k] = coupling
                end
                original = copy(Aind)
                F = MultiFloatLinearAlgebra.ldlt!(Aind)
                @test MultiFloatLinearAlgebra.issuccess(F)
                @test any(==(UInt8(2)), F.blocks)

                rhs = T.(randn(nld))
                rhs_original = copy(rhs)
                solution = MultiFloatLinearAlgebra.solve(F, rhs; config=config)
                residual = original * solution - rhs_original
                @test max_relative_error(residual, zeros(T, nld)) <=
                    tolerance(T, 64nld)

                rhs_matrix = T.(randn(nld, 5))
                rhs_matrix_original = copy(rhs_matrix)
                solution_matrix = MultiFloatLinearAlgebra.solve(
                    F, rhs_matrix; config=config,
                )
                matrix_residual = original * solution_matrix - rhs_matrix_original
                @test max_relative_error(
                    matrix_residual,
                    zeros(T, size(matrix_residual)),
                ) <= tolerance(T, 96nld)
            end

            @testset "LDLT 1x1 pivots" begin
                R = randn(n, n)
                S = 0.02 .* (R + transpose(R))
                for i in 1:n
                    S[i, i] += isodd(i) ? n : -n
                end
                Aind = T.(Matrix(S))
                original = copy(Aind)
                F = MultiFloatLinearAlgebra.ldlt!(Aind)
                @test MultiFloatLinearAlgebra.issuccess(F)
                @test any(==(UInt8(1)), F.blocks)
                rhs = T.(randn(n))
                rhs_original = copy(rhs)
                solution = MultiFloatLinearAlgebra.solve(F, rhs; config=config)
                residual = original * solution - rhs_original
                @test max_relative_error(residual, zeros(T, n)) <=
                    tolerance(T, 128n)
            end

            @testset "blocked LDLT with lazy panel updates" begin
                nld = 24
                R = randn(nld, nld)
                S = 0.01 .* (R + transpose(R))
                for i in 1:nld
                    S[i, i] += isodd(i) ? nld : -nld
                end
                Aind = T.(Matrix(S))
                original = copy(Aind)
                blocked_config = KernelConfig(
                    thread_count=2,
                    ldlt_strategy=:blocked,
                    ldlt_block=5,
                    ldlt_blocked_crossover=1,
                    gemm_strategy=:direct,
                )
                plan = ldlt_plan(T, nld, blocked_config)
                @test plan.strategy === :blocked
                @test plan.block_size == 5
                F = MultiFloatLinearAlgebra.ldlt!(
                    Aind;
                    config=blocked_config,
                )
                @test MultiFloatLinearAlgebra.issuccess(F)
                rhs = T.(randn(nld, 4))
                rhs_original = copy(rhs)
                solution = MultiFloatLinearAlgebra.solve(
                    F, rhs; config=blocked_config,
                )
                residual = original * solution - rhs_original
                @test max_relative_error(
                    residual,
                    zeros(T, size(residual)),
                ) <= tolerance(T, 512nld)

                crossing_n = 16
                crossing = zeros(T, crossing_n, crossing_n)
                for k in 1:2:crossing_n
                    coupling = T(1 + k / 20)
                    crossing[k, k + 1] = coupling
                    crossing[k + 1, k] = coupling
                end
                crossing_original = copy(crossing)
                crossing_config = KernelConfig(
                    thread_count=2,
                    ldlt_strategy=:blocked,
                    ldlt_block=3,
                    ldlt_blocked_crossover=1,
                    gemm_strategy=:direct,
                )
                crossing_factor = MultiFloatLinearAlgebra.ldlt!(
                    crossing;
                    config=crossing_config,
                )
                @test MultiFloatLinearAlgebra.issuccess(crossing_factor)
                @test count(==(UInt8(2)), crossing_factor.blocks) == crossing_n ÷ 2
                crossing_rhs = T.(randn(crossing_n))
                crossing_solution = MultiFloatLinearAlgebra.solve(
                    crossing_factor,
                    crossing_rhs;
                    config=crossing_config,
                )
                crossing_residual =
                    crossing_original * crossing_solution - crossing_rhs
                @test max_relative_error(
                    crossing_residual,
                    zeros(T, crossing_n),
                ) <= tolerance(T, 256crossing_n)
            end
        end
    end
end
