using Test
using Random
using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra
using Aqua

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
                @test builtin.thread_count == min(2, Threads.nthreads())
                @test builtin.fingerprint.julia_threads_available == Threads.nthreads()
                @test profile_compatible(builtin)

                mismatched = GemmProfile{T}(
                    :auto,
                    builtin.panel_columns,
                    builtin.micro_columns,
                    builtin.packed_crossover,
                    builtin.thread_count,
                    :builtin,
                    merge(
                        builtin.fingerprint,
                        (; cpu_model="definitely-not-this-machine"),
                    ),
                )
                @test !profile_compatible(mismatched)
                @test_throws ArgumentError with_gemm_profile(config, mismatched)
                forced = with_gemm_profile(
                    config, mismatched; strict=false,
                )
                @test forced.gemm_strategy === mismatched.strategy

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

            @testset "near-square auto policy" begin
                profile = GemmProfile{T}(
                    :auto,
                    16,
                    2,
                    8,
                    1,
                    :calibrated,
                    machine_fingerprint(),
                )
                cfg = with_gemm_profile(config, profile)
                square = gemm_plan(T, 64, 64, 64, cfg)
                tall = gemm_plan(T, 4096, 64, 64, cfg)
                reduction = gemm_plan(T, 32, 16, 32, cfg)
                @test square.strategy === :packed
                @test square.reason === :auto_above_crossover
                @test tall.strategy === :direct
                @test tall.reason === :auto_outside_calibrated_shape
                @test reduction.strategy === :direct
                @test reduction.reason === :auto_reduction_too_small

                large_crossover = GemmProfile{T}(
                    :auto,
                    16,
                    2,
                    512,
                    1,
                    :calibrated,
                    machine_fingerprint(),
                )
                large_cfg = with_gemm_profile(config, large_crossover)
                below = gemm_plan(T, 256, 256, 256, large_cfg)
                @test below.strategy === :direct
                @test below.reason === :auto_below_crossover
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

            @testset "non-finite input" begin
                Anan = T.(randn(n, n))

                Alu = copy(Anan)
                Alu[1, 1] = T(NaN)
                @test_throws DomainError MultiFloatLinearAlgebra.lu!(Alu; config=config)
                Alu = copy(Anan)
                Alu[1, 1] = T(NaN)
                Flu = MultiFloatLinearAlgebra.lu!(Alu; check=false, config=config)
                @test !MultiFloatLinearAlgebra.issuccess(Flu)
                @test Flu.info == -1

                Alu = copy(Anan)
                Alu[1, 1] = T(Inf)
                @test_throws DomainError MultiFloatLinearAlgebra.lu!(Alu; config=config)

                R = randn(n, n)
                S = R * transpose(R) + n * I
                Aspd = T.(Matrix(S))
                Aspd[2, 2] = T(NaN)
                @test_throws DomainError MultiFloatLinearAlgebra.cholesky!(Aspd; config=config)
                Aspd = T.(Matrix(S))
                Aspd[2, 2] = T(NaN)
                Fc = MultiFloatLinearAlgebra.cholesky!(Aspd; check=false, config=config)
                @test !MultiFloatLinearAlgebra.issuccess(Fc)
                @test Fc.info == -1

                Rind = randn(n, n)
                Sind = Rind + transpose(Rind)
                for i in 1:n
                    Sind[i, i] += isodd(i) ? 5 : -5
                end
                Aind = T.(Matrix(Sind))
                Aind[1, 1] = T(NaN)
                @test_throws DomainError MultiFloatLinearAlgebra.ldlt!(Aind; config=config)
                Aind = T.(Matrix(Sind))
                Aind[1, 1] = T(NaN)
                Fld = MultiFloatLinearAlgebra.ldlt!(Aind; check=false, config=config)
                @test !MultiFloatLinearAlgebra.issuccess(Fld)
                @test Fld.info == -1
            end

            @testset "alpha/beta kernels" begin
                A = T.(randn(n, n))
                B = T.(randn(n, n))
                C0 = T.(randn(n, n))
                C = copy(C0)
                MultiFloatLinearAlgebra.gemm!(C, A, B, T(2), T(3); config=config)
                @test max_relative_error(C, T(2) * (A * B) + T(3) * C0) <=
                    tolerance(T, n)

                x = T.(randn(n))
                y0 = T.(randn(n))
                y = copy(y0)
                MultiFloatLinearAlgebra.gemv!(y, A, x, T(2), T(3); config=config)
                @test max_relative_error(y, T(2) * (A * x) + T(3) * y0) <=
                    tolerance(T, n)
            end

            @testset "check=false singular" begin
                Ssing = zeros(T, n, n)
                Ssing[2, 2] = T(1)
                Flu = MultiFloatLinearAlgebra.lu!(copy(Ssing); check=false, config=config)
                @test !MultiFloatLinearAlgebra.issuccess(Flu)
                @test Flu.info >= 1
                @test_throws LinearAlgebra.SingularException MultiFloatLinearAlgebra.lu!(
                    copy(Ssing); check=true, config=config,
                )

                Aindef = T.([1 2; 2 1])
                Fc = MultiFloatLinearAlgebra.cholesky!(copy(Aindef); check=false, config=config)
                @test !MultiFloatLinearAlgebra.issuccess(Fc)
                @test Fc.info >= 1
            end

            @testset "single-thread path" begin
                single = KernelConfig(
                    thread_count=1,
                    cholesky_block=4,
                    lu_block=4,
                    column_tile=4,
                )
                A = T.(randn(n, n))
                B = T.(randn(n, n))
                C = zeros(T, n, n)
                MultiFloatLinearAlgebra.gemm!(C, A, B; config=single)
                @test max_relative_error(C, A * B) <= tolerance(T, n)

                R = randn(n, n)
                S = R * transpose(R) + n * I
                Aspd = T.(Matrix(S))
                original = copy(Aspd)
                F = MultiFloatLinearAlgebra.cholesky!(Aspd; config=single)
                rhs = T.(randn(n))
                rhs_original = copy(rhs)
                solution = MultiFloatLinearAlgebra.solve(F, rhs; config=single)
                @test max_relative_error(original * solution - rhs_original, zeros(T, n)) <=
                    tolerance(T, 128n)
            end

            @testset "vector/matrix solve equivalence" begin
                R = randn(n, n)
                S = R * transpose(R) + n * I
                Aspd = T.(Matrix(S))
                Fc = MultiFloatLinearAlgebra.cholesky!(Aspd; config=config)
                b = T.(randn(n))
                xv = MultiFloatLinearAlgebra.solve(Fc, b; config=config)
                xm = MultiFloatLinearAlgebra.solve(
                    Fc, reshape(copy(b), n, 1); config=config,
                )
                @test xv == vec(xm)

                Alu = T.(randn(n, n))
                for i in 1:n
                    Alu[i, i] += T(4)
                end
                Flu = MultiFloatLinearAlgebra.lu!(Alu; config=config)
                yv = MultiFloatLinearAlgebra.solve(Flu, b; config=config)
                ym = MultiFloatLinearAlgebra.solve(
                    Flu, reshape(copy(b), n, 1); config=config,
                )
                @test yv == vec(ym)

                Rind = randn(n, n)
                Sind = Rind + transpose(Rind)
                for i in 1:n
                    Sind[i, i] += isodd(i) ? 5 : -5
                end
                Aind = T.(Matrix(Sind))
                Fld = MultiFloatLinearAlgebra.ldlt!(Aind; config=config)
                zv = MultiFloatLinearAlgebra.solve(Fld, b; config=config)
                zm = MultiFloatLinearAlgebra.solve(
                    Fld, reshape(copy(b), n, 1); config=config,
                )
                @test zv == vec(zm)
            end

            @testset "factor public protocol" begin
                R = randn(n, n)
                S = R * transpose(R) + n * I
                Fc = MultiFloatLinearAlgebra.cholesky!(T.(Matrix(S)); config=config)
                @test Fc isa MultiFloatLinearAlgebra.AbstractMFFactorization
                @test MultiFloatLinearAlgebra.factor_kind(Fc) === :cholesky
                @test MultiFloatLinearAlgebra.factor_status(Fc) == 0
                @test MultiFloatLinearAlgebra.factor_matrix(Fc) === Fc.factors
                @test size(Fc) == (n, n)
                @test eltype(Fc) === T

                Alu = T.(randn(n, n))
                for i in 1:n
                    Alu[i, i] += T(4)
                end
                Flu = MultiFloatLinearAlgebra.lu!(Alu; config=config)
                @test MultiFloatLinearAlgebra.factor_kind(Flu) === :lu
                @test size(Flu) == (n, n)

                Rind = randn(n, n)
                Sind = Rind + transpose(Rind)
                for i in 1:n
                    Sind[i, i] += isodd(i) ? 5 : -5
                end
                Fld = MultiFloatLinearAlgebra.ldlt!(T.(Matrix(Sind)); config=config)
                @test MultiFloatLinearAlgebra.factor_kind(Fld) === :ldlt
                @test MultiFloatLinearAlgebra.factor_status(Fld) == 0
                @test eltype(Fld) === T
            end
        end
    end

    @testset "fused x3 mulacc" begin
        T3 = Float64x3
        V3 = MultiFloatVec{4,Float64,3}

        # Full three-limb data so the x3 product is not trivially exact.
        function rich_big(rng, exponent; sign=1)
            base = ldexp(BigFloat(0.5 + rand(rng)), exponent)
            c1 = ldexp(BigFloat(randn(rng)), exponent - 70)
            c2 = ldexp(BigFloat(randn(rng)), exponent - 135)
            return sign * (base + c1 + c2)
        end

        rng = MersenneTwister(0x3eed)
        @testset "bitwise equality with standard accumulation" begin
            neq = Ref(0)
            total = Ref(0)
            setprecision(BigFloat, 512) do
                for trial in 1:64
                    lanes_acc = Vector{T3}(undef, 4)
                    lanes_x = Vector{T3}(undef, 4)
                    lanes_y = Vector{T3}(undef, 4)
                    for lane in 1:4
                        lanes_acc[lane] = T3(rich_big(rng, rand(rng, -40:40); sign=rand(rng, Bool) ? 1 : -1))
                        lanes_x[lane] = T3(rich_big(rng, rand(rng, -40:40); sign=rand(rng, Bool) ? 1 : -1))
                        lanes_y[lane] = T3(rich_big(rng, rand(rng, -40:40); sign=rand(rng, Bool) ? 1 : -1))
                    end
                    acc = V3(lanes_acc...)
                    x = V3(lanes_x...)
                    y = V3(lanes_y...)
                    standard = acc + x * y
                    fused = MultiFloatLinearAlgebra.mulacc_x3(acc, x, y)
                    for lane in 1:4
                        total[] += 1
                        standard[lane] != fused[lane] && (neq[] += 1)
                    end
                end
            end
            @test neq[] == 0
            @test total[] == 256
        end

        @testset "fused GEMM strategy" begin
            n = 24
            A = Matrix{T3}(undef, n, n)
            B = Matrix{T3}(undef, n, n)
            setprecision(BigFloat, 512) do
                for j in 1:n, i in 1:n
                    A[i, j] = T3(rich_big(rng, rand(rng, -40:40); sign=rand(rng, Bool) ? 1 : -1))
                    B[i, j] = T3(rich_big(rng, rand(rng, -40:40); sign=rand(rng, Bool) ? 1 : -1))
                end
            end

            Cstandard = zeros(T3, n, n)
            Cfused = zeros(T3, n, n)
            MultiFloatLinearAlgebra.gemm!(
                Cstandard, A, B;
                config=KernelConfig(thread_count=1, gemm_strategy=:direct),
            )
            MultiFloatLinearAlgebra.gemm!(
                Cfused, A, B;
                config=KernelConfig(thread_count=1, gemm_strategy=:fused),
            )
            @test Cfused == Cstandard

            # Fused is x3-only.
            A2 = Float64x2.(randn(n, n))
            B2 = Float64x2.(randn(n, n))
            C2 = zeros(Float64x2, n, n)
            @test_throws ArgumentError MultiFloatLinearAlgebra.gemm!(
                C2, A2, B2;
                config=KernelConfig(thread_count=1, gemm_strategy=:fused),
            )
        end
    end

    @testset "Aqua" begin
        Aqua.test_all(
            MultiFloatLinearAlgebra;
            piracies=true,
            stale_deps=true,
            deps_compat=true,
        )
    end
end
