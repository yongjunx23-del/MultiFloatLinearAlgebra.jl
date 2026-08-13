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

            @testset "transpose gemv!" begin
                # Rectangular tall, wide, square, and odd shapes plus alpha/beta.
                for (m, nn) in ((8, 12), (12, 8), (7, 7), (1, 9), (9, 1), (11, 13))
                    At = T.(randn(m, nn))
                    xt = T.(randn(m))
                    y0 = T.(randn(nn))
                    yt = copy(y0)
                    gemv!(yt, At, xt, T(2), T(3); trans=:T, config=config)
                    reference = T(2) * (transpose(At) * xt) + T(3) * y0
                    @test max_relative_error(yt, reference) <= tolerance(T, 4 * max(m, nn))
                end

                # Deterministic ascending reduction: stale memory must not matter.
                At = T.(randn(16, 16))
                xt = T.(randn(16))
                clean = zeros(T, 16)
                dirty = fill(T(NaN), 16)
                gemv!(clean, At, xt; trans=:T, config=config)
                gemv!(dirty, At, xt; trans=:T, config=config)
                @test clean == dirty

                @test_throws ArgumentError gemv!(
                    zeros(T, 4), T.(randn(4, 4)), T.(randn(4)); trans=:bad, config=config,
                )
                @test_throws DimensionMismatch gemv!(
                    zeros(T, 3), T.(randn(4, 4)), T.(randn(4)); trans=:T, config=config,
                )
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
                @test plan.strategy in (:direct, :fused, :packed)
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
                fused_default = T === Float64x3
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
                @test tall.strategy === (fused_default ? :fused : :direct)
                @test tall.reason === (fused_default ? :auto_fused_direct : :auto_outside_calibrated_shape)
                @test reduction.strategy === (fused_default ? :fused : :direct)
                @test reduction.reason === (fused_default ? :auto_fused_direct : :auto_reduction_too_small)

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
                @test below.strategy === (fused_default ? :fused : :direct)
                @test below.reason === (fused_default ? :auto_fused_direct : :auto_below_crossover)
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

            @testset "trsv!" begin
                # Compare the single-RHS trsv! against the one-column trsm!
                # reference across every uplo/trans/diag combination, well- and
                # badly-scaled, plus explicit forward/back substitution checks.
                for uplo in (:lower, :upper), trans in (:N, :T), unit in (false, true)
                    triangular = triangular_matrix(T, n, uplo; unit=unit)
                    opA = trans === :N ? triangular : transpose(triangular)
                    x0 = T.(randn(n))
                    b = opA * x0

                    via_trsv = copy(b)
                    trsv!(
                        via_trsv, triangular;
                        uplo=uplo, trans=trans,
                        diag=unit ? :unit : :nonunit, config=config,
                    )
                    @test max_relative_error(via_trsv, x0) <= tolerance(T, 8n)

                    # Exact agreement with the one-column TRSM path.
                    via_trsm = copy(b)
                    trsm!(
                        reshape(via_trsm, n, 1), triangular;
                        side=:left, uplo=uplo, trans=trans,
                        diag=unit ? :unit : :nonunit, config=config,
                    )
                    @test via_trsv == via_trsm
                end

                # Badly scaled triangular matrix still solves.
                bad = triangular_matrix(T, n, :lower)
                for i in 1:n
                    bad[i, i] *= T(1e-20 + 1e20 * (i / n))
                end
                x0 = T.(randn(n))
                rhs = bad * x0
                trsv!(rhs, bad; uplo=:lower, trans=:N, diag=:nonunit, config=config)
                @test max_relative_error(rhs, x0) <= tolerance(T, 32n)

                @test_throws DimensionMismatch trsv!(
                    zeros(T, n - 1), bad; uplo=:lower, config=config,
                )
            end

            @testset "symv!" begin
                for uplo in (:lower, :upper)
                    Asym = T.(randn(n, n))
                    S = T.(Matrix(Asym + transpose(Asym)))

                    # Corrupt the inactive strict triangle with NaN; symv! must
                    # read only the authoritative triangle.
                    if uplo === :lower
                        for r in 1:n, c in (r + 1):n
                            S[r, c] = T(NaN)
                        end
                    else
                        for r in 1:n, c in 1:(r - 1)
                            S[r, c] = T(NaN)
                        end
                    end

                    x = T.(randn(n))
                    y0 = T.(randn(n))
                    y = copy(y0)
                    symv!(y, S, x, T(2), T(3); uplo=uplo, config=config)

                    reference = zeros(T, n)
                    for i in 1:n
                        acc = zero(T)
                        for j in 1:n
                            v = uplo === :lower ?
                                (j <= i ? S[i, j] : S[j, i]) :
                                (j >= i ? S[i, j] : S[j, i])
                            acc += v * x[j]
                        end
                        reference[i] = T(2) * acc + T(3) * y0[i]
                    end
                    @test max_relative_error(y, reference) <= tolerance(T, 4n)
                end

                @test_throws ArgumentError symv!(
                    zeros(T, n), T.(randn(n, n)), T.(randn(n));
                    uplo=:bad, config=config,
                )
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

            @testset "multithread transpose gemv / symv" begin
                single = KernelConfig(thread_count=1)
                multi = KernelConfig(thread_count=2)

                # transpose GEMV on a shape large enough to thread.
                At = T.(randn(96, 80))
                xt = T.(randn(96))
                ys = zeros(T, 80)
                ym = zeros(T, 80)
                gemv!(ys, At, xt; trans=:T, config=single)
                gemv!(ym, At, xt; trans=:T, config=multi)
                @test ys == ym

                # SYMV threaded matches single-thread on the authoritative triangle.
                R = randn(96, 96)
                S = T.(Matrix(R + transpose(R)))
                xs = T.(randn(96))
                ysv = zeros(T, 96)
                ymv = zeros(T, 96)
                symv!(ysv, S, xs; uplo=:lower, config=single)
                symv!(ymv, S, xs; uplo=:lower, config=multi)
                @test ysv == ymv
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

    @testset "structured mulacc policy" begin
        # The structured (GEMMT/SYRK) x3 accumulation is fused only on
        # AArch64, where it measured positive; x86_64 keeps standard `acc+x*y`.
        @test MultiFloatLinearAlgebra._structured_fuses_x3() == (Sys.ARCH === :aarch64)

        # For every supported type, GEMMT/SYRK write only the authoritative
        # lower triangle and reproduce the standard ascending reduction. On
        # x3 this holds on both architectures because the fused network is
        # bitwise-identical to the reference in the committed adversarial suite.
        for T in (Float64x2, Float64x3, Float64x4)
            reduction, rows = 11, 16
            left = T.(randn(rows, reduction))
            right = T.(randn(rows, reduction))

            Cstd = zeros(T, rows, rows)
            Cfus = zeros(T, rows, rows)
            for column in 1:rows
                for row in column:rows
                    acc = zero(T)
                    for k in 1:reduction
                        acc += left[row, k] * right[column, k]
                    end
                    Cstd[row, column] = acc
                end
            end
            gemmt!(Cfus, left, right; config=KernelConfig(thread_count=1))
            @test Cfus == Cstd
            @test iszero(triu(Cfus, 1))

            panel = T.(randn(reduction, rows))
            Sstd = zeros(T, rows, rows)
            Sfus = zeros(T, rows, rows)
            for column in 1:rows
                for row in column:rows
                    acc = zero(T)
                    for k in 1:reduction
                        acc += panel[k, row] * panel[k, column]
                    end
                    Sstd[row, column] = acc
                end
            end
            syrk!(Sfus, panel; config=KernelConfig(thread_count=1))
            @test Sfus == Sstd
            @test iszero(triu(Sfus, 1))
        end
    end

    @testset "fused x3 mulacc" begin
        T3 = Float64x3
        V3 = MultiFloatVec{4,Float64,3}

        function rich_big(rng, exponent; sign=1)
            base = ldexp(BigFloat(0.5 + rand(rng)), exponent)
            c1 = ldexp(BigFloat(randn(rng)), exponent - 70)
            c2 = ldexp(BigFloat(randn(rng)), exponent - 135)
            return sign * (base + c1 + c2)
        end

        function mode_triple(rng, mode, trial, lane)
            sign = rand(rng, Bool) ? 1 : -1
            if mode === :random
                ex = rand(rng, -40:40); ey = rand(rng, -40:40); ea = rand(rng, -40:40)
                return (T3(rich_big(rng, ea; sign=sign)),
                        T3(rich_big(rng, ex; sign=sign)),
                        T3(rich_big(rng, ey; sign=sign)))
            elseif mode === :wide
                ex = rand(rng, -300:300); ey = rand(rng, -300:300); ea = rand(rng, -500:500)
                return (T3(rich_big(rng, ea; sign=sign)),
                        T3(rich_big(rng, ex; sign=sign)),
                        T3(rich_big(rng, ey; sign=sign)))
            elseif mode === :near_underflow
                ex = rand(rng, -1080:-1040); ey = rand(rng, -1080:-1040); ea = rand(rng, -1080:-1040)
                return (T3(rich_big(rng, ea; sign=sign)),
                        T3(rich_big(rng, ex; sign=sign)),
                        T3(rich_big(rng, ey; sign=sign)))
            elseif mode === :near_overflow
                ex = rand(rng, 1010:1020); ey = rand(rng, 1010:1020); ea = rand(rng, 1010:1020)
                return (T3(rich_big(rng, ea; sign=sign)),
                        T3(rich_big(rng, ex; sign=sign)),
                        T3(rich_big(rng, ey; sign=sign)))
            else
                error("unknown mode")
            end
        end

        function limbs_bitwise_equal(a::T3, b::T3)
            la = a._limbs
            lb = b._limbs
            return all(
                reinterpret(UInt64, la[i]) == reinterpret(UInt64, lb[i]) for i in 1:3
            )
        end

        rng = MersenneTwister(0x3eed)
        @testset "bitwise equality with standard accumulation" begin
            neq = Ref(0)
            total = Ref(0)
            setprecision(BigFloat, 512) do
                for mode in (:random, :wide, :near_underflow, :near_overflow)
                    for trial in 1:64
                        lanes_acc = Vector{T3}(undef, 4)
                        lanes_x = Vector{T3}(undef, 4)
                        lanes_y = Vector{T3}(undef, 4)
                        for lane in 1:4
                            lanes_acc[lane], lanes_x[lane], lanes_y[lane] =
                                mode_triple(rng, mode, trial, lane)
                        end
                        acc = V3(lanes_acc...)
                        x = V3(lanes_x...)
                        y = V3(lanes_y...)
                        standard = acc + x * y
                        fused = MultiFloatLinearAlgebra.mulacc_x3(acc, x, y)
                        for lane in 1:4
                            total[] += 1
                            limbs_bitwise_equal(standard[lane], fused[lane]) || (neq[] += 1)
                        end
                    end
                end
            end
            @test neq[] == 0
            @test total[] == 1024
        end

        @testset "fused GEMM correctness (odd shapes and panels)" begin
            n = 24
            A = Matrix{T3}(undef, n, n)
            B = Matrix{T3}(undef, n, n)
            setprecision(BigFloat, 512) do
                for j in 1:n, i in 1:n
                    A[i, j] = T3(rich_big(rng, rand(rng, -40:40); sign=rand(rng, Bool) ? 1 : -1))
                    B[i, j] = T3(rich_big(rng, rand(rng, -40:40); sign=rand(rng, Bool) ? 1 : -1))
                end
            end

            for (m, k, nn) in ((24, 24, 24), (23, 24, 25), (25, 23, 24), (5, 5, 5), (24, 24, 5))
                A = Matrix{T3}(undef, m, k)
                B = Matrix{T3}(undef, k, nn)
                setprecision(BigFloat, 512) do
                    for j in 1:k, i in 1:m
                        A[i, j] = T3(rich_big(rng, rand(rng, -40:40); sign=rand(rng, Bool) ? 1 : -1))
                    end
                    for j in 1:nn, i in 1:k
                        B[i, j] = T3(rich_big(rng, rand(rng, -40:40); sign=rand(rng, Bool) ? 1 : -1))
                    end
                end
                for panel in (3, 5, 16)
                    Cstandard = zeros(T3, m, nn)
                    Cfused = zeros(T3, m, nn)
                    MultiFloatLinearAlgebra.gemm!(
                        Cstandard, A, B;
                        config=KernelConfig(thread_count=1, gemm_strategy=:direct),
                    )
                    MultiFloatLinearAlgebra.gemm!(
                        Cfused, A, B;
                        config=KernelConfig(
                            thread_count=2,
                            gemm_strategy=:fused,
                            gemm_panel_columns=panel,
                        ),
                    )
                    @test Cfused == Cstandard
                end
            end

            # alpha/beta on the fused path.
            A = Matrix{T3}(undef, 24, 24)
            B = Matrix{T3}(undef, 24, 24)
            C0 = zeros(T3, 24, 24)
            setprecision(BigFloat, 512) do
                for j in 1:24, i in 1:24
                    A[i, j] = T3(rich_big(rng, rand(rng, -40:40); sign=rand(rng, Bool) ? 1 : -1))
                    B[i, j] = T3(rich_big(rng, rand(rng, -40:40); sign=rand(rng, Bool) ? 1 : -1))
                    C0[i, j] = T3(rich_big(rng, rand(rng, -40:40); sign=rand(rng, Bool) ? 1 : -1))
                end
            end
            Cstandard = copy(C0)
            Cfused = copy(C0)
            MultiFloatLinearAlgebra.gemm!(
                Cstandard, A, B, T3(2), T3(3);
                config=KernelConfig(thread_count=1, gemm_strategy=:direct),
            )
            MultiFloatLinearAlgebra.gemm!(
                Cfused, A, B, T3(2), T3(3);
                config=KernelConfig(thread_count=1, gemm_strategy=:fused),
            )
            @test Cfused == Cstandard

            # Fused is x3-only.
            A2 = Float64x2.(randn(8, 8))
            B2 = Float64x2.(randn(8, 8))
            C2 = zeros(Float64x2, 8, 8)
            @test_throws ArgumentError MultiFloatLinearAlgebra.gemm!(
                C2, A2, B2;
                config=KernelConfig(thread_count=1, gemm_strategy=:fused),
            )

            # :auto routes x3 to fused, while :direct stays the reference path.
            plan_auto = MultiFloatLinearAlgebra.gemm_plan(
                T3, 32, 32, 32, KernelConfig(thread_count=1),
            )
            @test plan_auto.strategy === :fused
            @test plan_auto.reason === :auto_fused_direct
            plan_direct = MultiFloatLinearAlgebra.gemm_plan(
                T3, 32, 32, 32, KernelConfig(thread_count=1, gemm_strategy=:direct),
            )
            @test plan_direct.strategy === :direct
        end

        @testset "fused gemmt/syrk bitwise equality" begin
            # GEMMT and SYRK use the same internal `_mulacc` specialization, so
            # confirm the fused x3 bulk is bitwise identical to the standard
            # accumulation on full-limb data.
            for rows in (16, 17), reduction in (8, 11)
                left = Matrix{T3}(undef, rows, reduction)
                right = Matrix{T3}(undef, rows, reduction)
                setprecision(BigFloat, 512) do
                    for j in 1:reduction, i in 1:rows
                        left[i, j] = T3(rich_big(rng, rand(rng, -40:40); sign=rand(rng, Bool) ? 1 : -1))
                        right[i, j] = T3(rich_big(rng, rand(rng, -40:40); sign=rand(rng, Bool) ? 1 : -1))
                    end
                end

                # Fused is the x3 default now, so compare against an explicit
                # standard reference computed by the same reduction order.
                Cstd = zeros(T3, rows, rows)
                Cfus = zeros(T3, rows, rows)
                for column in 1:rows
                    for row in column:rows
                        acc = zero(T3)
                        for k in 1:reduction
                            acc += left[row, k] * right[column, k]
                        end
                        Cstd[row, column] = acc
                    end
                end
                MultiFloatLinearAlgebra.gemmt!(
                    Cfus, left, right;
                    config=KernelConfig(thread_count=1),
                )
                @test Cfus == Cstd

                panel = Matrix{T3}(undef, reduction, rows)
                setprecision(BigFloat, 512) do
                    for j in 1:rows, i in 1:reduction
                        panel[i, j] = T3(rich_big(rng, rand(rng, -40:40); sign=rand(rng, Bool) ? 1 : -1))
                    end
                end
                Sstd = zeros(T3, rows, rows)
                Sfus = zeros(T3, rows, rows)
                for column in 1:rows
                    for row in column:rows
                        acc = zero(T3)
                        for k in 1:reduction
                            acc += panel[k, row] * panel[k, column]
                        end
                        Sstd[row, column] = acc
                    end
                end
                MultiFloatLinearAlgebra.syrk!(
                    Sfus, panel;
                    config=KernelConfig(thread_count=1),
                )
                @test Sfus == Sstd
            end
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
