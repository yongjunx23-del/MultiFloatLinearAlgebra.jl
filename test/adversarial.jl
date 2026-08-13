function limb_bitwise_equal(left, right)
    size(left) == size(right) || return false
    for index in eachindex(left, right)
        left_limbs = left[index]._limbs
        right_limbs = right[index]._limbs
        length(left_limbs) == length(right_limbs) || return false
        for limb in eachindex(left_limbs, right_limbs)
            reinterpret(UInt64, left_limbs[limb]) ==
                reinterpret(UInt64, right_limbs[limb]) || return false
        end
    end
    return true
end

function lower_limb_bitwise_equal(left, right)
    size(left) == size(right) || return false
    for column in axes(left, 2), row in column:size(left, 1)
        limb_bitwise_equal(
            view(left, row:row, column:column),
            view(right, row:row, column:column),
        ) || return false
    end
    return true
end

@testset "adversarial numerical validation" begin
    for T in (Float64x2, Float64x3, Float64x4)
        @testset "$T factorizations" begin
            single = KernelConfig(
                thread_count=1,
                cholesky_block=2,
                lu_block=2,
                ldlt_strategy=:unblocked,
            )

            # A diagonal SPD matrix with condition number 2^600 remains a
            # valid, deterministic Cholesky input without relying on a random
            # generator to preserve positive definiteness after rounding.
            exponents = (-300, -150, 0, 150, 300)
            ill_spd = zeros(T, length(exponents), length(exponents))
            for (index, exponent) in enumerate(exponents)
                ill_spd[index, index] = T(ldexp(1.0, exponent))
            end
            truth = T[1, -2, 3, -4, 5]
            rhs = ill_spd * truth
            chol_ill = MultiFloatLinearAlgebra.cholesky!(
                copy(ill_spd); config=single,
            )
            @test MultiFloatLinearAlgebra.issuccess(chol_ill)
            @test solve(chol_ill, rhs; config=single) == truth
            chol_ill_diagnostics = factor_diagnostics(chol_ill)
            @test chol_ill_diagnostics.diagonal_spread == T(ldexp(1.0, 300))

            # Cholesky reads only the lower triangle. Corrupted upper storage
            # must neither reject the factorization nor enter a trailing update.
            clean_spd = T[
                8 2 1 0
                2 7 1 1
                1 1 6 2
                0 1 2 5
            ]
            lower_spd = copy(clean_spd)
            for column in axes(lower_spd, 2), row in 1:(column - 1)
                lower_spd[row, column] = T(NaN)
            end
            chol_lower = MultiFloatLinearAlgebra.cholesky!(
                lower_spd; config=single,
            )
            lower_truth = T[2, -1, 3, -2]
            lower_solution = solve(
                chol_lower, clean_spd * lower_truth; config=single,
            )
            @test max_relative_error(lower_solution, lower_truth) <=
                tolerance(T, 64)

            # The first pivot is accepted and the Schur complement fails at
            # pivot two by a representable amount for every limb count.
            delta = T(16) * eps(T)
            nearly_non_spd = T[1 0; 1 1 - delta]
            nearly_non_spd[1, 2] = T(NaN)
            failed_chol = MultiFloatLinearAlgebra.cholesky!(
                copy(nearly_non_spd); check=false, config=single,
            )
            @test factor_status(failed_chol) == 2
            @test factor_diagnostics(failed_chol).failure_location == 2
            @test_throws LinearAlgebra.PosDefException MultiFloatLinearAlgebra.cholesky!(
                copy(nearly_non_spd); config=single,
            )

            # Row and column scales span 2^600 while the unscaled core is
            # modest. The backward error, rather than forward error, is the
            # stable contract for this badly scaled solve.
            lu_core = T[
                4 1 0 1
                2 5 1 0
                0 2 6 1
                1 0 1 7
            ]
            row_scale = T.(ldexp.(1.0, (-200, -60, 60, 200)))
            column_scale = T.(ldexp.(1.0, (100, 30, -30, -100)))
            badly_scaled_lu = similar(lu_core)
            for column in axes(lu_core, 2), row in axes(lu_core, 1)
                badly_scaled_lu[row, column] =
                    row_scale[row] * lu_core[row, column] * column_scale[column]
            end
            lu_truth = T[1, -2, 3, -1]
            lu_rhs = badly_scaled_lu * lu_truth
            lu_scaled = MultiFloatLinearAlgebra.lu!(
                copy(badly_scaled_lu); config=single,
            )
            lu_solution = solve(lu_scaled, lu_rhs; config=single)
            lu_residual = similar(lu_rhs)
            residual!(lu_residual, badly_scaled_lu, lu_solution, lu_rhs)
            @test normwise_backward_error(
                badly_scaled_lu, lu_solution, lu_rhs, lu_residual,
            ) <= T(4096) * eps(T)

            tiny = T(ldexp(1.0, -300))
            tiny_first = T[tiny 1; 1 1]
            tiny_factor = MultiFloatLinearAlgebra.lu!(
                copy(tiny_first); config=single,
            )
            @test factor_diagnostics(tiny_factor).pivots[1] == 2
            tiny_truth = T[2, -3]
            @test max_relative_error(
                solve(tiny_factor, tiny_first * tiny_truth; config=single),
                tiny_truth,
            ) <= tolerance(T, 16)

            nearly_singular = T[1 1; 1 1 + delta]
            near_lu = MultiFloatLinearAlgebra.lu!(
                copy(nearly_singular); config=single,
            )
            near_diagnostics = factor_diagnostics(near_lu)
            @test near_diagnostics.minimum_pivot == delta
            near_truth = T[1, -1]
            @test solve(
                near_lu, nearly_singular * near_truth; config=single,
            ) == near_truth

            # Wilkinson's matrix realizes the classic 2^(n-1) partial-pivot
            # growth example without any random or architecture-sensitive data.
            growth_order = 8
            growth_matrix = zeros(T, growth_order, growth_order)
            for row in 1:growth_order
                growth_matrix[row, row] = one(T)
                growth_matrix[row, growth_order] = one(T)
                for column in 1:(row - 1)
                    growth_matrix[row, column] = -one(T)
                end
            end
            growth_factor = MultiFloatLinearAlgebra.lu!(
                copy(growth_matrix); config=single,
            )
            @test factor_diagnostics(growth_factor).pivot_growth == T(128)
            growth_truth = T.(collect(1:growth_order))
            growth_rhs = growth_matrix * growth_truth
            growth_solution = solve(growth_factor, growth_rhs; config=single)
            growth_residual = similar(growth_rhs)
            residual!(
                growth_residual, growth_matrix, growth_solution, growth_rhs,
            )
            @test normwise_backward_error(
                growth_matrix, growth_solution, growth_rhs, growth_residual,
            ) <= T(256) * eps(T)

            # A small saddle-point system has four positive and two negative
            # eigenvalues. Its upper triangle is intentionally invalid.
            hessian = T[
                4 0 0 0
                0 5 0 0
                0 0 6 0
                0 0 0 7
            ]
            constraints = T[1 2 0 1; 0 1 1 -1]
            clean_kkt = [
                hessian transpose(constraints)
                constraints zeros(T, 2, 2)
            ]
            kkt_truth = T[1, -2, 3, -1, 2, -3]
            kkt_rhs = clean_kkt * kkt_truth
            for strategy in (:unblocked, :blocked)
                kkt = copy(clean_kkt)
                for column in axes(kkt, 2), row in 1:(column - 1)
                    kkt[row, column] = T(NaN)
                end
                kkt_config = KernelConfig(
                    thread_count=2,
                    ldlt_strategy=strategy,
                    ldlt_block=3,
                    ldlt_blocked_crossover=1,
                )
                kkt_factor = MultiFloatLinearAlgebra.ldlt!(
                    kkt; config=kkt_config,
                )
                @test factor_diagnostics(kkt_factor).inertia ==
                    (positive=4, negative=2, zero=0)
                kkt_solution = solve(
                    kkt_factor, kkt_rhs; config=kkt_config,
                )
                kkt_residual = similar(kkt_rhs)
                residual!(
                    kkt_residual,
                    clean_kkt,
                    kkt_solution,
                    kkt_rhs;
                    uplo=:lower,
                    config=kkt_config,
                )
                @test normwise_backward_error(
                    clean_kkt,
                    kkt_solution,
                    kkt_rhs,
                    kkt_residual;
                    uplo=:lower,
                ) <= T(4096) * eps(T)
            end

            # Tiny 1x1 pivots and badly scaled forced 2x2 blocks exercise both
            # D solve paths and exact inertia reporting.
            diagonal_values = T[
                ldexp(1.0, -300),
                -ldexp(1.0, -100),
                ldexp(1.0, 100),
                -ldexp(1.0, 300),
            ]
            tiny_diagonal = Matrix(Diagonal(diagonal_values))
            diagonal_factor = MultiFloatLinearAlgebra.ldlt!(
                copy(tiny_diagonal); config=single,
            )
            diagonal_diagnostics = factor_diagnostics(diagonal_factor)
            @test diagonal_diagnostics.one_by_one_pivots == 4
            @test diagonal_diagnostics.inertia ==
                (positive=2, negative=2, zero=0)
            @test diagonal_diagnostics.minimum_block_eigenvalue_magnitude == tiny
            diagonal_truth = T[1, -2, 3, -4]
            @test solve(
                diagonal_factor,
                tiny_diagonal * diagonal_truth;
                config=single,
            ) == diagonal_truth

            couplings = T.(ldexp.(1.0, (-300, -100, 100, 300)))
            scaled_indefinite = zeros(T, 8, 8)
            for (block, coupling) in enumerate(couplings)
                first = 2block - 1
                scaled_indefinite[first + 1, first] = coupling
                scaled_indefinite[first, first + 1] = T(NaN)
            end
            scaled_factor = MultiFloatLinearAlgebra.ldlt!(
                copy(scaled_indefinite); config=single,
            )
            scaled_diagnostics = factor_diagnostics(scaled_factor)
            @test scaled_diagnostics.two_by_two_pivots == 4
            @test scaled_diagnostics.inertia ==
                (positive=4, negative=4, zero=0)
            @test scaled_diagnostics.minimum_block_eigenvalue_magnitude == tiny
            scaled_truth = T.(collect(1:8))
            scaled_rhs = zeros(T, 8)
            for block in 1:4
                first = 2block - 1
                coupling = couplings[block]
                scaled_rhs[first] = coupling * scaled_truth[first + 1]
                scaled_rhs[first + 1] = coupling * scaled_truth[first]
            end
            @test solve(
                scaled_factor, scaled_rhs; config=single,
            ) == scaled_truth

            singular_ldlt = MultiFloatLinearAlgebra.ldlt!(
                T[1 1; 1 1]; check=false, config=single,
            )
            @test factor_status(singular_ldlt) == 2
            @test factor_diagnostics(singular_ldlt).failure_location == 2
        end

        @testset "$T deterministic kernels and aliases" begin
            serial = KernelConfig(
                thread_count=1,
                gemm_strategy=:direct,
                gemm_panel_columns=8,
            )
            parallel = KernelConfig(
                thread_count=4,
                gemm_strategy=:direct,
                gemm_panel_columns=8,
            )

            normal_matrix = T.(randn(97, 31))
            normal_x = T.(randn(31))
            normal_serial = zeros(T, 97)
            normal_parallel = zeros(T, 97)
            gemv!(normal_serial, normal_matrix, normal_x; config=serial)
            gemv!(normal_parallel, normal_matrix, normal_x; config=parallel)
            @test limb_bitwise_equal(normal_serial, normal_parallel)

            transpose_matrix = T.(randn(31, 97))
            transpose_x = T.(randn(31))
            transpose_serial = zeros(T, 97)
            transpose_parallel = zeros(T, 97)
            gemv!(
                transpose_serial,
                transpose_matrix,
                transpose_x;
                trans=:T,
                config=serial,
            )
            gemv!(
                transpose_parallel,
                transpose_matrix,
                transpose_x;
                trans=:T,
                config=parallel,
            )
            @test limb_bitwise_equal(transpose_serial, transpose_parallel)

            symmetric_raw = T.(randn(97, 97))
            symmetric = T.(Matrix(symmetric_raw + transpose(symmetric_raw)))
            for column in axes(symmetric, 2), row in 1:(column - 1)
                symmetric[row, column] = T(NaN)
            end
            symmetric_x = T.(randn(97))
            symmetric_serial = zeros(T, 97)
            symmetric_parallel = zeros(T, 97)
            symv!(
                symmetric_serial,
                symmetric,
                symmetric_x;
                uplo=:lower,
                config=serial,
            )
            symv!(
                symmetric_parallel,
                symmetric,
                symmetric_x;
                uplo=:lower,
                config=parallel,
            )
            @test limb_bitwise_equal(symmetric_serial, symmetric_parallel)

            left = T.(randn(17, 13))
            right = T.(randn(13, 35))
            product_serial = zeros(T, 17, 35)
            product_parallel = zeros(T, 17, 35)
            gemm!(product_serial, left, right; config=serial)
            gemm!(product_parallel, left, right; config=parallel)
            @test limb_bitwise_equal(product_serial, product_parallel)

            gemmt_left = T.(randn(29, 13))
            gemmt_right = T.(randn(29, 13))
            initial_triangle = T.(randn(29, 29))
            for column in axes(initial_triangle, 2), row in 1:(column - 1)
                initial_triangle[row, column] = T(NaN)
            end
            gemmt_serial = copy(initial_triangle)
            gemmt_parallel = copy(initial_triangle)
            gemmt!(
                gemmt_serial,
                gemmt_left,
                gemmt_right,
                T(2),
                T(-3);
                config=serial,
            )
            gemmt!(
                gemmt_parallel,
                gemmt_left,
                gemmt_right,
                T(2),
                T(-3);
                config=parallel,
            )
            @test lower_limb_bitwise_equal(gemmt_serial, gemmt_parallel)
            @test all(
                column -> all(
                    row -> isnan(gemmt_serial[row, column]),
                    1:(column - 1),
                ),
                axes(gemmt_serial, 2),
            )

            panel = T.(randn(13, 29))
            syrk_serial = copy(initial_triangle)
            syrk_parallel = copy(initial_triangle)
            syrk!(syrk_serial, panel, T(-2), T(3); config=serial)
            syrk!(syrk_parallel, panel, T(-2), T(3); config=parallel)
            @test lower_limb_bitwise_equal(syrk_serial, syrk_parallel)
            @test all(
                column -> all(
                    row -> isnan(syrk_serial[row, column]),
                    1:(column - 1),
                ),
                axes(syrk_serial, 2),
            )

            packed_length = 29 * 30 ÷ 2
            packed_initial = T.(randn(packed_length))
            packed_serial = copy(packed_initial)
            packed_parallel = copy(packed_initial)
            syrk_packed!(
                packed_serial, panel, T(2), T(-1); config=serial,
            )
            syrk_packed!(
                packed_parallel, panel, T(2), T(-1); config=parallel,
            )
            @test limb_bitwise_equal(packed_serial, packed_parallel)

            alias_matrix = T.(randn(5, 5))
            alias_vector = T.(randn(5))
            @test_throws ArgumentError gemv!(
                view(alias_matrix, :, 1), alias_matrix, alias_vector,
            )
            @test_throws ArgumentError symv!(
                view(alias_matrix, :, 1), alias_matrix, alias_vector,
            )
            @test_throws ArgumentError gemm!(
                alias_matrix, alias_matrix, copy(alias_matrix),
            )
            @test_throws ArgumentError gemmt!(
                alias_matrix, alias_matrix, copy(alias_matrix),
            )
            @test_throws ArgumentError syrk!(alias_matrix, alias_matrix)
            packed_alias = view(vec(alias_matrix), 1:15)
            @test_throws ArgumentError syrk_packed!(
                packed_alias, alias_matrix,
            )
            @test_throws ArgumentError trsv!(
                view(alias_matrix, :, 1), alias_matrix,
            )
            @test_throws ArgumentError trsm!(alias_matrix, alias_matrix)
            @test_throws ArgumentError trmm!(alias_matrix, alias_matrix)
        end
    end

    @testset "stagnating explicit correction" begin
        for (Source, Residual) in (
            (Float64x2, Float64x3),
            (Float64x2, Float64x4),
            (Float64x3, Float64x4),
        )
            q = 55
            perturbation = Source(ldexp(1.0, -q))
            value = one(Source) + perturbation
            if Source === Float64x3
                value += Source(ldexp(1.0, -2q))
            end
            system = reshape(Source[value], 1, 1)
            approximation = Source[value]
            right_hand_side = Source[value * value]
            exposed = zeros(Residual, 1)
            residual_mixed!(
                exposed, system, approximation, right_hand_side,
            )
            @test !iszero(exposed[1])

            factor = MultiFloatLinearAlgebra.lu!(copy(system))
            correction = Source.(exposed)
            refinement_correction!(correction, factor, correction)
            @test !iszero(correction[1])

            updated = approximation + correction
            @test updated == approximation
            repeated = zeros(Residual, 1)
            residual_mixed!(
                repeated, system, updated, right_hand_side,
            )
            @test limb_bitwise_equal(repeated, exposed)
        end
    end
end
