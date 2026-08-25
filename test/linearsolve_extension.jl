import LinearSolve
import SciMLBase

@testset "LinearSolve package extension" begin
    extension = Base.get_extension(MultiFloatLinearAlgebra, :MultiFloatLinearSolveExt)
    @test extension !== nothing

    for T in (Float64x2, Float64x4)
        @testset "$T" begin
            expected = T[1, -2, 3, -1]
            cases = (
                (
                    MultiFloatLU(config=KernelConfig(thread_count=1)),
                    T[
                        4 1 2 0
                        2 5 1 1
                        1 0 3 1
                        0 1 2 6
                    ],
                ),
                (
                    MultiFloatCholesky(config=KernelConfig(thread_count=1)),
                    T[
                        6 1 0 0
                        1 5 1 0
                        0 1 4 1
                        0 0 1 3
                    ],
                ),
            )

            for (algorithm, A) in cases
                @test algorithm isa LinearSolve.SciMLLinearSolveAlgorithm
                if isdefined(LinearSolve, :algorithm_interface_issues)
                    @test isempty(LinearSolve.algorithm_interface_issues(algorithm))
                else
                    @test LinearSolve.needs_concrete_A(algorithm)
                end
                original_A = copy(A)
                b = A * expected
                problem = LinearSolve.LinearProblem(A, b)

                solution = LinearSolve.solve(problem, algorithm)
                @test solution.retcode == SciMLBase.ReturnCode.Success
                @test max_relative_error(solution.u, expected) <= tolerance(T, 16)
                @test A == original_A

                cache = LinearSolve.init(problem, algorithm)
                # `init` reserves the O(n^2) storage at the matrix size so the
                # first `solve!` never grows/re-allocates the factor matrix.
                @test size(factor_matrix(cache.cacheval)) == (size(A, 1), size(A, 2))
                @test cache.cacheval.config == algorithm.config
                init_storage = factor_matrix(cache.cacheval)
                first = SciMLBase.solve!(cache)
                @test first.retcode == SciMLBase.ReturnCode.Success
                @test !cache.isfresh
                factor = cache.cacheval
                factor_storage = factor_matrix(factor)
                # First solve wrote into the storage reserved at `init`.
                @test factor_storage === init_storage

                expected_second = T[-1, 2, 1, 3]
                cache.b = A * expected_second
                @test !cache.isfresh
                second = SciMLBase.solve!(cache)
                @test second.retcode == SciMLBase.ReturnCode.Success
                @test max_relative_error(second.u, expected_second) <= tolerance(T, 16)
                @test cache.cacheval === factor
                @test factor_matrix(cache.cacheval) === factor_storage

                updated_A = copy(A)
                updated_A[1, 1] += one(T)
                cache.A = updated_A
                cache.b = updated_A * expected
                @test cache.isfresh
                refreshed = SciMLBase.solve!(cache)
                @test refreshed.retcode == SciMLBase.ReturnCode.Success
                @test max_relative_error(refreshed.u, expected) <= tolerance(T, 16)
                # The cache reuses its owned factor storage on an in-place A
                # update: the numeric factor matrix object is preserved.
                @test factor_matrix(cache.cacheval) === factor_storage
                @test !cache.isfresh
                @test updated_A[1, 1] == original_A[1, 1] + one(T)

                # multi-RHS shares the same cache semantics: one factor, many RHS
                expected_matrix = T[
                    1 -2 3
                    -1 2 1
                    3 1 -2
                    2 0 4
                ]
                B = A * expected_matrix
                problem_matrix = LinearSolve.LinearProblem(A, B)
                matrix_cache = LinearSolve.init(problem_matrix, algorithm)
                matrix_solution = SciMLBase.solve!(matrix_cache)
                @test matrix_solution.retcode == SciMLBase.ReturnCode.Success
                @test size(matrix_solution.u) == size(B)
                @test max_relative_error(matrix_solution.u, expected_matrix) <= tolerance(T, 16)
                @test !matrix_cache.isfresh
                matrix_storage = factor_matrix(matrix_cache.cacheval)
                # RHS-only multi-RHS update reuses the factor
                expected_matrix2 = T[
                    -2 1 0
                    4 -3 2
                    0 1 1
                    2 5 -1
                ]
                matrix_cache.b = A * expected_matrix2
                matrix_solution2 = SciMLBase.solve!(matrix_cache)
                @test matrix_solution2.retcode == SciMLBase.ReturnCode.Success
                @test max_relative_error(matrix_solution2.u, expected_matrix2) <= tolerance(T, 16)
                @test factor_matrix(matrix_cache.cacheval) === matrix_storage

                # Explicit public refresh for *in-place* A mutation: modifying an
                # entry of the matrix the cache holds (not reassigning `cache.A`)
                # must mark the cache fresh and re-factorize into the same
                # storage. LinearSolve owns its concrete `cache.A` copy, so the
                # in-place mutation must target that object.
                inplace_cache = LinearSolve.init(
                    LinearSolve.LinearProblem(copy(A), b), algorithm,
                )
                inplace_storage = factor_matrix(inplace_cache.cacheval)
                inplace_first = SciMLBase.solve!(inplace_cache)
                @test inplace_first.retcode == SciMLBase.ReturnCode.Success
                inplace_cache.A[1, 1] += one(T)
                refresh!(inplace_cache)
                @test inplace_cache.isfresh
                inplace_cache.b = inplace_cache.A * expected
                inplace_refreshed = SciMLBase.solve!(inplace_cache)
                @test inplace_refreshed.retcode == SciMLBase.ReturnCode.Success
                @test max_relative_error(inplace_refreshed.u, expected) <= tolerance(T, 16)
                @test !inplace_cache.isfresh
                @test factor_matrix(inplace_cache.cacheval) === inplace_storage
            end

            singular = T[1 2; 2 4]
            failed_lu_cache = LinearSolve.init(
                LinearSolve.LinearProblem(singular, T[1, 2]),
                MultiFloatLU(),
            )
            failed_lu = SciMLBase.solve!(failed_lu_cache)
            @test failed_lu.retcode == SciMLBase.ReturnCode.Failure
            @test failed_lu_cache.isfresh
            # Fail-closed: the failed factorization must not retain a prior
            # success status.
            @test !MultiFloatLinearAlgebra.issuccess(failed_lu_cache.cacheval)
            @test factor_status(failed_lu_cache.cacheval) != 0
            # Replacing A (via the official cache-update path) lets the retry
            # succeed from the same owned factor storage.
            good_lu = T[2 0; 0 3]
            failed_lu_cache.A = good_lu
            failed_lu_cache.b = good_lu * T[1, 1]
            retried_lu = SciMLBase.solve!(failed_lu_cache)
            @test retried_lu.retcode == SciMLBase.ReturnCode.Success
            @test max_relative_error(retried_lu.u, T[1, 1]) <= tolerance(T, 16)
            @test factor_status(failed_lu_cache.cacheval) == 0
            @test !failed_lu_cache.isfresh

            indefinite = T[1 2; 2 1]
            failed_cholesky_cache = LinearSolve.init(
                LinearSolve.LinearProblem(indefinite, T[1, 1]),
                MultiFloatCholesky(),
            )
            failed_cholesky = SciMLBase.solve!(failed_cholesky_cache)
            @test failed_cholesky.retcode == SciMLBase.ReturnCode.Failure
            @test failed_cholesky_cache.isfresh
            @test !MultiFloatLinearAlgebra.issuccess(failed_cholesky_cache.cacheval)
            @test factor_status(failed_cholesky_cache.cacheval) != 0
        end
    end
end
