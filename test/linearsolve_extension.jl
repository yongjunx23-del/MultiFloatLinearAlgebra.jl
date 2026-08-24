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
                @test size(factor_matrix(cache.cacheval)) == (0, 0)
                first = SciMLBase.solve!(cache)
                @test first.retcode == SciMLBase.ReturnCode.Success
                @test !cache.isfresh
                factor = cache.cacheval
                factor_storage = factor_matrix(factor)

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
                @test factor_matrix(cache.cacheval) !== factor_storage
                @test !cache.isfresh
                @test updated_A[1, 1] == original_A[1, 1] + one(T)
            end

            singular = T[1 2; 2 4]
            failed_lu_cache = LinearSolve.init(
                LinearSolve.LinearProblem(singular, T[1, 2]),
                MultiFloatLU(),
            )
            failed_lu = SciMLBase.solve!(failed_lu_cache)
            @test failed_lu.retcode == SciMLBase.ReturnCode.Failure
            @test failed_lu_cache.isfresh

            indefinite = T[1 2; 2 1]
            failed_cholesky_cache = LinearSolve.init(
                LinearSolve.LinearProblem(indefinite, T[1, 1]),
                MultiFloatCholesky(),
            )
            failed_cholesky = SciMLBase.solve!(failed_cholesky_cache)
            @test failed_cholesky.retcode == SciMLBase.ReturnCode.Failure
            @test failed_cholesky_cache.isfresh
        end
    end
end
