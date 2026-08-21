using Test
using MultiFloatLinearAlgebra
using MultiFloats
using LinearAlgebra
using Random

@testset "fixed-permutation QR" begin
    T = Float64x2
    m, n = 40, 12
    rng = MersenneTwister(0x5eed)
    A0 = [T(randn(rng)) * T(randn(rng)) for _ in 1:m, _ in 1:n]
    perm = collect(n:-1:1)  # reversed order, non-identity

    # Fixed-order factorization: A[:, perm] = Q * R
    A1 = copy(A0)
    F1 = MultiFloatLinearAlgebra.qr!(A1, perm)
    @test F1.permutation == perm
    R1 = triu(F1.factors)
    # End-to-end: the fixed-order ldiv! solves the permuted system exactly.
    expected_x = T[T(i) for i in 1:n]
    rhs_full = A0[:, perm] * expected_x
    @test norm(apply_q!(copy(rhs_full), F1; trans=:T) -
               R1 * expected_x) <= T(1e-25) * norm(rhs_full)

    # Equivalence with explicit reorder + unpivoted qr!
    A2 = copy(A0)
    F2 = MultiFloatLinearAlgebra.qr!(A2[:, perm])
    @test triu(F2.factors) ≈ R1 rtol = T(1e-30) atol = T(1e-30)

    # Solve semantics: rectangular route via apply_q!/solve_r! with the
    # caller's permutation -- exactly how a full-rank equality consumer
    # drives the factorization (A0[:, perm] * x = rhs).
    factor_order_solution = T[T(i) for i in 1:n]
    rhs = A0[:, perm] * factor_order_solution
    work = copy(rhs)
    MultiFloatLinearAlgebra.apply_q!(work, F1; trans=:T)
    # Back-substitute in the FACTOR's column order, then scatter each
    # component to its original column identity: the full solution satisfies
    # x_full[permutation[i]] = factor_order_solution[i].
    xp = zeros(T, n)
    @inbounds for i in n:-1:1
        acc = work[i]
        for j in (i+1):n
            acc -= F1.factors[i, j] * xp[j]
        end
        xp[i] = acc / F1.factors[i, i]
    end
    x_full = zeros(T, n)
    @inbounds for i in 1:n
        x_full[Int(F1.permutation[i])] = xp[i]
    end
    @test norm(x_full - factor_order_solution[perm]) <=
          T(1e-28) * norm(x_full)

    # Validation: bad permutations fail closed.
    A3 = copy(A0)
    @test_throws DimensionMismatch MultiFloatLinearAlgebra.qr!(A3, ones(Int, n + 1))
    @test_throws ArgumentError MultiFloatLinearAlgebra.qr!(A3, [1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    @test_throws ArgumentError MultiFloatLinearAlgebra.qr!(A3, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
end
