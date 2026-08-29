using QDLDL
using SparseArrays

function _qdldl_upper(::Type{T}; shift=zero(T)) where {T}
    # Upper triangle of a quasi-definite (+, -, -) matrix. Structural zeros
    # remain stored, as required by repeated same-pattern refactors.
    values = T[2 + shift, 1, -2, T(0.25), -T(1.5)]
    return SparseMatrixCSC(3, 3, [1, 2, 4, 6], [1, 1, 2, 2, 3], values)
end

function _qdldl_dense(A)
    K = Matrix(A)
    return K + transpose(K) - Diagonal(diag(K))
end

@testset "optional QDLDL sparse LDL cache" begin
    @test MultiFloatLinearAlgebra.sparse_ldlt_available(Float64x2)
    for T in (Float64x2, Float64x4)
        @testset "$T" begin
            A = _qdldl_upper(T)
            cache = sparse_ldlt_cache(
                T, A; dsigns=Int[1, -1, -1], nrhs=2,
                regularize_eps=T(1e-30), regularize_delta=T(1e-20),
            )
            @test factor_state(cache) === :invalidated
            factorize!(cache, A)
            @test factor_state(cache) === :success
            diagnostics = factor_diagnostics(cache)
            @test diagnostics.provider === :qdldl
            @test diagnostics.symbolic_count == 1
            @test diagnostics.numeric_factor_count == 1
            @test diagnostics.positive_inertia == 1
            @test eltype(factor_matrix(cache)) === T

            b = T[1, 2, -1]
            x = similar(b)
            MultiFloatLinearAlgebra.solve!(cache, x, b)
            residual = norm(_qdldl_dense(A) * x - b, Inf)
            @test residual <= T(2048) * eps(T) * max(one(T), norm(b, Inf))

            # Same pattern, second numeric refill (including a numeric zero).
            A2 = _qdldl_upper(T; shift=T(0.125))
            A2.nzval[4] = zero(T)
            factorize!(cache, A2)
            diagnostics = factor_diagnostics(cache)
            @test diagnostics.symbolic_count == 1
            @test diagnostics.numeric_factor_count == 2

            B = T[1 2; 2 -1; -1 1]
            X = similar(B)
            MultiFloatLinearAlgebra.solve!(cache, X, B)
            @test norm(_qdldl_dense(A2) * X - B, Inf) <=
                  T(4096) * eps(T) * max(one(T), norm(B, Inf))
            @test factor_diagnostics(cache).solve_count == 3

            # Exact self-alias is supported; partial overlap is rejected.
            Balias = copy(B)
            MultiFloatLinearAlgebra.solve!(cache, Balias, Balias)
            storage = zeros(T, 3, 3)
            @test_throws ArgumentError MultiFloatLinearAlgebra.solve!(
                cache, @view(storage[:, 1:2]), @view(storage[:, 2:3]),
            )

            drift = copy(A2)
            drift = SparseMatrixCSC(
                3, 3, [1, 2, 4, 5], drift.rowval[1:4], drift.nzval[1:4],
            )
            @test_throws ArgumentError factorize!(cache, drift)
            @test factor_state(cache) !== :success

            # A failed refactor is stale and cannot solve; a valid refill recovers.
            bad = copy(A2)
            bad.nzval[1] = T(NaN)
            factorize!(cache, bad; check=false)
            @test factor_state(cache) !== :success
            @test_throws ArgumentError MultiFloatLinearAlgebra.solve!(cache, x, b)
            factorize!(cache, A2)
            MultiFloatLinearAlgebra.solve!(cache, x, b)
            @test eltype(x) === T
        end
    end
end
