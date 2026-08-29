using QDLDL
using SparseArrays

function _qdldl_upper(::Type{T}; shift=zero(T)) where {T}
    values = T[2 + shift, 1, -2, T(0.25), -T(1.5)]
    return SparseMatrixCSC(3, 3, [1, 2, 4, 6], [1, 1, 2, 2, 3], values)
end

function _qdldl_tiny_upper(::Type{T}) where {T}
    return SparseMatrixCSC(
        3, 3, [1, 2, 3, 4], [1, 2, 3], T[T(1e-20), -2, -3],
    )
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
            @test_throws ArgumentError sparse_ldlt_cache(
                T, A; dsigns=Int[1, -1, -1],
                regularize_eps=T(1e-30), regularize_delta=T(1e-20),
            )
            cache = sparse_ldlt_cache(
                T, A; dsigns=Int[1, -1, -1], nrhs=2,
            )
            @test factor_state(cache) === :invalidated
            factorize!(cache, A)
            @test factor_state(cache) === :success
            diagnostics = factor_diagnostics(cache)
            @test diagnostics.provider === :qdldl
            @test diagnostics.symbolic_count == 1
            @test diagnostics.numeric_factor_count == 1
            @test diagnostics.positive_inertia == 1
            @test diagnostics.regularized_entries == 0
            @test eltype(factor_matrix(cache)) === T

            # Public factor_matrix is a detached snapshot, never cache authority.
            exposed = factor_matrix(cache)
            exposed.colptr[end] -= 1
            @test cache.matrix.colptr == cache.frozen_colptr

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

            # Destination must not alias any provider factor storage.
            factor = something(cache.factor)
            @test_throws ArgumentError MultiFloatLinearAlgebra.solve!(
                cache, factor.Dinv.diag, b,
            )

            # Non-finite RHS is rejected before provider execution.
            nonfinite = copy(b)
            nonfinite[2] = T(NaN)
            before_solves = factor_diagnostics(cache).solve_count
            @test_throws ArgumentError MultiFloatLinearAlgebra.solve!(
                cache, x, nonfinite,
            )
            @test factor_diagnostics(cache).solve_count == before_solves

            # Numeric mutation after factorization invalidates solve authority.
            cache.matrix.nzval[1] += one(T)
            @test_throws ArgumentError MultiFloatLinearAlgebra.solve!(cache, x, b)
            @test factor_state(cache) === :invalidated
            factorize!(cache, A2)

            # Internal structural mutation is detected independently of input.
            saved_row = cache.matrix.rowval[2]
            cache.matrix.rowval[2] = saved_row == 1 ? 2 : 1
            @test_throws ArgumentError factorize!(cache, A2)
            @test factor_state(cache) === :invalidated
            cache.matrix.rowval[2] = saved_row
            factorize!(cache, A2)

            drift = SparseMatrixCSC(
                3, 3, [1, 2, 4, 5], copy(A2.rowval[1:4]), copy(A2.nzval[1:4]),
            )
            @test_throws ArgumentError factorize!(cache, drift)
            @test factor_state(cache) === :invalidated

            # A failed refactor is stale; success-only diagnostics are cleared.
            factorize!(cache, A2)
            bad = SparseMatrixCSC(
                3, 3, copy(A2.colptr), copy(A2.rowval), zeros(T, length(A2.nzval)),
            )
            factorize!(cache, bad; check=false)
            @test factor_state(cache) !== :success
            failed_diag = factor_diagnostics(cache)
            @test failed_diag.positive_inertia == -1
            @test failed_diag.regularized_entries == 0
            @test_throws ArgumentError MultiFloatLinearAlgebra.solve!(cache, x, b)
            factorize!(cache, A2)

            # Tiny legitimate pivots are not absolutely regularized.
            tiny = _qdldl_tiny_upper(T)
            tiny_cache = sparse_ldlt_cache(
                T, tiny; dsigns=Int[1, -1, -1], nrhs=1,
            )
            factorize!(tiny_cache, tiny)
            tiny_b = T[T(1e-20), -2, -3]
            tiny_x = similar(tiny_b)
            MultiFloatLinearAlgebra.solve!(tiny_cache, tiny_x, tiny_b)
            tiny_residual = norm(_qdldl_dense(tiny) * tiny_x - tiny_b, Inf)
            @test tiny_residual <= T(4096) * eps(T) * max(
                one(T), norm(tiny_b, Inf), norm(tiny, Inf) * norm(tiny_x, Inf),
            )
            @test factor_diagnostics(tiny_cache).regularized_entries == 0
            @test eltype(tiny_x) === T
        end
    end
end
