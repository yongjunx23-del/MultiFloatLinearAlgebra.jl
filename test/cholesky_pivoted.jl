using Test
using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra

function pstrf_relative_error(A, B)
    numerator = zero(eltype(A))
    denominator = zero(eltype(A))
    @inbounds for index in eachindex(A, B)
        numerator = max(numerator, abs(A[index] - B[index]))
        denominator = max(denominator, abs(B[index]))
    end
    return numerator / max(denominator, one(eltype(A)))
end

@testset "dynamic pivoted Cholesky" begin
    for T in (Float64x2, Float64x4)
        n, r = 7, 3
        B = zeros(T, n, r)
        B[1, 1] = T(1)
        B[2, 2] = T(4)
        B[3, 3] = T(2)
        B[4, :] .= T[2, 4, 1]
        B[5, :] .= T[-1, 1, 3]
        B[6, :] .= T[3, -2, 1]
        B[7, :] .= T[1, 2, -2]
        original = zeros(T, n, n)
        @inbounds for column in 1:n, row in 1:n
            value = zero(T)
            for k in 1:r
                value += B[row, k] * B[column, k]
            end
            original[row, column] = value
        end

        # The largest diagonal is row 4 (21), so a real max-residual PSTRF
        # must move it to the first pivot.  A static-order Cholesky would fail
        # this assertion and can reveal the wrong rank on this panel.
        F = cholesky_pivoted!(copy(original); tol=T(1e-12))
        @test factor_kind(F) === :cholesky_pivoted
        @test factor_status(F) == 0
        @test MultiFloatLinearAlgebra.issuccess(F)
        @test F.rank == r
        diagnostics = factor_diagnostics(F)
        @test diagnostics.kind === :cholesky_pivoted
        @test diagnostics.rank == r
        @test diagnostics.permutation == factor_permutation(F)
        permutation = factor_permutation(F)
        @test sort(permutation) == collect(1:n)
        @test permutation[1] == 4

        permuted = original[permutation, permutation]
        retained = tril(factor_matrix(F))[:, 1:F.rank]
        @test pstrf_relative_error(
            retained * transpose(retained), permuted,
        ) <= T(1e-20)

        # The tolerance stop is a rank decision, not a factorization failure.
        @test all(factor_rdiag(F)[1:F.rank] .> zero(T))
        @test_throws ArgumentError cholesky_pivoted!(
            copy(original); tol=T(-1), check=false,
        )

        indefinite = -Matrix{T}(I, n, n)
        Fbad = cholesky_pivoted!(indefinite; check=false)
        @test factor_status(Fbad) > 0
        @test !MultiFloatLinearAlgebra.issuccess(Fbad)
        @test Fbad.rank == 0
        @test_throws LinearAlgebra.PosDefException cholesky_pivoted!(
            copy(indefinite),
        )

        # Only the lower triangle is authoritative.  The upper triangle may
        # contain arbitrary finite data and must be replaced before pivoting.
        lower_only = copy(original)
        @inbounds for column in 1:n, row in 1:(column - 1)
            lower_only[row, column] = T(17 + row + 3column)
        end
        Flower = cholesky_pivoted!(lower_only; tol=T(1e-12))
        @test Flower.rank == r
        @test factor_permutation(Flower) == permutation
    end
end
