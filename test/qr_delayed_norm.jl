using Test
using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra

@testset "delayed RRQR norm state" begin
    T = Float64x4
    rows, columns = 96, 6
    A = Matrix{T}(undef, rows, columns)
    for column in 1:columns, row in 1:rows
        A[row, column] = T(
            sin(0.013 * row + 0.41 * column) +
            0.07 * cos(0.002 * row * column),
        )
    end
    Ftranspose = Matrix{T}(undef, 2, columns)
    for column in 1:columns
        Ftranspose[1, column] = T(0.17 * column - 0.03)
        Ftranspose[2, column] = T(-0.11 * column + 0.02)
    end
    actual = MultiFloatLinearAlgebra._qr_delayed_column_norm_state(
        A, Ftranspose, 1, 2, 3, 5,
    )
    scale = zero(T)
    scaled_sum = one(T)
    seen = false
    for row in 3:rows
        value = A[row, 5] - A[row, 1] * Ftranspose[1, 5] -
                A[row, 2] * Ftranspose[2, 5]
        magnitude = abs(value)
        if !iszero(magnitude)
            if !seen
                scale = magnitude
                scaled_sum = one(T)
                seen = true
            elseif magnitude > scale
                ratio = scale / magnitude
                scaled_sum = one(T) + scaled_sum * ratio * ratio
                scale = magnitude
            else
                ratio = magnitude / scale
                scaled_sum += ratio * ratio
            end
        end
    end
    @test actual == (scale, scaled_sum)

    # The same deterministic panel must preserve the compact factor exactly
    # across serial, threaded, and workspace-reuse routes.
    source = Matrix{T}(undef, 500, 40)
    for column in axes(source, 2), row in axes(source, 1)
        source[row, column] = T(
            sin(row * 0.01 + column * 0.7) +
            0.1 * cos(row * 0.003 * column),
        )
    end
    serial = rrqr!(copy(source); threads=1)
    threaded = rrqr!(copy(source); threads=2)
    workspace = MFWorkspace(T; thread_count=2)
    reused = rrqr!(copy(source); threads=2, workspace=workspace)
    @test factor_matrix(threaded) == factor_matrix(serial)
    @test factor_permutation(threaded) == factor_permutation(serial)
    @test factor_matrix(reused) == factor_matrix(threaded)
    @test factor_permutation(reused) == factor_permutation(threaded)
    @test numerical_rank(threaded; rtol=T(1e-30)) == 40
    permuted = source[:, factor_permutation(threaded)]
    apply_q!(permuted, threaded; trans=:T)
    @test maximum(abs, tril(permuted[1:40, :], -1)) <= T(1e-50)

    # Regression for cancellation-sensitive rank reveal: omitting the
    # current reflector from the exact delayed norm rebuild reports rank 2
    # instead of the true rank 30 on this deterministic duplicated-column
    # panel.  The current reflector row must be included in Ftranspose.
    rows, columns, true_rank = 256, 64, 30
    deficient = zeros(T, rows, columns)
    core = zeros(T, rows, true_rank)
    for column in axes(core, 2), row in axes(core, 1)
        core[row, column] = T(
            sin(0.013 * row + 0.23 * column) +
            0.21 * cos(0.005 * row * column),
        )
    end
    deficient[:, 1:true_rank] .= core
    deficient[:, 31:40] .= core[:, 2:11]
    deficient_factor = rrqr!(copy(deficient); threads=2)
    @test numerical_rank(
        deficient_factor; rtol=sqrt(eps(T)),
    ) == true_rank
    deficient_permuted = deficient[:, factor_permutation(deficient_factor)]
    apply_q!(deficient_permuted, deficient_factor; trans=:T)
    @test maximum(abs, tril(deficient_permuted[1:columns, :], -1)) <= T(1e-50)
end
