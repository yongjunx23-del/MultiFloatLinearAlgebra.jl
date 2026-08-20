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

function trmm_reference(B, A, alpha; side, uplo, trans, diag)
    result = similar(B)
    effective_lower = (uplo === :lower) == (trans === :N)
    unit_diagonal = diag === :unit

    if side === :left
        @inbounds for j in axes(B, 2), i in axes(B, 1)
            accumulator = zero(eltype(B))
            for k in (effective_lower ? (1:i) : (i:size(A, 1)))
                coefficient = k == i && unit_diagonal ? one(eltype(B)) :
                    (trans === :N ? A[i, k] : A[k, i])
                accumulator += coefficient * B[k, j]
            end
            result[i, j] = alpha * accumulator
        end
    else
        @inbounds for i in axes(B, 1), j in axes(B, 2)
            accumulator = zero(eltype(B))
            for k in (effective_lower ? (j:size(A, 1)) : (1:j))
                coefficient = k == j && unit_diagonal ? one(eltype(B)) :
                    (trans === :N ? A[k, j] : A[j, k])
                accumulator += B[i, k] * coefficient
            end
            result[i, j] = alpha * accumulator
        end
    end
    return result
end

function qr_explicit_r(F)
    rows, columns = size(F)
    R = zeros(eltype(F), rows, columns)
    factors = factor_matrix(F)
    @inbounds for column in 1:columns
        for row in 1:min(column, rows)
            R[row, column] = factors[row, column]
        end
    end
    return R
end

# A deliberately scalar reference path for blocked-RRQR validation.  This
# exercises the package's unblocked panel update at the same precision while
# keeping the blocked implementation under test independent of its delayed
# WY/GEMM update.  The reference returns compact Householder storage rather
# than an MFQR wrapper so the test does not depend on private constructors.
function qr_unblocked_reference(A)
    T = eltype(A)
    B = copy(A)
    m, n = size(B)
    reflector_count = min(m, n)
    tau = zeros(T, reflector_count)
    permutation = collect(1:n)
    norm_scale, norm_sum, norm_dirty =
        MultiFloatLinearAlgebra._prepare_qr_norm_state(T, n, nothing)
    MultiFloatLinearAlgebra._qr_initialize_norm_state!(
        B, norm_scale, norm_sum, norm_dirty,
    )
    MultiFloatLinearAlgebra._rrqr_unblocked!(
        B, tau, permutation,
        norm_scale, norm_sum, norm_dirty,
        sqrt(eps(T)), T(16) * eps(T), 1,
    )
    return B, tau, permutation
end

function qr_reference_apply_qt!(X, factors, tau)
    T = eltype(X)
    @inbounds for step in eachindex(tau)
        iszero(tau[step]) && continue
        for column in axes(X, 2)
            projection = X[step, column]
            for row in (step + 1):size(factors, 1)
                projection += factors[row, step] * X[row, column]
            end
            projection *= tau[step]
            X[step, column] -= projection
            for row in (step + 1):size(factors, 1)
                X[row, column] -= factors[row, step] * projection
            end
        end
    end
    return X
end

function qr_reference_rank(factors; rtol=zero(eltype(factors)), atol=zero(eltype(factors)))
    T = eltype(factors)
    diagonal_count = min(size(factors)...)
    largest = zero(T)
    @inbounds for index in 1:diagonal_count
        largest = max(largest, abs(factors[index, index]))
    end
    threshold = max(T(atol), T(rtol) * largest)
    rank = 0
    @inbounds for index in 1:diagonal_count
        abs(factors[index, index]) > threshold || break
        rank += 1
    end
    return rank
end

function blocked_rrqr_adversarial_matrix(T, kind; rows=256, columns=64)
    A = zeros(T, rows, columns)
    if kind === :near_tie
        # All columns have comparable norms, with close angular separations;
        # this stresses pivot tie handling without relying on random state.
        for column in 1:columns, row in 1:rows
            A[row, column] = T(
                sin(0.017 * row + 0.19 * column) +
                0.18 * cos(0.011 * row * column) +
                0.03 * sin(0.003 * row^2 + column),
            )
        end
    elseif kind === :cancellation
        # The second column is almost parallel to the first.  Its residual
        # norm is below the downdate reliability floor, forcing an exact
        # rebuild at a blocked-panel boundary.
        for row in 1:rows
            u = T(sin(0.021 * row) + 0.4 * cos(0.007 * row))
            v = T(cos(0.013 * row) - 0.2 * sin(0.005 * row))
            A[row, 1] = u
            A[row, 2] = u + T(ldexp(1.0, -60)) * v
        end
        for column in 3:columns, row in 1:rows
            A[row, column] = T(
                sin(0.019 * row + 0.17 * column) +
                0.11 * cos(0.004 * row * column),
            )
        end
    elseif kind === :rank_deficient
        core = zeros(T, rows, 30)
        for column in axes(core, 2), row in axes(core, 1)
            core[row, column] = T(
                sin(0.013 * row + 0.23 * column) +
                0.21 * cos(0.005 * row * column),
            )
        end
        A[:, 1:30] .= core
        A[:, 31:40] .= core[:, 2:11]
        A[:, 41:columns] .= zero(T)
    elseif kind === :extreme_scale
        for column in 1:columns, row in 1:rows
            A[row, column] = T(
                sin(0.015 * row + 0.07 * column) +
                0.17 * cos(0.009 * row * column),
            )
        end
        exponents = (-300, -150, 0, 150, 300)
        for column in 1:columns
            A[:, column] .*= T(ldexp(1.0, exponents[mod1(column, length(exponents))]))
        end
    else
        throw(ArgumentError("unknown adversarial QR matrix kind: $kind"))
    end
    return A
end

include("mulacc_x3_proof_vectors.jl")
include("adversarial.jl")

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

                matrix_reference = zero(T)
                for index in eachindex(A, B)
                    matrix_reference += A[index] * B[index]
                end
                @test mfdot(A, B) == matrix_reference
                @test_throws DimensionMismatch mfdot(A, view(B, :, 1))
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

            @testset "beta-zero GEMM destination semantics" begin
                rows, reduction, columns = 9, 7, 15
                beta_A = T.(randn(rows, reduction))
                beta_B = T.(randn(reduction, columns))
                for thread_count in unique((1, min(4, Threads.nthreads())))
                    strategies = T === Float64x3 ?
                        (:direct, :fused, :packed) : (:direct, :packed)
                    for strategy in strategies
                        beta_config = KernelConfig(
                            thread_count=thread_count,
                            gemm_strategy=strategy,
                            gemm_panel_columns=5,
                            gemm_micro_columns=4,
                        )
                        clean = zeros(T, rows, columns)
                        stale = fill(T(NaN), rows, columns)
                        workspace = strategy === :packed ? GemmWorkspace(
                            T;
                            thread_count=thread_count,
                            capacity=reduction * 5,
                        ) : nothing
                        gemm!(
                            clean, beta_A, beta_B, T(2), zero(T);
                            config=beta_config, workspace=workspace,
                        )
                        gemm!(
                            stale, beta_A, beta_B, T(2), zero(T);
                            config=beta_config, workspace=workspace,
                        )
                        @test stale == clean
                    end

                    initial = T.(randn(rows, columns))
                    direct_update = copy(initial)
                    packed_update = copy(initial)
                    direct_config = KernelConfig(
                        thread_count=thread_count,
                        gemm_strategy=:direct,
                        gemm_panel_columns=5,
                    )
                    packed_config = KernelConfig(
                        thread_count=thread_count,
                        gemm_strategy=:packed,
                        gemm_panel_columns=5,
                        gemm_micro_columns=4,
                    )
                    packed_workspace = GemmWorkspace(
                        T;
                        thread_count=thread_count,
                        capacity=reduction * 5,
                    )
                    gemm!(
                        direct_update, beta_A, beta_B, T(2), T(3);
                        config=direct_config,
                    )
                    gemm!(
                        packed_update, beta_A, beta_B, T(2), T(3);
                        config=packed_config, workspace=packed_workspace,
                    )
                    @test limb_bitwise_equal(packed_update, direct_update)

                    gemmt_rows = 25
                    left = T.(randn(gemmt_rows, reduction))
                    right = T.(randn(gemmt_rows, reduction))
                    clean = fill(T(17), gemmt_rows, gemmt_rows)
                    stale = copy(clean)
                    for column in 1:gemmt_rows, row in column:gemmt_rows
                        clean[row, column] = zero(T)
                        stale[row, column] = T(NaN)
                    end
                    gemmt_config = KernelConfig(thread_count=thread_count)
                    gemmt!(
                        clean, left, right, T(2), zero(T);
                        config=gemmt_config,
                    )
                    gemmt!(
                        stale, left, right, T(2), zero(T);
                        config=gemmt_config,
                    )
                    @test stale == clean
                end
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
                wrong_type = T === Float64x2 ? Float64x4 : Float64x2
                @test_throws ArgumentError gemm_plan(
                    wrong_type, 16, 16, 16, resolved,
                )

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

            @testset "syrk_packed!" begin
                reduction = 11
                columns = 9
                panel = T.(randn(reduction, columns))
                first = 2
                last = 10
                alpha = T(2)
                beta = T(-3)
                initial = T.(randn(columns * (columns + 1) ÷ 2))
                expected = similar(initial)
                output_index = 0
                for column in 1:columns, row in column:columns
                    output_index += 1
                    accumulator = zero(T)
                    for k in first:last
                        accumulator += panel[k, row] * panel[k, column]
                    end
                    expected[output_index] =
                        alpha * accumulator + beta * initial[output_index]
                end

                output = copy(initial)
                syrk_packed!(
                    output,
                    panel,
                    alpha,
                    beta;
                    reduction_first=first,
                    reduction_last=last,
                    config=config,
                )
                @test output == expected

                # beta=0 must not read stale/nonfinite packed storage.
                clean = zeros(T, length(output))
                stale = fill(T(NaN), length(output))
                single = KernelConfig(thread_count=1)
                syrk_packed!(clean, panel; config=single)
                syrk_packed!(stale, panel; config=single)
                @test stale == clean

                wide_panel = T.(randn(13, 25))
                wide_entries = size(wide_panel, 2) * (size(wide_panel, 2) + 1) ÷ 2
                serial_wide = zeros(T, wide_entries)
                threaded_wide = fill(T(NaN), wide_entries)
                syrk_packed!(serial_wide, wide_panel; config=single)
                syrk_packed!(
                    threaded_wide,
                    wide_panel;
                    config=KernelConfig(thread_count=2),
                )
                @test threaded_wide == serial_wide

                @test_throws DimensionMismatch syrk_packed!(
                    zeros(T, length(output) - 1), panel; config=config,
                )
                @test_throws BoundsError syrk_packed!(
                    zeros(T, length(output)), panel;
                    reduction_first=0, reduction_last=last, config=config,
                )
                @test_throws BoundsError syrk_packed!(
                    zeros(T, length(output)), panel;
                    reduction_first=first, reduction_last=reduction + 1,
                    config=config,
                )
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

            @testset "trmm!" begin
                order = 7
                alpha = T(-2)
                for side in (:left, :right), uplo in (:lower, :upper),
                    trans in (:N, :T), diag in (:unit, :nonunit)
                    triangular = triangular_matrix(T, order, uplo; unit=diag === :unit)

                    # Neither the inactive triangle nor a declared unit
                    # diagonal is authoritative.
                    if uplo === :lower
                        for row in 1:order, column in (row + 1):order
                            triangular[row, column] = T(NaN)
                        end
                    else
                        for row in 1:order, column in 1:(row - 1)
                            triangular[row, column] = T(Inf)
                        end
                    end
                    if diag === :unit
                        for index in 1:order
                            triangular[index, index] = T(NaN)
                        end
                    end

                    input = side === :left ?
                        T.(randn(order, 5)) : T.(randn(5, order))
                    expected = trmm_reference(
                        input, triangular, alpha;
                        side=side, uplo=uplo, trans=trans, diag=diag,
                    )
                    output = copy(input)
                    trmm!(
                        output, triangular, alpha;
                        side=side, uplo=uplo, trans=trans, diag=diag,
                        config=config,
                    )
                    @test output == expected
                end

                square = triangular_matrix(T, order, :lower)
                @test_throws DimensionMismatch trmm!(
                    zeros(T, order - 1, 2), square; side=:left, config=config,
                )
                @test_throws DimensionMismatch trmm!(
                    zeros(T, 2, order - 1), square; side=:right, config=config,
                )
                @test_throws DimensionMismatch trmm!(
                    zeros(T, order, 2), zeros(T, order, order - 1); config=config,
                )
                for (keyword, value) in (
                    (:side, :bad), (:uplo, :bad), (:trans, :bad), (:diag, :bad),
                )
                    arguments = (; config=config, keyword => value)
                    @test_throws ArgumentError trmm!(
                        zeros(T, order, 2), square; arguments...,
                    )
                end
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

            @testset "rank-revealing QR" begin
                for (rows, columns) in ((15, 7), (7, 15), (9, 9))
                    original = T.(randn(rows, columns))
                    F = rrqr!(copy(original))
                    @test MultiFloatLinearAlgebra.issuccess(F)
                    @test factor_kind(F) === :qr
                    @test factor_status(F) == 0
                    @test size(F) == (rows, columns)
                    @test eltype(F) === T

                    permutation = factor_permutation(F)
                    @test sort(permutation) == collect(1:columns)
                    permutation[1] = 0
                    @test factor_permutation(F)[1] != 0
                    @test factor_rdiag(F) == [
                        factor_matrix(F)[i, i] for i in 1:min(rows, columns)
                    ]

                    Q = Matrix{T}(I, rows, rows)
                    apply_q!(Q, F)
                    R = qr_explicit_r(F)
                    reconstruction = Q * R
                    @test max_relative_error(
                        reconstruction, original[:, factor_permutation(F)],
                    ) <= tolerance(T, 64 * max(rows, columns))
                    @test max_relative_error(
                        transpose(Q) * Q, Matrix{T}(I, rows, rows),
                    ) <= tolerance(T, 64rows)
                    @test numerical_rank(F; atol=zero(T), rtol=zero(T)) ==
                        min(rows, columns)

                    vector = T.(randn(rows))
                    vector_roundtrip = copy(vector)
                    apply_q!(vector_roundtrip, F; trans=:T)
                    apply_q!(vector_roundtrip, F; trans=:N)
                    @test max_relative_error(vector_roundtrip, vector) <=
                        tolerance(T, 64rows)

                    matrix = T.(randn(rows, 3))
                    matrix_roundtrip = copy(matrix)
                    apply_q!(matrix_roundtrip, F; trans=:T)
                    apply_q!(matrix_roundtrip, F; trans=:N)
                    @test max_relative_error(matrix_roundtrip, matrix) <=
                        tolerance(T, 64rows)
                end

                # Rank deficiency is a successful factorization. Exact zero
                # columns and duplicate columns are classified only by the
                # threshold explicitly supplied by the caller.
                rank_source = T.(randn(13, 4))
                deficient = hcat(
                    rank_source,
                    copy(rank_source[:, 2]),
                    zeros(T, size(rank_source, 1)),
                )
                Fdef = rrqr!(copy(deficient))
                @test MultiFloatLinearAlgebra.issuccess(Fdef)
                @test numerical_rank(Fdef; rtol=sqrt(eps(T))) == 4
                @test numerical_rank(rrqr!(zeros(T, 5, 8))) == 0

                # A perturbation near sqrt(eps(T)) remains visible under an
                # exact-nonzero test, while a coarser caller threshold treats
                # the same column as dependent.
                nearly = T.(randn(12, 5))
                nearly[:, 5] .= nearly[:, 1]
                nearly[1, 5] += sqrt(eps(T)) / T(4)
                Fnear = rrqr!(copy(nearly))
                @test numerical_rank(Fnear) == 5
                @test numerical_rank(Fnear; rtol=sqrt(eps(T))) == 4

                # Equal exact norms retain the smallest original-column tie
                # rule even though noncompetitive norms use downdated state.
                equal_norm = zeros(T, 8, 4)
                for column in 1:4
                    equal_norm[column, column] = one(T)
                end
                Fequal = rrqr!(copy(equal_norm))
                @test factor_permutation(Fequal) == collect(1:4)

                # The blocked DLAQPS path must preserve a column that leaves
                # and later re-enters a panel. This deterministic matrix hits
                # that history and has numerical rank 31 at rtol=1e-12.
                blocked_source = Matrix{T}(undef, 500, 40)
                for column in axes(blocked_source, 2),
                    row in axes(blocked_source, 1)
                    blocked_source[row, column] = T(
                        sin(row * 0.01 + column * 0.7) +
                        0.1 * cos(row * 0.003 * column),
                    )
                end
                Fblocked = rrqr!(copy(blocked_source); threads=2)
                blocked_workspace = MFWorkspace(T; thread_count=2)
                Fblocked_workspace = rrqr!(
                    copy(blocked_source);
                    threads=2,
                    workspace=blocked_workspace,
                )
                @test factor_matrix(Fblocked_workspace) ==
                    factor_matrix(Fblocked)
                @test factor_permutation(Fblocked_workspace) ==
                    factor_permutation(Fblocked)
                @test numerical_rank(Fblocked; rtol=T(1e-12)) == 31
                @test sort(factor_permutation(Fblocked)) == collect(1:40)

                transformed = copy(
                    blocked_source[:, factor_permutation(Fblocked)],
                )
                apply_q!(transformed, Fblocked; trans=:T)
                transformed_leading = transformed[1:40, :]
                @test maximum(abs, tril(transformed_leading, -1)) <=
                    tolerance(T, 512 * size(blocked_source, 1))
                @test max_relative_error(
                    triu(transformed_leading),
                    triu(factor_matrix(Fblocked)[1:40, :]),
                ) <= tolerance(T, 512 * size(blocked_source, 1))
                Rblocked = qr_explicit_r(Fblocked)
                permuted_source =
                    blocked_source[:, factor_permutation(Fblocked)]
                @test max_relative_error(
                    transpose(Rblocked) * Rblocked,
                    transpose(permuted_source) * permuted_source,
                ) <= tolerance(T, 512 * size(blocked_source, 1))

                # Scaled sum-of-squares pivot norms avoid squaring these
                # extreme column scales directly.
                scaled = T.(randn(10, 4))
                scales = T.((1e-100, 1e100, 1e-40, 1e40))
                for column in axes(scaled, 2)
                    scaled[:, column] .*= scales[column]
                end
                Fscaled = rrqr!(copy(scaled))
                @test MultiFloatLinearAlgebra.issuccess(Fscaled)
                @test factor_permutation(Fscaled)[1] == 2
                @test numerical_rank(Fscaled) == 4
                Qscaled = Matrix{T}(I, size(scaled, 1), size(scaled, 1))
                apply_q!(Qscaled, Fscaled)
                @test max_relative_error(
                    Qscaled * qr_explicit_r(Fscaled),
                    scaled[:, factor_permutation(Fscaled)],
                ) <= tolerance(T, 256 * size(scaled, 1))

                # Solve both leading R and R' for vector and matrix RHS.
                solve_source = T.(randn(14, 6))
                Fsolve = rrqr!(copy(solve_source))
                rank = 6
                Rleading = qr_explicit_r(Fsolve)[1:rank, 1:rank]
                expected_vector = T.(randn(rank))
                rhs_vector = Rleading * expected_vector
                solve_r!(rhs_vector, Fsolve, rank; trans=:N, config=config)
                @test max_relative_error(rhs_vector, expected_vector) <=
                    tolerance(T, 32rank)

                expected_matrix = T.(randn(rank, 3))
                rhs_matrix = transpose(Rleading) * expected_matrix
                solve_r!(rhs_matrix, Fsolve, rank; trans=:T, config=config)
                @test max_relative_error(rhs_matrix, expected_matrix) <=
                    tolerance(T, 32rank)

                @test_throws ArgumentError numerical_rank(Fsolve; atol=-1)
                @test_throws ArgumentError numerical_rank(Fsolve; rtol=Inf)
                @test_throws ArgumentError apply_q!(
                    zeros(T, 14), Fsolve; trans=:bad,
                )
                @test_throws DimensionMismatch apply_q!(zeros(T, 13), Fsolve)
                @test_throws ArgumentError apply_q!(
                    view(factor_matrix(Fsolve), :, 1), Fsolve,
                )
                @test_throws ArgumentError solve_r!(
                    zeros(T, rank), Fsolve, rank + 1; config=config,
                )
                @test_throws DimensionMismatch solve_r!(
                    zeros(T, rank - 1), Fsolve, rank; config=config,
                )

                nonfinite = T.(randn(7, 4))
                nonfinite[2, 3] = T(NaN)
                @test_throws DomainError rrqr!(copy(nonfinite))
                Fbad = rrqr!(copy(nonfinite); check=false)
                @test !MultiFloatLinearAlgebra.issuccess(Fbad)
                @test factor_status(Fbad) == -1
            end

            @testset "blocked RRQR adversarial stress" begin
                # Keep this panel just above the blocked crossover.  The
                # matrices are deterministic and small enough for CI while
                # still exercising several delayed panels.
                rows, columns = 256, 64
                thread_four = min(4, Threads.nthreads())
                workspace = MFWorkspace(
                    T;
                    factor_capacity=columns,
                    thread_count=thread_four,
                )
                for kind in (:near_tie, :cancellation, :rank_deficient, :extreme_scale)
                    source = blocked_rrqr_adversarial_matrix(
                        T, kind; rows=rows, columns=columns,
                    )
                    blocked_serial = rrqr!(copy(source); threads=1)
                    blocked_threaded = rrqr!(
                        copy(source); threads=thread_four,
                    )
                    blocked_reused = rrqr!(
                        copy(source);
                        threads=thread_four,
                        workspace=workspace,
                    )
                    reference_factors, reference_tau, reference_permutation =
                        qr_unblocked_reference(source)

                    # Threading and caller-owned workspace must not alter the
                    # compact factor, permutation, or Householder scalars.
                    @test factor_matrix(blocked_threaded) ==
                        factor_matrix(blocked_serial)
                    @test blocked_threaded.tau == blocked_serial.tau
                    @test factor_permutation(blocked_threaded) ==
                        factor_permutation(blocked_serial)
                    @test factor_matrix(blocked_reused) ==
                        factor_matrix(blocked_threaded)
                    @test blocked_reused.tau == blocked_threaded.tau
                    @test factor_permutation(blocked_reused) ==
                        factor_permutation(blocked_threaded)
                    @test sort(factor_permutation(blocked_threaded)) ==
                        collect(1:columns)
                    @test sort(reference_permutation) == collect(1:columns)

                    rank_rtol = kind === :rank_deficient ? sqrt(eps(T)) : zero(T)
                    blocked_rank = numerical_rank(
                        blocked_threaded; rtol=rank_rtol,
                    )
                    reference_rank = qr_reference_rank(
                        reference_factors; rtol=rank_rtol,
                    )
                    @test blocked_rank == reference_rank
                    if kind === :rank_deficient
                        @test blocked_rank == 30
                    end

                    # Q' A[:,p] must be triangular and agree with the
                    # compact R storage.  Scaling by the largest source
                    # entry keeps the extreme-scale case meaningful.
                    permuted = source[:, factor_permutation(blocked_threaded)]
                    apply_q!(permuted, blocked_threaded; trans=:T)
                    leading = permuted[1:min(rows, columns), :]
                    source_scale = max(one(T), maximum(abs, source))
                    residual_scale = T(8192 * max(rows, columns)) * eps(T)
                    @test maximum(abs, tril(leading, -1)) / source_scale <=
                        residual_scale
                    compact_r = qr_explicit_r(blocked_threaded)
                    @test max_relative_error(
                        triu(leading),
                        triu(compact_r[1:min(rows, columns), :]),
                    ) <= tolerance(T, 8192 * max(rows, columns))

                    reference_permuted = source[:, reference_permutation]
                    qr_reference_apply_qt!(
                        reference_permuted, reference_factors, reference_tau,
                    )
                    @test maximum(
                        abs, tril(reference_permuted[1:min(rows, columns), :], -1),
                    ) / source_scale <= residual_scale

                    # The Gram matrix is a permutation-invariant check on the
                    # reconstructed R, catching silent delayed-update errors
                    # even when the pivot order differs at a near tie.
                    @test max_relative_error(
                        transpose(compact_r) * compact_r,
                        transpose(source[:, factor_permutation(blocked_threaded)]) *
                        source[:, factor_permutation(blocked_threaded)],
                    ) <= tolerance(T, 8192 * max(rows, columns))
                end
            end

            @testset "cholesky!" begin
                R = randn(n, n)
                A64 = R * transpose(R) + n * I
                Aspd = T.(Matrix(A64))
                original = copy(Aspd)
                F = MultiFloatLinearAlgebra.cholesky!(Aspd; config=config)
                @test MultiFloatLinearAlgebra.issuccess(F)
                L = Matrix(LowerTriangular(factor_matrix(F)))
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
                @test any(==(UInt8(2)), factor_diagnostics(F).blocks)

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
                @test any(==(UInt8(1)), factor_diagnostics(F).blocks)
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
                @test count(
                    ==(UInt8(2)), factor_diagnostics(crossing_factor).blocks,
                ) == crossing_n ÷ 2
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
                @test factor_status(Flu) == -1

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
                @test factor_status(Fc) == -1

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
                @test factor_status(Fld) == -1
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
                @test factor_status(Flu) >= 1
                @test_throws LinearAlgebra.SingularException MultiFloatLinearAlgebra.lu!(
                    copy(Ssing); check=true, config=config,
                )

                Aindef = T.([1 2; 2 1])
                Fc = MultiFloatLinearAlgebra.cholesky!(copy(Aindef); check=false, config=config)
                @test !MultiFloatLinearAlgebra.issuccess(Fc)
                @test factor_status(Fc) >= 1
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
                xout = fill(T(17), n)
                @test MultiFloatLinearAlgebra.ldiv!(
                    xout, Fc, b; config=config,
                ) === xout
                @test xout == xv
                Bout = hcat(b, T(2) .* b)
                Xout = similar(Bout)
                MultiFloatLinearAlgebra.ldiv!(Xout, Fc, Bout; config=config)
                @test Xout == solve(Fc, Bout; config=config)

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
                yout = similar(b)
                MultiFloatLinearAlgebra.ldiv!(yout, Flu, b; config=config)
                @test yout == yv

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
                zout = similar(b)
                MultiFloatLinearAlgebra.ldiv!(zout, Fld, b; config=config)
                @test zout == zv

                failed = MultiFloatLinearAlgebra.lu!(
                    zeros(T, n, n); check=false, config=config,
                )
                untouched = fill(T(23), n)
                source = copy(b)
                @test_throws LinearAlgebra.SingularException MultiFloatLinearAlgebra.ldiv!(
                    untouched, failed, source; config=config,
                )
                @test untouched == fill(T(23), n)
                @test source == b

                factor_column = view(factor_matrix(Flu), :, 1)
                factor_before = copy(factor_matrix(Flu))
                @test_throws ArgumentError MultiFloatLinearAlgebra.ldiv!(
                    factor_column, Flu, b; config=config,
                )
                @test factor_matrix(Flu) == factor_before
            end

            @testset "factor public protocol" begin
                R = randn(n, n)
                S = R * transpose(R) + n * I
                Fc = MultiFloatLinearAlgebra.cholesky!(T.(Matrix(S)); config=config)
                @test Fc isa MultiFloatLinearAlgebra.AbstractMFFactorization
                @test MultiFloatLinearAlgebra.factor_kind(Fc) === :cholesky
                @test MultiFloatLinearAlgebra.factor_status(Fc) == 0
                @test MultiFloatLinearAlgebra.factor_matrix(Fc) isa AbstractMatrix{T}
                @test factor_state(Fc) === :success
                @test factor_precision(Fc) === T
                @test factor_provider(Fc) === :mfla
                @test size(Fc) == (n, n)
                @test size(Fc, 1) == n
                @test size(Fc, 2) == n
                @test eltype(Fc) === T

                Alu = T.(randn(n, n))
                for i in 1:n
                    Alu[i, i] += T(4)
                end
                Flu = MultiFloatLinearAlgebra.lu!(Alu; config=config)
                @test MultiFloatLinearAlgebra.factor_kind(Flu) === :lu
                @test factor_state(Flu) === :success
                @test factor_precision(Flu) === T
                @test factor_provider(Flu) === :mfla
                @test size(Flu) == (n, n)

                Rind = randn(n, n)
                Sind = Rind + transpose(Rind)
                for i in 1:n
                    Sind[i, i] += isodd(i) ? 5 : -5
                end
                Fld = MultiFloatLinearAlgebra.ldlt!(T.(Matrix(Sind)); config=config)
                @test MultiFloatLinearAlgebra.factor_kind(Fld) === :ldlt
                @test MultiFloatLinearAlgebra.factor_status(Fld) == 0
                @test factor_state(Fld) === :success
                @test factor_precision(Fld) === T
                @test factor_provider(Fld) === :mfla
                @test eltype(Fld) === T

                Fqr = rrqr!(T.(randn(n + 2, n)))
                @test factor_kind(Fqr) === :qr
                @test factor_state(Fqr) === :success
                @test factor_precision(Fqr) === T
                @test factor_provider(Fqr) === :mfla

                qr_matrix = T.(randn(n, n))
                for index in 1:n
                    qr_matrix[index, index] += T(4)
                end
                Fqr_square = rrqr!(copy(qr_matrix))
                qr_truth = T.(randn(n))
                qr_rhs = qr_matrix * qr_truth
                qr_solution = similar(qr_rhs)
                MultiFloatLinearAlgebra.ldiv!(
                    qr_solution, Fqr_square, qr_rhs; config=config,
                )
                @test max_relative_error(qr_solution, qr_truth) <=
                    tolerance(T, 128n)

                qr_inplace = copy(qr_rhs)
                MultiFloatLinearAlgebra.ldiv!(
                    qr_inplace, Fqr_square; config=config,
                )
                @test max_relative_error(qr_inplace, qr_truth) <=
                    tolerance(T, 128n)

                qr_truth_matrix = T.(randn(n, 3))
                qr_rhs_matrix = qr_matrix * qr_truth_matrix
                qr_solution_matrix = similar(qr_rhs_matrix)
                MultiFloatLinearAlgebra.ldiv!(
                    qr_solution_matrix,
                    Fqr_square,
                    qr_rhs_matrix;
                    config=config,
                )
                @test max_relative_error(
                    qr_solution_matrix, qr_truth_matrix,
                ) <= tolerance(T, 128n)

                qr_inplace_matrix = copy(qr_rhs_matrix)
                MultiFloatLinearAlgebra.ldiv!(
                    qr_inplace_matrix, Fqr_square; config=config,
                )
                @test max_relative_error(
                    qr_inplace_matrix, qr_truth_matrix,
                ) <= tolerance(T, 128n)

                qr_factor_source = view(factor_matrix(Fqr_square), :, 1)
                qr_factor_source_expected = solve(
                    Fqr_square, copy(qr_factor_source); config=config,
                )
                qr_factor_source_solution = similar(qr_factor_source)
                MultiFloatLinearAlgebra.ldiv!(
                    qr_factor_source_solution,
                    Fqr_square,
                    qr_factor_source;
                    config=config,
                )
                @test qr_factor_source_solution == qr_factor_source_expected

                qr_factor_before = copy(factor_matrix(Fqr_square))
                @test_throws ArgumentError MultiFloatLinearAlgebra.ldiv!(
                    view(factor_matrix(Fqr_square), :, 1),
                    Fqr_square,
                    qr_rhs;
                    config=config,
                )
                @test factor_matrix(Fqr_square) == qr_factor_before

                qr_overlap = T.(randn(n + 1))
                qr_overlap_before = copy(qr_overlap)
                @test_throws ArgumentError MultiFloatLinearAlgebra.ldiv!(
                    view(qr_overlap, 2:(n + 1)),
                    Fqr_square,
                    view(qr_overlap, 1:n);
                    config=config,
                )
                @test qr_overlap == qr_overlap_before
                @test_throws DimensionMismatch MultiFloatLinearAlgebra.ldiv!(
                    zeros(T, n + 2), Fqr, zeros(T, n + 2); config=config,
                )

                cycle_permutation = [2, 3, 1, 4, 6, 5]
                cycle_matrix = zeros(T, 6, 6)
                for (rank, column) in enumerate(cycle_permutation)
                    cycle_matrix[column, column] = T(7 - rank)
                end
                Fqr_cycles = rrqr!(copy(cycle_matrix))
                @test factor_permutation(Fqr_cycles) == cycle_permutation
                cycle_truth = T.(1:6)
                cycle_rhs = cycle_matrix * cycle_truth
                MultiFloatLinearAlgebra.ldiv!(
                    cycle_rhs, Fqr_cycles; config=config,
                )
                @test max_relative_error(cycle_rhs, cycle_truth) <=
                    tolerance(T, 128)

                qr_rank_deficient = copy(qr_matrix)
                qr_rank_deficient[:, end] .= zero(T)
                Fqr_rank_deficient = rrqr!(qr_rank_deficient)
                qr_untouched = fill(T(29), n)
                @test_throws LinearAlgebra.SingularException MultiFloatLinearAlgebra.ldiv!(
                    qr_untouched, Fqr_rank_deficient, qr_rhs; config=config,
                )
                @test qr_untouched == fill(T(29), n)

                qr_nonfinite = fill(T(NaN), n, n)
                Fqr_nonfinite = rrqr!(qr_nonfinite; check=false)
                @test factor_state(Fqr_nonfinite) === :nonfinite_input
                qr_failed_untouched = fill(T(31), n)
                @test_throws ArgumentError MultiFloatLinearAlgebra.ldiv!(
                    qr_failed_untouched, Fqr_nonfinite, qr_rhs; config=config,
                )
                @test qr_failed_untouched == fill(T(31), n)
            end

            @testset "factor diagnostics" begin
                # Cholesky reports accepted diagonal range and an explicit
                # failure location without making any fallback decision.
                R = randn(n, n)
                spd = T.(Matrix(R * transpose(R) + n * I))
                Fc = MultiFloatLinearAlgebra.cholesky!(copy(spd); config=config)
                dc = factor_diagnostics(Fc)
                @test dc.kind === :cholesky
                @test dc.success
                @test dc.status == 0
                @test dc.state === :success
                @test dc.precision === T
                @test dc.provider === :mfla
                @test dc.failure_location === nothing
                @test dc.accepted_pivots == n
                @test dc.minimum_diagonal > zero(T)
                @test dc.maximum_diagonal >= dc.minimum_diagonal
                @test dc.diagonal_spread ==
                    dc.maximum_diagonal / dc.minimum_diagonal
                @test dc.finite

                Fc_fail = MultiFloatLinearAlgebra.cholesky!(
                    T.([1 2; 2 1]); check=false, config=config,
                )
                dc_fail = factor_diagnostics(Fc_fail)
                @test !dc_fail.success
                @test dc_fail.state === :not_posdef
                @test dc_fail.failure_location == 2
                @test dc_fail.accepted_pivots == 1

                # LU growth is relative to the scalar input maximum recorded
                # before overwrite. Returned pivots are caller-owned copies.
                lu_input = T.(randn(n, n))
                for i in 1:n
                    lu_input[i, i] += T(4)
                end
                Flu = MultiFloatLinearAlgebra.lu!(copy(lu_input); config=config)
                du = factor_diagnostics(Flu)
                @test du.kind === :lu
                @test du.success
                @test du.original_maximum == maximum(abs, lu_input)
                @test du.maximum_u >= du.minimum_pivot > zero(T)
                @test du.pivot_growth == du.maximum_u / du.original_maximum
                @test du.finite
                du.pivots[1] = 0
                @test factor_diagnostics(Flu).pivots[1] != 0

                singular_lu = zeros(T, 4, 4)
                singular_lu[1, 1] = one(T)
                Flu_fail = MultiFloatLinearAlgebra.lu!(
                    singular_lu; check=false, config=config,
                )
                du_fail = factor_diagnostics(Flu_fail)
                @test !du_fail.success
                @test du_fail.state === :singular
                @test du_fail.failure_location == 2
                @test du_fail.accepted_pivots == 1

                # Pure off-diagonal 2x2 blocks have one positive and one
                # negative eigenvalue each, yielding exact known inertia.
                order = 8
                indefinite = zeros(T, order, order)
                for k in 1:2:order
                    indefinite[k, k + 1] = T(k)
                    indefinite[k + 1, k] = T(k)
                end
                Fld = MultiFloatLinearAlgebra.ldlt!(
                    copy(indefinite); config=config,
                )
                dd = factor_diagnostics(Fld)
                @test dd.kind === :ldlt
                @test dd.success
                @test dd.one_by_one_pivots == 0
                @test dd.two_by_two_pivots == order ÷ 2
                @test dd.inertia == (positive=order ÷ 2, negative=order ÷ 2, zero=0)
                @test dd.minimum_block_eigenvalue_magnitude == one(T)
                @test dd.minimum_scaled_block == one(T) / T(order - 1)
                @test abs(dd.block_growth - one(T)) <= T(8) * eps(T)
                @test dd.finite
                @test factor_inertia(Fld) == dd.inertia
                @test factor_permutation(Fld) == collect(1:order)
                @test factor_blocks(Fld) == dd.blocks
                @test factor_pivots(Fld) == dd.pivots
                copied_pivots = factor_pivots(Fld)
                copied_blocks = factor_blocks(Fld)
                copied_permutation = factor_permutation(Fld)
                copied_pivots[1] = 0
                copied_blocks[1] = 0
                copied_permutation[1] = 0
                @test factor_pivots(Fld)[1] != 0
                @test factor_blocks(Fld)[1] == UInt8(2)
                @test factor_permutation(Fld)[1] == 1
                dd.pivots[1] = 0
                dd.blocks[1] = 0
                @test factor_diagnostics(Fld).pivots[1] != 0
                @test factor_diagnostics(Fld).blocks[1] == UInt8(2)

                swapped = T[1 10; 10 101]
                Fswapped = MultiFloatLinearAlgebra.ldlt!(
                    copy(swapped); config=config,
                )
                @test factor_permutation(Fswapped) == [2, 1]
                @test factor_inertia(Fswapped) ==
                    factor_diagnostics(Fswapped).inertia

                # QR rank is reported only at the explicit threshold supplied
                # to factor_diagnostics; the stored factor remains unchanged.
                qr_source = T.(randn(11, 5))
                qr_source[:, 5] .= qr_source[:, 1]
                Fqr = rrqr!(copy(qr_source))
                dq_exact = factor_diagnostics(Fqr)
                dq_threshold = factor_diagnostics(Fqr; rtol=sqrt(eps(T)))
                @test dq_exact.kind === :qr
                @test dq_exact.success
                @test dq_exact.state === :success
                @test dq_threshold.rank_at_threshold == 4
                @test dq_threshold.rtol == sqrt(eps(T))
                @test dq_threshold.rdiag == factor_rdiag(Fqr)
                dq_threshold.permutation[1] = 0
                dq_threshold.rdiag[1] = zero(T)
                @test factor_permutation(Fqr)[1] != 0
                @test factor_rdiag(Fqr)[1] != zero(T)
            end

            @testset "residual and backward error" begin
                rows, columns = 9, 7
                Ares = T.(randn(rows, columns))
                xres = T.(randn(columns))
                bres = T.(randn(rows))
                expected = similar(bres)
                for row in 1:rows
                    accumulator = zero(T)
                    for column in 1:columns
                        accumulator += Ares[row, column] * xres[column]
                    end
                    expected[row] = -one(T) * accumulator + bres[row]
                end

                r = fill(T(NaN), rows)
                residual!(r, Ares, xres, bres; config=config)
                @test r == expected

                in_place = copy(bres)
                residual!(in_place, Ares, xres, in_place; config=config)
                @test in_place == expected

                Xres = T.(randn(columns, 3))
                Bres = T.(randn(rows, 3))
                Rres = fill(T(NaN), rows, 3)
                residual!(
                    Rres,
                    Ares,
                    Xres,
                    Bres;
                    config=KernelConfig(thread_count=2, gemm_strategy=:direct),
                )
                matrix_reference = Bres - Ares * Xres
                @test max_relative_error(Rres, matrix_reference) <=
                    tolerance(T, 8columns)

                # Symmetric residuals ignore a corrupted inactive triangle.
                order = 9
                raw = T.(randn(order, order))
                symmetric = T.(Matrix(raw + transpose(raw)))
                symmetric_clean = copy(symmetric)
                for row in 1:order, column in (row + 1):order
                    symmetric[row, column] = T(NaN)
                end
                xsym = T.(randn(order))
                bsym = T.(randn(order))
                rsym = similar(bsym)
                residual!(
                    rsym,
                    symmetric,
                    xsym,
                    bsym;
                    uplo=:lower,
                    config=config,
                )
                @test max_relative_error(
                    rsym, bsym - symmetric_clean * xsym,
                ) <= tolerance(T, 8order)

                # Backward error follows the documented infinity-norm formula.
                simple_A = T.([2 0; 0 -3])
                simple_x = T.([1, -2])
                simple_b = T.([3, 5])
                simple_r = similar(simple_b)
                residual!(simple_r, simple_A, simple_x, simple_b)
                simple_error = normwise_backward_error(
                    simple_A, simple_x, simple_b, simple_r,
                )
                setprecision(BigFloat, 512) do
                    expected_error = BigFloat(maximum(abs, simple_r)) /
                        (BigFloat(T(3)) * BigFloat(T(2)) + BigFloat(T(5)))
                    @test abs(BigFloat(simple_error) - expected_error) <=
                        BigFloat(64) * BigFloat(eps(T)) * abs(expected_error)
                end

                matrix_errors = normwise_backward_error(
                    Ares, Xres, Bres, Rres,
                )
                @test length(matrix_errors) == size(Xres, 2)
                for column in axes(Xres, 2)
                    @test matrix_errors[column] == normwise_backward_error(
                        Ares,
                        view(Xres, :, column),
                        view(Bres, :, column),
                        view(Rres, :, column),
                    )
                end

                # Inactive NaNs do not poison the norm, but authoritative
                # nonfinite values produce an explicit NaN fact.
                symmetric_error = normwise_backward_error(
                    symmetric, xsym, bsym, rsym; uplo=:lower,
                )
                @test isfinite(symmetric_error)
                bad_symmetric = copy(symmetric)
                bad_symmetric[2, 1] = T(Inf)
                @test isnan(normwise_backward_error(
                    bad_symmetric, xsym, bsym, rsym; uplo=:lower,
                ))

                zero_matrix = zeros(T, 3, 3)
                zero_vector = zeros(T, 3)
                @test iszero(normwise_backward_error(
                    zero_matrix, zero_vector, zero_vector, zero_vector,
                ))
                @test isinf(normwise_backward_error(
                    zero_matrix, zero_vector, zero_vector, ones(T, 3),
                ))

                nonzero_matrix = T.([2 0 0; 0 -3 0; 0 0 4])
                @test iszero(normwise_backward_error(
                    nonzero_matrix, zero_vector, zero_vector, zero_vector,
                ))
                @test isinf(normwise_backward_error(
                    nonzero_matrix, zero_vector, zero_vector, ones(T, 3),
                ))
                zero_solutions = zeros(T, 3, 2)
                zero_rhs = zeros(T, 3, 2)
                mixed_residuals = zeros(T, 3, 2)
                mixed_residuals[2, 2] = one(T)
                zero_denominator_errors = normwise_backward_error(
                    nonzero_matrix,
                    zero_solutions,
                    zero_rhs,
                    mixed_residuals,
                )
                @test iszero(zero_denominator_errors[1])
                @test isinf(zero_denominator_errors[2])

                # The exact denominator exceeds Float64 exponent range, but
                # the scaled backward error is finite and representable.
                huge_A = fill(T(1e200), 1, 1)
                huge_x = fill(T(1e200), 1)
                huge_b = zeros(T, 1)
                huge_r = fill(T(1e300), 1)
                huge_error = normwise_backward_error(
                    huge_A, huge_x, huge_b, huge_r,
                )
                @test isfinite(huge_error)
                setprecision(BigFloat, 512) do
                    expected_huge = BigFloat(huge_r[1]) /
                        (BigFloat(huge_A[1]) * BigFloat(huge_x[1]))
                    @test abs(BigFloat(huge_error) - expected_huge) <=
                        BigFloat(64) * BigFloat(eps(T)) * abs(expected_huge)
                end

                @test_throws ArgumentError residual!(
                    zeros(T, rows), Ares, xres, bres; uplo=:bad,
                )
                @test_throws DimensionMismatch residual!(
                    zeros(T, rows - 1), Ares, xres, bres,
                )
                square_alias = T.(randn(5, 5))
                alias_x = T.(randn(5))
                @test_throws ArgumentError residual!(
                    alias_x, square_alias, alias_x, T.(randn(5)),
                )
                overlap = zeros(T, rows + 1)
                @test_throws ArgumentError residual!(
                    view(overlap, 1:rows),
                    Ares,
                    xres,
                    view(overlap, 2:(rows + 1)),
                )
            end

            @testset "refinement correction" begin
                cases = (
                    (
                        T.([4 1; 1 3]),
                        A -> MultiFloatLinearAlgebra.cholesky!(A; config=config),
                    ),
                    (
                        T.([4 1; 2 3]),
                        A -> MultiFloatLinearAlgebra.lu!(A; config=config),
                    ),
                    (
                        T.([0 2; 2 1]),
                        A -> MultiFloatLinearAlgebra.ldlt!(A; config=config),
                    ),
                )
                for (system, factorize) in cases
                    factor = factorize(copy(system))
                    truth = T.([1, -2])
                    right_hand_side = system * truth
                    approximation = zeros(T, 2)
                    residual_before = similar(right_hand_side)
                    residual!(
                        residual_before,
                        system,
                        approximation,
                        right_hand_side,
                    )
                    error_before = normwise_backward_error(
                        system,
                        approximation,
                        right_hand_side,
                        residual_before,
                    )
                    correction = similar(residual_before)
                    refinement_correction!(
                        correction, factor, residual_before; config=config,
                    )
                    @test correction == MultiFloatLinearAlgebra.solve(
                        factor, residual_before; config=config,
                    )
                    approximation .+= correction
                    residual_after = similar(residual_before)
                    residual!(
                        residual_after,
                        system,
                        approximation,
                        right_hand_side,
                    )
                    error_after = normwise_backward_error(
                        system,
                        approximation,
                        right_hand_side,
                        residual_after,
                    )
                    @test error_after < error_before

                    in_place = copy(residual_before)
                    refinement_correction!(
                        in_place, factor, in_place; config=config,
                    )
                    @test in_place == correction
                end

                matrix_system = T.([5 1; 1 4])
                factor = MultiFloatLinearAlgebra.cholesky!(
                    copy(matrix_system); config=config,
                )
                matrix_residual = T.(randn(2, 3))
                correction = similar(matrix_residual)
                refinement_correction!(
                    correction, factor, matrix_residual; config=config,
                )
                @test correction == MultiFloatLinearAlgebra.solve(
                    factor, matrix_residual; config=config,
                )
                @test_throws DimensionMismatch refinement_correction!(
                    zeros(T, 1, 3), factor, matrix_residual; config=config,
                )

                failed_factor = MultiFloatLinearAlgebra.lu!(
                    zeros(T, 2, 2); check=false, config=config,
                )
                untouched = fill(T(7), 2)
                @test_throws LinearAlgebra.SingularException refinement_correction!(
                    untouched, failed_factor, ones(T, 2); config=config,
                )
                @test untouched == fill(T(7), 2)

                aliased = view(factor_matrix(factor), :, 1)
                @test_throws ArgumentError refinement_correction!(
                    aliased, factor, ones(T, 2); config=config,
                )
            end
        end
    end

    @testset "explicit mixed-MultiFloat residual" begin
        pairs = (
            (Float64x2, Float64x3),
            (Float64x2, Float64x4),
            (Float64x3, Float64x4),
        )
        for (Source, Residual) in pairs
            @testset "$Source -> $Residual" begin
                q = 55
                perturbation = Source(ldexp(1.0, -q))
                t = one(Source) + perturbation
                if Source === Float64x3
                    t += Source(ldexp(1.0, -2q))
                end
                A = reshape(Source[t], 1, 1)
                x = Source[t]
                b = Source[t * t]

                low = zeros(Source, 1)
                residual!(low, A, x, b)
                @test iszero(low[1])

                high = zeros(Residual, 1)
                residual_mixed!(high, A, x, b)
                @test !iszero(high[1])
                setprecision(BigFloat, 1024) do
                    expected = BigFloat(b[1]) - BigFloat(A[1]) * BigFloat(x[1])
                    scale = abs(BigFloat(b[1])) +
                        abs(BigFloat(A[1])) * abs(BigFloat(x[1]))
                    @test abs(BigFloat(high[1]) - expected) <=
                        BigFloat(64) * BigFloat(eps(Residual)) * scale
                end

                # General vector and matrix paths match explicit conversion
                # and ascending high-precision accumulation.
                rows, columns, right_hand_sides = 53, 17, 3
                general = Source.(randn(rows, columns))
                solution = Source.(randn(columns))
                rhs = Source.(randn(rows))
                expected_vector = Vector{Residual}(undef, rows)
                for row in 1:rows
                    accumulator = Residual(rhs[row])
                    for column in 1:columns
                        accumulator -= Residual(general[row, column]) *
                            Residual(solution[column])
                    end
                    expected_vector[row] = accumulator
                end
                serial = zeros(Residual, rows)
                threaded = zeros(Residual, rows)
                residual_mixed!(
                    serial,
                    general,
                    solution,
                    rhs;
                    config=KernelConfig(thread_count=1),
                )
                residual_mixed!(
                    threaded,
                    general,
                    solution,
                    rhs;
                    config=KernelConfig(thread_count=2),
                )
                @test serial == expected_vector
                @test threaded == serial
                for index in eachindex(serial)
                    @test all(
                        reinterpret(UInt64, serial[index]._limbs[limb]) ==
                        reinterpret(UInt64, threaded[index]._limbs[limb])
                        for limb in 1:length(serial[index]._limbs)
                    )
                end

                solutions = Source.(randn(columns, right_hand_sides))
                right_sides = Source.(randn(rows, right_hand_sides))
                expected_matrix = Matrix{Residual}(undef, rows, right_hand_sides)
                for rhs_column in 1:right_hand_sides, row in 1:rows
                    accumulator = Residual(right_sides[row, rhs_column])
                    for column in 1:columns
                        accumulator -= Residual(general[row, column]) *
                            Residual(solutions[column, rhs_column])
                    end
                    expected_matrix[row, rhs_column] = accumulator
                end
                matrix_output = zeros(Residual, rows, right_hand_sides)
                residual_mixed!(
                    matrix_output,
                    general,
                    solutions,
                    right_sides;
                    config=KernelConfig(thread_count=2),
                )
                @test matrix_output == expected_matrix

                # Only the selected symmetric triangle is converted/read.
                order = 51
                raw = Source.(randn(order, order))
                symmetric = Source.(Matrix(raw + transpose(raw)))
                clean = copy(symmetric)
                for row in 1:order, column in (row + 1):order
                    symmetric[row, column] = Source(NaN)
                end
                symmetric_x = Source.(randn(order))
                symmetric_b = Source.(randn(order))
                symmetric_r = zeros(Residual, order)
                residual_mixed!(
                    symmetric_r,
                    symmetric,
                    symmetric_x,
                    symmetric_b;
                    uplo=:lower,
                    config=KernelConfig(thread_count=2),
                )
                explicit_symmetric = Vector{Residual}(undef, order)
                for row in 1:order
                    accumulator = Residual(symmetric_b[row])
                    for column in 1:order
                        accumulator -= Residual(clean[row, column]) *
                            Residual(symmetric_x[column])
                    end
                    explicit_symmetric[row] = accumulator
                end
                @test symmetric_r == explicit_symmetric
                @test all(isfinite, symmetric_r)
            end
        end

        A2 = Float64x2.(randn(4, 4))
        x2 = Float64x2.(randn(4))
        b2 = Float64x2.(randn(4))
        @test_throws ArgumentError residual_mixed!(
            zeros(Float64x2, 4), A2, x2, b2,
        )

        A3 = Float64x3.(A2)
        x3 = Float64x3.(x2)
        b3 = Float64x3.(b2)
        @test_throws ArgumentError residual_mixed!(
            zeros(Float64x2, 4), A3, x3, b3,
        )
        A4 = Float64x4.(A2)
        x4 = Float64x4.(x2)
        b4 = Float64x4.(b2)
        @test_throws ArgumentError residual_mixed!(
            zeros(MultiFloat{Float64,5}, 4), A4, x4, b4,
        )
        @test_throws ArgumentError residual_mixed!(
            zeros(Float64x4, 4), A2, x3, b2,
        )
        @test_throws DimensionMismatch residual_mixed!(
            zeros(Float64x4, 3), A2, x2, b2,
        )
        @test_throws ArgumentError residual_mixed!(
            zeros(Float64x4, 4), A2, x2, b2; uplo=:bad,
        )

        # An ill-conditioned x2 system hides the product of two 2^-60
        # perturbations in its ordinary residual. Evaluating that residual in
        # x3 exposes 2^-120; one explicitly requested x2 correction then
        # removes the amplified 2^-60 solution error.
        Source = Float64x2
        Residual = Float64x3
        perturbation = Source(ldexp(1.0, -60))
        system = Source[
            1 1
            1 1 + perturbation
        ]
        truth = Source[1, -1]
        right_hand_side = Source[0, -perturbation]
        approximation = Source[
            1 + perturbation,
            -1 - perturbation,
        ]
        hidden = zeros(Source, 2)
        residual!(hidden, system, approximation, right_hand_side)
        @test all(iszero, hidden)

        exposed = zeros(Residual, 2)
        residual_mixed!(
            exposed, system, approximation, right_hand_side;
            config=KernelConfig(thread_count=1),
        )
        @test iszero(exposed[1])
        @test exposed[2] == Residual(ldexp(1.0, -120))
        backward_before = normwise_backward_error(
            Residual.(system),
            Residual.(approximation),
            Residual.(right_hand_side),
            exposed,
        )

        factor = MultiFloatLinearAlgebra.lu!(
            copy(system); config=KernelConfig(thread_count=1),
        )
        correction = Source.(exposed)
        refinement_correction!(
            correction,
            factor,
            correction;
            config=KernelConfig(thread_count=1),
        )
        error_before = maximum(abs, BigFloat.(approximation) .- BigFloat.(truth))
        approximation .+= correction
        error_after = maximum(abs, BigFloat.(approximation) .- BigFloat.(truth))
        @test error_after < error_before
        @test approximation == truth

        corrected_residual = zeros(Residual, 2)
        residual_mixed!(
            corrected_residual,
            system,
            approximation,
            right_hand_side;
            config=KernelConfig(thread_count=1),
        )
        @test all(iszero, corrected_residual)
        backward_after = normwise_backward_error(
            Residual.(system),
            Residual.(approximation),
            Residual.(right_hand_side),
            corrected_residual,
        )
        @test backward_after < backward_before
    end

    @testset "caller-owned workspace" begin
        for T in (Float64x2, Float64x3, Float64x4)
            @testset "$T" begin
                n = 18
                workspace = MFWorkspace(
                    T;
                    factor_capacity=n,
                    ldlt_block_capacity=4,
                    thread_count=2,
                    gemm_capacity=n * 6,
                )
                initial_capacity = workspace_capacity(workspace)
                @test initial_capacity == (
                    factor=n,
                    ldlt_block=4,
                    gemm_workers=2,
                    gemm_elements_per_worker=n * 6,
                )

                @test_throws ArgumentError MFWorkspace(T; factor_capacity=-1)
                @test_throws ArgumentError ensure_workspace_capacity!(
                    workspace; gemm_workers=0,
                )
                ensure_workspace_capacity!(
                    workspace;
                    factor_capacity=n - 1,
                    ldlt_block_capacity=3,
                    gemm_workers=1,
                    gemm_capacity=n,
                )
                @test workspace_capacity(workspace) == initial_capacity

                # The general workspace forwards its packed buffers to GEMM
                # and multi-RHS residual evaluation.
                rows, reduction, columns = 13, 11, 9
                left = T.(randn(rows, reduction))
                right = T.(randn(reduction, columns))
                direct = zeros(T, rows, columns)
                packed = zeros(T, rows, columns)
                gemm!(
                    direct,
                    left,
                    right;
                    config=KernelConfig(
                        thread_count=1,
                        gemm_strategy=:direct,
                    ),
                )
                gemm!(
                    packed,
                    left,
                    right;
                    config=KernelConfig(
                        thread_count=2,
                        gemm_strategy=:packed,
                        gemm_panel_columns=6,
                        gemm_micro_columns=2,
                    ),
                    workspace=workspace,
                )
                @test packed == direct

                solutions = T.(randn(reduction, 3))
                right_hand_sides = T.(randn(rows, 3))
                ordinary_residual = similar(right_hand_sides)
                workspace_residual = similar(right_hand_sides)
                residual!(
                    ordinary_residual,
                    left,
                    solutions,
                    right_hand_sides;
                    config=KernelConfig(
                        thread_count=1,
                        gemm_strategy=:direct,
                    ),
                )
                residual!(
                    workspace_residual,
                    left,
                    solutions,
                    right_hand_sides;
                    config=KernelConfig(
                        thread_count=1,
                        gemm_strategy=:packed,
                        gemm_panel_columns=6,
                        gemm_micro_columns=2,
                    ),
                    workspace=workspace,
                )
                @test workspace_residual == ordinary_residual

                lu_source = T.(randn(n, n))
                for index in 1:n
                    lu_source[index, index] += T(5)
                end
                lu_config = KernelConfig(
                    thread_count=1,
                    lu_block=5,
                    gemm_strategy=:packed,
                    gemm_panel_columns=6,
                    gemm_micro_columns=2,
                )
                lu_owned = MultiFloatLinearAlgebra.lu!(
                    copy(lu_source); config=lu_config,
                )
                lu_borrowed = MultiFloatLinearAlgebra.lu!(
                    copy(lu_source); config=lu_config, workspace=workspace,
                )
                @test factor_matrix(lu_borrowed) == factor_matrix(lu_owned)
                @test factor_diagnostics(lu_borrowed).pivots ==
                    factor_diagnostics(lu_owned).pivots
                lu_rhs = T.(randn(n))
                @test solve(lu_borrowed, lu_rhs; config=lu_config) ==
                    solve(lu_owned, lu_rhs; config=lu_config)

                indefinite = zeros(T, n, n)
                for index in 1:2:n
                    indefinite[index, index + 1] = T(index)
                    indefinite[index + 1, index] = T(index)
                end
                for strategy in (:unblocked, :blocked)
                    ldlt_config = KernelConfig(
                        thread_count=1,
                        ldlt_strategy=strategy,
                        ldlt_block=4,
                    )
                    owned = MultiFloatLinearAlgebra.ldlt!(
                        copy(indefinite); config=ldlt_config,
                    )
                    borrowed = MultiFloatLinearAlgebra.ldlt!(
                        copy(indefinite);
                        config=ldlt_config,
                        workspace=workspace,
                    )
                    @test factor_matrix(borrowed) == factor_matrix(owned)
                    owned_diagnostics = factor_diagnostics(owned)
                    borrowed_diagnostics = factor_diagnostics(borrowed)
                    @test borrowed_diagnostics.pivots == owned_diagnostics.pivots
                    @test borrowed_diagnostics.blocks == owned_diagnostics.blocks
                    @test borrowed_diagnostics.inertia == owned_diagnostics.inertia
                    @test factor_pivots(borrowed) == factor_pivots(owned)
                    @test factor_blocks(borrowed) == factor_blocks(owned)
                    @test factor_permutation(borrowed) ==
                        factor_permutation(owned)
                    @test factor_inertia(borrowed) == factor_inertia(owned)
                    ldlt_rhs = T.(randn(n))
                    @test solve(borrowed, ldlt_rhs; config=ldlt_config) ==
                        solve(owned, ldlt_rhs; config=ldlt_config)
                end

                qr_source = T.(randn(n + 3, n - 2))
                qr_owned = rrqr!(copy(qr_source))
                qr_borrowed = rrqr!(copy(qr_source); workspace=workspace)
                @test factor_matrix(qr_borrowed) == factor_matrix(qr_owned)
                @test factor_permutation(qr_borrowed) ==
                    factor_permutation(qr_owned)
                @test factor_rdiag(qr_borrowed) == factor_rdiag(qr_owned)

                # Reusing a workspace for a smaller blocked panel must ignore
                # stale norm flags beyond the current column count.  The
                # workspace deliberately retains capacity after the large
                # factorization; force its unused tail dirty to exercise the
                # bound in _qr_has_unreliable_trailing_norm.
                shrink_large = Matrix{T}(undef, 500, 48)
                for column in axes(shrink_large, 2),
                    row in axes(shrink_large, 1)
                    shrink_large[row, column] = T(
                        sin(row * 0.013 + column * 0.41) +
                        0.07 * cos(row * 0.002 * column),
                    )
                end
                shrink_small = shrink_large[:, 1:40]
                shrink_reference = rrqr!(
                    copy(shrink_small); threads=1,
                )
                shrink_workspace = MFWorkspace(T; thread_count=4)
                rrqr!(
                    copy(shrink_large);
                    threads=4,
                    workspace=shrink_workspace,
                )
                @inbounds for column in 41:48
                    shrink_workspace.qr_norm_dirty[column] = true
                end
                @test MultiFloatLinearAlgebra._qr_has_unreliable_trailing_norm(
                    shrink_workspace.qr_norm_dirty, 41, 40,
                ) == false
                @test MultiFloatLinearAlgebra._qr_has_unreliable_trailing_norm(
                    shrink_workspace.qr_norm_dirty, 41,
                ) == true
                shrink_reused_1 = rrqr!(
                    copy(shrink_small);
                    threads=1,
                    workspace=shrink_workspace,
                )
                shrink_reused_4 = rrqr!(
                    copy(shrink_small);
                    threads=4,
                    workspace=shrink_workspace,
                )
                @test factor_matrix(shrink_reused_1) ==
                    factor_matrix(shrink_reference)
                @test factor_permutation(shrink_reused_1) ==
                    factor_permutation(shrink_reference)
                @test factor_matrix(shrink_reused_4) ==
                    factor_matrix(shrink_reused_1)
                @test factor_permutation(shrink_reused_4) ==
                    factor_permutation(shrink_reused_1)

                # Every returned factor owns its required metadata. Reusing
                # the same workspace leaves all live factors valid.
                live_qr = qr_borrowed
                live_lu = lu_borrowed
                replacement = MultiFloatLinearAlgebra.lu!(
                    copy(lu_source); config=lu_config, workspace=workspace,
                )
                @test MultiFloatLinearAlgebra.issuccess(replacement)
                @test factor_kind(live_qr) === :qr
                @test MultiFloatLinearAlgebra.issuccess(live_qr)
                @test factor_status(live_qr) == 0
                @test factor_permutation(live_qr) == factor_permutation(qr_owned)
                @test solve(live_lu, lu_rhs; config=lu_config) ==
                    solve(lu_owned, lu_rhs; config=lu_config)

                # GEMM and factor-capacity growth also leave live factors valid.
                ensure_workspace_capacity!(
                    workspace;
                    gemm_workers=2,
                    gemm_capacity=2n * 6,
                )
                @test MultiFloatLinearAlgebra.issuccess(replacement)
                ensure_workspace_capacity!(
                    workspace; factor_capacity=n + 5,
                )
                @test MultiFloatLinearAlgebra.issuccess(replacement)
                @test MultiFloatLinearAlgebra.issuccess(live_qr)
                @test MultiFloatLinearAlgebra.issuccess(live_lu)
                @test factor_diagnostics(live_qr).success
                @test solve(live_lu, lu_rhs; config=lu_config) ==
                    solve(lu_owned, lu_rhs; config=lu_config)
                @test workspace_capacity(workspace).factor == n + 5

                nonfinite = fill(T(NaN), 4, 4)
                failed = MultiFloatLinearAlgebra.lu!(
                    copy(nonfinite); check=false, workspace=workspace,
                )
                @test factor_status(failed) == -1
                @test_throws DomainError MultiFloatLinearAlgebra.lu!(
                    copy(nonfinite); check=true, workspace=workspace,
                )
                @test factor_status(failed) == -1
                @test factor_state(failed) === :nonfinite_input
                @test factor_diagnostics(failed).state === :nonfinite_input
            end
        end
    end

    @testset "machine-readable capabilities" begin
        expected_properties = (
            :provider,
            :scalar_type,
            :base_type,
            :supported,
            :limb_count,
            :dot,
            :gemv,
            :transpose_gemv,
            :symv,
            :gemm,
            :gemmt,
            :syrk,
            :syrk_packed,
            :trsv,
            :trsm,
            :trmm,
            :cholesky,
            :lu,
            :ldlt,
            :rrqr,
            :ldlt_lightweight_metadata,
            :factor_diagnostics,
            :apply_q,
            :solve_r,
            :vector_rhs,
            :multi_rhs,
            :residual,
            :mixed_precision_residual,
            :mixed_residual_targets,
            :mixed_residual_target_types,
            :refinement_correction,
            :reusable_workspace,
            :factor_metadata_ownership,
            :factor_matrix_ownership,
            :factorization_destructive,
            :factor_solve_mutates_factor,
            :shared_gemm_workspace_concurrency,
            :concurrent_factor_workspace,
            :syrk_authoritative_triangle,
            :syrk_inactive_triangle,
            :threading,
        )
        operation_properties = filter(
            property -> !(property in (
                :provider, :scalar_type, :base_type, :supported, :limb_count,
                :mixed_residual_targets, :mixed_residual_target_types,
                :factor_metadata_ownership, :factor_matrix_ownership,
                :factorization_destructive, :factor_solve_mutates_factor,
                :shared_gemm_workspace_concurrency,
                :concurrent_factor_workspace, :syrk_authoritative_triangle,
                :syrk_inactive_triangle,
            )),
            expected_properties,
        )

        for limbs in 1:4
            T = MultiFloat{Float64,limbs}
            first_query = capabilities(T)
            second_query = capabilities(T)
            @test first_query == second_query
            @test propertynames(first_query) == expected_properties
            @test first_query.provider === :mfla
            @test first_query.scalar_type === T
            @test first_query.base_type === Float64
            @test first_query.supported
            @test first_query.limb_count == limbs
            @test all(getproperty(first_query, property) isa Bool
                      for property in operation_properties)
            @test first_query.mixed_residual_targets == (
                x2=false,
                x3=limbs == 2,
                x4=limbs in (2, 3),
            )
            @test first_query.mixed_precision_residual == (limbs in (2, 3))
            expected_target_types = limbs == 2 ? (Float64x3, Float64x4) :
                limbs == 3 ? (Float64x4,) : ()
            @test first_query.mixed_residual_target_types == expected_target_types
            @test first_query.factor_metadata_ownership === :factor_owned
            @test first_query.factor_matrix_ownership === :borrowed_input
            @test first_query.factorization_destructive
            @test !first_query.factor_solve_mutates_factor
            @test first_query.shared_gemm_workspace_concurrency === :serialized_safe
            @test !first_query.concurrent_factor_workspace
            @test first_query.syrk_authoritative_triangle === :lower
            @test first_query.syrk_inactive_triangle === :preserved
            for property in operation_properties
                property === :mixed_precision_residual && continue
                @test getproperty(first_query, property)
            end
        end

        Unsupported = MultiFloat{Float64,5}
        unsupported = capabilities(Unsupported)
        @test !unsupported.supported
        @test unsupported.limb_count == 5
        @test unsupported.mixed_residual_targets == (
            x2=false, x3=false, x4=false,
        )
        @test unsupported.mixed_residual_target_types == ()
        @test all(!getproperty(unsupported, property)
                  for property in operation_properties)
        @test_throws ArgumentError gemm_plan(
            Unsupported, 1, 1, 1, KernelConfig(thread_count=1),
        )
        @test_throws ArgumentError MFWorkspace(Unsupported)
        @test_throws MethodError capabilities(Float64)
    end

    @testset "shared packed GEMM workspace concurrency" begin
        for T in (Float64x2, Float64x3, Float64x4)
            rows, reduction, columns = 33, 29, 31
            A1 = T.(randn(rows, reduction))
            B1 = T.(randn(reduction, columns))
            A2 = T.(randn(rows + 2, reduction + 4))
            B2 = T.(randn(reduction + 4, columns + 2))
            reference1 = zeros(T, rows, columns)
            reference2 = zeros(T, rows + 2, columns + 2)
            direct = KernelConfig(thread_count=1, gemm_strategy=:direct)
            packed = KernelConfig(
                thread_count=min(2, Threads.nthreads()),
                gemm_strategy=:packed,
                gemm_panel_columns=7,
                gemm_micro_columns=2,
            )
            gemm!(reference1, A1, B1; config=direct)
            gemm!(reference2, A2, B2; config=direct)

            shared = GemmWorkspace(T; thread_count=1, capacity=0)
            for _ in 1:4
                output1 = zeros(T, rows, columns)
                output2 = zeros(T, rows + 2, columns + 2)
                ready = Channel{Nothing}(2)
                start = Base.Event()
                task1 = Threads.@spawn begin
                    put!(ready, nothing)
                    wait(start)
                    gemm!(output1, A1, B1; config=packed, workspace=shared)
                end
                task2 = Threads.@spawn begin
                    put!(ready, nothing)
                    wait(start)
                    gemm!(output2, A2, B2; config=packed, workspace=shared)
                end
                take!(ready)
                take!(ready)
                notify(start)
                fetch(task1)
                fetch(task2)
                @test limb_bitwise_equal(output1, reference1)
                @test limb_bitwise_equal(output2, reference2)
            end
        end
    end

    @testset "structured mulacc policy" begin
        # The structured (GEMMT/SYRK) x3 accumulation is fused only on
        # AArch64, where it measured positive; x86_64 keeps standard `acc+x*y`.
        @test MultiFloatLinearAlgebra._structured_fuses_x3() ==
            (Sys.isapple() && Sys.ARCH === :aarch64)

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
