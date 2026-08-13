using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra
using Printf
using Random

Random.seed!(0x534f4c564552)
BLAS.set_num_threads(1)

const SUITE_TYPES = (Float64x2, Float64x3, Float64x4)

type_label(::Type{MultiFloat{Float64,N}}) where {N} = "x$N"

function metric_string(value)
    value === nothing && return "na"
    value isa AbstractFloat && return @sprintf("%.9e", value)
    return replace(string(value), '\t' => ' ', '\n' => ' ')
end

function measurement(function_call, samples)
    function_call()
    elapsed = Vector{Float64}(undef, samples)
    for sample in 1:samples
        GC.gc()
        start = time_ns()
        function_call()
        elapsed[sample] = (time_ns() - start) / 1.0e9
    end
    sort!(elapsed)
    GC.gc()
    bytes = @allocated function_call()
    return (
        seconds=elapsed[cld(samples, 2)],
        allocations=bytes,
        peak_rss=Sys.maxrss(),
    )
end

function emit_row(
    outputs;
    section,
    operation,
    arithmetic,
    threads,
    shape,
    route,
    measured,
    workspace="none",
    residual=nothing,
    backward_error=nothing,
    diagnostics="none",
)
    fields = (
        section,
        operation,
        arithmetic,
        threads,
        shape,
        route,
        measured.seconds,
        measured.allocations,
        measured.peak_rss,
        workspace,
        residual,
        backward_error,
        diagnostics,
    )
    line = join(metric_string.(fields), '\t')
    for output in outputs
        println(output, line)
        flush(output)
    end
    return nothing
end

function max_relative_error(left, right)
    size(left) == size(right) || error("quality arrays differ in size")
    return setprecision(BigFloat, 512) do
        numerator = BigFloat(0)
        denominator = BigFloat(0)
        for index in eachindex(left, right)
            observed = BigFloat(left[index])
            expected = BigFloat(right[index])
            numerator = max(numerator, abs(observed - expected))
            denominator = max(denominator, abs(expected))
        end
        Float64(numerator / max(denominator, BigFloat(1)))
    end
end

function lower_relative_error(left, right)
    size(left) == size(right) || error("quality matrices differ in size")
    return setprecision(BigFloat, 512) do
        numerator = BigFloat(0)
        denominator = BigFloat(0)
        for column in axes(left, 2), row in column:size(left, 1)
            observed = BigFloat(left[row, column])
            expected = BigFloat(right[row, column])
            numerator = max(numerator, abs(observed - expected))
            denominator = max(denominator, abs(expected))
        end
        Float64(numerator / max(denominator, BigFloat(1)))
    end
end

function quality_gate(label, value, ::Type{T}, work) where {T}
    threshold = setprecision(BigFloat, 512) do
        Float64(BigFloat(8192 * max(work, 1)) * BigFloat(eps(T)))
    end
    isfinite(value) && value <= threshold || error(
        "$label correctness gate failed: value=$value threshold=$threshold",
    )
    return value
end

function reference_gemv(A, x; trans=false)
    T = eltype(A)
    output_length = trans ? size(A, 2) : size(A, 1)
    reduction = trans ? size(A, 1) : size(A, 2)
    output = zeros(T, output_length)
    for index in 1:output_length
        accumulator = zero(T)
        for k in 1:reduction
            accumulator += (trans ? A[k, index] : A[index, k]) * x[k]
        end
        output[index] = accumulator
    end
    return output
end

function reference_gemm(A, B)
    T = eltype(A)
    output = zeros(T, size(A, 1), size(B, 2))
    for column in axes(B, 2), row in axes(A, 1)
        accumulator = zero(T)
        for k in axes(A, 2)
            accumulator += A[row, k] * B[k, column]
        end
        output[row, column] = accumulator
    end
    return output
end

function reference_gemmt(left, right)
    T = eltype(left)
    rows = size(left, 1)
    output = zeros(T, rows, rows)
    for column in 1:rows, row in column:rows
        accumulator = zero(T)
        for k in axes(left, 2)
            accumulator += left[row, k] * right[column, k]
        end
        output[row, column] = accumulator
    end
    return output
end

function reference_syrk(panel)
    T = eltype(panel)
    columns = size(panel, 2)
    output = zeros(T, columns, columns)
    for column in 1:columns, row in column:columns
        accumulator = zero(T)
        for k in axes(panel, 1)
            accumulator += panel[k, row] * panel[k, column]
        end
        output[row, column] = accumulator
    end
    return output
end

function reference_symv(A, x)
    T = eltype(A)
    n = size(A, 1)
    output = zeros(T, n)
    for row in 1:n
        accumulator = zero(T)
        for column in 1:n
            coefficient = column <= row ? A[row, column] : A[column, row]
            accumulator += coefficient * x[column]
        end
        output[row] = accumulator
    end
    return output
end

function reference_trmm(A, B)
    T = eltype(A)
    output = zeros(T, size(B))
    for column in axes(B, 2), row in axes(B, 1)
        accumulator = zero(T)
        for k in 1:row
            accumulator += A[row, k] * B[k, column]
        end
        output[row, column] = accumulator
    end
    return output
end

function make_lower(::Type{T}, n) where {T}
    output = zeros(T, n, n)
    for column in 1:n, row in column:n
        output[row, column] = T(randn())
    end
    for index in 1:n
        output[index, index] += T(4)
    end
    return output
end

function make_spd(::Type{T}, n) where {T}
    raw = randn(n, n)
    matrix = (raw * transpose(raw)) / n
    for index in 1:n
        matrix[index, index] += 2
    end
    return T.(matrix)
end

function make_general(::Type{T}, n) where {T}
    matrix = T.(randn(n, n))
    for index in 1:n
        matrix[index, index] += T(5)
    end
    return matrix
end

function make_kkt(::Type{T}, n) where {T}
    constraints = max(4, div(n, 4))
    primal = n - constraints
    raw = randn(primal, primal)
    hessian = (raw * transpose(raw)) / primal
    for index in 1:primal
        hessian[index, index] += 1
    end
    equality = randn(constraints, primal) / sqrt(primal)
    matrix = zeros(Float64, n, n)
    matrix[1:primal, 1:primal] .= hessian
    matrix[1:primal, (primal + 1):n] .= transpose(equality)
    matrix[(primal + 1):n, 1:primal] .= equality
    for index in 1:constraints
        matrix[primal + index, primal + index] = -1.0e-4
    end
    return T.(matrix)
end

function packed_lower_reference(panel)
    T = eltype(panel)
    n = size(panel, 2)
    output = zeros(T, div(n * (n + 1), 2))
    index = 0
    for column in 1:n, row in column:n
        index += 1
        accumulator = zero(T)
        for k in axes(panel, 1)
            accumulator += panel[k, row] * panel[k, column]
        end
        output[index] = accumulator
    end
    return output
end

function benchmark_kernels(outputs, ::Type{T}, n, threads, samples) where {T}
    config = KernelConfig(
        thread_count=threads,
        gemm_strategy=:auto,
        gemm_panel_columns=max(8, min(24, n)),
    )
    reduction = max(9, div(n, 3) + 1)
    rows = n + 3
    columns = n + 5

    normal_matrix = T.(randn(rows, reduction))
    normal_x = T.(randn(reduction))
    normal_output = zeros(T, rows)
    normal_reference = reference_gemv(normal_matrix, normal_x)
    gemv!(normal_output, normal_matrix, normal_x; config=config)
    normal_quality = quality_gate(
        "gemv", max_relative_error(normal_output, normal_reference), T, reduction,
    )
    emit_row(
        outputs;
        section="kernel",
        operation="gemv_n",
        arithmetic=type_label(T),
        threads=threads,
        shape="$(rows)x$(reduction)",
        route="row_owned",
        measured=measurement(
            () -> gemv!(normal_output, normal_matrix, normal_x; config=config),
            samples,
        ),
        residual=normal_quality,
    )

    transpose_matrix = T.(randn(reduction, columns))
    transpose_x = T.(randn(reduction))
    transpose_output = zeros(T, columns)
    transpose_reference = reference_gemv(
        transpose_matrix, transpose_x; trans=true,
    )
    gemv!(
        transpose_output,
        transpose_matrix,
        transpose_x;
        trans=:T,
        config=config,
    )
    transpose_quality = quality_gate(
        "gemv_t",
        max_relative_error(transpose_output, transpose_reference),
        T,
        reduction,
    )
    emit_row(
        outputs;
        section="kernel",
        operation="gemv_t",
        arithmetic=type_label(T),
        threads=threads,
        shape="$(reduction)x$(columns)",
        route="column_lane",
        measured=measurement(
            () -> gemv!(
                transpose_output,
                transpose_matrix,
                transpose_x;
                trans=:T,
                config=config,
            ),
            samples,
        ),
        residual=transpose_quality,
    )

    raw_symmetric = T.(randn(n, n))
    symmetric = T.(Matrix(raw_symmetric + transpose(raw_symmetric)))
    symmetric_x = T.(randn(n))
    symmetric_output = zeros(T, n)
    symmetric_reference = reference_symv(symmetric, symmetric_x)
    symv!(
        symmetric_output, symmetric, symmetric_x;
        uplo=:lower,
        config=config,
    )
    symmetric_quality = quality_gate(
        "symv", max_relative_error(symmetric_output, symmetric_reference), T, n,
    )
    emit_row(
        outputs;
        section="kernel",
        operation="symv",
        arithmetic=type_label(T),
        threads=threads,
        shape="$(n)x$(n)",
        route="lower_authoritative",
        measured=measurement(
            () -> symv!(
                symmetric_output,
                symmetric,
                symmetric_x;
                uplo=:lower,
                config=config,
            ),
            samples,
        ),
        residual=symmetric_quality,
    )

    left = T.(randn(n, reduction))
    right = T.(randn(reduction, n + 1))
    product = zeros(T, n, n + 1)
    product_reference = reference_gemm(left, right)
    gemm!(product, left, right; config=config)
    product_quality = quality_gate(
        "gemm", max_relative_error(product, product_reference), T, reduction,
    )
    plan = gemm_plan(T, n, reduction, n + 1, config)
    emit_row(
        outputs;
        section="kernel",
        operation="gemm",
        arithmetic=type_label(T),
        threads=threads,
        shape="$(n)x$(reduction)x$(n + 1)",
        route="$(plan.strategy)/$(plan.reason)",
        measured=measurement(
            () -> gemm!(product, left, right; config=config), samples,
        ),
        residual=product_quality,
    )

    gemmt_left = T.(randn(n, reduction))
    gemmt_right = T.(randn(n, reduction))
    triangle = zeros(T, n, n)
    triangle_reference = reference_gemmt(gemmt_left, gemmt_right)
    gemmt!(triangle, gemmt_left, gemmt_right; config=config)
    triangle_quality = quality_gate(
        "gemmt",
        lower_relative_error(triangle, triangle_reference),
        T,
        reduction,
    )
    emit_row(
        outputs;
        section="kernel",
        operation="gemmt",
        arithmetic=type_label(T),
        threads=threads,
        shape="$(n)x$(reduction)",
        route=MultiFloatLinearAlgebra._structured_fuses_x3() && T === Float64x3 ?
            "fused_x3_lower" : "standard_lower",
        measured=measurement(
            () -> gemmt!(
                triangle, gemmt_left, gemmt_right; config=config,
            ),
            samples,
        ),
        residual=triangle_quality,
    )

    panel = T.(randn(reduction, n))
    gram = zeros(T, n, n)
    gram_reference = reference_syrk(panel)
    syrk!(gram, panel; config=config)
    gram_quality = quality_gate(
        "syrk", lower_relative_error(gram, gram_reference), T, reduction,
    )
    emit_row(
        outputs;
        section="kernel",
        operation="syrk",
        arithmetic=type_label(T),
        threads=threads,
        shape="$(reduction)x$(n)",
        route=MultiFloatLinearAlgebra._structured_fuses_x3() && T === Float64x3 ?
            "fused_x3_lower" : "standard_lower",
        measured=measurement(
            () -> syrk!(gram, panel; config=config), samples,
        ),
        residual=gram_quality,
    )

    packed = zeros(T, div(n * (n + 1), 2))
    packed_reference = packed_lower_reference(panel)
    syrk_packed!(packed, panel; config=config)
    packed_quality = quality_gate(
        "syrk_packed",
        max_relative_error(packed, packed_reference),
        T,
        reduction,
    )
    emit_row(
        outputs;
        section="kernel",
        operation="syrk_packed",
        arithmetic=type_label(T),
        threads=threads,
        shape="$(reduction)x$(n)",
        route="packed_lower",
        measured=measurement(
            () -> syrk_packed!(packed, panel; config=config), samples,
        ),
        residual=packed_quality,
    )

    triangular = make_lower(T, n)
    vector_truth = T.(randn(n))
    vector_rhs = reference_gemv(triangular, vector_truth)
    vector_work = similar(vector_rhs)
    copyto!(vector_work, vector_rhs)
    trsv!(vector_work, triangular; uplo=:lower, config=config)
    trsv_quality = quality_gate(
        "trsv", max_relative_error(vector_work, vector_truth), T, n,
    )
    emit_row(
        outputs;
        section="kernel",
        operation="trsv",
        arithmetic=type_label(T),
        threads=threads,
        shape="n=$(n),rhs=1",
        route="lower_serial",
        measured=measurement(
            () -> begin
                copyto!(vector_work, vector_rhs)
                trsv!(vector_work, triangular; uplo=:lower, config=config)
            end,
            samples,
        ),
        residual=trsv_quality,
    )

    matrix_truth = T.(randn(n, 4))
    matrix_rhs = reference_gemm(triangular, matrix_truth)
    matrix_work = similar(matrix_rhs)
    copyto!(matrix_work, matrix_rhs)
    trsm!(matrix_work, triangular; uplo=:lower, config=config)
    trsm_quality = quality_gate(
        "trsm", max_relative_error(matrix_work, matrix_truth), T, n,
    )
    emit_row(
        outputs;
        section="kernel",
        operation="trsm",
        arithmetic=type_label(T),
        threads=threads,
        shape="n=$(n),rhs=4",
        route="left_lower",
        measured=measurement(
            () -> begin
                copyto!(matrix_work, matrix_rhs)
                trsm!(matrix_work, triangular; uplo=:lower, config=config)
            end,
            samples,
        ),
        residual=trsm_quality,
    )

    multiply_input = T.(randn(n, 4))
    multiply_reference = reference_trmm(triangular, multiply_input)
    multiply_work = similar(multiply_input)
    copyto!(multiply_work, multiply_input)
    trmm!(multiply_work, triangular; uplo=:lower, config=config)
    trmm_quality = quality_gate(
        "trmm", max_relative_error(multiply_work, multiply_reference), T, n,
    )
    emit_row(
        outputs;
        section="kernel",
        operation="trmm",
        arithmetic=type_label(T),
        threads=threads,
        shape="n=$(n),rhs=4",
        route="left_lower_serial",
        measured=measurement(
            () -> begin
                copyto!(multiply_work, multiply_input)
                trmm!(multiply_work, triangular; uplo=:lower, config=config)
            end,
            samples,
        ),
        residual=trmm_quality,
    )
end

function diagnostics_summary(diagnostics)
    kind = diagnostics.kind
    if kind === :cholesky
        return "status=$(diagnostics.status),spread=$(diagnostics.diagonal_spread)"
    elseif kind === :lu
        return "status=$(diagnostics.status),growth=$(diagnostics.pivot_growth)"
    elseif kind === :ldlt
        return "status=$(diagnostics.status),blocks=$(diagnostics.one_by_one_pivots)+2x$(diagnostics.two_by_two_pivots),inertia=$(diagnostics.inertia)"
    end
    return "status=$(diagnostics.status),rank=$(diagnostics.rank_at_threshold),spread=$(diagnostics.rdiag_spread)"
end

function solve_component_rows(
    outputs,
    label,
    matrix,
    factor,
    config,
    threads,
    samples;
    uplo=:general,
)
    T = eltype(matrix)
    n = size(matrix, 1)
    vector_truth = T.(randn(n))
    vector_rhs = uplo === :general ?
        reference_gemv(matrix, vector_truth) :
        reference_symv(matrix, vector_truth)
    vector_work = similar(vector_rhs)
    vector_measured = measurement(
        () -> begin
            copyto!(vector_work, vector_rhs)
            MultiFloatLinearAlgebra.ldiv!(vector_work, factor; config=config)
        end,
        samples,
    )
    vector_quality = quality_gate(
        "$(label)_vector_solve",
        max_relative_error(vector_work, vector_truth),
        T,
        n,
    )
    emit_row(
        outputs;
        section="solve",
        operation="$(label)_predictor",
        arithmetic=type_label(T),
        threads=threads,
        shape="n=$(n),rhs=1",
        route="trsv",
        measured=vector_measured,
        residual=vector_quality,
    )

    matrix_truth = T.(randn(n, 4))
    matrix_rhs = if uplo === :general
        reference_gemm(matrix, matrix_truth)
    else
        output = zeros(T, n, 4)
        for column in 1:4
            output[:, column] .= reference_symv(
                matrix, view(matrix_truth, :, column),
            )
        end
        output
    end
    matrix_work = similar(matrix_rhs)
    matrix_measured = measurement(
        () -> begin
            copyto!(matrix_work, matrix_rhs)
            MultiFloatLinearAlgebra.ldiv!(matrix_work, factor; config=config)
        end,
        samples,
    )
    matrix_quality = quality_gate(
        "$(label)_matrix_solve",
        max_relative_error(matrix_work, matrix_truth),
        T,
        n,
    )
    emit_row(
        outputs;
        section="solve",
        operation="$(label)_corrector",
        arithmetic=type_label(T),
        threads=threads,
        shape="n=$(n),rhs=4",
        route="trsm",
        measured=matrix_measured,
        residual=matrix_quality,
    )

    residual_storage = similar(vector_rhs)
    residual_arguments = uplo === :general ?
        (; config=config) : (; uplo=:lower, config=config)
    residual_measured = measurement(
        () -> residual!(
            residual_storage,
            matrix,
            vector_work,
            vector_rhs;
            residual_arguments...,
        ),
        samples,
    )
    backward = Float64(normwise_backward_error(
        matrix,
        vector_work,
        vector_rhs,
        residual_storage;
        (uplo === :general ? (;) : (; uplo=:lower))...,
    ))
    quality_gate("$(label)_backward", backward, T, n)
    emit_row(
        outputs;
        section="solve",
        operation="$(label)_residual",
        arithmetic=type_label(T),
        threads=threads,
        shape="n=$(n),rhs=1",
        route=uplo === :general ? "general" : "lower_authoritative",
        measured=residual_measured,
        backward_error=backward,
    )

    correction = similar(residual_storage)
    correction_measured = measurement(
        () -> refinement_correction!(
            correction, factor, residual_storage; config=config,
        ),
        samples,
    )
    emit_row(
        outputs;
        section="solve",
        operation="$(label)_correction",
        arithmetic=type_label(T),
        threads=threads,
        shape="n=$(n),rhs=1",
        route="one_explicit_solve",
        measured=correction_measured,
        backward_error=backward,
    )
    return nothing
end

function qr_explicit_r(factor)
    rows, columns = size(factor)
    output = zeros(eltype(factor), rows, columns)
    compact = factor_matrix(factor)
    for column in 1:columns, row in 1:min(column, rows)
        output[row, column] = compact[row, column]
    end
    return output
end

function benchmark_factors(outputs, ::Type{T}, n, threads, samples) where {T}
    config = KernelConfig(
        thread_count=threads,
        cholesky_block=min(16, n),
        lu_block=min(16, n),
        ldlt_strategy=:auto,
        gemm_strategy=:auto,
    )

    spd = make_spd(T, n)
    spd_buffer = similar(spd)
    cholesky_measured = measurement(
        () -> begin
            copyto!(spd_buffer, spd)
            MultiFloatLinearAlgebra.cholesky!(spd_buffer; config=config)
        end,
        samples,
    )
    cholesky_factor = MultiFloatLinearAlgebra.cholesky!(
        copy(spd); config=config,
    )
    cholesky_diagnostics = factor_diagnostics(cholesky_factor)
    emit_row(
        outputs;
        section="factor",
        operation="cholesky",
        arithmetic=type_label(T),
        threads=threads,
        shape="$(n)x$(n)",
        route="blocked_lower",
        measured=cholesky_measured,
        diagnostics=diagnostics_summary(cholesky_diagnostics),
    )
    solve_component_rows(
        outputs,
        "cholesky",
        spd,
        cholesky_factor,
        config,
        threads,
        samples;
        uplo=:lower,
    )

    general = make_general(T, n)
    general_buffer = similar(general)
    lu_measured = measurement(
        () -> begin
            copyto!(general_buffer, general)
            MultiFloatLinearAlgebra.lu!(general_buffer; config=config)
        end,
        samples,
    )
    lu_factor = MultiFloatLinearAlgebra.lu!(copy(general); config=config)
    lu_diagnostics = factor_diagnostics(lu_factor)
    emit_row(
        outputs;
        section="factor",
        operation="lu",
        arithmetic=type_label(T),
        threads=threads,
        shape="$(n)x$(n)",
        route="blocked_partial_pivot",
        measured=lu_measured,
        diagnostics=diagnostics_summary(lu_diagnostics),
    )
    solve_component_rows(
        outputs,
        "lu",
        general,
        lu_factor,
        config,
        threads,
        samples,
    )

    kkt = make_kkt(T, n)
    kkt_buffer = similar(kkt)
    ldlt_plan_value = ldlt_plan(T, n, config)
    ldlt_measured = measurement(
        () -> begin
            copyto!(kkt_buffer, kkt)
            MultiFloatLinearAlgebra.ldlt!(kkt_buffer; config=config)
        end,
        samples,
    )
    ldlt_factor = MultiFloatLinearAlgebra.ldlt!(copy(kkt); config=config)
    ldlt_diagnostics = factor_diagnostics(ldlt_factor)
    emit_row(
        outputs;
        section="factor",
        operation="ldlt_kkt",
        arithmetic=type_label(T),
        threads=threads,
        shape="$(n)x$(n)",
        route="$(ldlt_plan_value.strategy)/$(ldlt_plan_value.reason)/block=$(ldlt_plan_value.block_size)",
        measured=ldlt_measured,
        diagnostics=diagnostics_summary(ldlt_diagnostics),
    )
    solve_component_rows(
        outputs,
        "ldlt",
        kkt,
        ldlt_factor,
        config,
        threads,
        samples;
        uplo=:lower,
    )

    qr_rows = n + 7
    qr_columns = max(4, div(n, 2))
    equality = T.(randn(qr_rows, qr_columns))
    equality_buffer = similar(equality)
    qr_measured = measurement(
        () -> begin
            copyto!(equality_buffer, equality)
            rrqr!(equality_buffer)
        end,
        samples,
    )
    qr_factor = rrqr!(copy(equality))
    qr_diagnostics = factor_diagnostics(
        qr_factor; rtol=sqrt(eps(T)),
    )
    q_matrix = Matrix{T}(I, qr_rows, qr_rows)
    apply_q!(q_matrix, qr_factor)
    reconstruction = reference_gemm(q_matrix, qr_explicit_r(qr_factor))
    qr_quality = quality_gate(
        "rrqr",
        max_relative_error(
            reconstruction, equality[:, factor_permutation(qr_factor)],
        ),
        T,
        qr_rows,
    )
    emit_row(
        outputs;
        section="factor",
        operation="rrqr_equality",
        arithmetic=type_label(T),
        threads=threads,
        shape="$(qr_rows)x$(qr_columns)",
        route="column_pivoted_householder",
        measured=qr_measured,
        residual=qr_quality,
        diagnostics=diagnostics_summary(qr_diagnostics),
    )

    qr_workspace = MFWorkspace(
        T; factor_capacity=qr_columns, thread_count=threads,
    )
    qr_workspace_measured = measurement(
        () -> begin
            copyto!(equality_buffer, equality)
            rrqr!(equality_buffer; workspace=qr_workspace)
        end,
        samples,
    )
    emit_row(
        outputs;
        section="factor",
        operation="rrqr_equality_workspace",
        arithmetic=type_label(T),
        threads=threads,
        shape="$(qr_rows)x$(qr_columns)",
        route="column_pivoted_householder",
        measured=qr_workspace_measured,
        workspace=workspace_capacity(qr_workspace),
        residual=qr_quality,
        diagnostics=diagnostics_summary(qr_diagnostics),
    )

    q_vector = T.(randn(qr_rows))
    q_work = similar(q_vector)
    q_measured = measurement(
        () -> begin
            copyto!(q_work, q_vector)
            apply_q!(q_work, qr_factor; trans=:T)
        end,
        samples,
    )
    emit_row(
        outputs;
        section="solve",
        operation="rrqr_apply_qt",
        arithmetic=type_label(T),
        threads=threads,
        shape="n=$(qr_rows),rhs=1",
        route="householder",
        measured=q_measured,
        diagnostics=diagnostics_summary(qr_diagnostics),
    )

    rank = qr_columns
    leading_r = qr_explicit_r(qr_factor)[1:rank, 1:rank]
    r_truth = T.(randn(rank))
    r_rhs = reference_gemv(leading_r, r_truth)
    r_work = similar(r_rhs)
    r_measured = measurement(
        () -> begin
            copyto!(r_work, r_rhs)
            solve_r!(r_work, qr_factor, rank; config=config)
        end,
        samples,
    )
    r_quality = quality_gate(
        "solve_r", max_relative_error(r_work, r_truth), T, rank,
    )
    emit_row(
        outputs;
        section="solve",
        operation="rrqr_solve_r",
        arithmetic=type_label(T),
        threads=threads,
        shape="n=$(rank),rhs=1",
        route="trsv_upper",
        measured=r_measured,
        residual=r_quality,
    )

    benchmark_cycles(
        outputs,
        T,
        n,
        threads,
        samples,
        config,
        general,
        kkt,
    )
    return nothing
end

function cycle_rhs(matrix, uplo)
    T = eltype(matrix)
    n = size(matrix, 1)
    predictor_truth = T.(randn(n))
    predictor_rhs = uplo === :general ?
        reference_gemv(matrix, predictor_truth) :
        reference_symv(matrix, predictor_truth)
    corrector_truth = T.(randn(n, 4))
    corrector_rhs = if uplo === :general
        reference_gemm(matrix, corrector_truth)
    else
        output = zeros(T, n, 4)
        for column in 1:4
            output[:, column] .= reference_symv(
                matrix, view(corrector_truth, :, column),
            )
        end
        output
    end
    return predictor_rhs, corrector_rhs
end

function benchmark_cycle_route(
    outputs,
    ::Type{T},
    label,
    matrix,
    config,
    threads,
    samples,
    workspace;
    uplo=:general,
    predictor_rhs,
    corrector_rhs,
) where {T}
    n = size(matrix, 1)
    factor_buffer = similar(matrix)
    predictor_solution = similar(predictor_rhs)
    corrector_solution = similar(corrector_rhs)
    residual_storage = similar(predictor_rhs)
    correction = similar(predictor_rhs)

    function cycle!()
        copyto!(factor_buffer, matrix)
        factor = if label == "lu"
            MultiFloatLinearAlgebra.lu!(
                factor_buffer; config=config, workspace=workspace,
            )
        else
            MultiFloatLinearAlgebra.ldlt!(
                factor_buffer; config=config, workspace=workspace,
            )
        end
        copyto!(predictor_solution, predictor_rhs)
        MultiFloatLinearAlgebra.ldiv!(
            predictor_solution, factor; config=config,
        )
        copyto!(corrector_solution, corrector_rhs)
        MultiFloatLinearAlgebra.ldiv!(
            corrector_solution, factor; config=config,
        )
        if uplo === :general
            residual!(
                residual_storage,
                matrix,
                predictor_solution,
                predictor_rhs;
                config=config,
                workspace=workspace,
            )
        else
            residual!(
                residual_storage,
                matrix,
                predictor_solution,
                predictor_rhs;
                uplo=:lower,
                config=config,
            )
        end
        refinement_correction!(
            correction, factor, residual_storage; config=config,
        )
        return factor
    end

    measured = measurement(cycle!, samples)
    cycle_factor = cycle!()
    cycle_diagnostics = factor_diagnostics(cycle_factor)
    backward = Float64(normwise_backward_error(
        matrix,
        predictor_solution,
        predictor_rhs,
        residual_storage;
        (uplo === :general ? (;) : (; uplo=:lower))...,
    ))
    quality_gate("$(label)_cycle", backward, T, n)
    workspace_description = workspace === nothing ?
        "owned" : workspace_capacity(workspace)
    route = label == "ldlt" ? begin
        plan = ldlt_plan(T, n, config)
        "$(plan.strategy)/$(plan.reason)"
    end : "partial_pivot"
    emit_row(
        outputs;
        section="cycle",
        operation="$(label)_factor_predictor_corrector_residual_correction",
        arithmetic=type_label(T),
        threads=threads,
        shape="n=$(n),rhs=1+4",
        route=route,
        measured=measured,
        workspace=workspace_description,
        backward_error=backward,
        diagnostics=diagnostics_summary(cycle_diagnostics),
    )
    return nothing
end

function benchmark_cycles(
    outputs,
    ::Type{T},
    n,
    threads,
    samples,
    config,
    general,
    kkt,
) where {T}
    block_capacity = max(2, ldlt_plan(T, n, config).block_size)
    workspace = MFWorkspace(
        T;
        factor_capacity=n,
        ldlt_block_capacity=block_capacity,
        thread_count=threads,
        gemm_capacity=n * min(24, n),
    )
    general_predictor_rhs, general_corrector_rhs = cycle_rhs(
        general, :general,
    )
    kkt_predictor_rhs, kkt_corrector_rhs = cycle_rhs(kkt, :lower)
    benchmark_cycle_route(
        outputs,
        T,
        "lu",
        general,
        config,
        threads,
        samples,
        nothing,
        predictor_rhs=general_predictor_rhs,
        corrector_rhs=general_corrector_rhs,
    )
    benchmark_cycle_route(
        outputs,
        T,
        "lu",
        general,
        config,
        threads,
        samples,
        workspace,
        predictor_rhs=general_predictor_rhs,
        corrector_rhs=general_corrector_rhs,
    )
    benchmark_cycle_route(
        outputs,
        T,
        "ldlt",
        kkt,
        config,
        threads,
        samples,
        nothing;
        uplo=:lower,
        predictor_rhs=kkt_predictor_rhs,
        corrector_rhs=kkt_corrector_rhs,
    )
    benchmark_cycle_route(
        outputs,
        T,
        "ldlt",
        kkt,
        config,
        threads,
        samples,
        workspace;
        uplo=:lower,
        predictor_rhs=kkt_predictor_rhs,
        corrector_rhs=kkt_corrector_rhs,
    )
    return nothing
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 96
    samples = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
    n >= 16 || throw(ArgumentError("suite dimension must be at least 16"))
    samples >= 1 || throw(ArgumentError("sample count must be positive"))

    outputs = IO[stdout]
    output_file = nothing
    if length(ARGS) >= 3
        output_file = open(ARGS[3], "w")
        push!(outputs, output_file)
    end
    header = join((
        "section",
        "operation",
        "arithmetic",
        "threads",
        "shape",
        "route",
        "median_seconds",
        "allocated_bytes",
        "process_peak_rss_bytes",
        "workspace_capacity",
        "relative_residual",
        "backward_error",
        "diagnostics",
    ), '\t')
    for output in outputs
        println(output, "# MFLA solver suite")
        println(
            output,
            "# julia=$(VERSION) arch=$(Sys.ARCH) os=$(Sys.KERNEL) " *
            "available_threads=$(Threads.nthreads()) blas_threads=1 " *
            "n=$n samples=$samples rss=monotonic_process_peak",
        )
        println(output, header)
    end

    thread_counts = unique((1, min(4, Threads.nthreads())))
    for threads in thread_counts
        for T in SUITE_TYPES
            # Thread-count comparisons use identical arithmetic inputs for a
            # given limb count. Timing order may change, but matrix values do not.
            Random.seed!(0x534f4c564552 + capabilities(T).limb_count)
            benchmark_kernels(outputs, T, n, threads, samples)
            benchmark_factors(outputs, T, n, threads, samples)
        end
    end
    output_file === nothing || close(output_file)
    return nothing
end

main()
