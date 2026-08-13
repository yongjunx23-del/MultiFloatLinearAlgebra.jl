using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra
using Printf
using Random

Random.seed!(0x776f726b)
BLAS.set_num_threads(1)

const TYPES = (Float64x2, Float64x3, Float64x4)

limb_count(::Type{MultiFloat{Float64,N}}) where {N} = N

function allocation_bytes(f)
    f()
    GC.gc()
    return @allocated f()
end

function report(operation, ::Type{T}, n, right_hand_sides, route, bytes) where {T}
    @printf(
        "%-24s x%d n=%4d nrhs=%2d %-12s %10d bytes\n",
        operation,
        limb_count(T),
        n,
        right_hand_sides,
        route,
        bytes,
    )
end

function make_general(::Type{T}, n) where {T}
    A = T.(randn(n, n))
    @inbounds for index in 1:n
        A[index, index] += T(5)
    end
    return A
end

function make_indefinite(::Type{T}, n) where {T}
    A = zeros(T, n, n)
    @inbounds for index in 1:2:(n - 1)
        value = T(1 + index / n)
        A[index, index + 1] = value
        A[index + 1, index] = value
    end
    isodd(n) && (A[n, n] = one(T))
    return A
end

function form_rhs!(B, A, X)
    gemm!(
        B,
        A,
        X;
        config=KernelConfig(thread_count=1, gemm_strategy=:direct),
    )
    return B
end

function lu_cycle!(
    factor_input,
    source,
    solution,
    right_hand_side,
    residual_storage,
    correction,
    config,
    workspace,
)
    copyto!(factor_input, source)
    factor = MultiFloatLinearAlgebra.lu!(
        factor_input;
        check=false,
        config=config,
        workspace=workspace,
    )
    copyto!(solution, right_hand_side)
    MultiFloatLinearAlgebra.ldiv!(solution, factor; config=config)
    residual!(
        residual_storage,
        source,
        solution,
        right_hand_side;
        config=config,
        workspace=workspace,
    )
    refinement_correction!(correction, factor, residual_storage; config=config)
    return factor
end

function ldlt_cycle!(
    factor_input,
    source,
    solution,
    right_hand_side,
    residual_storage,
    correction,
    config,
    workspace,
)
    copyto!(factor_input, source)
    factor = MultiFloatLinearAlgebra.ldlt!(
        factor_input;
        check=false,
        config=config,
        workspace=workspace,
    )
    copyto!(solution, right_hand_side)
    MultiFloatLinearAlgebra.ldiv!(solution, factor; config=config)
    residual!(
        residual_storage,
        source,
        solution,
        right_hand_side;
        config=config,
    )
    refinement_correction!(correction, factor, residual_storage; config=config)
    return factor
end

function audit_type(::Type{T}, n, right_hand_sides) where {T}
    panel_columns = min(16, n)
    block = min(16, n)
    workspace = MFWorkspace(
        T;
        factor_capacity=n,
        ldlt_block_capacity=block,
        thread_count=1,
        gemm_capacity=n * panel_columns,
    )

    general = make_general(T, n)
    indefinite = make_indefinite(T, n)
    truth = T.(randn(n, right_hand_sides))
    right_hand_side = zeros(T, n, right_hand_sides)
    form_rhs!(right_hand_side, general, truth)
    indefinite_rhs = zeros(T, n, right_hand_sides)
    form_rhs!(indefinite_rhs, indefinite, truth)

    factor_input = similar(general)
    solution = similar(right_hand_side)
    residual_storage = similar(right_hand_side)
    correction = similar(right_hand_side)
    direct_config = KernelConfig(
        thread_count=1,
        lu_block=block,
        gemm_strategy=:direct,
    )
    packed_config = KernelConfig(
        thread_count=1,
        lu_block=block,
        gemm_strategy=:packed,
        gemm_panel_columns=panel_columns,
        gemm_micro_columns=2,
    )
    ldlt_config = KernelConfig(
        thread_count=1,
        ldlt_strategy=:blocked,
        ldlt_block=block,
    )

    report(
        "LU cycle",
        T,
        n,
        right_hand_sides,
        "owned/direct",
        allocation_bytes(() -> lu_cycle!(
            factor_input,
            general,
            solution,
            right_hand_side,
            residual_storage,
            correction,
            direct_config,
            nothing,
        )),
    )
    report(
        "LU cycle",
        T,
        n,
        right_hand_sides,
        "reuse/direct",
        allocation_bytes(() -> lu_cycle!(
            factor_input,
            general,
            solution,
            right_hand_side,
            residual_storage,
            correction,
            direct_config,
            workspace,
        )),
    )
    report(
        "LU cycle",
        T,
        n,
        right_hand_sides,
        "owned/packed",
        allocation_bytes(() -> lu_cycle!(
            factor_input,
            general,
            solution,
            right_hand_side,
            residual_storage,
            correction,
            packed_config,
            nothing,
        )),
    )
    report(
        "LU cycle",
        T,
        n,
        right_hand_sides,
        "reuse/packed",
        allocation_bytes(() -> lu_cycle!(
            factor_input,
            general,
            solution,
            right_hand_side,
            residual_storage,
            correction,
            packed_config,
            workspace,
        )),
    )
    report(
        "LDLT blocked cycle",
        T,
        n,
        right_hand_sides,
        "owned",
        allocation_bytes(() -> ldlt_cycle!(
            factor_input,
            indefinite,
            solution,
            indefinite_rhs,
            residual_storage,
            correction,
            ldlt_config,
            nothing,
        )),
    )
    report(
        "LDLT blocked cycle",
        T,
        n,
        right_hand_sides,
        "reuse",
        allocation_bytes(() -> ldlt_cycle!(
            factor_input,
            indefinite,
            solution,
            indefinite_rhs,
            residual_storage,
            correction,
            ldlt_config,
            workspace,
        )),
    )

    copyto!(factor_input, general)
    live_factor = MultiFloatLinearAlgebra.lu!(
        factor_input;
        config=direct_config,
        workspace=workspace,
    )
    copyto!(solution, right_hand_side)
    MultiFloatLinearAlgebra.ldiv!(solution, live_factor; config=direct_config)
    report(
        "residual only",
        T,
        n,
        right_hand_sides,
        "preallocated",
        allocation_bytes(() -> residual!(
            residual_storage,
            general,
            solution,
            right_hand_side;
            config=direct_config,
        )),
    )
    report(
        "correction only",
        T,
        n,
        right_hand_sides,
        "preallocated",
        allocation_bytes(() -> refinement_correction!(
            correction,
            live_factor,
            residual_storage;
            config=direct_config,
        )),
    )

    qr_input = T.(randn(n + 7, n))
    qr_buffer = similar(qr_input)
    report(
        "RRQR factor",
        T,
        n,
        0,
        "owned",
        allocation_bytes(() -> begin
            copyto!(qr_buffer, qr_input)
            rrqr!(qr_buffer)
        end),
    )
    report(
        "RRQR factor",
        T,
        n,
        0,
        "reuse",
        allocation_bytes(() -> begin
            copyto!(qr_buffer, qr_input)
            rrqr!(qr_buffer; workspace=workspace)
        end),
    )
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128
    right_hand_sides = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
    n > 1 || throw(ArgumentError("matrix size must exceed one"))
    right_hand_sides > 0 ||
        throw(ArgumentError("right-hand-side count must be positive"))

    println("MFLA caller-owned workspace cycle allocation audit")
    println(
        "Julia $(VERSION), Julia threads=$(Threads.nthreads()), " *
        "BLAS threads=1, n=$n, nrhs=$right_hand_sides",
    )
    println("All numerical inputs and destinations are preallocated.")
    println()
    for T in TYPES
        audit_type(T, n, right_hand_sides)
        println()
    end
end

main()
