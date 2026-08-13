using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra
using Printf
using Random

Random.seed!(0xa110ca7e)
BLAS.set_num_threads(1)

const TYPES = (Float64x2, Float64x3, Float64x4)

limb_count(::Type{MultiFloat{Float64,N}}) where {N} = N

function allocation_bytes(f)
    f()
    GC.gc()
    return @allocated f()
end

function report(operation, ::Type{T}, threads, shape, bytes) where {T}
    @printf(
        "%-18s %-10s threads=%d  %-14s %10d bytes\n",
        operation,
        "x$(limb_count(T))",
        threads,
        shape,
        bytes,
    )
end

function make_lower(::Type{T}, n) where {T}
    A = zeros(T, n, n)
    @inbounds for column in 1:n
        for row in column:n
            A[row, column] = T(randn())
        end
        A[column, column] += T(4)
    end
    return A
end

function make_spd(::Type{T}, n) where {T}
    R = randn(n, n)
    A = R * transpose(R)
    @inbounds for i in 1:n
        A[i, i] += n
    end
    return T.(A)
end

function make_indefinite(::Type{T}, n) where {T}
    R = 0.01 .* randn(n, n)
    A = R + transpose(R)
    @inbounds for i in 1:n
        A[i, i] += isodd(i) ? n : -n
    end
    return T.(A)
end

function audit_type(::Type{T}, n, threads) where {T}
    reduction = min(24, n)
    rhs_count = min(8, n)
    config = KernelConfig(thread_count=threads)

    A = T.(randn(n, n))
    B = T.(randn(n, n))
    C = zeros(T, n, n)
    report(
        "gemm/auto",
        T,
        threads,
        "$(n)x$(n)",
        allocation_bytes(() -> gemm!(C, A, B; config=config)),
    )

    packed_config = KernelConfig(
        thread_count=threads,
        gemm_strategy=:packed,
        gemm_panel_columns=16,
        gemm_micro_columns=4,
    )
    packed_workspace = GemmWorkspace(
        T;
        thread_count=threads,
        capacity=n * min(16, n),
    )
    report(
        "gemm/packed-reuse",
        T,
        threads,
        "$(n)x$(n)",
        allocation_bytes(
            () -> gemm!(C, A, B; config=packed_config, workspace=packed_workspace),
        ),
    )

    left = T.(randn(n, reduction))
    right = T.(randn(n, reduction))
    triangular_output = zeros(T, n, n)
    report(
        "gemmt",
        T,
        threads,
        "n=$n,k=$reduction",
        allocation_bytes(
            () -> gemmt!(triangular_output, left, right; config=config),
        ),
    )

    panel = T.(randn(reduction, n))
    report(
        "syrk",
        T,
        threads,
        "k=$reduction,n=$n",
        allocation_bytes(() -> syrk!(triangular_output, panel; config=config)),
    )

    packed_triangle = zeros(T, n * (n + 1) ÷ 2)
    report(
        "syrk/packed",
        T,
        threads,
        "k=$reduction,n=$n",
        allocation_bytes(
            () -> syrk_packed!(packed_triangle, panel; config=config),
        ),
    )

    lower = make_lower(T, n)
    rhs_source = T.(randn(n, rhs_count))
    rhs = similar(rhs_source)
    report(
        "trsm",
        T,
        threads,
        "n=$n,nrhs=$rhs_count",
        allocation_bytes(() -> begin
            copyto!(rhs, rhs_source)
            trsm!(rhs, lower; config=config)
        end),
    )

    vector_source = T.(randn(n))
    vector_rhs = similar(vector_source)
    report(
        "trsv",
        T,
        threads,
        "n=$n",
        allocation_bytes(() -> begin
            copyto!(vector_rhs, vector_source)
            trsv!(vector_rhs, lower; config=config)
        end),
    )

    product_source = T.(randn(rhs_count, n))
    product = similar(product_source)
    report(
        "trmm/right",
        T,
        threads,
        "nrhs=$rhs_count,n=$n",
        allocation_bytes(() -> begin
            copyto!(product, product_source)
            trmm!(product, lower; side=:right, config=config)
        end),
    )

    spd_source = make_spd(T, n)
    factor_input = similar(spd_source)
    report(
        "cholesky",
        T,
        threads,
        "n=$n",
        allocation_bytes(() -> begin
            copyto!(factor_input, spd_source)
            MultiFloatLinearAlgebra.cholesky!(
                factor_input; check=false, config=config,
            )
        end),
    )

    lu_source = T.(randn(n, n))
    @inbounds for i in 1:n
        lu_source[i, i] += T(4)
    end
    report(
        "lu",
        T,
        threads,
        "n=$n",
        allocation_bytes(() -> begin
            copyto!(factor_input, lu_source)
            MultiFloatLinearAlgebra.lu!(
                factor_input; check=false, config=config,
            )
        end),
    )

    ldlt_source = make_indefinite(T, n)
    report(
        "ldlt",
        T,
        threads,
        "n=$n",
        allocation_bytes(() -> begin
            copyto!(factor_input, ldlt_source)
            MultiFloatLinearAlgebra.ldlt!(
                factor_input; check=false, config=config,
            )
        end),
    )
end

function main()
    n = isempty(ARGS) ? 128 : parse(Int, ARGS[1])
    n > 0 || throw(ArgumentError("matrix size must be positive"))
    thread_counts = unique((1, min(4, Threads.nthreads())))

    println("MFLA repeated-call allocation audit")
    println("Julia $(VERSION), Julia threads=$(Threads.nthreads()), BLAS threads=1, n=$n")
    println("Input reset is included; all arrays and GEMM workspace are preallocated.")
    println()
    for threads in thread_counts, T in TYPES
        audit_type(T, n, threads)
        println()
    end
end

main()
