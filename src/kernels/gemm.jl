@inline function _default_gemm_panel_columns(
    ::Type{MultiFloat{T,N}},
    threads::Int,
) where {T,N}
    base = N <= 1 ? 48 : N == 2 ? 32 : N == 3 ? 24 : 16
    return threads >= 8 ? max(12, base * 3 ÷ 4) : base
end

@inline function _default_gemm_micro_columns(
    ::Type{MultiFloat{T,N}},
) where {T,N}
    return N <= 3 ? 4 : 2
end

@inline function _near_square_shape(m::Int, k::Int, n::Int, ratio::Int=4)
    all(dim -> dim > 0, (m, k, n)) || return false
    return maximum((m, k, n)) <= ratio * minimum((m, k, n))
end

"""
    gemm_plan(T, m, k, n, config=KernelConfig())

Resolve the inspectable dense-GEMM route. `:auto` never benchmarks implicitly;
it uses the stored crossover and the type-specific panel/microkernel defaults.
Use `calibrate_gemm` to produce a machine-specific profile explicitly.

`gemm_strategy` accepts:

- `:auto` — packed when calibration clears the crossover, otherwise the direct
  family (the fused `mulacc_x3` kernel for Float64x3, the standard kernel
  otherwise);
- `:direct` — the standard direct kernel, kept as a reference baseline;
- `:packed` — the B-panel-packed kernel;
- `:fused` — the fused Float64x3 direct kernel (x3 only).

`gemm_packed_crossover` is expressed as an equivalent square edge length: the
packed route is eligible once the total `m*k*n` work reaches `crossover^3`, the
reduction dimension `k` is at least half that edge, and the shape is near-square
(no dimension more than four times another). Calibration only measures square
GEMM, so the calibrated `:auto` route is intentionally restricted to near-square
shapes; strongly tall or wide GEMM stays on the direct route until shape-aware
calibration exists.
"""
function gemm_plan(
    ::Type{MF},
    m::Int,
    k::Int,
    n::Int,
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    _check_supported(MF)
    strategy = config.gemm_strategy
    strategy in (:auto, :direct, :packed, :fused) ||
        throw(ArgumentError("gemm_strategy must be :auto, :direct, :packed, or :fused"))

    if strategy === :fused
        _supports_fused_mulacc(MF) ||
            throw(ArgumentError("gemm_strategy=:fused is only available for Float64x3"))
        panel_columns = config.gemm_panel_columns > 0 ?
                        config.gemm_panel_columns :
                        _default_gemm_panel_columns(MF, config.thread_count)
        panel_columns = max(panel_columns, 1)
        jobs = cld(max(n, 0), panel_columns)
        workers = _workers(config, jobs)
        return GemmPlan(:fused, :forced_fused, panel_columns, 2, workers, 0)
    end

    panel_columns = config.gemm_panel_columns > 0 ?
                    config.gemm_panel_columns :
                    _default_gemm_panel_columns(MF, config.thread_count)
    panel_columns = max(panel_columns, 1)
    requested_micro = config.gemm_micro_columns > 0 ?
                      config.gemm_micro_columns :
                      _default_gemm_micro_columns(MF)
    requested_micro in (1, 2, 4) ||
        throw(ArgumentError("gemm_micro_columns must be 1, 2, or 4"))
    micro_columns = min(requested_micro, panel_columns)
    micro_columns = micro_columns >= 4 ? 4 : micro_columns >= 2 ? 2 : 1
    jobs = cld(max(n, 0), panel_columns)
    workers = _workers(config, jobs)

    if strategy === :direct
        return GemmPlan(
            :direct,
            :forced_direct,
            panel_columns,
            micro_columns,
            workers,
            0,
        )
    elseif strategy === :packed
        return GemmPlan(
            :packed,
            :forced_packed,
            panel_columns,
            micro_columns,
            workers,
            max(k, 0) * panel_columns,
        )
    end

    # crossover is an equivalent square edge length; compare total work against
    # its cube so non-square shapes use the same volume-based trigger.
    crossover = max(config.gemm_packed_crossover, 1)
    work = Float64(max(m, 0)) * Float64(max(k, 0)) * Float64(max(n, 0))
    minimum_reduction = max(32, crossover ÷ 2)
    reason = if !_near_square_shape(m, k, n)
        :auto_outside_calibrated_shape
    elseif k < minimum_reduction
        :auto_reduction_too_small
    elseif work < Float64(crossover)^3
        :auto_below_crossover
    else
        :auto_above_crossover
    end
    use_packed = reason === :auto_above_crossover
    strategy = use_packed ? :packed :
               _supports_fused_mulacc(MF) ? :fused : :direct
    resolved_reason = use_packed ? reason :
                      _supports_fused_mulacc(MF) ? :auto_fused_direct : reason
    return GemmPlan(
        strategy,
        resolved_reason,
        panel_columns,
        micro_columns,
        workers,
        use_packed ? max(k, 0) * panel_columns : 0,
    )
end

@inline function _gemm_store_pair!(
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    B::AbstractMatrix{MF},
    row::Int,
    column::Int,
    alpha::MF,
    beta::MF,
) where {T,N,MF<:MultiFloat{T,N}}
    V4 = MultiFloatVec{4,T,N}
    first_accumulator = zero(V4)
    second_accumulator = zero(V4)
    @inbounds for k in axes(A, 2)
        values = V4(
            A[row, k],
            A[row + 1, k],
            A[row + 2, k],
            A[row + 3, k],
        )
        first_accumulator += values * V4(B[k, column])
        second_accumulator += values * V4(B[k, column + 1])
    end

    first_result = V4(alpha) * first_accumulator + V4(beta) * V4(
        C[row, column],
        C[row + 1, column],
        C[row + 2, column],
        C[row + 3, column],
    )
    second_result = V4(alpha) * second_accumulator + V4(beta) * V4(
        C[row, column + 1],
        C[row + 1, column + 1],
        C[row + 2, column + 1],
        C[row + 3, column + 1],
    )
    @inbounds for lane in 1:4
        C[row + lane - 1, column] = first_result[lane]
        C[row + lane - 1, column + 1] = second_result[lane]
    end
    return nothing
end

function _gemm_direct_column_range!(
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    B::AbstractMatrix{MF},
    alpha::MF,
    beta::MF,
    first_column::Int,
    last_column::Int,
) where {T,N,MF<:MultiFloat{T,N}}
    m = size(A, 1)
    V4 = MultiFloatVec{4,T,N}
    column = first_column

    @inbounds while column + 1 <= last_column
        row = 1
        while row + 3 <= m
            _gemm_store_pair!(C, A, B, row, column, alpha, beta)
            row += 4
        end
        while row <= m
            first_accumulator = zero(MF)
            second_accumulator = zero(MF)
            for k in axes(A, 2)
                a = A[row, k]
                first_accumulator += a * B[k, column]
                second_accumulator += a * B[k, column + 1]
            end
            C[row, column] =
                alpha * first_accumulator + beta * C[row, column]
            C[row, column + 1] =
                alpha * second_accumulator + beta * C[row, column + 1]
            row += 1
        end
        column += 2
    end

    if column <= last_column
        row = 1
        while row + 3 <= m
            accumulator = zero(V4)
            for k in axes(A, 2)
                values = V4(
                    A[row, k],
                    A[row + 1, k],
                    A[row + 2, k],
                    A[row + 3, k],
                )
                accumulator += values * V4(B[k, column])
            end
            result = V4(alpha) * accumulator + V4(beta) * V4(
                C[row, column],
                C[row + 1, column],
                C[row + 2, column],
                C[row + 3, column],
            )
            for lane in 1:4
                C[row + lane - 1, column] = result[lane]
            end
            row += 4
        end
        while row <= m
            accumulator = zero(MF)
            for k in axes(A, 2)
                accumulator += A[row, k] * B[k, column]
            end
            C[row, column] = alpha * accumulator + beta * C[row, column]
            row += 1
        end
    end
    return nothing
end

function _gemm_direct!(
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    B::AbstractMatrix{MF},
    alpha::MF,
    beta::MF,
    plan::GemmPlan,
) where {MF<:MultiFloat}
    n = size(B, 2)
    jobs = cld(n, plan.panel_columns)
    workers = plan.workers
    if workers == 1 || jobs <= 1
        _gemm_direct_column_range!(C, A, B, alpha, beta, 1, n)
        return C
    end

    @sync for worker in 1:workers
        Threads.@spawn begin
            for job in worker:workers:jobs
                first_column = (job - 1) * plan.panel_columns + 1
                last_column = min(job * plan.panel_columns, n)
                _gemm_direct_column_range!(
                    C,
                    A,
                    B,
                    alpha,
                    beta,
                    first_column,
                    last_column,
                )
            end
        end
    end
    return C
end

# ---------------------------------------------------------------------------
# Fused x3 direct GEMM
# ---------------------------------------------------------------------------

@inline function _gemm_store_pair_fused!(
    C::AbstractMatrix{MultiFloat{Float64,3}},
    A::AbstractMatrix{MultiFloat{Float64,3}},
    B::AbstractMatrix{MultiFloat{Float64,3}},
    row::Int,
    column::Int,
    alpha::MultiFloat{Float64,3},
    beta::MultiFloat{Float64,3},
)
    V3 = MultiFloatVec{4,Float64,3}
    first_accumulator = zero(V3)
    second_accumulator = zero(V3)
    @inbounds for k in axes(A, 2)
        values = V3(
            A[row, k],
            A[row + 1, k],
            A[row + 2, k],
            A[row + 3, k],
        )
        first_accumulator = mulacc_x3(first_accumulator, values, V3(B[k, column]))
        second_accumulator = mulacc_x3(second_accumulator, values, V3(B[k, column + 1]))
    end

    first_result = V3(alpha) * first_accumulator + V3(beta) * V3(
        C[row, column],
        C[row + 1, column],
        C[row + 2, column],
        C[row + 3, column],
    )
    second_result = V3(alpha) * second_accumulator + V3(beta) * V3(
        C[row, column + 1],
        C[row + 1, column + 1],
        C[row + 2, column + 1],
        C[row + 3, column + 1],
    )
    @inbounds for lane in 1:4
        C[row + lane - 1, column] = first_result[lane]
        C[row + lane - 1, column + 1] = second_result[lane]
    end
    return nothing
end

@inline function _gemm_store_single_fused!(
    C::AbstractMatrix{MultiFloat{Float64,3}},
    A::AbstractMatrix{MultiFloat{Float64,3}},
    B::AbstractMatrix{MultiFloat{Float64,3}},
    row::Int,
    column::Int,
    alpha::MultiFloat{Float64,3},
    beta::MultiFloat{Float64,3},
)
    V3 = MultiFloatVec{4,Float64,3}
    accumulator = zero(V3)
    @inbounds for k in axes(A, 2)
        values = V3(
            A[row, k],
            A[row + 1, k],
            A[row + 2, k],
            A[row + 3, k],
        )
        accumulator = mulacc_x3(accumulator, values, V3(B[k, column]))
    end
    result = V3(alpha) * accumulator + V3(beta) * V3(
        C[row, column],
        C[row + 1, column],
        C[row + 2, column],
        C[row + 3, column],
    )
    @inbounds for lane in 1:4
        C[row + lane - 1, column] = result[lane]
    end
    return nothing
end

function _gemm_direct_column_range_fused!(
    C::AbstractMatrix{MultiFloat{Float64,3}},
    A::AbstractMatrix{MultiFloat{Float64,3}},
    B::AbstractMatrix{MultiFloat{Float64,3}},
    alpha::MultiFloat{Float64,3},
    beta::MultiFloat{Float64,3},
    first_column::Int,
    last_column::Int,
)
    MF3 = MultiFloat{Float64,3}
    m = size(A, 1)
    column = first_column

    @inbounds while column + 1 <= last_column
        row = 1
        while row + 3 <= m
            _gemm_store_pair_fused!(C, A, B, row, column, alpha, beta)
            row += 4
        end
        while row <= m
            first_accumulator = zero(MF3)
            second_accumulator = zero(MF3)
            for k in axes(A, 2)
                a = A[row, k]
                first_accumulator += a * B[k, column]
                second_accumulator += a * B[k, column + 1]
            end
            C[row, column] =
                alpha * first_accumulator + beta * C[row, column]
            C[row, column + 1] =
                alpha * second_accumulator + beta * C[row, column + 1]
            row += 1
        end
        column += 2
    end

    if column <= last_column
        row = 1
        while row + 3 <= m
            _gemm_store_single_fused!(C, A, B, row, column, alpha, beta)
            row += 4
        end
        while row <= m
            accumulator = zero(MF3)
            for k in axes(A, 2)
                accumulator += A[row, k] * B[k, column]
            end
            C[row, column] = alpha * accumulator + beta * C[row, column]
            row += 1
        end
    end
    return nothing
end

function _gemm_direct_fused!(
    C::AbstractMatrix{MultiFloat{Float64,3}},
    A::AbstractMatrix{MultiFloat{Float64,3}},
    B::AbstractMatrix{MultiFloat{Float64,3}},
    alpha::MultiFloat{Float64,3},
    beta::MultiFloat{Float64,3},
    plan::GemmPlan,
)
    n = size(B, 2)
    jobs = cld(n, plan.panel_columns)
    workers = plan.workers

    if workers == 1 || jobs <= 1
        _gemm_direct_column_range_fused!(C, A, B, alpha, beta, 1, n)
        return C
    end

    @sync for worker in 1:workers
        Threads.@spawn begin
            for job in worker:workers:jobs
                first_column = (job - 1) * plan.panel_columns + 1
                last_column = min(job * plan.panel_columns, n)
                _gemm_direct_column_range_fused!(
                    C, A, B, alpha, beta, first_column, last_column,
                )
            end
        end
    end
    return C
end

function _pack_b_panel!(
    destination::AbstractVector{MF},
    B::AbstractMatrix{MF},
    first_column::Int,
    last_column::Int,
) where {MF<:MultiFloat}
    width = last_column - first_column + 1
    reduction = size(B, 1)
    @inbounds for k in 1:reduction
        offset = (k - 1) * width
        for local_column in 1:width
            destination[offset + local_column] =
                B[k, first_column + local_column - 1]
        end
    end
    return width
end

function _packed_gemm_column_group!(
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    packed_b::AbstractVector{MF},
    width::Int,
    local_column::Int,
    global_column::Int,
    alpha::MF,
    beta::MF,
    columns::Val{NR},
) where {MF<:MultiFloat,NR}
    m = size(A, 1)
    row = 1
    @inbounds while row + 3 <= m
        _packed_gemm_vector_block!(
            C,
            A,
            packed_b,
            width,
            row,
            local_column,
            global_column,
            alpha,
            beta,
            columns,
        )
        row += 4
    end
    while row <= m
        _packed_gemm_scalar_block!(
            C,
            A,
            packed_b,
            width,
            row,
            local_column,
            global_column,
            alpha,
            beta,
            columns,
        )
        row += 1
    end
    return nothing
end

function _packed_gemm_panel_nr!(
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    packed_b::AbstractVector{MF},
    first_column::Int,
    width::Int,
    alpha::MF,
    beta::MF,
    ::Val{NR},
) where {MF<:MultiFloat,NR}
    local_column = 1
    while local_column + NR - 1 <= width
        _packed_gemm_column_group!(
            C,
            A,
            packed_b,
            width,
            local_column,
            first_column + local_column - 1,
            alpha,
            beta,
            Val(NR),
        )
        local_column += NR
    end
    while local_column + 1 <= width
        _packed_gemm_column_group!(
            C,
            A,
            packed_b,
            width,
            local_column,
            first_column + local_column - 1,
            alpha,
            beta,
            Val(2),
        )
        local_column += 2
    end
    if local_column <= width
        _packed_gemm_column_group!(
            C,
            A,
            packed_b,
            width,
            local_column,
            first_column + local_column - 1,
            alpha,
            beta,
            Val(1),
        )
    end
    return C
end

function _packed_gemm_panel!(
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    packed_b::AbstractVector{MF},
    first_column::Int,
    width::Int,
    alpha::MF,
    beta::MF,
    micro_columns::Int,
) where {MF<:MultiFloat}
    if micro_columns == 4
        return _packed_gemm_panel_nr!(
            C, A, packed_b, first_column, width, alpha, beta, Val(4),
        )
    elseif micro_columns == 2
        return _packed_gemm_panel_nr!(
            C, A, packed_b, first_column, width, alpha, beta, Val(2),
        )
    end
    return _packed_gemm_panel_nr!(
        C, A, packed_b, first_column, width, alpha, beta, Val(1),
    )
end

function _gemm_packed_worker!(
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    B::AbstractMatrix{MF},
    alpha::MF,
    beta::MF,
    plan::GemmPlan,
    workspace::GemmWorkspace{MF},
    worker::Int,
    jobs::Int,
) where {MF<:MultiFloat}
    buffer = workspace.buffers[worker]
    @inbounds for job in worker:plan.workers:jobs
        first_column = (job - 1) * plan.panel_columns + 1
        last_column = min(job * plan.panel_columns, size(B, 2))
        width = _pack_b_panel!(buffer, B, first_column, last_column)
        _packed_gemm_panel!(
            C,
            A,
            buffer,
            first_column,
            width,
            alpha,
            beta,
            plan.micro_columns,
        )
    end
    return nothing
end

function _gemm_packed!(
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    B::AbstractMatrix{MF},
    alpha::MF,
    beta::MF,
    plan::GemmPlan,
    workspace::Union{Nothing,GemmWorkspace{MF}},
) where {MF<:MultiFloat}
    jobs = cld(size(B, 2), plan.panel_columns)
    owned_workspace = workspace === nothing ?
                      GemmWorkspace(
                          MF;
                          thread_count=plan.workers,
                          capacity=plan.packed_elements_per_worker,
                      ) :
                      workspace
    _prepare_gemm_workspace!(
        owned_workspace,
        plan.workers,
        plan.packed_elements_per_worker,
    )
    if plan.workers == 1 || jobs <= 1
        _gemm_packed_worker!(
            C, A, B, alpha, beta, plan, owned_workspace, 1, jobs,
        )
        return C
    end

    @sync for worker in 1:plan.workers
        Threads.@spawn _gemm_packed_worker!(
            C,
            A,
            B,
            alpha,
            beta,
            plan,
            owned_workspace,
            worker,
            jobs,
        )
    end
    return C
end

"""
    gemm!(C, A, B, alpha=one(eltype(A)), beta=zero(eltype(A));
          config=KernelConfig(), workspace=nothing)

CPU matrix multiplication specialized for `MultiFloat{T,N}`. The direct route
uses four SIMD row lanes and two output columns. The packed route stores each
B panel in reduction-major order, then reuses every A lane load across a
2- or 4-column register microkernel. The fused Float64x3 route uses the same
four-lane direct layout but replaces `acc += x*y` with the `mulacc_x3` fused
multiply-accumulate network. Output-column panels have disjoint task ownership.
Passing `GemmWorkspace` removes repeated packing-buffer allocation.

The kernel requires one-based indexing and does not support aliasing: `C` must
not share storage with `A` or `B`.
"""
function gemm!(
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    B::AbstractMatrix{MF},
    alpha::MF=one(MF),
    beta::MF=zero(MF);
    config::KernelConfig=KernelConfig(),
    workspace::Union{Nothing,GemmWorkspace{MF}}=nothing,
) where {MF<:MultiFloat}
    m, k = size(A)
    size(B, 1) == k || throw(DimensionMismatch("gemm! inner dimensions differ"))
    n = size(B, 2)
    size(C) == (m, n) || throw(DimensionMismatch("gemm! output dimensions differ"))
    _check_supported(MF)
    Base.require_one_based_indexing(C, A, B)

    plan = gemm_plan(MF, m, k, n, config)
    if plan.strategy === :packed
        return _gemm_packed!(C, A, B, alpha, beta, plan, workspace)
    elseif plan.strategy === :fused
        return _gemm_direct_fused!(C, A, B, alpha, beta, plan)
    end
    return _gemm_direct!(C, A, B, alpha, beta, plan)
end
