# Optional dense A+B packed GEMM specialization.
#
# The established packed route stores B in reduction-major order. For
# strided dense matrices we additionally pack every complete four-row A group
# as one MultiFloatVec. This removes the long column-major stride from
# A[row,k] while preserving the exact ascending-k reduction order used by the
# direct and B-only kernels. Generic AbstractMatrix inputs keep the B-only
# implementation in gemm.jl.

function _pack_a_vec4(
    A::StridedMatrix{MF},
) where {T,N,MF<:MultiFloat{T,N}}
    m, reduction = size(A)
    row_groups = m ÷ 4
    V4 = MultiFloatVec{4,T,N}
    packed = Vector{V4}(undef, row_groups * reduction)
    @inbounds for group in 1:row_groups
        row = 4 * (group - 1) + 1
        offset = (group - 1) * reduction
        for k in 1:reduction
            packed[offset + k] = V4(
                A[row, k],
                A[row + 1, k],
                A[row + 2, k],
                A[row + 3, k],
            )
        end
    end
    return packed, row_groups
end

@inline function _packed_ab_vector_block!(
    C::StridedMatrix{MF},
    packed_a::AbstractVector{V4},
    packed_b::AbstractVector{MF},
    reduction::Int,
    width::Int,
    row_group::Int,
    local_column::Int,
    global_column::Int,
    alpha::MF,
    beta::MF,
    ::Val{NR},
) where {T,N,MF<:MultiFloat{T,N},V4<:MultiFloatVec{4,T,N},NR}
    accumulators = ntuple(_ -> zero(V4), Val(NR))
    a_offset = (row_group - 1) * reduction
    @inbounds for k in 1:reduction
        values = packed_a[a_offset + k]
        b_offset = (k - 1) * width + local_column - 1
        accumulators = ntuple(
            column -> accumulators[column] +
                      values * V4(packed_b[b_offset + column]),
            Val(NR),
        )
    end

    row = 4 * (row_group - 1) + 1
    alpha_vector = V4(alpha)
    beta_vector = V4(beta)
    @inbounds for column in 1:NR
        output_column = global_column + column - 1
        result = alpha_vector * accumulators[column] + beta_vector * V4(
            C[row, output_column],
            C[row + 1, output_column],
            C[row + 2, output_column],
            C[row + 3, output_column],
        )
        for lane in 1:4
            C[row + lane - 1, output_column] = result[lane]
        end
    end
    return nothing
end

function _packed_ab_column_group!(
    C::StridedMatrix{MF},
    A::StridedMatrix{MF},
    packed_a::AbstractVector{V4},
    row_groups::Int,
    packed_b::AbstractVector{MF},
    width::Int,
    local_column::Int,
    global_column::Int,
    alpha::MF,
    beta::MF,
    columns::Val{NR},
) where {T,N,MF<:MultiFloat{T,N},V4<:MultiFloatVec{4,T,N},NR}
    reduction = size(A, 2)
    @inbounds for row_group in 1:row_groups
        _packed_ab_vector_block!(
            C, packed_a, packed_b, reduction, width, row_group,
            local_column, global_column, alpha, beta, columns,
        )
    end

    row = 4 * row_groups + 1
    @inbounds while row <= size(A, 1)
        _packed_gemm_scalar_block!(
            C, A, packed_b, width, row, local_column, global_column,
            alpha, beta, columns,
        )
        row += 1
    end
    return nothing
end

function _packed_ab_panel_nr!(
    C::StridedMatrix{MF},
    A::StridedMatrix{MF},
    packed_a::AbstractVector{V4},
    row_groups::Int,
    packed_b::AbstractVector{MF},
    first_column::Int,
    width::Int,
    alpha::MF,
    beta::MF,
    ::Val{NR},
) where {T,N,MF<:MultiFloat{T,N},V4<:MultiFloatVec{4,T,N},NR}
    local_column = 1
    while local_column + NR - 1 <= width
        _packed_ab_column_group!(
            C, A, packed_a, row_groups, packed_b, width, local_column,
            first_column + local_column - 1, alpha, beta, Val(NR),
        )
        local_column += NR
    end
    while local_column + 1 <= width
        _packed_ab_column_group!(
            C, A, packed_a, row_groups, packed_b, width, local_column,
            first_column + local_column - 1, alpha, beta, Val(2),
        )
        local_column += 2
    end
    if local_column <= width
        _packed_ab_column_group!(
            C, A, packed_a, row_groups, packed_b, width, local_column,
            first_column + local_column - 1, alpha, beta, Val(1),
        )
    end
    return C
end

function _packed_ab_panel!(
    C::StridedMatrix{MF},
    A::StridedMatrix{MF},
    packed_a::AbstractVector{V4},
    row_groups::Int,
    packed_b::AbstractVector{MF},
    first_column::Int,
    width::Int,
    alpha::MF,
    beta::MF,
    micro_columns::Int,
) where {T,N,MF<:MultiFloat{T,N},V4<:MultiFloatVec{4,T,N}}
    if micro_columns == 4
        return _packed_ab_panel_nr!(
            C, A, packed_a, row_groups, packed_b,
            first_column, width, alpha, beta, Val(4),
        )
    elseif micro_columns == 2
        return _packed_ab_panel_nr!(
            C, A, packed_a, row_groups, packed_b,
            first_column, width, alpha, beta, Val(2),
        )
    end
    return _packed_ab_panel_nr!(
        C, A, packed_a, row_groups, packed_b,
        first_column, width, alpha, beta, Val(1),
    )
end

function _gemm_ab_worker!(
    C::StridedMatrix{MF},
    A::StridedMatrix{MF},
    B::StridedMatrix{MF},
    packed_a::AbstractVector{V4},
    row_groups::Int,
    alpha::MF,
    beta::MF,
    plan::GemmPlan,
    workspace::GemmWorkspace{MF},
    worker::Int,
    jobs::Int,
) where {T,N,MF<:MultiFloat{T,N},V4<:MultiFloatVec{4,T,N}}
    buffer = workspace.buffers[worker]
    @inbounds for job in worker:plan.workers:jobs
        first_column = (job - 1) * plan.panel_columns + 1
        last_column = min(job * plan.panel_columns, size(B, 2))
        width = _pack_b_panel!(buffer, B, first_column, last_column)
        _packed_ab_panel!(
            C, A, packed_a, row_groups, buffer, first_column, width,
            alpha, beta, plan.micro_columns,
        )
    end
    return nothing
end

# More-specific dense implementation of the existing packed route. The A pack
# is immutable after construction and shared by disjoint output-panel workers;
# GemmWorkspace continues to own the mutable per-worker B buffers.
function _gemm_packed!(
    C::StridedMatrix{MF},
    A::StridedMatrix{MF},
    B::StridedMatrix{MF},
    alpha::MF,
    beta::MF,
    plan::GemmPlan,
    workspace::Union{Nothing,GemmWorkspace{MF}},
) where {T,N,MF<:MultiFloat{T,N}}
    jobs = cld(size(B, 2), plan.panel_columns)
    owned_workspace = workspace === nothing ?
                      GemmWorkspace(
                          MF;
                          thread_count=plan.workers,
                          capacity=plan.packed_elements_per_worker,
                      ) : workspace
    _prepare_gemm_workspace!(
        owned_workspace, plan.workers, plan.packed_elements_per_worker,
    )

    packed_a, row_groups = _pack_a_vec4(A)
    if plan.workers == 1 || jobs <= 1
        _gemm_ab_worker!(
            C, A, B, packed_a, row_groups,
            alpha, beta, plan, owned_workspace, 1, jobs,
        )
        return C
    end

    @sync for worker in 1:plan.workers
        Threads.@spawn _gemm_ab_worker!(
            C, A, B, packed_a, row_groups,
            alpha, beta, plan, owned_workspace, worker, jobs,
        )
    end
    return C
end
