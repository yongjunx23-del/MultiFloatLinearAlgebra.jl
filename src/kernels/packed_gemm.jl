@inline function _packed_gemm_vector_block!(
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    packed_b::AbstractVector{MF},
    width::Int,
    row::Int,
    local_column::Int,
    global_column::Int,
    alpha::MF,
    beta::MF,
    ::Val{1},
    ::Val{OVERWRITE},
) where {T,N,MF<:MultiFloat{T,N},OVERWRITE}
    V4 = MultiFloatVec{4,T,N}
    accumulator1 = zero(V4)
    reduction = size(A, 2)
    @inbounds for k in 1:reduction
        values = V4(
            A[row, k],
            A[row + 1, k],
            A[row + 2, k],
            A[row + 3, k],
        )
        offset = (k - 1) * width + local_column
        accumulator1 += values * V4(packed_b[offset])
    end
    result1 = if OVERWRITE
        V4(alpha) * accumulator1
    else
        V4(alpha) * accumulator1 + V4(beta) * V4(
            C[row, global_column],
            C[row + 1, global_column],
            C[row + 2, global_column],
            C[row + 3, global_column],
        )
    end
    @inbounds for lane in 1:4
        C[row + lane - 1, global_column] = result1[lane]
    end
    return nothing
end

@inline function _packed_gemm_vector_block!(
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    packed_b::AbstractVector{MF},
    width::Int,
    row::Int,
    local_column::Int,
    global_column::Int,
    alpha::MF,
    beta::MF,
    ::Val{2},
    ::Val{OVERWRITE},
) where {T,N,MF<:MultiFloat{T,N},OVERWRITE}
    V4 = MultiFloatVec{4,T,N}
    accumulator1 = zero(V4)
    accumulator2 = zero(V4)
    reduction = size(A, 2)
    @inbounds for k in 1:reduction
        values = V4(
            A[row, k],
            A[row + 1, k],
            A[row + 2, k],
            A[row + 3, k],
        )
        offset = (k - 1) * width + local_column
        accumulator1 += values * V4(packed_b[offset])
        accumulator2 += values * V4(packed_b[offset + 1])
    end
    alpha_vector = V4(alpha)
    beta_vector = V4(beta)
    result1 = if OVERWRITE
        alpha_vector * accumulator1
    else
        alpha_vector * accumulator1 + beta_vector * V4(
            C[row, global_column],
            C[row + 1, global_column],
            C[row + 2, global_column],
            C[row + 3, global_column],
        )
    end
    result2 = if OVERWRITE
        alpha_vector * accumulator2
    else
        alpha_vector * accumulator2 + beta_vector * V4(
            C[row, global_column + 1],
            C[row + 1, global_column + 1],
            C[row + 2, global_column + 1],
            C[row + 3, global_column + 1],
        )
    end
    @inbounds for lane in 1:4
        C[row + lane - 1, global_column] = result1[lane]
        C[row + lane - 1, global_column + 1] = result2[lane]
    end
    return nothing
end

@inline function _packed_gemm_vector_block!(
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    packed_b::AbstractVector{MF},
    width::Int,
    row::Int,
    local_column::Int,
    global_column::Int,
    alpha::MF,
    beta::MF,
    ::Val{4},
    ::Val{OVERWRITE},
) where {T,N,MF<:MultiFloat{T,N},OVERWRITE}
    V4 = MultiFloatVec{4,T,N}
    accumulator1 = zero(V4)
    accumulator2 = zero(V4)
    accumulator3 = zero(V4)
    accumulator4 = zero(V4)
    reduction = size(A, 2)
    @inbounds for k in 1:reduction
        values = V4(
            A[row, k],
            A[row + 1, k],
            A[row + 2, k],
            A[row + 3, k],
        )
        offset = (k - 1) * width + local_column
        accumulator1 += values * V4(packed_b[offset])
        accumulator2 += values * V4(packed_b[offset + 1])
        accumulator3 += values * V4(packed_b[offset + 2])
        accumulator4 += values * V4(packed_b[offset + 3])
    end
    alpha_vector = V4(alpha)
    beta_vector = V4(beta)
    result1 = if OVERWRITE
        alpha_vector * accumulator1
    else
        alpha_vector * accumulator1 + beta_vector * V4(
            C[row, global_column],
            C[row + 1, global_column],
            C[row + 2, global_column],
            C[row + 3, global_column],
        )
    end
    result2 = if OVERWRITE
        alpha_vector * accumulator2
    else
        alpha_vector * accumulator2 + beta_vector * V4(
            C[row, global_column + 1],
            C[row + 1, global_column + 1],
            C[row + 2, global_column + 1],
            C[row + 3, global_column + 1],
        )
    end
    result3 = if OVERWRITE
        alpha_vector * accumulator3
    else
        alpha_vector * accumulator3 + beta_vector * V4(
            C[row, global_column + 2],
            C[row + 1, global_column + 2],
            C[row + 2, global_column + 2],
            C[row + 3, global_column + 2],
        )
    end
    result4 = if OVERWRITE
        alpha_vector * accumulator4
    else
        alpha_vector * accumulator4 + beta_vector * V4(
            C[row, global_column + 3],
            C[row + 1, global_column + 3],
            C[row + 2, global_column + 3],
            C[row + 3, global_column + 3],
        )
    end
    @inbounds for lane in 1:4
        output_row = row + lane - 1
        C[output_row, global_column] = result1[lane]
        C[output_row, global_column + 1] = result2[lane]
        C[output_row, global_column + 2] = result3[lane]
        C[output_row, global_column + 3] = result4[lane]
    end
    return nothing
end

@inline function _packed_gemm_scalar_block!(
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    packed_b::AbstractVector{MF},
    width::Int,
    row::Int,
    local_column::Int,
    global_column::Int,
    alpha::MF,
    beta::MF,
    ::Val{1},
    ::Val{OVERWRITE},
) where {MF<:MultiFloat,OVERWRITE}
    accumulator1 = zero(MF)
    reduction = size(A, 2)
    @inbounds for k in 1:reduction
        offset = (k - 1) * width + local_column
        accumulator1 += A[row, k] * packed_b[offset]
    end
    C[row, global_column] = OVERWRITE ? alpha * accumulator1 :
        alpha * accumulator1 + beta * C[row, global_column]
    return nothing
end

@inline function _packed_gemm_scalar_block!(
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    packed_b::AbstractVector{MF},
    width::Int,
    row::Int,
    local_column::Int,
    global_column::Int,
    alpha::MF,
    beta::MF,
    ::Val{2},
    ::Val{OVERWRITE},
) where {MF<:MultiFloat,OVERWRITE}
    accumulator1 = zero(MF)
    accumulator2 = zero(MF)
    reduction = size(A, 2)
    @inbounds for k in 1:reduction
        value = A[row, k]
        offset = (k - 1) * width + local_column
        accumulator1 += value * packed_b[offset]
        accumulator2 += value * packed_b[offset + 1]
    end
    C[row, global_column] = OVERWRITE ? alpha * accumulator1 :
        alpha * accumulator1 + beta * C[row, global_column]
    C[row, global_column + 1] = OVERWRITE ? alpha * accumulator2 :
        alpha * accumulator2 + beta * C[row, global_column + 1]
    return nothing
end

@inline function _packed_gemm_scalar_block!(
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    packed_b::AbstractVector{MF},
    width::Int,
    row::Int,
    local_column::Int,
    global_column::Int,
    alpha::MF,
    beta::MF,
    ::Val{4},
    ::Val{OVERWRITE},
) where {MF<:MultiFloat,OVERWRITE}
    accumulator1 = zero(MF)
    accumulator2 = zero(MF)
    accumulator3 = zero(MF)
    accumulator4 = zero(MF)
    reduction = size(A, 2)
    @inbounds for k in 1:reduction
        value = A[row, k]
        offset = (k - 1) * width + local_column
        accumulator1 += value * packed_b[offset]
        accumulator2 += value * packed_b[offset + 1]
        accumulator3 += value * packed_b[offset + 2]
        accumulator4 += value * packed_b[offset + 3]
    end
    C[row, global_column] = OVERWRITE ? alpha * accumulator1 :
        alpha * accumulator1 + beta * C[row, global_column]
    C[row, global_column + 1] = OVERWRITE ? alpha * accumulator2 :
        alpha * accumulator2 + beta * C[row, global_column + 1]
    C[row, global_column + 2] = OVERWRITE ? alpha * accumulator3 :
        alpha * accumulator3 + beta * C[row, global_column + 2]
    C[row, global_column + 3] = OVERWRITE ? alpha * accumulator4 :
        alpha * accumulator4 + beta * C[row, global_column + 3]
    return nothing
end
