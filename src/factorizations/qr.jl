struct MFQR{
    MF<:MultiFloat,
    M<:AbstractMatrix{MF},
    T<:AbstractVector{MF},
    P<:AbstractVector{Int},
} <: AbstractMFFactorization{MF}
    factors::M
    tau::T
    permutation::P
    permutation_cycle_leaders::Vector{Int}
    info::Int
end

factor_kind(::MFQR) = :qr
factor_status(F::MFQR) = F.info
factor_matrix(F::MFQR) = F.factors

function _prepare_qr_metadata!(
    ::Type{MF},
    reflector_count::Int,
    column_count::Int,
    workspace::Union{Nothing,MFWorkspace{MF}},
) where {MF<:MultiFloat}
    if workspace === nothing
        return zeros(MF, reflector_count), collect(1:column_count)
    end
    _acquire_factor_workspace!(
        workspace, max(reflector_count, column_count),
    )
    tau = @view workspace.qr_tau[1:reflector_count]
    permutation = @view workspace.qr_permutation[1:column_count]
    fill!(tau, zero(MF))
    @inbounds for column in 1:column_count
        permutation[column] = column
    end
    return tau, permutation
end

@inline _owned_qr_metadata(tau, permutation, ::Nothing) = (tau, permutation)
@inline _owned_qr_metadata(tau, permutation, ::MFWorkspace) =
    (copy(tau), copy(permutation))

function _qr_permutation_cycle_leaders!(permutation::AbstractVector{Int})
    leaders = Int[]
    @inbounds for start in eachindex(permutation)
        permutation[start] > 0 || continue
        current = start
        cycle_length = 0
        while permutation[current] > 0
            next = permutation[current]
            permutation[current] = -next
            current = next
            cycle_length += 1
        end
        cycle_length > 1 && push!(leaders, start)
    end
    @inbounds for index in eachindex(permutation)
        permutation[index] = -permutation[index]
    end
    return leaders
end

"""
    factor_permutation(F::MFQR) -> Vector{Int}

Return the column permutation `p` satisfying `A[:, p] = Q * R`. The returned
vector is a copy and may be mutated by the caller.
"""
function factor_permutation(F::MFQR)
    return copy(F.permutation)
end

"""
    factor_rdiag(F::MFQR) -> Vector

Return a copy of the signed diagonal of the compactly stored `R` factor.
"""
function factor_rdiag(F::MFQR{MF}) where {MF<:MultiFloat}
    diagonal_count = min(size(F.factors)...)
    diagonal = Vector{MF}(undef, diagonal_count)
    @inbounds for index in 1:diagonal_count
        diagonal[index] = F.factors[index, index]
    end
    return diagonal
end

function _qr_column_norm(
    A::AbstractMatrix{MF},
    first_row::Int,
    column::Int,
) where {MF<:MultiFloat}
    scale, scaled_sum = _qr_column_norm_state(A, first_row, column)
    return iszero(scale) ? zero(MF) : scale * sqrt(scaled_sum)
end

function _qr_column_norm_state(
    A::AbstractMatrix{MF},
    first_row::Int,
    column::Int,
) where {MF<:MultiFloat}
    scale = zero(MF)
    scaled_sum = one(MF)
    nonzero_seen = false
    @inbounds for row in first_row:size(A, 1)
        value = abs(A[row, column])
        if !iszero(value)
            if !nonzero_seen
                scale = value
                scaled_sum = one(MF)
                nonzero_seen = true
            elseif value > scale
                ratio = scale / value
                scaled_sum = one(MF) + scaled_sum * ratio * ratio
                scale = value
            else
                ratio = value / scale
                scaled_sum += ratio * ratio
            end
        end
    end
    return scale, scaled_sum
end

function _prepare_qr_norm_state(
    ::Type{MF},
    column_count::Int,
    workspace::Union{Nothing,MFWorkspace{MF}},
) where {MF<:MultiFloat}
    if workspace === nothing
        return (
            Vector{MF}(undef, column_count),
            Vector{MF}(undef, column_count),
            Vector{Bool}(undef, column_count),
        )
    end
    return (
        workspace.qr_norm_scale,
        workspace.qr_norm_sum,
        workspace.qr_norm_dirty,
    )
end

function _qr_initialize_norm_state!(
    A::AbstractMatrix{MF},
    scale::AbstractVector{MF},
    scaled_sum::AbstractVector{MF},
    dirty::AbstractVector{Bool},
) where {MF<:MultiFloat}
    @inbounds for column in axes(A, 2)
        scale[column], scaled_sum[column] =
            _qr_column_norm_state(A, 1, column)
        dirty[column] = false
    end
    return nothing
end

@inline function _qr_norm_from_state(
    scale::AbstractVector{MF},
    scaled_sum::AbstractVector{MF},
    column::Int,
) where {MF<:MultiFloat}
    current_scale = scale[column]
    return iszero(current_scale) ? zero(MF) :
           current_scale * sqrt(scaled_sum[column])
end

function _qr_recompute_norm!(
    A::AbstractMatrix{MF},
    scale::AbstractVector{MF},
    scaled_sum::AbstractVector{MF},
    dirty::AbstractVector{Bool},
    first_row::Int,
    column::Int,
) where {MF<:MultiFloat}
    current_scale, current_sum =
        _qr_column_norm_state(A, first_row, column)
    scale[column] = current_scale
    scaled_sum[column] = current_sum
    dirty[column] = false
    return iszero(current_scale) ? zero(MF) :
           current_scale * sqrt(current_sum)
end

function _qr_swap_columns!(A::AbstractMatrix, first::Int, second::Int)
    first == second && return nothing
    @inbounds for row in axes(A, 1)
        A[row, first], A[row, second] = A[row, second], A[row, first]
    end
    return nothing
end

function _qr_select_pivot_hybrid!(
    A::AbstractMatrix{MF},
    permutation::AbstractVector{Int},
    step::Int,
    scale::AbstractVector{MF},
    scaled_sum::AbstractVector{MF},
    dirty::AbstractVector{Bool},
    margin::MF,
) where {MF<:MultiFloat}
    pivot = step
    pivot_norm = _qr_recompute_norm!(
        A, scale, scaled_sum, dirty, step, step,
    )
    @inbounds for column in (step + 1):size(A, 2)
        candidate_norm = _qr_norm_from_state(scale, scaled_sum, column)
        if dirty[column] || iszero(pivot_norm) ||
           candidate_norm >= (one(MF) - margin) * pivot_norm
            candidate_norm = _qr_recompute_norm!(
                A, scale, scaled_sum, dirty, step, column,
            )
        end
        if candidate_norm > pivot_norm ||
           (candidate_norm == pivot_norm &&
            permutation[column] < permutation[pivot])
            pivot = column
            pivot_norm = candidate_norm
        end
    end
    return pivot
end

# Exact column norm with the current panel's delayed prefix corrections
# (DLAQPS).  The bottom update is not applied yet, so the effective column
# value at row r is A[r, c] - sum_prior A[r, b+prior-1] * F[prior, c].
function _qr_delayed_column_norm_state(
    A::AbstractMatrix{MF},
    Ftranspose::Union{Nothing,AbstractMatrix{MF}},
    block_start::Int,
    prior_count::Int,
    first_row::Int,
    column::Int,
) where {MF<:MultiFloat}
    scale = zero(MF)
    scaled_sum = one(MF)
    nonzero_seen = false
    @inbounds for row in first_row:size(A, 1)
        value = A[row, column]
        if Ftranspose !== nothing
            for prior in 1:prior_count
                value -= A[row, block_start + prior - 1] *
                         Ftranspose[prior, column]
            end
        end
        magnitude = abs(value)
        if !iszero(magnitude)
            if !nonzero_seen
                scale = magnitude
                scaled_sum = one(MF)
                nonzero_seen = true
            elseif magnitude > scale
                ratio = scale / magnitude
                scaled_sum = one(MF) + scaled_sum * ratio * ratio
                scale = magnitude
            else
                ratio = magnitude / scale
                scaled_sum += ratio * ratio
            end
        end
    end
    return scale, scaled_sum
end

function _qr_update_norm_state!(
    A::AbstractMatrix{MF},
    tau::MF,
    step::Int,
    scale::AbstractVector{MF},
    scaled_sum::AbstractVector{MF},
    dirty::AbstractVector{Bool},
    reliability_floor::MF,
    Ftranspose::Union{Nothing,AbstractMatrix{MF}},
    block_start::Int,
    prior_count::Int,
) where {MF<:MultiFloat}
    iszero(tau) && return nothing
    @inbounds for column in (step + 1):size(A, 2)
        dirty[column] && continue
        current_scale = scale[column]
        removed = abs(A[step, column])
        if iszero(current_scale)
            dirty[column] = !iszero(removed)
            continue
        end
        ratio = removed / current_scale
        new_sum = scaled_sum[column] - ratio * ratio
        if !isfinite(new_sum) || new_sum <= reliability_floor
            # Downdate overshoot: the stored scale/sum no longer describe the
            # column after the delayed prefix. Rebuild this column's exact
            # norm inline (including Ftranspose corrections) instead of
            # forcing a full panel break and O(rows*columns) rebuild.
            scale[column], scaled_sum[column] = _qr_delayed_column_norm_state(
                A, Ftranspose, block_start, prior_count, step + 1, column,
            )
            dirty[column] = false
        else
            scaled_sum[column] = new_sum
        end
    end
    return nothing
end

const _QR_BLOCK_SIZE = 16
const _QR_BLOCK_WORK_CROSSOVER = 16_384

@inline function _qr_use_blocked_panel(
    rows::Int,
    columns::Int,
    reflector_count::Int,
)
    return reflector_count >= _QR_BLOCK_SIZE &&
           Int128(rows) * Int128(columns) >=
           Int128(_QR_BLOCK_WORK_CROSSOVER)
end

function _prepare_qr_block_scratch!(
    ::Type{MF},
    block_size::Int,
    column_count::Int,
    workspace::Union{Nothing,MFWorkspace{MF}},
) where {MF<:MultiFloat}
    if workspace === nothing
        return (
            zeros(MF, block_size, column_count),
            Vector{MF}(undef, block_size),
        )
    end
    current_rows, current_columns = size(workspace.qr_ftranspose)
    if current_rows < block_size || current_columns < column_count
        workspace.qr_ftranspose = Matrix{MF}(
            undef,
            max(current_rows, block_size),
            max(current_columns, column_count),
        )
    end
    length(workspace.qr_aux) < block_size &&
        resize!(workspace.qr_aux, block_size)
    return (
        @view(workspace.qr_ftranspose[1:block_size, 1:column_count]),
        @view(workspace.qr_aux[1:block_size]),
    )
end

@inline function _qr_select_pivot_downdated(
    permutation::AbstractVector{Int},
    step::Int,
    scale::AbstractVector{MF},
    scaled_sum::AbstractVector{MF},
) where {MF<:MultiFloat}
    pivot = step
    pivot_norm = _qr_norm_from_state(scale, scaled_sum, step)
    @inbounds for column in (step + 1):length(permutation)
        candidate_norm = _qr_norm_from_state(scale, scaled_sum, column)
        if candidate_norm > pivot_norm ||
           (candidate_norm == pivot_norm &&
            permutation[column] < permutation[pivot])
            pivot = column
            pivot_norm = candidate_norm
        end
    end
    return pivot
end

function _rrqr_unblocked!(
    A::AbstractMatrix{MF},
    tau::AbstractVector{MF},
    permutation::AbstractVector{Int},
    norm_scale::AbstractVector{MF},
    norm_sum::AbstractVector{MF},
    norm_dirty::AbstractVector{Bool},
    norm_margin::MF,
    norm_reliability_floor::MF,
    threads::Int,
) where {MF<:MultiFloat}
    reflector_count = length(tau)
    @inbounds for step in 1:reflector_count
        pivot = _qr_select_pivot_hybrid!(
            A, permutation, step, norm_scale, norm_sum, norm_dirty,
            norm_margin,
        )
        _qr_swap_columns!(A, step, pivot)
        permutation[step], permutation[pivot] =
            permutation[pivot], permutation[step]
        norm_scale[step], norm_scale[pivot] =
            norm_scale[pivot], norm_scale[step]
        norm_sum[step], norm_sum[pivot] =
            norm_sum[pivot], norm_sum[step]
        norm_dirty[step], norm_dirty[pivot] =
            norm_dirty[pivot], norm_dirty[step]
        tau[step] = _qr_make_reflector!(A, step)
        _qr_apply_reflector_to_trailing!(A, tau[step], step, threads)
        _qr_update_norm_state!(
            A, tau[step], step, norm_scale, norm_sum, norm_dirty,
            norm_reliability_floor, nothing, step, 0,
        )
    end
    return nothing
end

@inline function _qr_has_unreliable_trailing_norm(
    dirty::AbstractVector{Bool},
    first_column::Int,
    last_column::Int=length(dirty),
)
    first_column > last_column && return false
    last = min(last_column, length(dirty))
    first_column > last && return false
    @inbounds for column in first_column:last
        dirty[column] && return true
    end
    return false
end

# Exact per-column norm rebuilds are independent. Each
# column keeps the scalar `_qr_recompute_norm!` row order; workers only split
# columns, so no reduction arithmetic or pivot state changes.
function _qr_recompute_norm_columns_parallel!(
    A::AbstractMatrix{MF},
    scale::AbstractVector{MF},
    scaled_sum::AbstractVector{MF},
    dirty::AbstractVector{Bool},
    first_row::Int,
    first_column::Int,
    last_column::Int,
    thread_count::Int,
) where {MF<:MultiFloat}
    first_column > last_column && return nothing
    workers = max(1, min(thread_count, Threads.nthreads(), last_column - first_column + 1))
    if workers == 1
        @inbounds for column in first_column:last_column
            _qr_recompute_norm!(A, scale, scaled_sum, dirty, first_row, column)
        end
        return nothing
    end
    jobs = last_column - first_column + 1
    chunk = cld(jobs, workers)
    @sync for worker in 1:workers
        first = first_column + (worker - 1) * chunk
        last = min(first + chunk - 1, last_column)
        first <= last || continue
        Threads.@spawn begin
            @inbounds for column in first:last
                _qr_recompute_norm!(A, scale, scaled_sum, dirty, first_row, column)
            end
        end
    end
    return nothing
end

function _rrqr_blocked!(
    A::AbstractMatrix{MF},
    tau::AbstractVector{MF},
    permutation::AbstractVector{Int},
    norm_scale::AbstractVector{MF},
    norm_sum::AbstractVector{MF},
    norm_dirty::AbstractVector{Bool},
    norm_reliability_floor::MF,
    threads::Int,
    workspace::Union{Nothing,MFWorkspace{MF}},
) where {MF<:MultiFloat}
    rows, columns = size(A)
    reflector_count = length(tau)
    block_size = min(_QR_BLOCK_SIZE, reflector_count)
    Ftranspose, auxiliary = _prepare_qr_block_scratch!(
        MF, block_size, columns, workspace,
    )
    kernel_config = KernelConfig(thread_count=max(threads, 1))

    block_start = 1
    while block_start <= reflector_count
        requested = min(
            block_size, reflector_count - block_start + 1,
        )
        fill!(Ftranspose, zero(MF))
        actual = 0

        for local_step in 1:requested
            step = block_start + local_step - 1
            pivot = _qr_select_pivot_downdated(
                permutation, step, norm_scale, norm_sum,
            )

            _qr_swap_columns!(A, step, pivot)
            @inbounds for prior in 1:(local_step - 1)
                Ftranspose[prior, step], Ftranspose[prior, pivot] =
                    Ftranspose[prior, pivot], Ftranspose[prior, step]
            end
            permutation[step], permutation[pivot] =
                permutation[pivot], permutation[step]
            norm_scale[step], norm_scale[pivot] =
                norm_scale[pivot], norm_scale[step]
            norm_sum[step], norm_sum[pivot] =
                norm_sum[pivot], norm_sum[step]
            norm_dirty[step], norm_dirty[pivot] =
                norm_dirty[pivot], norm_dirty[step]

            # Apply the delayed prefix to the selected column. Swapping the
            # corresponding F entries above makes this correct even when a
            # column leaves and later re-enters the current panel.
            if local_step > 1
                @inbounds for row in step:rows
                    correction = zero(MF)
                    for prior in 1:(local_step - 1)
                        correction +=
                            A[row, block_start + prior - 1] *
                            Ftranspose[prior, step]
                    end
                    A[row, step] -= correction
                end
            end

            tau[step] = _qr_make_reflector!(A, step)
            diagonal = A[step, step]
            A[step, step] = one(MF)

            # DLAQPS F recurrence. The new row records the delayed action on
            # every column to the right while the panel itself stays narrow.
            if step < columns
                gemv!(
                    view(Ftranspose, local_step, (step + 1):columns),
                    view(A, step:rows, (step + 1):columns),
                    view(A, step:rows, step),
                    tau[step], zero(MF);
                    trans=:T,
                    config=kernel_config,
                )
            end
            if local_step > 1
                gemv!(
                    view(auxiliary, 1:(local_step - 1)),
                    view(
                        A,
                        step:rows,
                        block_start:(step - 1),
                    ),
                    view(A, step:rows, step),
                    -tau[step], zero(MF);
                    trans=:T,
                    config=kernel_config,
                )
                @inbounds for column in (step + 1):columns
                    correction = zero(MF)
                    for prior in 1:(local_step - 1)
                        correction +=
                            Ftranspose[prior, column] * auxiliary[prior]
                    end
                    Ftranspose[local_step, column] += correction
                end
            end

            # Materialize the current R row; rows below it remain delayed
            # until the level-3 update at the panel boundary.
            @inbounds for column in (step + 1):columns
                correction = zero(MF)
                for panel_column in 1:local_step
                    correction +=
                        A[step, block_start + panel_column - 1] *
                        Ftranspose[panel_column, column]
                end
                A[step, column] -= correction
            end
            A[step, step] = diagonal

            _qr_update_norm_state!(
                A, tau[step], step, norm_scale, norm_sum, norm_dirty,
                norm_reliability_floor, Ftranspose, block_start,
                local_step,
            )
            actual = local_step

            # A cancellation-sensitive norm cannot be recomputed while the
            # bottom update is delayed. End this panel, apply it, then rebuild
            # every trailing norm exactly from the updated matrix.
            _qr_has_unreliable_trailing_norm(norm_dirty, step + 1, columns) &&
                break
        end

        block_end = block_start + actual - 1
        if block_end < rows && block_end < columns
            gemm!(
                view(
                    A,
                    (block_end + 1):rows,
                    (block_end + 1):columns,
                ),
                view(A, (block_end + 1):rows, block_start:block_end),
                view(
                    Ftranspose,
                    1:actual,
                    (block_end + 1):columns,
                ),
                -one(MF), one(MF);
                config=kernel_config,
                workspace=workspace,
            )
        end
        _qr_recompute_norm_columns_parallel!(
            A, norm_scale, norm_sum, norm_dirty,
            block_end + 1, block_end + 1, columns, threads,
        )
        block_start = block_end + 1
    end
    return nothing
end

function _qr_make_reflector!(
    A::AbstractMatrix{MF},
    step::Int,
) where {MF<:MultiFloat}
    norm_value = _qr_column_norm(A, step, step)
    iszero(norm_value) && return zero(MF)

    alpha = A[step, step]
    beta = alpha >= zero(MF) ? -norm_value : norm_value
    leading = alpha - beta
    tau = (beta - alpha) / beta
    @inbounds for row in (step + 1):size(A, 1)
        A[row, step] /= leading
    end
    A[step, step] = beta
    return tau
end

@inline function _qr_apply_reflector_to_column!(
    A::AbstractMatrix{MF},
    tau::MF,
    step::Int,
    column::Int,
) where {MF<:MultiFloat}
    @inbounds begin
        projection = A[step, column]
        for row in (step + 1):size(A, 1)
            projection += A[row, step] * A[row, column]
        end
        projection *= tau
        A[step, column] -= projection
        for row in (step + 1):size(A, 1)
            A[row, column] -= A[row, step] * projection
        end
    end
    return nothing
end

function _qr_apply_reflector_to_trailing!(
    A::AbstractMatrix{MF},
    tau::MF,
    step::Int,
    threads::Int=Threads.nthreads(),
) where {MF<:MultiFloat}
    iszero(tau) && return nothing
    columns = size(A, 2)
    trailing_columns = columns - step
    trailing_rows = size(A, 1) - step
    # The per-column updates are independent (each reads the fixed
    # reflector column and writes only its own column), so the dominant
    # O(rows x trailing_columns) trailing update parallelizes across
    # columns with bit-identical results.  Keep the serial path for
    # small trailing panels where task-dispatch overhead dominates.
    if trailing_columns * trailing_rows >= 16384 && threads > 1
        Threads.@threads for column in (step + 1):columns
            _qr_apply_reflector_to_column!(A, tau, step, column)
        end
    else
        for column in (step + 1):columns
            _qr_apply_reflector_to_column!(A, tau, step, column)
        end
    end
    return nothing
end

"""
    rrqr!(A; check=true, workspace=nothing)

Compute a deterministic column-pivoted Householder QR factorization in place.
The result satisfies `A_original[:, factor_permutation(F)] = Q * R`.
Householder vectors are stored below the diagonal of `factor_matrix(F)` and
`R` is stored on and above it.

Column norms use scale-safe downdates. Small problems use the direct hybrid
path; larger panels use a DLAQPS-style delayed update and finish each panel
with a level-3 matrix multiplication. A panel ends early whenever a norm
downdate loses reliability, after which all trailing norms are recomputed from
the fully updated matrix. Exact norm ties choose the smallest original column
index. Rank deficiency is a successful factorization;
[`numerical_rank`](@ref) applies a caller-supplied threshold after
factorization. No fallback or rank policy is performed here.

With `workspace=MFWorkspace(T)`, temporary metadata storage is reused during
factorization and the returned factor owns its reflector coefficients and
permutation.
"""
function rrqr!(
    A::AbstractMatrix{MF};
    check::Bool=true,
    workspace::Union{Nothing,MFWorkspace{MF}}=nothing,
    threads::Int=Threads.nthreads(),
) where {MF<:MultiFloat}
    _check_supported(MF)
    Base.require_one_based_indexing(A)
    m, n = size(A)
    reflector_count = min(m, n)
    finite_input = _all_finite(A)
    !finite_input && check &&
        throw(DomainError(A, "rrqr!: input matrix contains non-finite entries"))
    tau, permutation = _prepare_qr_metadata!(
        MF, reflector_count, n, workspace,
    )

    if !finite_input
        owned_tau, owned_permutation =
            _owned_qr_metadata(tau, permutation, workspace)
        cycle_leaders = _qr_permutation_cycle_leaders!(owned_permutation)
        return MFQR{MF,typeof(A),typeof(owned_tau),typeof(owned_permutation)}(
            A, owned_tau, owned_permutation, cycle_leaders, -1,
        )
    end

    norm_scale, norm_sum, norm_dirty =
        _prepare_qr_norm_state(MF, n, workspace)
    _qr_initialize_norm_state!(A, norm_scale, norm_sum, norm_dirty)
    norm_margin = sqrt(eps(MF))
    norm_reliability_floor = MF(16) * eps(MF)

    if _qr_use_blocked_panel(m, n, reflector_count)
        _rrqr_blocked!(
            A, tau, permutation,
            norm_scale, norm_sum, norm_dirty,
            norm_reliability_floor, threads, workspace,
        )
    else
        _rrqr_unblocked!(
            A, tau, permutation,
            norm_scale, norm_sum, norm_dirty,
            norm_margin, norm_reliability_floor, threads,
        )
    end
    owned_tau, owned_permutation =
        _owned_qr_metadata(tau, permutation, workspace)
    cycle_leaders = _qr_permutation_cycle_leaders!(owned_permutation)
    return MFQR{MF,typeof(A),typeof(owned_tau),typeof(owned_permutation)}(
        A, owned_tau, owned_permutation, cycle_leaders, 0,
    )
end

"""
    qr!(A; check=true, workspace=nothing)

Compute a deterministic non-pivoted Householder QR factorization in place.
The result satisfies `A = Q * R`.  Householder vectors are stored below the
diagonal of `factor_matrix(F)` and `R` is stored on and above it; the
permutation is the identity.

This is the fixed-order counterpart to [`rrqr!`](@ref): it performs no column
pivoting, so it does not reveal numerical rank.  Callers that need
rank-revealing behavior must use `rrqr!`, or verify the fixed-order result
themselves (e.g. by checking the `R` diagonal against a caller-supplied
threshold).

With `workspace=MFWorkspace(T)`, temporary metadata storage is reused during
factorization and the returned factor owns its reflector coefficients and
permutation.
"""
function qr!(
    A::AbstractMatrix{MF};
    check::Bool=true,
    workspace::Union{Nothing,MFWorkspace{MF}}=nothing,
    threads::Int=Threads.nthreads(),
) where {MF<:MultiFloat}
    _check_supported(MF)
    Base.require_one_based_indexing(A)
    m, n = size(A)
    reflector_count = min(m, n)
    finite_input = _all_finite(A)
    !finite_input && check &&
        throw(DomainError(A, "qr!: input matrix contains non-finite entries"))
    tau, permutation = _prepare_qr_metadata!(
        MF, reflector_count, n, workspace,
    )

    if !finite_input
        owned_tau, owned_permutation =
            _owned_qr_metadata(tau, permutation, workspace)
        cycle_leaders = _qr_permutation_cycle_leaders!(owned_permutation)
        return MFQR{MF,typeof(A),typeof(owned_tau),typeof(owned_permutation)}(
            A, owned_tau, owned_permutation, cycle_leaders, -1,
        )
    end

    @inbounds for step in 1:reflector_count
        tau[step] = _qr_make_reflector!(A, step)
        _qr_apply_reflector_to_trailing!(A, tau[step], step, threads)
    end
    owned_tau, owned_permutation =
        _owned_qr_metadata(tau, permutation, workspace)
    cycle_leaders = _qr_permutation_cycle_leaders!(owned_permutation)
    return MFQR{MF,typeof(A),typeof(owned_tau),typeof(owned_permutation)}(
        A, owned_tau, owned_permutation, cycle_leaders, 0,
    )
end

"""
    qr!(A, permutation; check=true, workspace=nothing, threads=nthreads())

Factor the columns of `A` in a caller-supplied order.

`permutation[i] = j` means factor column `i` is original column `j`; the
columns of `A` are physically reordered in place (the same destructive
contract as the unpivoted entry point), and the returned factor records
`permutation` so pivoted-style solves and `factor_permutation` consumers see
the original column identity. Deterministic fixed-order route for callers
that already know a good column order -- no norm search, no swaps.
"""
function qr!(
    A::AbstractMatrix{MF},
    permutation::AbstractVector{<:Integer};
    check::Bool=true,
    workspace::Union{Nothing,MFWorkspace{MF}}=nothing,
    threads::Int=Threads.nthreads(),
) where {MF<:MultiFloat}
    _check_supported(MF)
    Base.require_one_based_indexing(A)
    m, n = size(A)
    length(permutation) == n || throw(DimensionMismatch(
        "permutation length $(length(permutation)) does not match $n columns",
    ))
    seen = falses(n)
    @inbounds for j in eachindex(permutation)
        column = Int(permutation[j])
        1 <= column <= n || throw(ArgumentError(
            "permutation entry $column is outside 1:$n",
        ))
        seen[column] && throw(ArgumentError(
            "permutation repeats column $column",
        ))
        seen[column] = true
    end
    # Physically reorder the columns into the requested order, then run the
    # unpivoted core on the reordered panel. The factor's permutation is the
    # caller's order, so pivoted-style solves recover original identities.
    ordered = A[:, Int.(permutation)]
    copyto!(A, ordered)
    factor = qr!(A; check=check, workspace=workspace, threads=threads)
    # MFQR is immutable and the unpivoted core records the identity
    # permutation; rebuild the factor with the caller's column identities so
    # pivoted-style solves recover original columns.
    return MFQR(
        factor.factors,
        factor.tau,
        Int.(permutation),
        _qr_permutation_cycle_leaders!(Int.(permutation)),
        factor.info,
    )
end

"""
    numerical_rank(F::MFQR; atol=0, rtol=0) -> Int

Evaluate the leading numerical rank using the caller's nonnegative absolute
and relative thresholds. Diagonal `i` is accepted while
`abs(R[i,i]) > max(atol, rtol * maximum(abs.(diag(R))))`. The zero defaults
mean exact nonzero rank; MFLA does not select a solver rank tolerance.
"""
function numerical_rank(
    F::MFQR{MF};
    atol::Real=zero(MF),
    rtol::Real=zero(MF),
) where {MF<:MultiFloat}
    issuccess(F) || throw(ArgumentError("numerical_rank requires a successful QR factorization"))
    absolute_tolerance = MF(atol)
    relative_tolerance = MF(rtol)
    isfinite(absolute_tolerance) && absolute_tolerance >= zero(MF) ||
        throw(ArgumentError("atol must be finite and nonnegative"))
    isfinite(relative_tolerance) && relative_tolerance >= zero(MF) ||
        throw(ArgumentError("rtol must be finite and nonnegative"))

    diagonal_count = min(size(F.factors)...)
    largest = zero(MF)
    @inbounds for index in 1:diagonal_count
        largest = max(largest, abs(F.factors[index, index]))
    end
    threshold = max(absolute_tolerance, relative_tolerance * largest)
    rank = 0
    @inbounds for index in 1:diagonal_count
        abs(F.factors[index, index]) > threshold || break
        rank += 1
    end
    return rank
end

function _apply_q_reflector!(
    destination::AbstractVector{MF},
    F::MFQR{MF},
    step::Int,
) where {MF<:MultiFloat}
    tau = F.tau[step]
    iszero(tau) && return nothing
    projection = destination[step]
    @inbounds for row in (step + 1):size(F.factors, 1)
        projection += F.factors[row, step] * destination[row]
    end
    projection *= tau
    destination[step] -= projection
    @inbounds for row in (step + 1):size(F.factors, 1)
        destination[row] -= F.factors[row, step] * projection
    end
    return nothing
end

function _apply_q_reflector!(
    destination::AbstractMatrix{MF},
    F::MFQR{MF},
    step::Int,
) where {MF<:MultiFloat}
    tau = F.tau[step]
    iszero(tau) && return nothing
    @inbounds for column in axes(destination, 2)
        projection = destination[step, column]
        for row in (step + 1):size(F.factors, 1)
            projection += F.factors[row, step] * destination[row, column]
        end
        projection *= tau
        destination[step, column] -= projection
        for row in (step + 1):size(F.factors, 1)
            destination[row, column] -=
                F.factors[row, step] * projection
        end
    end
    return nothing
end

"""
    apply_q!(B, F::MFQR; trans=:N)

Overwrite a vector or matrix `B` with `Q*B` (`trans=:N`) or `Q'*B`
(`trans=:T`) using the compact Householder representation. `B` must have one
row per row of the factorized matrix and must not alias `factor_matrix(F)`.
"""
function apply_q!(
    destination::Union{AbstractVector{MF},AbstractMatrix{MF}},
    F::MFQR{MF};
    trans::Symbol=:N,
) where {MF<:MultiFloat}
    issuccess(F) || throw(ArgumentError("apply_q! requires a successful QR factorization"))
    trans in (:N, :T) || throw(ArgumentError("trans must be :N or :T"))
    Base.require_one_based_indexing(destination, F.factors)
    size(destination, 1) == size(F.factors, 1) ||
        throw(DimensionMismatch("apply_q! destination row count differs"))
    Base.mightalias(destination, F.factors) &&
        throw(ArgumentError("apply_q! destination must not alias QR storage"))

    steps = trans === :N ? reverse(eachindex(F.tau)) : eachindex(F.tau)
    for step in steps
        _apply_q_reflector!(destination, F, step)
    end
    return destination
end

function _check_r_solve(F::MFQR, destination, rank::Int, trans::Symbol)
    issuccess(F) || throw(ArgumentError("solve_r! requires a successful QR factorization"))
    trans in (:N, :T) || throw(ArgumentError("trans must be :N or :T"))
    maximum_rank = min(size(F.factors)...)
    0 <= rank <= maximum_rank ||
        throw(ArgumentError("rank must be between zero and $maximum_rank"))
    size(destination, 1) == rank ||
        throw(DimensionMismatch("solve_r! destination row count must equal rank"))
    @inbounds for index in 1:rank
        iszero(F.factors[index, index]) &&
            throw(LinearAlgebra.SingularException(index))
    end
    return nothing
end

"""
    solve_r!(B, F::MFQR, rank; trans=:N, config=KernelConfig())

Solve the caller-selected leading `rank` block of `R` in place. `trans=:N`
solves `R[1:rank,1:rank] * X = B`; `trans=:T` solves the transposed system.
The destination must have exactly `rank` rows. This primitive does not choose
the rank.
"""
function solve_r!(
    destination::Union{AbstractVector{MF},AbstractMatrix{MF}},
    F::MFQR{MF},
    rank::Integer;
    trans::Symbol=:N,
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    rank_value = Int(rank)
    Base.require_one_based_indexing(destination, F.factors)
    _check_r_solve(F, destination, rank_value, trans)
    rank_value == 0 && return destination
    leading = @view F.factors[1:rank_value, 1:rank_value]
    if destination isa AbstractVector
        trsv!(
            destination,
            leading;
            uplo=:upper,
            trans=trans,
            diag=:nonunit,
            config=config,
        )
    else
        trsm!(
            destination,
            leading;
            side=:left,
            uplo=:upper,
            trans=trans,
            diag=:nonunit,
            config=config,
        )
    end
    return destination
end
