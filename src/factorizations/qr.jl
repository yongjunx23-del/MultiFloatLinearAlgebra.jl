struct MFQR{
    MF<:MultiFloat,
    M<:AbstractMatrix{MF},
    T<:AbstractVector{MF},
    P<:AbstractVector{Int},
} <: AbstractMFFactorization{MF}
    factors::M
    tau::T
    permutation::P
    info::Int
    lease::Union{Nothing,_FactorWorkspaceLease{MF}}
end

factor_kind(::MFQR) = :qr
@inline _check_factor_valid(F::MFQR) = _check_factor_lease(F.lease)
factor_status(F::MFQR) = (_check_factor_lease(F.lease); F.info)
factor_matrix(F::MFQR) = (_check_factor_lease(F.lease); F.factors)

function _prepare_qr_metadata!(
    ::Type{MF},
    reflector_count::Int,
    column_count::Int,
    workspace::Union{Nothing,MFWorkspace{MF}},
) where {MF<:MultiFloat}
    if workspace === nothing
        return zeros(MF, reflector_count), collect(1:column_count), nothing
    end
    lease = _acquire_factor_workspace!(
        workspace, max(reflector_count, column_count),
    )
    tau = @view workspace.qr_tau[1:reflector_count]
    permutation = @view workspace.qr_permutation[1:column_count]
    fill!(tau, zero(MF))
    @inbounds for column in 1:column_count
        permutation[column] = column
    end
    return tau, permutation, lease
end

"""
    factor_permutation(F::MFQR) -> Vector{Int}

Return the column permutation `p` satisfying `A[:, p] = Q * R`. The returned
vector is a copy and may be mutated by the caller.
"""
function factor_permutation(F::MFQR)
    _check_factor_lease(F.lease)
    return copy(F.permutation)
end

"""
    factor_rdiag(F::MFQR) -> Vector

Return a copy of the signed diagonal of the compactly stored `R` factor.
"""
function factor_rdiag(F::MFQR{MF}) where {MF<:MultiFloat}
    _check_factor_lease(F.lease)
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
    return nonzero_seen ? scale * sqrt(scaled_sum) : zero(MF)
end

function _qr_swap_columns!(A::AbstractMatrix, first::Int, second::Int)
    first == second && return nothing
    @inbounds for row in axes(A, 1)
        A[row, first], A[row, second] = A[row, second], A[row, first]
    end
    return nothing
end

function _qr_select_pivot(
    A::AbstractMatrix{MF},
    permutation::AbstractVector{Int},
    step::Int,
) where {MF<:MultiFloat}
    pivot = step
    pivot_norm = _qr_column_norm(A, step, step)
    @inbounds for column in (step + 1):size(A, 2)
        candidate_norm = _qr_column_norm(A, step, column)
        if candidate_norm > pivot_norm ||
           (candidate_norm == pivot_norm &&
            permutation[column] < permutation[pivot])
            pivot = column
            pivot_norm = candidate_norm
        end
    end
    return pivot
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

function _qr_apply_reflector_to_trailing!(
    A::AbstractMatrix{MF},
    tau::MF,
    step::Int,
) where {MF<:MultiFloat}
    iszero(tau) && return nothing
    @inbounds for column in (step + 1):size(A, 2)
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

"""
    rrqr!(A; check=true, workspace=nothing)

Compute a deterministic column-pivoted Householder QR factorization in place.
The result satisfies `A_original[:, factor_permutation(F)] = Q * R`.
Householder vectors are stored below the diagonal of `factor_matrix(F)` and
`R` is stored on and above it.

Column norms are recomputed with scaled sum-of-squares arithmetic at every
step. Exact pivot-norm ties choose the smallest original column index. Rank
deficiency is a successful factorization; [`numerical_rank`](@ref) applies a
caller-supplied threshold after factorization. No fallback or rank policy is
performed here.

With `workspace=MFWorkspace(T)`, `tau` and the permutation borrow caller-owned
storage until that workspace starts another factorization.
"""
function rrqr!(
    A::AbstractMatrix{MF};
    check::Bool=true,
    workspace::Union{Nothing,MFWorkspace{MF}}=nothing,
) where {MF<:MultiFloat}
    _check_supported(MF)
    Base.require_one_based_indexing(A)
    m, n = size(A)
    reflector_count = min(m, n)
    finite_input = _all_finite(A)
    !finite_input && check &&
        throw(DomainError(A, "rrqr!: input matrix contains non-finite entries"))
    tau, permutation, lease = _prepare_qr_metadata!(
        MF, reflector_count, n, workspace,
    )

    if !finite_input
        return MFQR{MF,typeof(A),typeof(tau),typeof(permutation)}(
            A, tau, permutation, -1, lease,
        )
    end

    @inbounds for step in 1:reflector_count
        pivot = _qr_select_pivot(A, permutation, step)
        _qr_swap_columns!(A, step, pivot)
        permutation[step], permutation[pivot] =
            permutation[pivot], permutation[step]
        tau[step] = _qr_make_reflector!(A, step)
        _qr_apply_reflector_to_trailing!(A, tau[step], step)
    end
    return MFQR{MF,typeof(A),typeof(tau),typeof(permutation)}(
        A, tau, permutation, 0, lease,
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
