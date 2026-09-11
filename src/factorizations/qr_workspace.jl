# M03 — QR workspace: an independent Householder reference, orthogonality and
# recomputed-norm checks, and the measured tall/wide support boundary.
#
# STATUS: add-only module file written by the M03 worker; not in the include
# graph at the time of writing (probed — see the header of `pivot_policy.jl`).
#
# THE NORMAL-EQUATIONS RULE
#
# `M03.md` forbids replacing the QR path with the normal equations. Nothing
# here routes, accelerates or falls back to `A'A`. The one function that forms
# `A'A` is `m03_qr_normal_equations_control`, it is named `_control`, it is
# only ever *called* by the driver to produce a deliberately-wrong comparison
# number, and its measured condition-number-squared loss is the evidence for
# why it must not be used. If the QR path cannot be exercised for some shape,
# the driver reports `not_run` for that shape instead of substituting this.
#
# WHY A SECOND HOUSEHOLDER IMPLEMENTATION
#
# `rrqr!` is column-pivoted (rank-revealing); `qr!` is fixed order. The
# orthogonality of the *produced* `Q` is a property of the reflectors, so the
# driver checks it directly with `apply_q!` on an identity block, which is the
# real production path. This file's own reflector routine exists so that the
# recomputed-norm and rank criteria have a reference that is not derived from
# the code under test — the same "higher-precision reference must be
# independent of the optimised path" rule the packet states in AGENTS.md.

const M03_QR_WORKSPACE_VERSION = v"1.0.0"

# Same include-time guard as pivot_policy.jl.
@isdefined(MultiFloatLinearAlgebra) || error(
    "M03: include this file into MultiFloatLinearAlgebra, not into a sandbox " *
    "module: Base.include(MultiFloatLinearAlgebra, " *
    "\"src/factorizations/qr_workspace.jl\")")

# `m03_mirror_lower_to_upper!` and `m03_orthogonality` rely on the M03 grammar
# helper for the normal-equations control; include it if it is not present yet.
@isdefined(M03PivotGrammar) || Base.include(
    @__MODULE__, joinpath(@__DIR__, "pivot_policy.jl"))
@isdefined(m03_mirror_lower_to_upper!) || Base.include(
    @__MODULE__, joinpath(@__DIR__, "panel_updates.jl"))

"""
    m03_householder_qr!(A, tau) -> tau

In-place Householder QR of `A` (m x n). On return the strict lower triangle of
`A` holds the reflector vectors, the upper triangle holds `R`, and `tau[i]`
holds the reflector coefficient. Satisfies `A_original = Q * R` with
`Q = H_1 * ... * H_r`, `r = min(m, n)`.

This is the plain, unblocked, unpivoted algorithm — deliberately the least
clever correct version, used as an independent reference. It performs no
column pivoting, so it is NOT rank-revealing and is compared against `qr!`
(also unpivoted), never against `rrqr!` directly.
"""
function m03_householder_qr!(A::AbstractMatrix{MF},
                             tau::AbstractVector{MF}) where {MF<:MultiFloat}
    m, n = size(A)
    r = min(m, n)
    length(tau) >= r || throw(DimensionMismatch("tau must have length >= min(m, n)"))
    @inbounds for step in 1:r
        # norm of A[step:m, step] with the same scale/sum-of-squares form the
        # production norm code uses, so no spurious overflow at extreme scales.
        scale = zero(MF)
        scaled_sum = one(MF)
        nonzero_seen = false
        for row in step:m
            value = abs(A[row, step])
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
        norm_value = iszero(scale) ? zero(MF) : scale * sqrt(scaled_sum)
        alpha = A[step, step]
        if iszero(norm_value)
            tau[step] = zero(MF)
            continue
        end
        beta = alpha >= zero(MF) ? -norm_value : norm_value
        tau[step] = (beta - alpha) / beta
        @inbounds for row in (step + 1):m
            A[row, step] /= (alpha - beta)
        end
        A[step, step] = beta
        # H = I - tau * v * v' with v[step] = 1: trailing update
        @inbounds for column in (step + 1):n
            projection = A[step, column]
            for row in (step + 1):m
                projection += A[row, step] * A[row, column]
            end
            projection *= tau[step]
            A[step, column] -= projection
            for row in (step + 1):m
                A[row, column] -= A[row, step] * projection
            end
        end
    end
    return tau
end

@inline function _m03_apply_reflector!(destination::AbstractVector{MF},
                                       factors::AbstractMatrix{MF},
                                       tau::AbstractVector{MF},
                                       step::Int) where {MF<:MultiFloat}
    coefficient = tau[step]
    iszero(coefficient) && return destination
    projection = destination[step]
    @inbounds for row in (step + 1):size(factors, 1)
        projection += factors[row, step] * destination[row]
    end
    projection *= coefficient
    destination[step] -= projection
    @inbounds for row in (step + 1):size(factors, 1)
        destination[row] -= factors[row, step] * projection
    end
    return destination
end

"""
    m03_apply_q!(B, factors, tau; trans=:N) -> B

Apply the Householder product described by `(factors, tau)`: `Q * B` for
`trans=:N`, `Q' * B` for `trans=:T`. `B` has one row per row of the factorized
matrix. `trans=:N` applies the reflectors in reverse order, matching MFLA's
`apply_q!` convention.
"""
function m03_apply_q!(B::AbstractVecOrMat{MF},
                      factors::AbstractMatrix{MF},
                      tau::AbstractVector{MF};
                      trans::Symbol=:N) where {MF<:MultiFloat}
    trans in (:N, :T) || throw(ArgumentError("trans must be :N or :T"))
    size(B, 1) == size(factors, 1) ||
        throw(DimensionMismatch("destination row count differs from the factor"))
    steps = trans === :N ? reverse(eachindex(tau)) : eachindex(tau)
    for step in steps
        if B isa AbstractVector
            _m03_apply_reflector!(B, factors, tau, step)
        else
            coefficient = tau[step]
            if !iszero(coefficient)
                @inbounds for column in axes(B, 2)
                    projection = B[step, column]
                    for row in (step + 1):size(factors, 1)
                        projection += factors[row, step] * B[row, column]
                    end
                    projection *= coefficient
                    B[step, column] -= projection
                    for row in (step + 1):size(factors, 1)
                        B[row, column] -= factors[row, step] * projection
                    end
                end
            end
        end
    end
    return B
end

"""
    m03_identity_block(::Type{MF}, n) -> Matrix{MF}

`n x n` identity, the object `apply_q!` is applied to when forming `Q`
explicitly.
"""
function m03_identity_block(::Type{MF}, n::Int) where {MF<:MultiFloat}
    out = zeros(MF, n, n)
    @inbounds for index in 1:n
        out[index, index] = one(MF)
    end
    return out
end

m03_identity_block(::Type{MF}, rows::Int, columns::Int) where {MF<:MultiFloat} =
    m03_identity_block(MF, max(rows, columns))[1:rows, 1:columns]

"""
    m03_orthogonality(::Type{MF}, Q) -> (gram_error, raw_gram_error)

`Q' * Q - I` measured two ways:

  * `gram_error` — the entrywise residual scaled by `max(1, max|Q'Q|)`, which is
    the honest small number to quote;
  * `raw_gram_error` — the unscaled entrywise residual `max|Q'Q - I|`, quoted
    alongside it so a reader can see the absolute defect rather than only the
    normalised one.

The reference product is computed in the caller's type; the driver additionally
recomputes `Q'Q` in `BigFloat` for a subset so the check is not made by the
same arithmetic that produced `Q`.
"""
function m03_orthogonality(::Type{MF}, Q::AbstractMatrix{MF}) where {MF<:MultiFloat}
    m, n = size(Q)
    gram = zeros(MF, n, n)
    largest = zero(MF)
    @inbounds for column in 1:n
        for other in 1:n
            accumulator = zero(MF)
            for row in 1:m
                accumulator += Q[row, column] * Q[row, other]
            end
            gram[other, column] = accumulator
            largest = max(largest, abs(accumulator))
        end
    end
    raw = zero(MF)
    @inbounds for column in 1:n
        for other in 1:n
            target = column == other ? one(MF) : zero(MF)
            raw = max(raw, abs(gram[other, column] - target))
        end
    end
    return (raw / max(one(MF), largest), raw)
end

"""
    m03_qr_recomputed_norms(F) -> Vector

Recompute `‖A[:, j]‖₂` for every column from `A = Q*R` using the stored `R`
only: with `p = factor_permutation(F)`, the pivot column's norm is
`‖R[1:min(m,j), j]‖₂`. No normal equations, no access to the original `A`.

This is the quantity MFLA's incremental norm downdate is supposed to track;
the driver compares it against the same norm recomputed from the original
matrix.
"""
function m03_qr_recomputed_norms(F)
    rows, columns = size(factor_matrix(F))
    diagonal_count = min(rows, columns)
    norms = Vector{eltype(factor_matrix(F))}(undef, columns)
    @inbounds for column in 1:columns
        scale = zero(eltype(norms))
        scaled_sum = one(eltype(norms))
        nonzero_seen = false
        for row in 1:min(column, diagonal_count)
            value = abs(factor_matrix(F)[row, column])
            if !iszero(value)
                if !nonzero_seen
                    scale = value
                    scaled_sum = one(eltype(norms))
                    nonzero_seen = true
                elseif value > scale
                    ratio = scale / value
                    scaled_sum = one(eltype(norms)) + scaled_sum * ratio * ratio
                    scale = value
                else
                    ratio = value / scale
                    scaled_sum += ratio * ratio
                end
            end
        end
        norms[column] = iszero(scale) ? zero(eltype(norms)) :
                        scale * sqrt(scaled_sum)
    end
    return norms
end

"""
    m03_qr_rank_by_diagonal(F; atol=0, rtol=0) -> Int

A *local* leading-rank criterion on the recomputed `|R[i,i]|` sequence: accept
while `|R[i,i]| > max(atol, rtol * max_i |R[i,i]|)`. Deliberately written here
rather than calling MFLA's `numerical_rank` so the driver can compare two
independently written criteria and report a disagreement as a defect.

The stopping test is `!(|R[i,i]| > threshold)` and therefore treats a NaN
diagonal entry as a rejection (rank stops) instead of propagating NaN into a
comparison; MFLA's version uses `>` as its continue test, so the two agree on
finite input and the driver asserts that agreement explicitly.
"""
function m03_qr_rank_by_diagonal(F; atol::Real=0.0, rtol::Real=0.0)
    rows, columns = size(factor_matrix(F))
    diagonal_count = min(rows, columns)
    MF = eltype(factor_matrix(F))
    absolute_tolerance = MF(atol)
    relative_tolerance = MF(rtol)
    largest = zero(MF)
    @inbounds for index in 1:diagonal_count
        largest = max(largest, abs(factor_matrix(F)[index, index]))
    end
    threshold = max(absolute_tolerance, relative_tolerance * largest)
    rank = 0
    @inbounds for index in 1:diagonal_count
        magnitude = abs(factor_matrix(F)[index, index])
        (magnitude > threshold) || break
        rank += 1
    end
    return rank
end

"""
    m03_qr_normal_equations_control(A) -> (gram, gram_error)

CONTROL, NOT A ROUTE. Forms the Gram matrix `A'A` (lower triangle only, then
mirrored) and returns the relative reconstruction error of the normal-equations
operator. It is called once per QR shape by the driver purely to measure how
much accuracy this substitution loses; it is never used to solve, to estimate a
rank, or to replace a QR solve. The loss is expected to be ~`eps(T) *
cond(A)^2`.
"""
function m03_qr_normal_equations_control(A::AbstractMatrix{MF}) where {MF<:MultiFloat}
    m, n = size(A)
    gram = zeros(MF, n, n)
    @inbounds for column in 1:n
        for other in 1:column
            accumulator = zero(MF)
            for row in 1:m
                accumulator += A[row, column] * A[row, other]
            end
            gram[other, column] = accumulator
        end
    end
    m03_mirror_lower_to_upper!(gram, 1)
    # Reconstruct A'A the cheap way a normal-equations route would already have
    # it, and compare against the exact-scalar recomputation of the same
    # product. The difference IS the normal-equations error.
    exact = zeros(BigFloat, n, n)
    for column in 1:n, other in 1:n
        accumulator = BigFloat(0)
        for row in 1:m
            accumulator += BigFloat(A[row, column]) * BigFloat(A[row, other])
        end
        exact[other, column] = accumulator
    end
    largest = maximum(abs, exact)
    difference = BigFloat(0)
    for index in eachindex(gram)
        difference = max(difference, abs(BigFloat(gram[index]) - exact[index]))
    end
    return (gram, eltype(gram)(difference / max(BigFloat(1), largest)))
end

"""
    m03_qr_shape_class(m, n) -> Symbol

The support boundary this task must state explicitly:

  * `:tall_or_square` — `m >= n`. The factor is `A[:, p] = Q * R` with
    `R` upper triangular `n x n`, `Q` `m x m` (economic `Q` is `m x n`), and the
    full-rank `n x n` triangular solve/solve_r! path is well posed.
  * `:wide` — `m < n`. Only `m` reflectors exist, so `R` is `m x n` upper
    trapezoidal with `min(m,n) = m` nonzeros per accepted row; a *full-column*
    triangular solve does not exist and `solve_r!` is restricted to
    `rank <= m`. Rank-deficient wide systems need a minimum-norm route that
    MFLA does not provide, so the driver reports that arm `not_run` rather than
    routing it through `A'A`.
"""
m03_qr_shape_class(m::Int, n::Int) = m >= n ? :tall_or_square : :wide
