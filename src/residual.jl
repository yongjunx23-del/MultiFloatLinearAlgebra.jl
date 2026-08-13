@inline function _check_residual_uplo(uplo::Symbol)
    uplo in (:general, :lower, :upper) ||
        throw(ArgumentError("uplo must be :general, :lower, or :upper"))
    return nothing
end

function _check_residual_vector_dimensions(r, A, x, b, uplo::Symbol)
    rows, columns = size(A)
    length(r) == rows || throw(DimensionMismatch("residual output length differs"))
    length(x) == columns || throw(DimensionMismatch("residual solution length differs"))
    length(b) == rows || throw(DimensionMismatch("residual right-hand side length differs"))
    uplo === :general || rows == columns ||
        throw(DimensionMismatch("symmetric residual requires a square matrix"))
    return nothing
end

function _check_residual_matrix_dimensions(R, A, X, B, uplo::Symbol)
    rows, columns = size(A)
    size(R) == size(B) || throw(DimensionMismatch("residual output dimensions differ"))
    size(R, 1) == rows || throw(DimensionMismatch("residual output row count differs"))
    size(X, 1) == columns || throw(DimensionMismatch("residual solution row count differs"))
    size(X, 2) == size(R, 2) ||
        throw(DimensionMismatch("residual right-hand-side count differs"))
    uplo === :general || rows == columns ||
        throw(DimensionMismatch("symmetric residual requires a square matrix"))
    return nothing
end

function _prepare_residual_destination!(destination, source)
    if destination !== source
        Base.mightalias(destination, source) &&
            throw(ArgumentError("residual output may equal b but must not partially alias it"))
        copyto!(destination, source)
    end
    return destination
end

"""
    residual!(r, A, x, b; uplo=:general, config=KernelConfig(), workspace=nothing)

Compute the solver-independent residual

`r = b - A*x`

in caller-owned storage. Vector and matrix right-hand sides are supported.
`uplo=:general` uses the full matrix. `uplo=:lower` or `:upper` treats `A` as
symmetric and reads only the selected authoritative triangle; the inactive
triangle may contain stale or nonfinite data.

The destination may be exactly `b`, but must not alias `x` or `A` and may not
partially overlap `b`. Reductions use the existing deterministic MFLA kernel
order. A `GemmWorkspace` or `MFWorkspace` is forwarded only to the general
multi-RHS GEMM path.
"""
function residual!(
    r::AbstractVector{MF},
    A::AbstractMatrix{MF},
    x::AbstractVector{MF},
    b::AbstractVector{MF};
    uplo::Symbol=:general,
    config::KernelConfig=KernelConfig(),
    workspace::Union{Nothing,GemmWorkspace{MF},MFWorkspace{MF}}=nothing,
) where {MF<:MultiFloat}
    _check_supported(MF)
    _check_residual_uplo(uplo)
    Base.require_one_based_indexing(r, A, x, b)
    _check_residual_vector_dimensions(r, A, x, b, uplo)
    Base.mightalias(r, x) &&
        throw(ArgumentError("residual output must not alias x"))
    Base.mightalias(r, A) &&
        throw(ArgumentError("residual output must not alias A"))
    _prepare_residual_destination!(r, b)

    if uplo === :general
        gemv!(r, A, x, -one(MF), one(MF); config=config)
    else
        symv!(r, A, x, -one(MF), one(MF); uplo=uplo, config=config)
    end
    return r
end

function residual!(
    R::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    X::AbstractMatrix{MF},
    B::AbstractMatrix{MF};
    uplo::Symbol=:general,
    config::KernelConfig=KernelConfig(),
    workspace::Union{Nothing,GemmWorkspace{MF},MFWorkspace{MF}}=nothing,
) where {MF<:MultiFloat}
    _check_supported(MF)
    _check_residual_uplo(uplo)
    Base.require_one_based_indexing(R, A, X, B)
    _check_residual_matrix_dimensions(R, A, X, B, uplo)
    Base.mightalias(R, X) &&
        throw(ArgumentError("residual output must not alias X"))
    Base.mightalias(R, A) &&
        throw(ArgumentError("residual output must not alias A"))
    _prepare_residual_destination!(R, B)

    if uplo === :general
        gemm!(
            R,
            A,
            X,
            -one(MF),
            one(MF);
            config=config,
            workspace=workspace,
        )
    else
        @inbounds for column in axes(R, 2)
            symv!(
                view(R, :, column),
                A,
                view(X, :, column),
                -one(MF),
                one(MF);
                uplo=uplo,
                config=config,
            )
        end
    end
    return R
end

function _vector_infinity_norm(x::AbstractVector{MF}) where {MF<:MultiFloat}
    value = zero(MF)
    @inbounds for index in eachindex(x)
        value = max(value, abs(x[index]))
    end
    return value
end

function _column_infinity_norm(X::AbstractMatrix{MF}, column::Int) where {MF<:MultiFloat}
    value = zero(MF)
    @inbounds for row in axes(X, 1)
        value = max(value, abs(X[row, column]))
    end
    return value
end

function _matrix_infinity_norm(
    A::AbstractMatrix{MF},
    uplo::Symbol,
) where {MF<:MultiFloat}
    rows, columns = size(A)
    value = zero(MF)
    @inbounds for row in 1:rows
        row_sum = zero(MF)
        for column in 1:columns
            element = if uplo === :general
                A[row, column]
            elseif uplo === :lower
                column <= row ? A[row, column] : A[column, row]
            else
                column >= row ? A[row, column] : A[column, row]
            end
            row_sum += abs(element)
        end
        value = max(value, row_sum)
    end
    return value
end

function _matrix_authoritative_finite(A::AbstractMatrix, uplo::Symbol)
    if uplo === :general
        return _all_finite(A)
    elseif uplo === :lower
        return _lower_triangle_finite(A)
    end
    @inbounds for column in axes(A, 2), row in 1:column
        isfinite(A[row, column]) || return false
    end
    return true
end

function _backward_error_value(
    matrix_norm::MF,
    solution_norm::MF,
    right_hand_side_norm::MF,
    residual_norm::MF,
    inputs_finite::Bool,
) where {MF<:MultiFloat}
    inputs_finite || return MF(NaN)
    primary_scale = max(matrix_norm, right_hand_side_norm)
    if iszero(primary_scale)
        return iszero(residual_norm) ? zero(MF) : MF(Inf)
    end
    solution_scale = max(solution_norm, one(MF))
    scaled_denominator =
        (matrix_norm / primary_scale) *
        (solution_norm / solution_scale) +
        (right_hand_side_norm / primary_scale) / solution_scale
    scaled_residual = (residual_norm / primary_scale) / solution_scale
    return scaled_residual / scaled_denominator
end

"""
    normwise_backward_error(A, x, b, r; uplo=:general)

Return the normwise scaled backward error

`norm(r, Inf) / (norm(A, Inf) * norm(x, Inf) + norm(b, Inf))`.

For matrix right-hand sides, return one value per column. Symmetric
authoritative-triangle semantics match [`residual!`](@ref). A zero denominator
returns zero for a zero residual and `Inf` otherwise. Any nonfinite value in an
authoritative input returns `NaN`; nonfinite inactive-triangle data is ignored.
No acceptance threshold or solver policy is applied.
"""
function normwise_backward_error(
    A::AbstractMatrix{MF},
    x::AbstractVector{MF},
    b::AbstractVector{MF},
    r::AbstractVector{MF};
    uplo::Symbol=:general,
) where {MF<:MultiFloat}
    _check_supported(MF)
    _check_residual_uplo(uplo)
    Base.require_one_based_indexing(A, x, b, r)
    _check_residual_vector_dimensions(r, A, x, b, uplo)
    matrix_finite = _matrix_authoritative_finite(A, uplo)
    inputs_finite = matrix_finite && _all_finite(x) &&
        _all_finite(b) && _all_finite(r)
    return _backward_error_value(
        _matrix_infinity_norm(A, uplo),
        _vector_infinity_norm(x),
        _vector_infinity_norm(b),
        _vector_infinity_norm(r),
        inputs_finite,
    )
end

function normwise_backward_error(
    A::AbstractMatrix{MF},
    X::AbstractMatrix{MF},
    B::AbstractMatrix{MF},
    R::AbstractMatrix{MF};
    uplo::Symbol=:general,
) where {MF<:MultiFloat}
    _check_supported(MF)
    _check_residual_uplo(uplo)
    Base.require_one_based_indexing(A, X, B, R)
    _check_residual_matrix_dimensions(R, A, X, B, uplo)
    matrix_norm = _matrix_infinity_norm(A, uplo)
    matrix_finite = _matrix_authoritative_finite(A, uplo)
    result = Vector{MF}(undef, size(R, 2))
    @inbounds for column in axes(R, 2)
        inputs_finite = matrix_finite
        for row in axes(X, 1)
            inputs_finite &= isfinite(X[row, column])
        end
        for row in axes(B, 1)
            inputs_finite &= isfinite(B[row, column]) && isfinite(R[row, column])
        end
        result[column] = _backward_error_value(
            matrix_norm,
            _column_infinity_norm(X, column),
            _column_infinity_norm(B, column),
            _column_infinity_norm(R, column),
            inputs_finite,
        )
    end
    return result
end

const _RefinableFactor{MF} = Union{MFCholesky{MF},MFLU{MF},MFLDLT{MF}}

function _preflight_correction(F::MFCholesky)
    issuccess(F) || throw(LinearAlgebra.PosDefException(factor_status(F)))
    return nothing
end

function _preflight_correction(F::Union{MFLU,MFLDLT})
    issuccess(F) || throw(LinearAlgebra.SingularException(factor_status(F)))
    return nothing
end

function _prepare_correction!(destination, residual, factors)
    Base.mightalias(destination, factors) &&
        throw(ArgumentError("correction destination must not alias factor storage"))
    if destination !== residual
        Base.mightalias(destination, residual) &&
            throw(ArgumentError("correction destination may equal r but must not partially alias it"))
        copyto!(destination, residual)
    end
    return destination
end

"""
    refinement_correction!(delta, F, r; config=KernelConfig())

Compute exactly one correction `delta = F \\ r` with an existing Cholesky,
LU, or LDLT factorization. Vector and matrix right-hand sides are supported.
The destination may equal `r`. This primitive performs no convergence test,
iteration, fallback, or precision escalation.
"""
function refinement_correction!(
    delta::AbstractVector{MF},
    F::_RefinableFactor{MF},
    r::AbstractVector{MF};
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    _preflight_correction(F)
    length(delta) == length(r) ||
        throw(DimensionMismatch("correction and residual lengths differ"))
    length(r) == size(F, 1) ||
        throw(DimensionMismatch("correction residual length differs from factor"))
    Base.require_one_based_indexing(delta, r, factor_matrix(F))
    _prepare_correction!(delta, r, factor_matrix(F))
    return ldiv!(delta, F; config=config)
end

function refinement_correction!(
    delta::AbstractMatrix{MF},
    F::_RefinableFactor{MF},
    r::AbstractMatrix{MF};
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    _preflight_correction(F)
    size(delta) == size(r) ||
        throw(DimensionMismatch("correction and residual dimensions differ"))
    size(r, 1) == size(F, 1) ||
        throw(DimensionMismatch("correction residual rows differ from factor"))
    Base.require_one_based_indexing(delta, r, factor_matrix(F))
    _prepare_correction!(delta, r, factor_matrix(F))
    return ldiv!(delta, F; config=config)
end

@inline function _mixed_residual_pair_supported(
    ::Type{MultiFloat{Float64,SourceLimbs}},
    ::Type{MultiFloat{Float64,ResidualLimbs}},
) where {SourceLimbs,ResidualLimbs}
    return (SourceLimbs == 2 && ResidualLimbs in (3, 4)) ||
           (SourceLimbs == 3 && ResidualLimbs == 4)
end

function _check_mixed_residual_pair(::Type{Source}, ::Type{Residual}) where {Source,Residual}
    _mixed_residual_pair_supported(Source, Residual) ||
        throw(ArgumentError(
            "residual_mixed! supports only Float64x2 to Float64x3/x4 " *
            "and Float64x3 to Float64x4",
        ))
    return nothing
end

@inline function _mixed_matrix_value(
    A::AbstractMatrix{Source},
    row::Int,
    column::Int,
    uplo::Symbol,
) where {Source<:MultiFloat}
    if uplo === :general
        return A[row, column]
    elseif uplo === :lower
        return column <= row ? A[row, column] : A[column, row]
    end
    return column >= row ? A[row, column] : A[column, row]
end

function _mixed_residual_vector_rows!(
    r::AbstractVector{Residual},
    A::AbstractMatrix{Source},
    x::AbstractVector{Source},
    b::AbstractVector{Source},
    first_row::Int,
    row_stride::Int,
    uplo::Symbol,
) where {Source<:MultiFloat,Residual<:MultiFloat}
    @inbounds for row in first_row:row_stride:size(A, 1)
        accumulator = Residual(b[row])
        for column in axes(A, 2)
            accumulator -=
                Residual(_mixed_matrix_value(A, row, column, uplo)) *
                Residual(x[column])
        end
        r[row] = accumulator
    end
    return nothing
end

function _mixed_residual_matrix_jobs!(
    R::AbstractMatrix{Residual},
    A::AbstractMatrix{Source},
    X::AbstractMatrix{Source},
    B::AbstractMatrix{Source},
    first_job::Int,
    job_stride::Int,
    uplo::Symbol,
) where {Source<:MultiFloat,Residual<:MultiFloat}
    rows = size(A, 1)
    jobs = rows * size(X, 2)
    @inbounds for job in first_job:job_stride:jobs
        column = (job - 1) ÷ rows + 1
        row = (job - 1) % rows + 1
        accumulator = Residual(B[row, column])
        for k in axes(A, 2)
            accumulator -=
                Residual(_mixed_matrix_value(A, row, k, uplo)) *
                Residual(X[k, column])
        end
        R[row, column] = accumulator
    end
    return nothing
end

"""
    residual_mixed!(r, A, x, b; uplo=:general, config=KernelConfig())

Compute `r = b - A*x` directly in an explicitly higher MultiFloat precision.
The supported source-to-residual pairs are:

- `Float64x2 -> Float64x3`
- `Float64x2 -> Float64x4`
- `Float64x3 -> Float64x4`

All source operands have one common lower-precision type and every operand is
converted before multiplication/subtraction. This function never computes a
low-precision residual and promotes it, never uses BigFloat, and never changes
the precision of ordinary kernels. Vector and matrix right-hand sides are
supported. `uplo=:lower/:upper` preserves authoritative-triangle semantics.

Each output accumulates source columns in ascending order. Threading assigns
independent outputs and therefore does not change arithmetic order. The output
must not alias any source operand.
"""
function residual_mixed!(
    r::AbstractVector{Residual},
    A::AbstractMatrix{Source},
    x::AbstractVector{Source},
    b::AbstractVector{Source};
    uplo::Symbol=:general,
    config::KernelConfig=KernelConfig(),
) where {
    SourceLimbs,
    ResidualLimbs,
    Source<:MultiFloat{Float64,SourceLimbs},
    Residual<:MultiFloat{Float64,ResidualLimbs},
}
    _check_mixed_residual_pair(Source, Residual)
    _check_residual_uplo(uplo)
    Base.require_one_based_indexing(r, A, x, b)
    _check_residual_vector_dimensions(r, A, x, b, uplo)
    (Base.mightalias(r, A) || Base.mightalias(r, x) || Base.mightalias(r, b)) &&
        throw(ArgumentError("mixed residual output must not alias source operands"))

    workers = _workers(config, size(A, 1))
    if workers == 1 || size(A, 1) < 48
        _mixed_residual_vector_rows!(r, A, x, b, 1, 1, uplo)
        return r
    end
    @sync for worker in 1:workers
        Threads.@spawn _mixed_residual_vector_rows!(
            r, A, x, b, worker, workers, uplo,
        )
    end
    return r
end

function residual_mixed!(
    R::AbstractMatrix{Residual},
    A::AbstractMatrix{Source},
    X::AbstractMatrix{Source},
    B::AbstractMatrix{Source};
    uplo::Symbol=:general,
    config::KernelConfig=KernelConfig(),
) where {
    SourceLimbs,
    ResidualLimbs,
    Source<:MultiFloat{Float64,SourceLimbs},
    Residual<:MultiFloat{Float64,ResidualLimbs},
}
    _check_mixed_residual_pair(Source, Residual)
    _check_residual_uplo(uplo)
    Base.require_one_based_indexing(R, A, X, B)
    _check_residual_matrix_dimensions(R, A, X, B, uplo)
    (Base.mightalias(R, A) || Base.mightalias(R, X) || Base.mightalias(R, B)) &&
        throw(ArgumentError("mixed residual output must not alias source operands"))

    jobs = size(R, 1) * size(R, 2)
    workers = _workers(config, jobs)
    if workers == 1 || jobs < 48
        _mixed_residual_matrix_jobs!(R, A, X, B, 1, 1, uplo)
        return R
    end
    @sync for worker in 1:workers
        Threads.@spawn _mixed_residual_matrix_jobs!(
            R, A, X, B, worker, workers, uplo,
        )
    end
    return R
end
