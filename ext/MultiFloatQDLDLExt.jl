module MultiFloatQDLDLExt

using MultiFloatLinearAlgebra
using MultiFloats
using QDLDL
using SparseArrays
using LinearAlgebra: AbstractVecOrMat, diag, istriu

const MFLA = MultiFloatLinearAlgebra

@inline function _mix(signature::UInt64, value::Integer)
    return (signature ⊻ reinterpret(UInt64, Int64(value))) * UInt64(0x100000001b3)
end

function _pattern_signature(A::SparseMatrixCSC, dsigns::AbstractVector{<:Integer})
    signature = UInt64(0xcbf29ce484222325)
    signature = _mix(signature, size(A, 1))
    signature = _mix(signature, size(A, 2))
    for value in A.colptr
        signature = _mix(signature, value)
    end
    for value in A.rowval
        signature = _mix(signature, value)
    end
    for value in dsigns
        signature = _mix(signature, value)
    end
    return signature
end

mutable struct MFSparseLDLCache{MF<:MultiFloat,Ti<:Integer} <:
               MFLA.AbstractMFFactorCache{MF}
    matrix::SparseMatrixCSC{MF,Ti}
    factor::Union{Nothing,QDLDL.QDLDLFactorisation{MF,Ti}}
    indices::Vector{Ti}
    dsigns::Vector{Ti}             # diagnostic only; never fed to QDLDL
    frozen_colptr::Vector{Ti}
    frozen_rowval::Vector{Ti}
    factored_values::Vector{MF}
    factor_values_valid::Bool
    pattern_signature::UInt64
    nrhs_capacity::Int
    status::Int
    config::MFLA.KernelConfig
    config_epoch::UInt
    prepared_epoch::UInt
    prepared_shape::Tuple{Int,Int}
    symbolic_count::Int
    numeric_factor_count::Int
    solve_count::Int
    positive_inertia::Int
    regularized_entries::Int
end

function _validate_pattern(
    ::Type{MF}, A::SparseMatrixCSC{MF,Ti}, dsigns::AbstractVector{<:Integer},
) where {MF<:MultiFloat,Ti<:Integer}
    n, m = size(A)
    n == m || throw(DimensionMismatch("QDLDL sparse LDL requires a square matrix"))
    istriu(A) || throw(ArgumentError(
        "QDLDL sparse LDL pattern must store the upper triangle",
    ))
    length(dsigns) == n || throw(DimensionMismatch(
        "QDLDL D-sign vector length must equal the matrix order",
    ))
    all(sign -> sign == -1 || sign == 1, dsigns) || throw(ArgumentError(
        "QDLDL D signs must be exactly +1 or -1",
    ))
    all(isfinite, A.nzval) || throw(ArgumentError(
        "QDLDL sparse LDL input contains non-finite values",
    ))
    @inbounds for column in 1:n
        A.colptr[column] < A.colptr[column + 1] || throw(ArgumentError(
            "QDLDL sparse LDL requires every structural column to be nonempty",
        ))
    end
    return nothing
end

function MFSparseLDLCache(
    ::Type{MF}, pattern::SparseMatrixCSC{MF,Ti};
    dsigns::AbstractVector{<:Integer}, nrhs::Integer=1,
    regularize_eps::MF=zero(MF), regularize_delta::MF=zero(MF),
) where {MF<:MultiFloat,Ti<:Integer}
    nrhs >= 1 || throw(ArgumentError("nrhs must be positive"))
    iszero(regularize_eps) && iszero(regularize_delta) || throw(ArgumentError(
        "QDLDL dynamic regularization is unsupported; apply an explicit signed " *
        "static shift to the input matrix",
    ))
    _validate_pattern(MF, pattern, dsigns)
    frozen_colptr = copy(pattern.colptr)
    frozen_rowval = copy(pattern.rowval)
    matrix = SparseMatrixCSC(
        size(pattern, 1), size(pattern, 2), copy(frozen_colptr),
        copy(frozen_rowval), copy(pattern.nzval),
    )
    signs = Ti[sign for sign in dsigns]
    indices = Ti.(eachindex(matrix.nzval))
    # Dynamic regularization is intentionally disabled. The caller owns any
    # signed static shift and SDPX validates against its retained original K.
    factor = QDLDL.qdldl(
        matrix; logical=true, Dsigns=nothing,
        regularize_eps=zero(MF), regularize_delta=zero(MF),
    )
    n = size(matrix, 1)
    return MFSparseLDLCache{MF,Ti}(
        matrix, factor, indices, signs, frozen_colptr, frozen_rowval,
        copy(matrix.nzval), false, _pattern_signature(matrix, signs),
        Int(nrhs), MFLA._FACTOR_CACHE_INVALID, MFLA.KernelConfig(),
        UInt(0), UInt(0), (n, n), 1, 0, 0, -1, 0,
    )
end

MFLA.factor_kind(::MFSparseLDLCache) = :sparse_ldlt
MFLA.factor_status(cache::MFSparseLDLCache) = cache.status
function MFLA.factor_matrix(cache::MFSparseLDLCache{MF,Ti}) where {MF,Ti}
    return SparseMatrixCSC(
        size(cache.matrix, 1), size(cache.matrix, 2), copy(cache.frozen_colptr),
        copy(cache.frozen_rowval), copy(cache.matrix.nzval),
    )
end

function _validate_authority(cache::MFSparseLDLCache)
    size(cache.matrix) == cache.prepared_shape || throw(ArgumentError(
        "QDLDL sparse LDL cache dimensions drifted from frozen authority",
    ))
    cache.matrix.colptr == cache.frozen_colptr &&
        cache.matrix.rowval == cache.frozen_rowval || throw(ArgumentError(
            "QDLDL sparse LDL internal pattern drift",
        ))
    _pattern_signature(cache.matrix, cache.dsigns) == cache.pattern_signature ||
        throw(ArgumentError("QDLDL sparse LDL pattern signature drift"))
    return nothing
end

function _validate_numeric(cache::MFSparseLDLCache{MF,Ti}, A) where {MF,Ti}
    _validate_authority(cache)
    A isa SparseMatrixCSC{MF,Ti} || throw(ArgumentError(
        "QDLDL sparse LDL numeric matrix must preserve scalar/index types",
    ))
    size(A) == cache.prepared_shape || throw(DimensionMismatch(
        "QDLDL sparse LDL numeric dimensions differ from the frozen pattern",
    ))
    A.colptr == cache.frozen_colptr && A.rowval == cache.frozen_rowval ||
        throw(ArgumentError("QDLDL sparse LDL pattern drift"))
    all(isfinite, A.nzval) || throw(ArgumentError(
        "QDLDL sparse LDL numeric input contains non-finite values",
    ))
    return nothing
end

function _revoke_factor!(cache::MFSparseLDLCache)
    cache.status = MFLA._FACTOR_CACHE_INVALID
    cache.factor_values_valid = false
    cache.positive_inertia = -1
    cache.regularized_entries = 0
    return nothing
end

function MFLA.factorize!(
    cache::MFSparseLDLCache{MF,Ti}, A::SparseMatrixCSC{MF,Ti};
    check::Bool=true,
) where {MF<:MultiFloat,Ti<:Integer}
    _revoke_factor!(cache)
    try
        _validate_numeric(cache, A)
        copyto!(cache.matrix.nzval, A.nzval)
        factor = something(cache.factor)
        QDLDL.update_values!(factor, cache.indices, cache.matrix.nzval)
        QDLDL.refactor!(factor)
        all(isfinite, diag(factor.Dinv)) && all(isfinite, factor.L.nzval) ||
            throw(ArgumentError("QDLDL produced non-finite factors"))
        copyto!(cache.factored_values, cache.matrix.nzval)
        cache.factor_values_valid = true
        cache.positive_inertia = QDLDL.positive_inertia(factor)
        cache.regularized_entries = QDLDL.regularized_entries(factor)
        cache.regularized_entries == 0 || throw(ArgumentError(
            "QDLDL unexpectedly regularized an entry",
        ))
        cache.numeric_factor_count += 1
        cache.status = 0
    catch
        _revoke_factor!(cache)
        cache.status = -3
        check && rethrow()
    end
    return cache
end

function _validate_solve_authority(cache::MFSparseLDLCache)
    _validate_authority(cache)
    cache.factor_values_valid && cache.matrix.nzval == cache.factored_values ||
        throw(ArgumentError("QDLDL sparse LDL numeric factor is stale"))
    return nothing
end

function _reject_factor_alias(cache::MFSparseLDLCache, destination)
    factor = something(cache.factor)
    for storage in (
        cache.matrix.nzval, cache.factored_values, factor.L.nzval,
        factor.Dinv.diag, factor.workspace.D, factor.workspace.Dinv,
        factor.workspace.fwork, factor.workspace.Lx,
    )
        Base.mightalias(destination, storage) && throw(ArgumentError(
            "QDLDL sparse LDL destination must not alias factor storage",
        ))
    end
    return nothing
end

function _solve!(
    cache::MFSparseLDLCache{MF}, destination::AbstractVecOrMat{MF},
    rhs::AbstractVecOrMat{MF},
) where {MF<:MultiFloat}
    MFLA.issuccess(cache) || throw(ArgumentError(
        "QDLDL sparse LDL cache does not hold a successful factor",
    ))
    _validate_solve_authority(cache)
    size(destination) == size(rhs) || throw(DimensionMismatch(
        "QDLDL sparse LDL destination/RHS dimensions differ",
    ))
    size(rhs, 1) == size(cache.matrix, 1) || throw(DimensionMismatch(
        "QDLDL sparse LDL RHS row count differs from the factor order",
    ))
    nrhs = ndims(rhs) == 1 ? 1 : size(rhs, 2)
    nrhs <= cache.nrhs_capacity || throw(DimensionMismatch(
        "QDLDL sparse LDL RHS width exceeds prepared capacity",
    ))
    all(isfinite, rhs) || throw(ArgumentError(
        "QDLDL sparse LDL RHS contains non-finite values",
    ))
    _reject_factor_alias(cache, destination)
    MFLA._copy_rhs!(destination, rhs)
    factor = something(cache.factor)
    if ndims(destination) == 1
        QDLDL.solve!(factor, destination)
    else
        @inbounds for column in axes(destination, 2)
            QDLDL.solve!(factor, view(destination, :, column))
        end
    end
    all(isfinite, destination) || throw(ArgumentError(
        "QDLDL sparse LDL solve produced non-finite values",
    ))
    cache.solve_count += nrhs
    return destination
end

MFLA.solve!(cache::MFSparseLDLCache{MF}, destination::AbstractVecOrMat{MF},
            rhs::AbstractVecOrMat{MF}) where {MF<:MultiFloat} =
    _solve!(cache, destination, rhs)

function MFLA.factor_diagnostics(cache::MFSparseLDLCache{MF}) where {MF}
    factor = cache.factor
    return (
        provider=:qdldl, kind=:sparse_ldlt, scalar_type=MF,
        pattern_signature=cache.pattern_signature,
        status=MFLA.factor_state(cache), symbolic_count=cache.symbolic_count,
        numeric_factor_count=cache.numeric_factor_count,
        solve_count=cache.solve_count, nnz_k=nnz(cache.matrix),
        nnz_l=factor === nothing ? 0 : nnz(factor.L),
        positive_inertia=cache.status == 0 ? cache.positive_inertia : -1,
        regularized_entries=cache.status == 0 ? cache.regularized_entries : 0,
    )
end

end
