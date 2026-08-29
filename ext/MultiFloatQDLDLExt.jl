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
    dsigns::Vector{Ti}
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
    regularize_eps::MF=MF(1e-12), regularize_delta::MF=MF(1e-7),
) where {MF<:MultiFloat,Ti<:Integer}
    nrhs >= 1 || throw(ArgumentError("nrhs must be positive"))
    _validate_pattern(MF, pattern, dsigns)
    matrix = SparseMatrixCSC(
        size(pattern, 1), size(pattern, 2), copy(pattern.colptr),
        copy(pattern.rowval), copy(pattern.nzval),
    )
    signs = Ti[sign for sign in dsigns]
    indices = Ti.(eachindex(matrix.nzval))
    factor = QDLDL.qdldl(
        matrix; logical=true, Dsigns=signs,
        regularize_eps=regularize_eps, regularize_delta=regularize_delta,
    )
    n = size(matrix, 1)
    return MFSparseLDLCache{MF,Ti}(
        matrix, factor, indices, signs, _pattern_signature(matrix, signs),
        Int(nrhs), MFLA._FACTOR_CACHE_INVALID, MFLA.KernelConfig(),
        UInt(0), UInt(0), (n, n), 1, 0, 0,
    )
end

MFLA.factor_kind(::MFSparseLDLCache) = :sparse_ldlt
MFLA.factor_status(cache::MFSparseLDLCache) = cache.status
MFLA.factor_matrix(cache::MFSparseLDLCache) = cache.matrix

function _validate_numeric(cache::MFSparseLDLCache{MF,Ti}, A) where {MF,Ti}
    A isa SparseMatrixCSC{MF,Ti} || throw(ArgumentError(
        "QDLDL sparse LDL numeric matrix must preserve scalar/index types",
    ))
    size(A) == cache.prepared_shape || throw(DimensionMismatch(
        "QDLDL sparse LDL numeric dimensions differ from the frozen pattern",
    ))
    A.colptr == cache.matrix.colptr && A.rowval == cache.matrix.rowval ||
        throw(ArgumentError("QDLDL sparse LDL pattern drift"))
    all(isfinite, A.nzval) || throw(ArgumentError(
        "QDLDL sparse LDL numeric input contains non-finite values",
    ))
    return nothing
end

function MFLA.factorize!(
    cache::MFSparseLDLCache{MF,Ti}, A::SparseMatrixCSC{MF,Ti};
    check::Bool=true,
) where {MF<:MultiFloat,Ti<:Integer}
    cache.status = MFLA._FACTOR_CACHE_INVALID
    try
        _validate_numeric(cache, A)
        copyto!(cache.matrix.nzval, A.nzval)
        factor = something(cache.factor)
        QDLDL.update_values!(factor, cache.indices, cache.matrix.nzval)
        QDLDL.refactor!(factor)
        all(isfinite, diag(factor.Dinv)) && all(isfinite, factor.L.nzval) ||
            throw(ArgumentError("QDLDL produced non-finite factors"))
        cache.numeric_factor_count += 1
        cache.status = 0
    catch
        cache.status = -3
        check && rethrow()
    end
    return cache
end

function _solve!(
    cache::MFSparseLDLCache{MF}, destination::AbstractVecOrMat{MF},
    rhs::AbstractVecOrMat{MF},
) where {MF<:MultiFloat}
    MFLA.issuccess(cache) || throw(ArgumentError(
        "QDLDL sparse LDL cache does not hold a successful factor",
    ))
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
        positive_inertia=factor === nothing ? -1 : QDLDL.positive_inertia(factor),
        regularized_entries=factor === nothing ? 0 : QDLDL.regularized_entries(factor),
    )
end

end
