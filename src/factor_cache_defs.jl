"""
    AbstractMFFactorCache{MF}

Reusable, caller-owned factorization storage. A factor cache owns every array
its factorization and solve need (factor matrix, pivots, block grammar,
permutation, reflectors, norm scratch, and the internal reusable workspace), so
the warm hot path performs no allocation:

  - repeated `factorize!` at the same size overwrites storage only;
  - growth happens only through the explicit `prepare!`/`reserve!` methods;
  - repeated `solve!` creates no temporary RHS, factor wrapper, or metadata copy.

A cache factor is *borrowed* storage owned by the cache and is invalidated by
the next `factorize!`. It is deliberately not a long-lived, independently
ownable factor: callers that need a persistent factor must continue to use the
safe standalone `lu!`/`ldlt!`/`cholesky!`/`rrqr!` API, which returns an object
that owns its metadata.

Caches are not concurrency-safe: one cache must not be used by concurrent
factorization or solve calls. Distinct caches retain call-level concurrency.
"""
abstract type AbstractMFFactorCache{MF<:MultiFloat} end

# status sentinel for a not-yet-factorized / invalidated cache
const _FACTOR_CACHE_INVALID = -2

"""
    MFCholeskyCache(::Type{MF}; config=KernelConfig())

Reusable cache for a lower Cholesky factorization. Owns its factor matrix and a
`KernelConfig`; repeated `factorize!` overwrites the owned factor matrix and
`status`. The authoritative triangle on output is the lower triangle.
"""
mutable struct MFCholeskyCache{MF<:MultiFloat} <: AbstractMFFactorCache{MF}
    factors::Matrix{MF}
    status::Int
    config::KernelConfig
end

"""
    MFLUCache(::Type{MF}; config=KernelConfig())

Reusable cache for a dense partial-pivoting LU factorization. Owns the factor
matrix and the pivot vector; repeated `factorize!` overwrites both in place.
"""
mutable struct MFLUCache{MF<:MultiFloat} <: AbstractMFFactorCache{MF}
    factors::Matrix{MF}
    ipiv::Vector{Int}
    status::Int
    original_maximum::MF
    gemm::GemmWorkspace{MF}
    config::KernelConfig
end

"""
    MFLDLTCache(::Type{MF}; config=KernelConfig())

Reusable cache for a symmetric-indefinite `L*D*L'` factorization. Owns the
factor matrix, the subdiagonal `D` entries, the Bunch-Kaufman pivot and block
vectors, and blocked-panel weighted storage. Repeated `factorize!` overwrites
all owned storage.
"""
mutable struct MFLDLTCache{MF<:MultiFloat} <: AbstractMFFactorCache{MF}
    factors::Matrix{MF}
    dsub::Vector{MF}
    pivots::Vector{Int}
    blocks::Vector{UInt8}
    weighted::Matrix{MF}
    status::Int
    original_maximum::MF
    gemm::GemmWorkspace{MF}
    config::KernelConfig
end

"""
    MFRRQRCache(::Type{MF}; config=KernelConfig())

Reusable cache for a deterministic column-pivoted Householder QR factorization.
Owns the factor matrix, reflectors (`tau`), the column permutation and its
cycle leaders, and the delayed-norm / blocked-panel scratch used during
factorization. `factorize!` overwrites all owned storage.
"""
mutable struct MFRRQRCache{MF<:MultiFloat} <: AbstractMFFactorCache{MF}
    factors::Matrix{MF}
    tau::Vector{MF}
    permutation::Vector{Int}
    cycle_leaders::Vector{Int}
    cycle_count::Int
    norm_scale::Vector{MF}
    norm_sum::Vector{MF}
    norm_dirty::Vector{Bool}
    ftranspose::Matrix{MF}
    auxiliary::Vector{MF}
    gemm::GemmWorkspace{MF}
    status::Int
    config::KernelConfig
end

factor_precision(::AbstractMFFactorCache{MF}) where {MF} = MF
factor_provider(::AbstractMFFactorCache) = :mfla

function factor_state(cache::AbstractMFFactorCache)
    status = factor_status(cache)
    status == _FACTOR_CACHE_INVALID && return :invalidated
    status == -1 && return :nonfinite_input
    status < 0 && return :numerical_breakdown
    iszero(status) && return :success
    return factor_kind(cache) === :cholesky ? :not_posdef : :singular
end

issuccess(cache::AbstractMFFactorCache) = iszero(factor_status(cache))

Base.size(cache::AbstractMFFactorCache) = size(factor_matrix(cache))
Base.size(cache::AbstractMFFactorCache, d::Integer) = size(factor_matrix(cache), d)
Base.eltype(::AbstractMFFactorCache{MF}) where {MF} = MF

"""
    invalidate!(cache)

Mark a cache as not holding a usable factorization. Subsequent `solve!` throws
until `factorize!` runs again. This does not resize or otherwise touch owned
storage, so it is allocation-free and is the explicit refresh signal after a
caller mutates `A` in place.
"""
function invalidate!(cache::AbstractMFFactorCache)
    cache.status = _FACTOR_CACHE_INVALID
    return cache
end

factor_kind(::MFCholeskyCache) = :cholesky
factor_status(cache::MFCholeskyCache) = cache.status
factor_matrix(cache::MFCholeskyCache) = cache.factors

factor_kind(::MFLUCache) = :lu
factor_status(cache::MFLUCache) = cache.status
factor_matrix(cache::MFLUCache) = cache.factors

factor_kind(::MFLDLTCache) = :ldlt
factor_status(cache::MFLDLTCache) = cache.status
factor_matrix(cache::MFLDLTCache) = cache.factors

factor_kind(::MFRRQRCache) = :rrqr
factor_status(cache::MFRRQRCache) = cache.status
factor_matrix(cache::MFRRQRCache) = cache.factors

@inline _check_no_alias(destination, storage) =
    Base.mightalias(destination, storage) &&
        throw(ArgumentError("cache solve does not support destination/factor aliasing"))

@inline function _copy_rhs!(destination, source)
    destination === source && return destination
    Base.mightalias(destination, source) &&
        throw(ArgumentError("cache solve does not support partially overlapping right-hand sides"))
    copyto!(destination, source)
    return destination
end
