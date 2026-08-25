"""
Cache factorization and solve methods. The cache structs and their basic
accessors live in [`factor_cache_defs.jl`](@ref); this file holds `prepare!`,
`factorize!`, `solve!`, and their internal helpers, which reuse the same panel
and triangular kernels as the standalone factor API so the warm path is
allocation-free and deterministic.
"""

################################################################################
# Cholesky cache
################################################################################

function MFCholeskyCache(::Type{MF}; config::KernelConfig=KernelConfig()) where {MF<:MultiFloat}
    _check_supported(MF)
    return MFCholeskyCache{MF}(
        Matrix{MF}(undef, 0, 0), _FACTOR_CACHE_INVALID, config,
        zero(UInt), zero(UInt), (0, 0),
    )
end

"""
    prepare!(cache, n; nrhs=1)

Reserve factor storage for an `n`-by-`n` matrix. `nrhs` is a reserved no-op
accepted for a uniform solver-facing signature: it does not affect storage or
the warm path, because `solve!` writes into caller-owned destinations. Growth
here is explicit and allowed to allocate. The hot `factorize!` path never grows.
"""
function prepare!(cache::MFCholeskyCache{MF}, n::Integer; nrhs::Integer=1) where {MF<:MultiFloat}
    n >= 0 || throw(ArgumentError("matrix size must be nonnegative"))
    nrhs >= 1 || throw(ArgumentError("nrhs must be positive"))
    size(cache.factors, 1) == n || (cache.factors = Matrix{MF}(undef, n, n))
    cache.prepared_shape = (n, n)
    cache.prepared_epoch = cache.config_epoch
    cache.status = _FACTOR_CACHE_INVALID
    return cache
end

function factorize!(
    cache::MFCholeskyCache{MF},
    A::AbstractMatrix{MF};
    check::Bool=true,
    config::KernelConfig=cache.config,
) where {MF<:MultiFloat}
    _check_config_frozen(cache, config)
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("cholesky cache requires a square matrix"))
    _check_supported(MF)
    Base.require_one_based_indexing(A)
    _check_prepared(cache, (n, n))
    # Fail-closed: invalidate before touching numerical storage so an unexpected
    # exception can never leave a stale success behind.
    invalidate!(cache)
    copyto!(cache.factors, A)
    status = _cholesky_factorize_core!(cache.factors, config, false)
    cache.status = status
    check && !iszero(status) && throw(_cholesky_exception(status))
    return cache
end

_cholesky_exception(status::Int) =
    status == -1 ? DomainError(status, "cholesky!: input contains non-finite entries") :
    LinearAlgebra.PosDefException(status)

"""
    solve!(x, cache::MFCholeskyCache, b)
    solve!(X, cache::MFCholeskyCache, B)

Solve into caller-owned `x`/`X` with the cached factor. The RHS is copied into
the destination before the solve; exact source/destination identity is allowed,
other overlap is rejected. No factor wrapper or temporary storage is created.
"""
function solve!(
    destination::AbstractVector{MF},
    cache::MFCholeskyCache{MF},
    source::AbstractVector{MF};
    config::KernelConfig=cache.config,
) where {MF<:MultiFloat}
    _check_config_frozen(cache, config)
    issuccess(cache) || throw(LinearAlgebra.PosDefException(factor_status(cache)))
    Base.require_one_based_indexing(destination, source)
    n = size(cache.factors, 1)
    length(destination) == n && length(source) == n ||
        throw(DimensionMismatch("right-hand side length differs"))
    _check_no_alias(destination, cache.factors)
    _copy_rhs!(destination, source)
    trsv!(destination, cache.factors; uplo=:lower, trans=:N, diag=:nonunit, config=config)
    trsv!(destination, cache.factors; uplo=:lower, trans=:T, diag=:nonunit, config=config)
    return destination
end

function solve!(
    destination::AbstractMatrix{MF},
    cache::MFCholeskyCache{MF},
    source::AbstractMatrix{MF};
    config::KernelConfig=cache.config,
) where {MF<:MultiFloat}
    _check_config_frozen(cache, config)
    issuccess(cache) || throw(LinearAlgebra.PosDefException(factor_status(cache)))
    Base.require_one_based_indexing(destination, source, cache.factors)
    size(destination) == size(source) || throw(DimensionMismatch("source and destination dimensions differ"))
    size(destination, 1) == size(cache.factors, 1) ||
        throw(DimensionMismatch("right-hand side dimensions differ"))
    _check_no_alias(destination, cache.factors)
    _copy_rhs!(destination, source)
    _trsm!(destination, cache.factors, one(MF), :left, :lower, :N, :nonunit, config)
    _trsm!(destination, cache.factors, one(MF), :left, :lower, :T, :nonunit, config)
    return destination
end

################################################################################
# LU cache
################################################################################

function MFLUCache(::Type{MF}; config::KernelConfig=KernelConfig()) where {MF<:MultiFloat}
    _check_supported(MF)
    return MFLUCache{MF}(
        Matrix{MF}(undef, 0, 0),
        Int[],
        _FACTOR_CACHE_INVALID,
        zero(MF),
        config,
        zero(UInt), zero(UInt), (0, 0),
    )
end

function prepare!(cache::MFLUCache{MF}, n::Integer; nrhs::Integer=1) where {MF<:MultiFloat}
    n >= 0 || throw(ArgumentError("matrix size must be nonnegative"))
    nrhs >= 1 || throw(ArgumentError("nrhs must be positive"))
    size(cache.factors, 1) == n || (cache.factors = Matrix{MF}(undef, n, n))
    length(cache.ipiv) == n || (cache.ipiv = Vector{Int}(undef, n))
    cache.prepared_shape = (n, n)
    cache.prepared_epoch = cache.config_epoch
    cache.status = _FACTOR_CACHE_INVALID
    return cache
end

function factorize!(
    cache::MFLUCache{MF},
    A::AbstractMatrix{MF};
    check::Bool=true,
    config::KernelConfig=cache.config,
) where {MF<:MultiFloat}
    _check_config_frozen(cache, config)
    m, n = size(A)
    _check_supported(MF)
    Base.require_one_based_indexing(A)
    _check_prepared(cache, (m, n))
    # Fail-closed: invalidate before touching numerical storage.
    invalidate!(cache)
    copyto!(cache.factors, A)
    cache.original_maximum = _maximum_abs(cache.factors)
    # Deterministically initialize pivots to identity so a nonfinite failure
    # (which returns before the core writes pivots) leaves a well-defined state
    # rather than stale data from a prior round. Allocation-free on the warm path.
    @inbounds for i in eachindex(cache.ipiv)
        cache.ipiv[i] = i
    end
    status = _lu_factorize_core_viewfree!(cache.factors, cache.ipiv, config, false)
    cache.status = status
    check && !iszero(status) && throw(LinearAlgebra.SingularException(status))
    return cache
end

function solve!(
    destination::AbstractVector{MF},
    cache::MFLUCache{MF},
    source::AbstractVector{MF};
    config::KernelConfig=cache.config,
) where {MF<:MultiFloat}
    _check_config_frozen(cache, config)
    issuccess(cache) || throw(LinearAlgebra.SingularException(factor_status(cache)))
    Base.require_one_based_indexing(destination, source)
    n = size(cache.factors, 1)
    size(cache.factors, 2) == n || throw(DimensionMismatch("solve requires a square LU factor"))
    length(destination) == n && length(source) == n ||
        throw(DimensionMismatch("right-hand side length differs"))
    _check_no_alias(destination, cache.factors)
    _copy_rhs!(destination, source)
    _apply_pivots!(destination, cache.ipiv)
    trsv!(destination, cache.factors; uplo=:lower, trans=:N, diag=:unit, config=config)
    trsv!(destination, cache.factors; uplo=:upper, trans=:N, diag=:nonunit, config=config)
    return destination
end

function solve!(
    destination::AbstractMatrix{MF},
    cache::MFLUCache{MF},
    source::AbstractMatrix{MF};
    config::KernelConfig=cache.config,
) where {MF<:MultiFloat}
    _check_config_frozen(cache, config)
    issuccess(cache) || throw(LinearAlgebra.SingularException(factor_status(cache)))
    Base.require_one_based_indexing(destination, source, cache.factors)
    size(destination) == size(source) || throw(DimensionMismatch("source and destination dimensions differ"))
    size(destination, 1) == size(cache.factors, 1) ||
        throw(DimensionMismatch("right-hand side dimensions differ"))
    _check_no_alias(destination, cache.factors)
    _copy_rhs!(destination, source)
    _apply_pivots!(destination, cache.ipiv)
    _trsm!(destination, cache.factors, one(MF), :left, :lower, :N, :unit, config)
    _trsm!(destination, cache.factors, one(MF), :left, :upper, :N, :nonunit, config)
    return destination
end

################################################################################
# LDLT cache
################################################################################

function MFLDLTCache(::Type{MF}; config::KernelConfig=KernelConfig()) where {MF<:MultiFloat}
    _check_supported(MF)
    return MFLDLTCache{MF}(
        Matrix{MF}(undef, 0, 0),
        Vector{MF}(undef, 0),
        Vector{Int}(undef, 0),
        Vector{UInt8}(undef, 0),
        Matrix{MF}(undef, 0, 0),
        _FACTOR_CACHE_INVALID,
        zero(MF),
        config,
        zero(UInt), zero(UInt), (0, 0),
    )
end

function prepare!(cache::MFLDLTCache{MF}, n::Integer; nrhs::Integer=1) where {MF<:MultiFloat}
    n >= 0 || throw(ArgumentError("matrix size must be nonnegative"))
    nrhs >= 1 || throw(ArgumentError("nrhs must be positive"))
    size(cache.factors, 1) == n || (cache.factors = Matrix{MF}(undef, n, n))
    length(cache.dsub) == n || (cache.dsub = Vector{MF}(undef, n))
    length(cache.pivots) == n || (cache.pivots = Vector{Int}(undef, n))
    length(cache.blocks) == n || (cache.blocks = Vector{UInt8}(undef, n))
    # Weighted panel is allocated in prepare! according to the frozen ldlt_plan,
    # so factorize! never grows it on the hot path.
    plan = ldlt_plan(MF, n, cache.config)
    block_capacity = plan.strategy === :blocked ? plan.block_size : 0
    if block_capacity > 0 && (size(cache.weighted, 1) != n || size(cache.weighted, 2) < block_capacity + 1)
        cache.weighted = Matrix{MF}(undef, n, block_capacity + 1)
    end
    cache.prepared_shape = (n, n)
    cache.prepared_epoch = cache.config_epoch
    cache.status = _FACTOR_CACHE_INVALID
    return cache
end

function factorize!(
    cache::MFLDLTCache{MF},
    A::AbstractMatrix{MF};
    check::Bool=true,
    config::KernelConfig=cache.config,
) where {MF<:MultiFloat}
    _check_config_frozen(cache, config)
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("ldlt cache requires a square matrix"))
    _check_supported(MF)
    Base.require_one_based_indexing(A)
    _check_prepared(cache, (n, n))
    # Fail-closed: invalidate before touching numerical storage.
    invalidate!(cache)
    copyto!(cache.factors, A)
    # Deterministically initialize metadata so a nonfinite failure (which
    # returns before the core initializes it) leaves a well-defined state
    # rather than stale data from a prior round. Allocation-free on the warm path.
    fill!(cache.dsub, zero(MF))
    fill!(cache.blocks, UInt8(0))
    @inbounds for i in eachindex(cache.pivots)
        cache.pivots[i] = i
    end
    info, original_maximum = _ldlt_factorize_core!(
        cache.factors, cache.dsub, cache.pivots, cache.blocks, cache.weighted,
        config, false,
    )
    cache.original_maximum = original_maximum
    cache.status = info
    check && !iszero(info) && throw(LinearAlgebra.SingularException(info))
    return cache
end

# LDLT forward/backward triangular solves: vector destinations use trsv!, matrix
# destinations use trsm! (the standalone LDLT solve does the same).
@inline function _ldlt_tri_solve!(destination::AbstractVector{MF}, factors, trans::Symbol, config) where {MF<:MultiFloat}
    trsv!(destination, factors; uplo=:lower, trans=trans, diag=:unit, config=config)
    return destination
end
@inline function _ldlt_tri_solve!(destination::AbstractMatrix{MF}, factors, trans::Symbol, config) where {MF<:MultiFloat}
    _trsm!(destination, factors, one(MF), :left, :lower, trans, :unit, config)
    return destination
end

function _ldlt_cache_solve!(destination, cache::MFLDLTCache{MF}, config::KernelConfig) where {MF<:MultiFloat}
    _ldlt_cache_apply_forward!(destination, cache)
    _ldlt_tri_solve!(destination, cache.factors, :N, config)
    _ldlt_cache_solve_d!(destination, cache)
    _ldlt_tri_solve!(destination, cache.factors, :T, config)
    _ldlt_cache_apply_reverse!(destination, cache)
    return destination
end

function _ldlt_cache_apply_forward!(destination, cache::MFLDLTCache)
    k = 1
    @inbounds while k <= length(cache.blocks)
        block = cache.blocks[k]
        if block == UInt8(1)
            _ldlt_swap!(destination, k, cache.pivots[k])
            k += 1
        elseif block == UInt8(2)
            _ldlt_swap!(destination, k + 1, cache.pivots[k])
            k += 2
        else
            throw(ArgumentError("invalid LDLT block structure"))
        end
    end
    return destination
end

function _ldlt_cache_apply_reverse!(destination, cache)
    k = length(cache.blocks)
    @inbounds while k >= 1
        if cache.blocks[k] == UInt8(1)
            _ldlt_swap!(destination, k, cache.pivots[k])
            k -= 1
        elseif cache.blocks[k] == UInt8(0) && k > 1 && cache.blocks[k - 1] == UInt8(2)
            _ldlt_swap!(destination, k, cache.pivots[k - 1])
            k -= 2
        else
            throw(ArgumentError("invalid LDLT block structure"))
        end
    end
    return destination
end

@inline function _ldlt_swap!(destination::AbstractVector, first::Int, second::Int)
    first == second && return nothing
    destination[first], destination[second] = destination[second], destination[first]
    return nothing
end

@inline function _ldlt_swap!(destination::AbstractMatrix, first::Int, second::Int)
    first == second && return nothing
    @inbounds for column in axes(destination, 2)
        destination[first, column], destination[second, column] =
            destination[second, column], destination[first, column]
    end
    return nothing
end

function _ldlt_cache_solve_d!(destination::AbstractVector{MF}, cache::MFLDLTCache{MF}) where {MF<:MultiFloat}
    n = length(cache.blocks)
    k = 1
    @inbounds while k <= n
        if cache.blocks[k] == UInt8(1)
            destination[k] /= cache.factors[k, k]
            k += 1
        else
            d11 = cache.factors[k, k]
            d21 = cache.dsub[k]
            d22 = cache.factors[k + 1, k + 1]
            first, second = destination[k], destination[k + 1]
            solved_first, solved_second, nonsingular = _ldlt_solve_2x2(d11, d21, d22, first, second)
            nonsingular || throw(LinearAlgebra.SingularException(k))
            destination[k] = solved_first
            destination[k + 1] = solved_second
            k += 2
        end
    end
    return destination
end

function _ldlt_cache_solve_d!(destination::AbstractMatrix{MF}, cache::MFLDLTCache{MF}) where {MF<:MultiFloat}
    n = length(cache.blocks)
    k = 1
    @inbounds while k <= n
        if cache.blocks[k] == UInt8(1)
            d = cache.factors[k, k]
            for column in axes(destination, 2)
                destination[k, column] /= d
            end
            k += 1
        else
            d11 = cache.factors[k, k]
            d21 = cache.dsub[k]
            d22 = cache.factors[k + 1, k + 1]
            for column in axes(destination, 2)
                first, second = destination[k, column], destination[k + 1, column]
                solved_first, solved_second, nonsingular = _ldlt_solve_2x2(d11, d21, d22, first, second)
                nonsingular || throw(LinearAlgebra.SingularException(k))
                destination[k, column] = solved_first
                destination[k + 1, column] = solved_second
            end
            k += 2
        end
    end
    return destination
end

function solve!(
    destination::AbstractVector{MF},
    cache::MFLDLTCache{MF},
    source::AbstractVector{MF};
    config::KernelConfig=cache.config,
) where {MF<:MultiFloat}
    _check_config_frozen(cache, config)
    issuccess(cache) || throw(LinearAlgebra.SingularException(factor_status(cache)))
    Base.require_one_based_indexing(destination, source, cache.factors)
    n = size(cache.factors, 1)
    length(destination) == n && length(source) == n ||
        throw(DimensionMismatch("right-hand side length differs"))
    _check_no_alias(destination, cache.factors)
    _copy_rhs!(destination, source)
    return _ldlt_cache_solve!(destination, cache, config)
end

function solve!(
    destination::AbstractMatrix{MF},
    cache::MFLDLTCache{MF},
    source::AbstractMatrix{MF};
    config::KernelConfig=cache.config,
) where {MF<:MultiFloat}
    _check_config_frozen(cache, config)
    issuccess(cache) || throw(LinearAlgebra.SingularException(factor_status(cache)))
    Base.require_one_based_indexing(destination, source, cache.factors)
    size(destination) == size(source) || throw(DimensionMismatch("source and destination dimensions differ"))
    size(destination, 1) == size(cache.factors, 1) ||
        throw(DimensionMismatch("right-hand side dimensions differ"))
    _check_no_alias(destination, cache.factors)
    _copy_rhs!(destination, source)
    return _ldlt_cache_solve!(destination, cache, config)
end

################################################################################
# RRQR cache
################################################################################

function MFRRQRCache(::Type{MF}; config::KernelConfig=KernelConfig()) where {MF<:MultiFloat}
    _check_supported(MF)
    return MFRRQRCache{MF}(
        Matrix{MF}(undef, 0, 0),
        Vector{MF}(undef, 0),
        Vector{Int}(undef, 0),
        Vector{Int}(undef, 0),
        0,
        Vector{MF}(undef, 0),
        Vector{MF}(undef, 0),
        Vector{Bool}(undef, 0),
        Matrix{MF}(undef, 0, 0),
        Vector{MF}(undef, 0),
        _FACTOR_CACHE_INVALID,
        config,
        zero(UInt), zero(UInt), (0, 0),
    )
end

function prepare!(cache::MFRRQRCache{MF}, n::Integer; nrhs::Integer=1) where {MF<:MultiFloat}
    prepare!(cache, n, n; nrhs=nrhs)
end

function prepare!(
    cache::MFRRQRCache{MF},
    m::Integer,
    n::Integer;
    nrhs::Integer=1,
) where {MF<:MultiFloat}
    m >= 0 && n >= 0 || throw(ArgumentError("matrix dimensions must be nonnegative"))
    nrhs >= 1 || throw(ArgumentError("nrhs must be positive"))
    reflector_count = min(m, n)
    size(cache.factors) == (m, n) || (cache.factors = Matrix{MF}(undef, m, n))
    length(cache.tau) == reflector_count || (cache.tau = Vector{MF}(undef, reflector_count))
    length(cache.permutation) == n || (cache.permutation = Vector{Int}(undef, n))
    length(cache.cycle_leaders) == n || (cache.cycle_leaders = Vector{Int}(undef, n))
    cache.cycle_count = 0
    length(cache.norm_scale) == n || (cache.norm_scale = Vector{MF}(undef, n))
    length(cache.norm_sum) == n || (cache.norm_sum = Vector{MF}(undef, n))
    length(cache.norm_dirty) == n || (cache.norm_dirty = Vector{Bool}(undef, n))
    block_size = min(_QR_BLOCK_SIZE, reflector_count)
    # Always reserve at least one row so the zero-reflector (m=0 or n=0) case
    # matches factor_cache_requirements exactly; the buffer is unused there.
    alloc_rows = max(block_size, 1)
    if size(cache.ftranspose, 1) < alloc_rows || size(cache.ftranspose, 2) < n
        cache.ftranspose = Matrix{MF}(undef, alloc_rows, n)
    end
    length(cache.auxiliary) < alloc_rows &&
        (cache.auxiliary = Vector{MF}(undef, alloc_rows))
    cache.prepared_shape = (m, n)
    cache.prepared_epoch = cache.config_epoch
    cache.status = _FACTOR_CACHE_INVALID
    return cache
end

function factorize!(
    cache::MFRRQRCache{MF},
    A::AbstractMatrix{MF};
    check::Bool=true,
    config::KernelConfig=cache.config,
) where {MF<:MultiFloat}
    _check_config_frozen(cache, config)
    m, n = size(A)
    _check_supported(MF)
    Base.require_one_based_indexing(A)
    _check_prepared(cache, (m, n))
    # Fail-closed: invalidate before touching numerical storage.
    invalidate!(cache)
    copyto!(cache.factors, A)
    status = _factorize_rrqr!(cache, false, cache.config.thread_count, config)
    cache.status = status
    check && !iszero(status) && throw(ArgumentError("rrqr!: input contains non-finite entries"))
    return cache
end

function _factorize_rrqr!(cache::MFRRQRCache{MF}, check::Bool, threads::Int, config::KernelConfig) where {MF<:MultiFloat}
    A = cache.factors
    m, n = size(A)
    reflector_count = min(m, n)
    # Deterministically initialize metadata BEFORE the nonfinite check so a
    # nonfinite failure leaves a well-defined (identity/zero) state rather than
    # stale data from a prior round. Allocation-free on the warm path.
    fill!(cache.tau, zero(MF))
    @inbounds for column in 1:n
        cache.permutation[column] = column
    end
    finite_input = _all_finite(A)
    if !finite_input
        check && throw(ArgumentError("rrqr!: input matrix contains non-finite entries"))
        return -1
    end
    _qr_initialize_norm_state!(A, cache.norm_scale, cache.norm_sum, cache.norm_dirty)
    norm_margin = sqrt(eps(MF))
    norm_reliability_floor = MF(16) * eps(MF)

    if _qr_use_blocked_panel(m, n, reflector_count)
        _rrqr_blocked!(
            A, cache.tau, cache.permutation,
            cache.norm_scale, cache.norm_sum, cache.norm_dirty,
            norm_reliability_floor, threads, cache,
        )
    else
        _rrqr_unblocked!(
            A, cache.tau, cache.permutation,
            cache.norm_scale, cache.norm_sum, cache.norm_dirty,
            norm_margin, norm_reliability_floor, threads,
        )
    end
    # Fixed-capacity cycle leaders: write into the owned buffer and record the
    # count; never allocate a fresh leaders vector on the hot path.
    cache.cycle_count = _qr_cycle_leaders_fixed!(cache.cycle_leaders, cache.permutation)
    return 0
end

# Writes the cycle leaders of `permutation` (p such that p[p[i]] ... cycles) into
# the caller-owned buffer `leaders` and returns the number written. Mirrors the
# standalone `_qr_permutation_cycle_leaders!` but never allocates a new vector.
function _qr_cycle_leaders_fixed!(leaders::Vector{Int}, permutation::AbstractVector{Int})
    count = 0
    n = length(permutation)
    # leaders has length >= n; cycle leaders are a subset.
    @inbounds for start in 1:n
        permutation[start] > 0 || continue
        current = start
        cycle_length = 0
        while permutation[current] > 0
            next = permutation[current]
            permutation[current] = -next
            current = next
            cycle_length += 1
        end
        if cycle_length > 1
            count += 1
            leaders[count] = start
        end
    end
    @inbounds for index in 1:n
        permutation[index] = -permutation[index]
    end
    return count
end

function _cache_qr_square_check(destination, cache::MFRRQRCache{MF}) where {MF<:MultiFloat}
    issuccess(cache) || throw(ArgumentError("solve! requires a successful QR cache"))
    n, m = size(cache.factors)
    n == m || throw(DimensionMismatch(
        "QR cache solve! is defined only for square factors; use apply_q!/solve_r! for caller-selected rectangular routes",
    ))
    size(destination, 1) == n || throw(DimensionMismatch("right-hand side dimensions differ"))
    @inbounds for index in 1:n
        iszero(cache.factors[index, index]) && throw(LinearAlgebra.SingularException(index))
    end
    return n
end

function _cache_apply_q!(destination, cache::MFRRQRCache{MF}; trans::Symbol=:N) where {MF<:MultiFloat}
    trans in (:N, :T) || throw(ArgumentError("trans must be :N or :T"))
    steps = trans === :N ? reverse(eachindex(cache.tau)) : eachindex(cache.tau)
    for step in steps
        _cache_apply_reflector!(destination, cache, step)
    end
    return nothing
end

function _cache_apply_reflector!(destination::AbstractVector{MF}, cache, step::Int) where {MF<:MultiFloat}
    tau = cache.tau[step]
    iszero(tau) && return nothing
    projection = destination[step]
    @inbounds for row in (step + 1):size(cache.factors, 1)
        projection += cache.factors[row, step] * destination[row]
    end
    projection *= tau
    destination[step] -= projection
    @inbounds for row in (step + 1):size(cache.factors, 1)
        destination[row] -= cache.factors[row, step] * projection
    end
    return nothing
end

function _cache_apply_reflector!(destination::AbstractMatrix{MF}, cache, step::Int) where {MF<:MultiFloat}
    tau = cache.tau[step]
    iszero(tau) && return nothing
    @inbounds for column in axes(destination, 2)
        projection = destination[step, column]
        for row in (step + 1):size(cache.factors, 1)
            projection += cache.factors[row, step] * destination[row, column]
        end
        projection *= tau
        destination[step, column] -= projection
        for row in (step + 1):size(cache.factors, 1)
            destination[row, column] -= cache.factors[row, step] * projection
        end
    end
    return nothing
end

function _cache_solve_r!(destination, cache::MFRRQRCache{MF}, rank::Int; trans::Symbol=:N, config) where {MF<:MultiFloat}
    rank == 0 && return nothing
    if destination isa AbstractVector
        if trans === :N
            _trsv_leading_upper!(destination, cache.factors, rank)
        else
            _trsv_leading_upper_trans!(destination, cache.factors, rank)
        end
    else
        if trans === :N
            _trsm_leading_upper!(destination, cache.factors, rank)
        else
            _trsm_leading_upper_trans!(destination, cache.factors, rank)
        end
    end
    return nothing
end

# Count-aware cycle permutation over the fixed-capacity leaders buffer, so no
# SubArray view is allocated on the solve path.
function _cache_permute_qr_solution!(destination, cache::MFRRQRCache{MF}) where {MF<:MultiFloat}
    leaders = cache.cycle_leaders
    @inbounds for c in 1:cache.cycle_count
        start = leaders[c]
        value = destination[start]
        current = start
        while true
            target = cache.permutation[current]
            if target == start
                destination[target] = value
                break
            end
            next_value = destination[target]
            destination[target] = value
            value = next_value
            current = target
        end
    end
    return destination
end

# Matrix-destination QR solution permutation: apply the column permutation to
# each right-hand-side column independently (mirrors the standalone
# `_permute_qr_solution!` matrix path, which the linear-index vector path would
# corrupt because a Matrix is column-major).
function _cache_permute_qr_solution!(destination::AbstractMatrix{MF}, cache::MFRRQRCache{MF}) where {MF<:MultiFloat}
    leaders = cache.cycle_leaders
    @inbounds for column in axes(destination, 2), c in 1:cache.cycle_count
        start = leaders[c]
        value = destination[start, column]
        current = start
        while true
            target = cache.permutation[current]
            if target == start
                destination[target, column] = value
                break
            end
            next_value = destination[target, column]
            destination[target, column] = value
            value = next_value
            current = target
        end
    end
    return destination
end

function solve!(
    destination::AbstractVector{MF},
    cache::MFRRQRCache{MF},
    source::AbstractVector{MF};
    config::KernelConfig=cache.config,
) where {MF<:MultiFloat}
    _check_config_frozen(cache, config)
    n = _cache_qr_square_check(destination, cache)
    length(source) == n || throw(DimensionMismatch("right-hand side length differs"))
    _check_no_alias(destination, cache.factors)
    _copy_rhs!(destination, source)
    _cache_apply_q!(destination, cache; trans=:T)
    _cache_solve_r!(destination, cache, n; config=config)
    _cache_permute_qr_solution!(destination, cache)
    return destination
end

function solve!(
    destination::AbstractMatrix{MF},
    cache::MFRRQRCache{MF},
    source::AbstractMatrix{MF};
    config::KernelConfig=cache.config,
) where {MF<:MultiFloat}
    _check_config_frozen(cache, config)
    n = _cache_qr_square_check(destination, cache)
    size(destination) == size(source) || throw(DimensionMismatch("source and destination dimensions differ"))
    size(destination, 1) == n || throw(DimensionMismatch("right-hand side dimensions differ"))
    _check_no_alias(destination, cache.factors)
    _copy_rhs!(destination, source)
    _cache_apply_q!(destination, cache; trans=:T)
    _cache_solve_r!(destination, cache, n; config=config)
    return _cache_permute_qr_solution!(destination, cache)
end

################################################################################
# Public rectangular RRQR cache route (SDPX equality/null-space use).
################################################################################

"""
    apply_q!(destination, cache::MFRRQRCache; trans=:N)

Overwrite a vector or matrix `B` with `Q*B` (`trans=:N`) or `Q'*B`
(`trans=:T`) using the cache's compact Householder representation. Supports
tall, wide, and square factors and vector or multi-RHS destinations. `B` must
have one row per row of the factorized matrix and must not alias the factor
storage. Zero allocation on the warm path.
"""
function apply_q!(
    destination::Union{AbstractVector{MF},AbstractMatrix{MF}},
    cache::MFRRQRCache{MF};
    trans::Symbol=:N,
) where {MF<:MultiFloat}
    _check_config_frozen(cache, cache.config)
    issuccess(cache) || throw(ArgumentError("apply_q! requires a successful QR cache"))
    trans in (:N, :T) || throw(ArgumentError("trans must be :N or :T"))
    Base.require_one_based_indexing(destination, cache.factors)
    size(destination, 1) == size(cache.factors, 1) ||
        throw(DimensionMismatch("apply_q! destination row count differs"))
    Base.mightalias(destination, cache.factors) &&
        throw(ArgumentError("apply_q! destination must not alias QR storage"))
    _cache_apply_q!(destination, cache; trans=trans)
    return destination
end

"""
    solve_r!(destination, cache::MFRRQRCache, rank; trans=:N, config=cache.config)

Solve the caller-selected leading `rank` block of `R` in place. `trans=:N`
solves `R[1:rank,1:rank] * X = B`; `trans=:T` solves the transposed system.
The destination must have exactly `rank` rows. The caller chooses the rank
threshold; MFLA never infers rank. Vector and matrix destinations are
supported.
"""
function solve_r!(
    destination::Union{AbstractVector{MF},AbstractMatrix{MF}},
    cache::MFRRQRCache{MF},
    rank::Integer;
    trans::Symbol=:N,
    config::KernelConfig=cache.config,
) where {MF<:MultiFloat}
    _check_config_frozen(cache, config)
    rank_value = Int(rank)
    issuccess(cache) || throw(ArgumentError("solve_r! requires a successful QR cache"))
    trans in (:N, :T) || throw(ArgumentError("trans must be :N or :T"))
    maximum_rank = min(size(cache.factors)...)
    0 <= rank_value <= maximum_rank ||
        throw(ArgumentError("rank must be between zero and $maximum_rank"))
    size(destination, 1) == rank_value ||
        throw(DimensionMismatch("solve_r! destination row count must equal rank"))
    Base.require_one_based_indexing(destination, cache.factors)
    @inbounds for index in 1:rank_value
        iszero(cache.factors[index, index]) &&
            throw(LinearAlgebra.SingularException(index))
    end
    _cache_solve_r!(destination, cache, rank_value; trans=trans, config=config)
    return destination
end

"""
    numerical_rank(cache::MFRRQRCache; atol=0, rtol=0) -> Int

Evaluate the leading numerical rank with the caller's nonnegative absolute and
relative thresholds, exactly as [`numerical_rank`](@ref) does for the standalone
RRQR factor. Diagonal `i` is accepted while
`abs(R[i,i]) > max(atol, rtol * maximum(abs.(diag(R))))`. MFLA does not select a
rank threshold.
"""
function numerical_rank(
    cache::MFRRQRCache{MF};
    atol::Real=zero(MF),
    rtol::Real=zero(MF),
) where {MF<:MultiFloat}
    issuccess(cache) || throw(ArgumentError("numerical_rank requires a successful QR cache"))
    absolute_tolerance = MF(atol)
    relative_tolerance = MF(rtol)
    isfinite(absolute_tolerance) && absolute_tolerance >= zero(MF) ||
        throw(ArgumentError("atol must be finite and nonnegative"))
    isfinite(relative_tolerance) && relative_tolerance >= zero(MF) ||
        throw(ArgumentError("rtol must be finite and nonnegative"))

    diagonal_count = min(size(cache.factors)...)
    largest = zero(MF)
    @inbounds for index in 1:diagonal_count
        largest = max(largest, abs(cache.factors[index, index]))
    end
    threshold = max(absolute_tolerance, relative_tolerance * largest)
    rank = 0
    @inbounds for index in 1:diagonal_count
        abs(cache.factors[index, index]) > threshold || break
        rank += 1
    end
    return rank
end

"""
    factor_permutation(cache::MFRRQRCache) -> Vector{Int}

Return a caller-owned copy of the column permutation `p` satisfying
`A_original[:, p] = Q * R`. Requires a successful QR cache.
"""
function factor_permutation(cache::MFRRQRCache)
    issuccess(cache) || throw(ArgumentError("factor_permutation requires a successful QR cache"))
    return copy(cache.permutation)
end

"""
    factor_rdiag(cache::MFRRQRCache) -> Vector{MF}

Return a copy of the signed diagonal of the compactly stored `R` factor.
Requires a successful QR cache.
"""
function factor_rdiag(cache::MFRRQRCache{MF}) where {MF<:MultiFloat}
    issuccess(cache) || throw(ArgumentError("factor_rdiag requires a successful QR cache"))
    diagonal_count = min(size(cache.factors)...)
    diagonal = Vector{MF}(undef, diagonal_count)
    @inbounds for index in 1:diagonal_count
        diagonal[index] = cache.factors[index, index]
    end
    return diagonal
end
