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
    return MFCholeskyCache{MF}(Matrix{MF}(undef, 0, 0), _FACTOR_CACHE_INVALID, config)
end

"""
    prepare!(cache, n; nrhs=1)

Reserve factor storage for an `n`-by-`n` matrix. `nrhs` is accepted for a
uniform solver-facing signature and records intended multi-RHS capacity; growth
here is explicit and allowed to allocate. The hot `factorize!` path never grows.
"""
function prepare!(cache::MFCholeskyCache{MF}, n::Integer; nrhs::Integer=1) where {MF<:MultiFloat}
    n >= 0 || throw(ArgumentError("matrix size must be nonnegative"))
    nrhs >= 1 || throw(ArgumentError("nrhs must be positive"))
    size(cache.factors, 1) == n || (cache.factors = Matrix{MF}(undef, n, n))
    cache.status = _FACTOR_CACHE_INVALID
    return cache
end

function factorize!(
    cache::MFCholeskyCache{MF},
    A::AbstractMatrix{MF};
    check::Bool=true,
    config::KernelConfig=cache.config,
) where {MF<:MultiFloat}
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("cholesky cache requires a square matrix"))
    _check_supported(MF)
    Base.require_one_based_indexing(A)
    size(cache.factors, 1) == n ||
        throw(ArgumentError("cache prepared for size $(size(cache.factors, 1)); call prepare!(cache, $n) first"))
    copyto!(cache.factors, A)
    cache.status = _factorize_cholesky!(cache.factors, config, check)
    return cache
end

function _factorize_cholesky!(A::Matrix{MF}, config::KernelConfig, check::Bool) where {MF<:MultiFloat}
    n = size(A, 1)
    if !_lower_triangle_finite(A)
        check && throw(DomainError(A, "cholesky!: input matrix contains non-finite entries"))
        return -1
    end
    block = max(config.cholesky_block, 1)
    info = 0
    for first in 1:block:n
        last = min(first + block - 1, n)
        info = _factor_cholesky_panel!(A, first, last)
        if !iszero(info)
            check && throw(LinearAlgebra.PosDefException(info))
            return info
        end
        if last < n
            _solve_cholesky_panel!(A, first, last, config)
            trailing = @view A[(last + 1):n, (last + 1):n]
            panel = transpose(@view A[(last + 1):n, first:last])
            syrk!(trailing, panel, -one(MF), one(MF); uplo=:lower, config=config)
        end
    end
    return 0
end

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
    issuccess(cache) || throw(LinearAlgebra.PosDefException(factor_status(cache)))
    Base.require_one_based_indexing(destination, source, cache.factors)
    size(destination) == size(source) || throw(DimensionMismatch("source and destination dimensions differ"))
    size(destination, 1) == size(cache.factors, 1) ||
        throw(DimensionMismatch("right-hand side dimensions differ"))
    _check_no_alias(destination, cache.factors)
    _copy_rhs!(destination, source)
    trsm!(destination, cache.factors; side=:left, uplo=:lower, trans=:N, diag=:nonunit, config=config)
    trsm!(destination, cache.factors; side=:left, uplo=:lower, trans=:T, diag=:nonunit, config=config)
    return destination
end

################################################################################
# LU cache
################################################################################

function MFLUCache(::Type{MF}; config::KernelConfig=KernelConfig()) where {MF<:MultiFloat}
    _check_supported(MF)
    return MFLUCache{MF}(Matrix{MF}(undef, 0, 0), Int[], _FACTOR_CACHE_INVALID, zero(MF), config)
end

function prepare!(cache::MFLUCache{MF}, n::Integer; nrhs::Integer=1) where {MF<:MultiFloat}
    n >= 0 || throw(ArgumentError("matrix size must be nonnegative"))
    nrhs >= 1 || throw(ArgumentError("nrhs must be positive"))
    size(cache.factors, 1) == n || (cache.factors = Matrix{MF}(undef, n, n))
    length(cache.ipiv) == n || (cache.ipiv = Vector{Int}(undef, n))
    cache.status = _FACTOR_CACHE_INVALID
    return cache
end

function factorize!(
    cache::MFLUCache{MF},
    A::AbstractMatrix{MF};
    check::Bool=true,
    config::KernelConfig=cache.config,
) where {MF<:MultiFloat}
    m, n = size(A)
    _check_supported(MF)
    Base.require_one_based_indexing(A)
    size(cache.factors) == (m, n) ||
        throw(ArgumentError("cache prepared for size $(size(cache.factors)); call prepare!(cache, $m) first"))
    copyto!(cache.factors, A)
    cache.status = _factorize_lu!(cache.factors, cache.ipiv, config, check)
    cache.original_maximum = _maximum_abs(cache.factors)
    return cache
end

function _factorize_lu!(A::Matrix{MF}, ipiv::Vector{Int}, config::KernelConfig, check::Bool) where {MF<:MultiFloat}
    m, n = size(A)
    kmax = min(m, n)
    if !_all_finite(A)
        check && throw(DomainError(A, "lu!: input matrix contains non-finite entries"))
        return -1
    end
    info = 0
    block = max(config.lu_block, 1)
    for first in 1:block:kmax
        last = min(first + block - 1, kmax)
        info = _factor_lu_panel!(A, first, last, ipiv)
        if !iszero(info)
            check && throw(LinearAlgebra.SingularException(info))
            return info
        end
        if last < n
            L11 = @view A[first:last, first:last]
            U12 = @view A[first:last, (last + 1):n]
            trsm!(U12, L11; side=:left, uplo=:lower, trans=:N, diag=:unit, config=config)
            if last < m
                A21 = @view A[(last + 1):m, first:last]
                A22 = @view A[(last + 1):m, (last + 1):n]
                gemm!(A22, A21, U12, -one(MF), one(MF); config=config)
            end
        end
    end
    return 0
end

function solve!(
    destination::AbstractVector{MF},
    cache::MFLUCache{MF},
    source::AbstractVector{MF};
    config::KernelConfig=cache.config,
) where {MF<:MultiFloat}
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
    issuccess(cache) || throw(LinearAlgebra.SingularException(factor_status(cache)))
    Base.require_one_based_indexing(destination, source, cache.factors)
    size(destination) == size(source) || throw(DimensionMismatch("source and destination dimensions differ"))
    size(destination, 1) == size(cache.factors, 1) ||
        throw(DimensionMismatch("right-hand side dimensions differ"))
    _check_no_alias(destination, cache.factors)
    _copy_rhs!(destination, source)
    _apply_pivots!(destination, cache.ipiv)
    trsm!(destination, cache.factors; side=:left, uplo=:lower, trans=:N, diag=:unit, config=config)
    trsm!(destination, cache.factors; side=:left, uplo=:upper, trans=:N, diag=:nonunit, config=config)
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
    )
end

function prepare!(cache::MFLDLTCache{MF}, n::Integer; nrhs::Integer=1) where {MF<:MultiFloat}
    n >= 0 || throw(ArgumentError("matrix size must be nonnegative"))
    nrhs >= 1 || throw(ArgumentError("nrhs must be positive"))
    size(cache.factors, 1) == n || (cache.factors = Matrix{MF}(undef, n, n))
    length(cache.dsub) == n || (cache.dsub = Vector{MF}(undef, n))
    length(cache.pivots) == n || (cache.pivots = Vector{Int}(undef, n))
    length(cache.blocks) == n || (cache.blocks = Vector{UInt8}(undef, n))
    cache.status = _FACTOR_CACHE_INVALID
    return cache
end

function factorize!(
    cache::MFLDLTCache{MF},
    A::AbstractMatrix{MF};
    check::Bool=true,
    config::KernelConfig=cache.config,
) where {MF<:MultiFloat}
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("ldlt cache requires a square matrix"))
    _check_supported(MF)
    Base.require_one_based_indexing(A)
    size(cache.factors, 1) == n ||
        throw(ArgumentError("cache prepared for size $(size(cache.factors, 1)); call prepare!(cache, $n) first"))
    copyto!(cache.factors, A)
    cache.status = _factorize_ldlt!(cache, check, config)
    return cache
end

function _factorize_ldlt!(cache::MFLDLTCache{MF}, check::Bool, config::KernelConfig) where {MF<:MultiFloat}
    A = cache.factors
    n = size(A, 1)
    if !_lower_triangle_finite(A)
        check && throw(DomainError(A, "ldlt!: input matrix contains non-finite entries"))
        return -1
    end
    _mirror_lower_to_upper!(A)
    cache.original_maximum = _lower_maximum_abs(A)

    plan = ldlt_plan(MF, n, config)
    block_capacity = plan.strategy === :blocked ? plan.block_size : 0
    if block_capacity > 0 && size(cache.weighted, 1) < n
        cache.weighted = Matrix{MF}(undef, n, block_capacity + 1)
    end
    weighted = block_capacity > 0 ? cache.weighted : Matrix{MF}(undef, 0, 0)

    dsub = cache.dsub
    pivots = cache.pivots
    blocks = cache.blocks
    fill!(dsub, zero(MF))
    fill!(blocks, UInt8(0))
    @inbounds for index in 1:n
        pivots[index] = index
    end

    alpha = (one(MF) + sqrt(MF(17))) / MF(8)
    info = if plan.strategy === :blocked
        _factor_ldlt_blocked!(A, dsub, pivots, blocks, weighted, alpha, plan, config)
    else
        _factor_ldlt_unblocked!(A, dsub, pivots, blocks, alpha)
    end
    if !iszero(info) && check
        throw(LinearAlgebra.SingularException(info))
    end
    return info
end

function _ldlt_cache_solve!(destination, cache::MFLDLTCache{MF}) where {MF<:MultiFloat}
    _ldlt_cache_apply_forward!(destination, cache)
    trsv!(destination, cache.factors; uplo=:lower, trans=:N, diag=:unit, config=cache.config)
    _ldlt_cache_solve_d!(destination, cache)
    trsv!(destination, cache.factors; uplo=:lower, trans=:T, diag=:unit, config=cache.config)
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
    issuccess(cache) || throw(LinearAlgebra.SingularException(factor_status(cache)))
    Base.require_one_based_indexing(destination, source, cache.factors)
    n = size(cache.factors, 1)
    length(destination) == n && length(source) == n ||
        throw(DimensionMismatch("right-hand side length differs"))
    _check_no_alias(destination, cache.factors)
    _copy_rhs!(destination, source)
    return _ldlt_cache_solve!(destination, cache)
end

function solve!(
    destination::AbstractMatrix{MF},
    cache::MFLDLTCache{MF},
    source::AbstractMatrix{MF};
    config::KernelConfig=cache.config,
) where {MF<:MultiFloat}
    issuccess(cache) || throw(LinearAlgebra.SingularException(factor_status(cache)))
    Base.require_one_based_indexing(destination, source, cache.factors)
    size(destination) == size(source) || throw(DimensionMismatch("source and destination dimensions differ"))
    size(destination, 1) == size(cache.factors, 1) ||
        throw(DimensionMismatch("right-hand side dimensions differ"))
    _check_no_alias(destination, cache.factors)
    _copy_rhs!(destination, source)
    return _ldlt_cache_solve!(destination, cache)
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
        Int[],
        Vector{MF}(undef, 0),
        Vector{MF}(undef, 0),
        Vector{Bool}(undef, 0),
        Matrix{MF}(undef, 0, 0),
        Vector{MF}(undef, 0),
        GemmWorkspace(MF; thread_count=config.thread_count, capacity=0),
        _FACTOR_CACHE_INVALID,
        config,
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
    length(cache.norm_scale) == n || (cache.norm_scale = Vector{MF}(undef, n))
    length(cache.norm_sum) == n || (cache.norm_sum = Vector{MF}(undef, n))
    length(cache.norm_dirty) == n || (cache.norm_dirty = Vector{Bool}(undef, n))
    block_size = min(_QR_BLOCK_SIZE, reflector_count)
    if size(cache.ftranspose, 1) < block_size || size(cache.ftranspose, 2) < n
        cache.ftranspose = Matrix{MF}(undef, max(block_size, 1), n)
    end
    length(cache.auxiliary) < block_size &&
        (cache.auxiliary = Vector{MF}(undef, max(block_size, 1)))
    _prepare_gemm_workspace!(cache.gemm, max(cache.config.thread_count, 1), 0)
    cache.status = _FACTOR_CACHE_INVALID
    return cache
end

function factorize!(
    cache::MFRRQRCache{MF},
    A::AbstractMatrix{MF};
    check::Bool=true,
    threads::Int=Threads.nthreads(),
    config::KernelConfig=cache.config,
) where {MF<:MultiFloat}
    m, n = size(A)
    _check_supported(MF)
    Base.require_one_based_indexing(A)
    size(cache.factors) == (m, n) ||
        throw(ArgumentError("cache prepared for size $(size(cache.factors)); call prepare!(cache, $m, $n) first"))
    copyto!(cache.factors, A)
    cache.status = _factorize_rrqr!(cache, check, threads, config)
    return cache
end

function _factorize_rrqr!(cache::MFRRQRCache{MF}, check::Bool, threads::Int, config::KernelConfig) where {MF<:MultiFloat}
    A = cache.factors
    m, n = size(A)
    reflector_count = min(m, n)
    finite_input = _all_finite(A)
    if !finite_input
        check && throw(ArgumentError("rrqr!: input matrix contains non-finite entries"))
        return -1
    end
    fill!(cache.tau, zero(MF))
    @inbounds for column in 1:n
        cache.permutation[column] = column
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
    empty!(cache.cycle_leaders)
    append!(cache.cycle_leaders, _qr_permutation_cycle_leaders!(cache.permutation))
    return 0
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
    return destination
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

function _cache_solve_r!(destination, cache::MFRRQRCache{MF}, rank::Int; config) where {MF<:MultiFloat}
    rank == 0 && return destination
    leading = @view cache.factors[1:rank, 1:rank]
    if destination isa AbstractVector
        trsv!(destination, leading; uplo=:upper, trans=:N, diag=:nonunit, config=config)
    else
        trsm!(destination, leading; side=:left, uplo=:upper, trans=:N, diag=:nonunit, config=config)
    end
    return destination
end

function solve!(
    destination::AbstractVector{MF},
    cache::MFRRQRCache{MF},
    source::AbstractVector{MF};
    config::KernelConfig=cache.config,
) where {MF<:MultiFloat}
    n = _cache_qr_square_check(destination, cache)
    length(source) == n || throw(DimensionMismatch("right-hand side length differs"))
    _check_no_alias(destination, cache.factors)
    _copy_rhs!(destination, source)
    _cache_apply_q!(destination, cache; trans=:T)
    _cache_solve_r!(destination, cache, n; config=config)
    _permute_qr_solution!(destination, cache.permutation, cache.cycle_leaders)
    return destination
end

function solve!(
    destination::AbstractMatrix{MF},
    cache::MFRRQRCache{MF},
    source::AbstractMatrix{MF};
    config::KernelConfig=cache.config,
) where {MF<:MultiFloat}
    n = _cache_qr_square_check(destination, cache)
    size(destination) == size(source) || throw(DimensionMismatch("source and destination dimensions differ"))
    size(destination, 1) == n || throw(DimensionMismatch("right-hand side dimensions differ"))
    _check_no_alias(destination, cache.factors)
    _copy_rhs!(destination, source)
    _cache_apply_q!(destination, cache; trans=:T)
    _cache_solve_r!(destination, cache, n; config=config)
    return _permute_qr_solution!(destination, cache.permutation, cache.cycle_leaders)
end
