module MultiFloatLinearSolveExt

import LinearSolve
import MultiFloatLinearAlgebra
import MultiFloats
import SciMLBase

const MFLA = MultiFloatLinearAlgebra

struct MultiFloatLU <: LinearSolve.SciMLLinearSolveAlgorithm
    config::MFLA.KernelConfig
end

MultiFloatLU(; config::MFLA.KernelConfig=MFLA.KernelConfig()) = MultiFloatLU(config)

struct MultiFloatCholesky <: LinearSolve.SciMLLinearSolveAlgorithm
    config::MFLA.KernelConfig
end

function MultiFloatCholesky(; config::MFLA.KernelConfig=MFLA.KernelConfig())
    return MultiFloatCholesky(config)
end

LinearSolve.needs_concrete_A(::MultiFloatLU) = true
LinearSolve.needs_concrete_A(::MultiFloatCholesky) = true
LinearSolve.needs_square_A(::Union{MultiFloatLU,MultiFloatCholesky}) = true
LinearSolve.default_alias_A(
    ::Union{MultiFloatLU,MultiFloatCholesky},
    ::Any,
    ::Any,
) = false
LinearSolve.default_alias_b(
    ::Union{MultiFloatLU,MultiFloatCholesky},
    ::Any,
    ::Any,
) = false

function _multifloat_matrix(A)
    matrix = convert(AbstractMatrix, A)
    eltype(matrix) <: MultiFloats.MultiFloat || throw(ArgumentError(
        "MultiFloat LinearSolve algorithms require a MultiFloat matrix",
    ))
    return matrix
end

# Build the reusable factor cache at the *matrix* size so the first `solve!`
# writes into preallocated storage instead of growing it. `init_cacheval` is
# called by `LinearSolve.init`, so `prepare!` here reserves the O(n^2) storage
# (factor matrix, pivots, workspaces) up front. No O(n^3) factorization runs at
# init time: the cache's status is left `_FACTOR_CACHE_INVALID`, and the first
# `solve!` performs the single numeric factorization.
function _build_cache(alg::Union{MultiFloatLU,MultiFloatCholesky}, A)
    MF = eltype(_multifloat_matrix(A))
    n = size(A, 1)
    if alg isa MultiFloatLU
        cache = MFLA.MFLUCache(MF; config=alg.config)
    else
        cache = MFLA.MFCholeskyCache(MF; config=alg.config)
    end
    MFLA.prepare!(cache, n)
    return cache
end

function LinearSolve.init_cacheval(
    alg::Union{MultiFloatLU,MultiFloatCholesky},
    A,
    b,
    u,
    Pl,
    Pr,
    maxiters::Int,
    abstol,
    reltol,
    verbose,
    assumptions,
)
    # `LinearCache` fixes the cache-value field type at initialization; build the
    # cache at the matrix size (reserving O(n^2) storage only) without running
    # the real O(n^3) factorization.
    return _build_cache(alg, A)
end

function _failure_solution(cache, alg)
    u = cache.u
    # On LinearSolve 2.22 a matrix RHS yields a Vector `cache.u`; shape the
    # failure solution like the RHS so success and failure agree.
    if cache.b isa AbstractMatrix && u isa AbstractVector
        u = Matrix{eltype(u)}(undef, size(cache.b))
    end
    return SciMLBase.build_linear_solution(
        alg,
        u,
        nothing,
        cache;
        retcode=SciMLBase.ReturnCode.Failure,
    )
end

function _factorize!(cache, alg)
    cacheval = cache.cacheval
    A = _multifloat_matrix(cache.A)
    n = size(A, 1)
    # Only a size change (e.g. a `reinit!` with a larger A) re-prepares storage;
    # an ordinary in-place A update at the init-time size overwrites the owned
    # factor matrix in place and never grows it.
    if size(MFLA.factor_matrix(cacheval), 1) != n
        MFLA.prepare!(cacheval, n)
    end
    # check=false: a numerical breakdown returns a non-zero status instead of
    # throwing, and `factorize!` is fail-closed (it invalidates the cache before
    # touching storage), so a failed factorization can never leave a stale
    # success behind.
    MFLA.factorize!(cacheval, A; check=false, config=alg.config)
    return cacheval
end

# Solve into a correctly-shaped destination. On LinearSolve 2.22 a matrix RHS
# yields a Vector `cache.u` (its `__init_u0_from_Ab` has no `b::AbstractMatrix`
# method), so we allocate a matrix and solve into it; on 5.x `cache.u` is
# already a Matrix and the zero-alloc fast path is preserved.
function _solve_into!(cache, cacheval, alg)
    b = cache.b
    u = cache.u
    if b isa AbstractMatrix && u isa AbstractVector
        X = Matrix{eltype(u)}(undef, size(b))
        MFLA.solve!(X, cacheval, b; config=alg.config)
        return X
    end
    MFLA.solve!(u, cacheval, b; config=alg.config)
    return u
end

function _solve!(cache, alg)
    cacheval = cache.cacheval
    if cache.isfresh
        _factorize!(cache, alg)
        # Only a successful factorization is a usable cache; on failure keep
        # `isfresh` true so a caller can replace A and retry, and return Failure.
        MFLA.issuccess(cacheval) || return _failure_solution(cache, alg)
        cache.isfresh = false
    end
    MFLA.issuccess(cacheval) || return _failure_solution(cache, alg)

    u = _solve_into!(cache, cacheval, alg)
    return SciMLBase.build_linear_solution(
        alg,
        u,
        nothing,
        cache;
        retcode=SciMLBase.ReturnCode.Success,
    )
end

"""
    refresh!(cache::LinearCache)

Mark a `LinearCache` produced by a `MultiFloatLU` / `MultiFloatCholesky`
algorithm as requiring re-factorization on the next `solve!`. Call this after
mutating `cache.A` **in place** (modifying entries of the matrix object the cache
already holds) instead of replacing it with `cache.A = newA`.

The two A-update routes are equivalent and both reuse the cache's owned factor
storage:

  - `cache.A = newA` (or `reinit!(cache; A = newA)`) — the standard LinearSolve
    cache-update path; it already sets the freshness flag through
    `Base.setproperty!`.
  - `refresh!(cache)` after in-place mutation — the only route that works when
    the matrix object is mutated rather than reassigned.

On the next `solve!` the cache re-factorizes exactly once into its existing
factor matrix (no reallocation at the same size) and is fail-closed: a failed
factorization returns `ReturnCode.Failure` and never leaves a stale success
behind.
"""
function MFLA.refresh!(cache::LinearSolve.LinearCache)
    cache.isfresh = true
    return cache
end

function SciMLBase.solve!(
    cache::LinearSolve.LinearCache,
    alg::MultiFloatLU;
    kwargs...,
)
    return _solve!(cache, alg)
end

function SciMLBase.solve!(
    cache::LinearSolve.LinearCache,
    alg::MultiFloatCholesky;
    kwargs...,
)
    return _solve!(cache, alg)
end

end
