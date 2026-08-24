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

# A fresh empty cache supplies the fixed `cacheval` field type without running
# the real O(n^3) factorization during `init`. Storage is grown inside `_solve!`
# on the first (or size-changing) factorization and reused thereafter.
function _empty_cache(alg::Union{MultiFloatLU,MultiFloatCholesky}, A)
    MF = eltype(_multifloat_matrix(A))
    if alg isa MultiFloatLU
        return MFLA.MFLUCache(MF)
    end
    return MFLA.MFCholeskyCache(MF)
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
    # `LinearCache` fixes the cache-value field type at initialization; an empty
    # cache supplies that type without doing the real O(n^3) factorization.
    return _empty_cache(alg, A)
end

function _failure_solution(cache, alg)
    return SciMLBase.build_linear_solution(
        alg,
        cache.u,
        nothing,
        cache;
        retcode=SciMLBase.ReturnCode.Failure,
    )
end

function _factorize!(cache, alg)
    cacheval = cache.cacheval
    A = _multifloat_matrix(cache.A)
    n = size(A, 1)
    # Explicit reserve on the first factorization or on a size change; the
    # warm repeated `factorize!` at an unchanged size only overwrites storage.
    if size(MFLA.factor_matrix(cacheval), 1) != n
        MFLA.prepare!(cacheval, n)
    end
    MFLA.factorize!(cacheval, A; check=false, config=alg.config)
    return cacheval
end

function _solve!(cache, alg)
    cacheval = cache.cacheval
    if cache.isfresh
        _factorize!(cache, alg)
        # Only a successful factorization is a usable cache; on failure keep
        # `isfresh` true so a caller can replace A and retry.
        MFLA.issuccess(cacheval) || return _failure_solution(cache, alg)
        cache.isfresh = false
    end
    MFLA.issuccess(cacheval) || return _failure_solution(cache, alg)

    MFLA.solve!(cache.u, cacheval, cache.b; config=alg.config)
    return SciMLBase.build_linear_solution(
        alg,
        cache.u,
        nothing,
        cache;
        retcode=SciMLBase.ReturnCode.Success,
    )
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
