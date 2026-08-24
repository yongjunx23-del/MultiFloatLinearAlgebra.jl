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

function _factorize(alg::MultiFloatLU, A)
    return MFLA.lu!(Matrix(_multifloat_matrix(A)); check=false, config=alg.config)
end

function _factorize(alg::MultiFloatCholesky, A)
    return MFLA.cholesky!(Matrix(_multifloat_matrix(A)); check=false, config=alg.config)
end

function _empty_factor(alg, A)
    matrix = _multifloat_matrix(A)
    placeholder = Matrix{eltype(matrix)}(undef, 0, 0)
    return _factorize(alg, placeholder)
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
    # `LinearCache` fixes the cache-value field type at initialization. An empty
    # factor supplies that type without doing the real O(n^3) factorization twice.
    return _empty_factor(alg, A)
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

function _solve!(cache, alg)
    factor = cache.cacheval
    if cache.isfresh
        factor = _factorize(alg, cache.A)
        cache.cacheval = factor
        MFLA.issuccess(factor) || return _failure_solution(cache, alg)
        cache.isfresh = false
    elseif !MFLA.issuccess(factor)
        return _failure_solution(cache, alg)
    end

    MFLA.ldiv!(cache.u, factor, cache.b; config=alg.config)
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
