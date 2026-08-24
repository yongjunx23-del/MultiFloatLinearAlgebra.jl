struct MFCholesky{MF<:MultiFloat,M<:AbstractMatrix{MF}} <: AbstractMFFactorization{MF}
    factors::M
    info::Int
end

factor_kind(::MFCholesky) = :cholesky
factor_status(F::MFCholesky) = F.info
factor_matrix(F::MFCholesky) = F.factors

function _factor_cholesky_panel!(
    A::AbstractMatrix{MF},
    first::Int,
    last::Int,
) where {MF<:MultiFloat}
    @inbounds for column in first:last
        diagonal = A[column, column]
        for k in first:(column - 1)
            value = A[column, k]
            diagonal -= value * value
        end
        diagonal > zero(MF) || return column
        pivot = sqrt(diagonal)
        A[column, column] = pivot

        for row in (column + 1):last
            value = A[row, column]
            for k in first:(column - 1)
                value -= A[row, k] * A[column, k]
            end
            A[row, column] = value / pivot
        end
    end
    return 0
end

function _solve_cholesky_panel!(
    A::AbstractMatrix{MF},
    panel_first::Int,
    panel_last::Int,
    config::KernelConfig,
) where {MF<:MultiFloat}
    n = size(A, 1)
    panel_last < n || return nothing
    L11 = @view A[panel_first:panel_last, panel_first:panel_last]
    A21 = @view A[(panel_last + 1):n, panel_first:panel_last]
    trsm!(
        A21,
        L11;
        side=:right,
        uplo=:lower,
        trans=:T,
        diag=:nonunit,
        config=config,
    )
    return nothing
end

"""
    _cholesky_factorize_core!(A, config; check)

Shared blocked lower Cholesky numerical core. Overwrites `A` in place with the
factor (lower triangle authoritative) and returns the `info` status: `0` on
success, `-1` for a non-finite authoritative triangle, or a positive pivot index
for a not-PD failure. When `check=false` it never throws a numerical exception
after partial writes; the caller owns writing `info` back to a factor/cache and
deciding whether to throw. The caller guarantees `A` is the correctly sized,
supported matrix.
"""
function _cholesky_factorize_core!(
    A::AbstractMatrix{MF},
    config::KernelConfig,
    check::Bool,
) where {MF<:MultiFloat}
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
    return info
end

"""
    cholesky!(A; check=true, config=KernelConfig())

Blocked lower Cholesky factorization for a dense MultiFloat SPD matrix.

Only the lower triangle is authoritative on return. Panel solves go through
the package-level `trsm!` backend and trailing updates go through `syrk!`, so
factorization and repeated solves share the same MultiFloat-specific kernels
exposed to caller packages.

This entry point wraps [`_cholesky_factorize_core!`](@ref), the single shared
numerical core, into the standalone factor API (safe, independently ownable
factor).
"""
function cholesky!(
    A::AbstractMatrix{MF};
    check::Bool=true,
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    n, m = size(A)
    n == m || throw(DimensionMismatch("cholesky! requires a square matrix"))
    _check_supported(MF)
    Base.require_one_based_indexing(A)
    info = _cholesky_factorize_core!(A, config, check)
    return MFCholesky{MF,typeof(A)}(A, info)
end
