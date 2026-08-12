struct MFCholesky{MF<:MultiFloat,M<:AbstractMatrix{MF}}
    factors::M
    info::Int
end

issuccess(F::MFCholesky) = iszero(F.info)

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
    cholesky!(A; check=true, config=KernelConfig())

Blocked lower Cholesky factorization for a dense MultiFloat SPD matrix.

Only the lower triangle is authoritative on return. Panel solves go through
the package-level `trsm!` backend and trailing updates go through `syrk!`, so
factorization and repeated solves share the same MultiFloat-specific kernels
exposed to caller packages.
"""
function cholesky!(
    A::AbstractMatrix{MF};
    check::Bool=true,
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    n, m = size(A)
    n == m || throw(DimensionMismatch("cholesky! requires a square matrix"))
    _check_supported(MF)
    if !_lower_triangle_finite(A)
        check && throw(DomainError(A, "cholesky!: input matrix contains non-finite entries"))
        return MFCholesky{MF,typeof(A)}(A, -1)
    end

    block = max(config.cholesky_block, 1)
    info = 0
    for first in 1:block:n
        last = min(first + block - 1, n)
        info = _factor_cholesky_panel!(A, first, last)
        if !iszero(info)
            check && throw(LinearAlgebra.PosDefException(info))
            return MFCholesky{MF,typeof(A)}(A, info)
        end

        if last < n
            _solve_cholesky_panel!(A, first, last, config)
            trailing = @view A[(last + 1):n, (last + 1):n]
            panel = transpose(@view A[(last + 1):n, first:last])
            syrk!(
                trailing,
                panel,
                -one(MF),
                one(MF);
                config=config,
            )
        end
    end
    return MFCholesky{MF,typeof(A)}(A, 0)
end
