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

function _solve_cholesky_panel_rows!(
    A::AbstractMatrix{MF},
    panel_first::Int,
    panel_last::Int,
    row_first::Int,
    row_last::Int,
) where {T,N,MF<:MultiFloat{T,N}}
    row_first > row_last && return nothing
    V4 = MultiFloatVec{4,T,N}
    row = row_first

    @inbounds while row + 3 <= row_last
        for column in panel_first:panel_last
            values = V4(
                A[row, column],
                A[row + 1, column],
                A[row + 2, column],
                A[row + 3, column],
            )
            for k in panel_first:(column - 1)
                values -= V4(
                    A[row, k],
                    A[row + 1, k],
                    A[row + 2, k],
                    A[row + 3, k],
                ) * V4(A[column, k])
            end
            values /= V4(A[column, column])
            for lane in 1:4
                A[row + lane - 1, column] = values[lane]
            end
        end
        row += 4
    end

    while row <= row_last
        for column in panel_first:panel_last
            value = A[row, column]
            for k in panel_first:(column - 1)
                value -= A[row, k] * A[column, k]
            end
            A[row, column] = value / A[column, column]
        end
        row += 1
    end
    return nothing
end

function _solve_cholesky_panel!(
    A::AbstractMatrix{MF},
    panel_first::Int,
    panel_last::Int,
    config::KernelConfig,
) where {MF<:MultiFloat}
    n = size(A, 1)
    row_first = panel_last + 1
    row_first > n && return nothing
    row_count = n - panel_last
    workers = _workers(config, cld(row_count, 32))
    if workers == 1 || row_count < 64
        _solve_cholesky_panel_rows!(
            A, panel_first, panel_last, row_first, n,
        )
        return nothing
    end

    chunk = cld(row_count, workers)
    @sync for worker in 1:workers
        first_row = row_first + (worker - 1) * chunk
        last_row = min(row_first + worker * chunk - 1, n)
        first_row <= last_row || continue
        Threads.@spawn _solve_cholesky_panel_rows!(
            A, panel_first, panel_last, first_row, last_row,
        )
    end
    return nothing
end

"""
    cholesky!(A; check=true, config=KernelConfig())

Blocked lower Cholesky factorization for a dense MultiFloat SPD matrix.

Only the lower triangle is authoritative on return. The trailing update is
delegated to the package's SIMD/threaded `syrk!`, making this factorization a
direct consumer of the same backend kernels exposed to solver packages.
"""
function cholesky!(
    A::AbstractMatrix{MF};
    check::Bool=true,
    config::KernelConfig=KernelConfig(),
) where {MF<:MultiFloat}
    n, m = size(A)
    n == m || throw(DimensionMismatch("cholesky! requires a square matrix"))
    _check_supported(MF)

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
