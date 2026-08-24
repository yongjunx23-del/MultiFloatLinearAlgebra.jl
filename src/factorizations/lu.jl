struct MFLU{
    MF<:MultiFloat,
    M<:AbstractMatrix{MF},
    P<:AbstractVector{Int},
} <: AbstractMFFactorization{MF}
    factors::M
    ipiv::P
    info::Int
    original_maximum::MF
end

factor_kind(::MFLU) = :lu
factor_status(F::MFLU) = F.info
factor_matrix(F::MFLU) = F.factors

function _prepare_lu_metadata!(
    ::Type{MF},
    count::Int,
    workspace::Union{Nothing,MFWorkspace{MF}},
) where {MF<:MultiFloat}
    if workspace === nothing
        return collect(1:count)
    end
    _acquire_factor_workspace!(workspace, count)
    @inbounds for index in 1:count
        workspace.lu_pivots[index] = index
    end
    return @view(workspace.lu_pivots[1:count])
end

@inline _owned_lu_pivots(ipiv, ::Nothing) = ipiv
@inline _owned_lu_pivots(ipiv, ::MFWorkspace) = copy(ipiv)

@inline function _swap_rows!(
    A::AbstractMatrix,
    first::Int,
    second::Int,
)
    first == second && return nothing
    @inbounds for column in axes(A, 2)
        A[first, column], A[second, column] =
            A[second, column], A[first, column]
    end
    return nothing
end

function _lu_panel_rank1_update!(
    A::AbstractMatrix{MF},
    k::Int,
    panel_last::Int,
) where {T,N,MF<:MultiFloat{T,N}}
    m = size(A, 1)
    V4 = MultiFloatVec{4,T,N}
    @inbounds for column in (k + 1):panel_last
        multiplier = V4(A[k, column])
        row = k + 1
        while row + 3 <= m
            values = V4(
                A[row, column],
                A[row + 1, column],
                A[row + 2, column],
                A[row + 3, column],
            )
            factors = V4(
                A[row, k],
                A[row + 1, k],
                A[row + 2, k],
                A[row + 3, k],
            )
            values -= factors * multiplier
            for lane in 1:4
                A[row + lane - 1, column] = values[lane]
            end
            row += 4
        end
        while row <= m
            A[row, column] -= A[row, k] * A[k, column]
            row += 1
        end
    end
    return nothing
end

function _factor_lu_panel!(
    A::AbstractMatrix{MF},
    first::Int,
    last::Int,
    ipiv::AbstractVector{Int},
) where {MF<:MultiFloat}
    m = size(A, 1)
    @inbounds for k in first:last
        pivot = k
        pivot_value = abs(A[k, k])
        for row in (k + 1):m
            candidate = abs(A[row, k])
            if candidate > pivot_value
                pivot = row
                pivot_value = candidate
            end
        end
        ipiv[k] = pivot
        _swap_rows!(A, k, pivot)

        iszero(A[k, k]) && return k
        inverse_pivot = inv(A[k, k])
        for row in (k + 1):m
            A[row, k] *= inverse_pivot
        end
        k < last && _lu_panel_rank1_update!(A, k, last)
    end
    return 0
end

"""
    _lu_factorize_core!(A, pivots, config, check)

Shared blocked dense partial-pivoting LU numerical core. Overwrites `A` in
place with the factor (unit-lower `L` below, `U` on/above) and writes the row
swaps into `pivots`. Returns the `info` status (`0` success, `-1` nonfinite, or
the failed pivot index). `pivots` must be length `min(m, n)`. When
`check=false` it never throws after partial writes. This is the single core
shared by the standalone [`lu!`](@ref) and the factor cache.
"""
function _lu_factorize_core!(
    A::AbstractMatrix{MF},
    pivots::AbstractVector{Int},
    config::KernelConfig,
    check::Bool,
    workspace::Union{Nothing,MFWorkspace{MF},GemmWorkspace{MF}}=nothing,
) where {MF<:MultiFloat}
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
        info = _factor_lu_panel!(A, first, last, pivots)
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
                gemm!(A22, A21, U12, -one(MF), one(MF); config=config, workspace=workspace)
            end
        end
    end
    return info
end

"""
    lu!(A; check=true, config=KernelConfig(), workspace=nothing)

Blocked dense partial-pivoting LU specialized for MultiFloat matrices. The
panel factorization performs pivoting and only updates the current panel.
After each panel, the upper-right block is formed with the package `trsm!`
kernel and the O(n^3) trailing update is delegated to the SIMD/threaded
`gemm!` backend.

The factorization stores unit-lower `L` below the diagonal and `U` on/above
the diagonal. `ipiv` records the row swap performed at each pivot. With
`workspace=MFWorkspace(T)`, pivot workspace is reused during factorization and
the returned factor owns its pivot metadata.

This entry point wraps [`_lu_factorize_core!`](@ref), the single shared
numerical core, into the standalone factor API.
"""
function lu!(
    A::AbstractMatrix{MF};
    check::Bool=true,
    config::KernelConfig=KernelConfig(),
    workspace::Union{Nothing,MFWorkspace{MF}}=nothing,
) where {MF<:MultiFloat}
    m, n = size(A)
    _check_supported(MF)
    Base.require_one_based_indexing(A)
    kmax = min(m, n)
    if !_all_finite(A)
        check && throw(DomainError(A, "lu!: input matrix contains non-finite entries"))
        ipiv = _prepare_lu_metadata!(MF, kmax, workspace)
        owned_ipiv = _owned_lu_pivots(ipiv, workspace)
        return MFLU{MF,typeof(A),typeof(owned_ipiv)}(
            A, owned_ipiv, -1, zero(MF),
        )
    end
    original_maximum = _maximum_abs(A)
    ipiv = _prepare_lu_metadata!(MF, kmax, workspace)
    info = _lu_factorize_core!(A, ipiv, config, check, workspace)
    owned_ipiv = _owned_lu_pivots(ipiv, workspace)
    return MFLU{MF,typeof(A),typeof(owned_ipiv)}(
        A, owned_ipiv, info, original_maximum,
    )
end
