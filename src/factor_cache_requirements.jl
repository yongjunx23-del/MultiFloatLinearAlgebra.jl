"""
Pure, pre-execution capacity queries. These never benchmark, calibrate, mutate
storage, or read solver state; they return an exact description of the storage a
caller must reserve (via `MFWorkspace` / `ensure_workspace_capacity!`, or
`prepare!` on a factor cache) so that the subsequent warm single-threaded core
path performs no allocation.

Every query is parameterized by the exact MultiFloat scalar type and the
frozen `KernelConfig`, so the reported capacities match what the warm
`factorize!`/`solve!` actually consume (including the resolved GEMM/LDLT/RRQR
route, worker count, and panel/microkernel widths).
"""

############################################################################
# workspace_requirements(::Type{MF}, operation, shape, config)
############################################################################

"""
    workspace_requirements(::Type{MF}, operation, shape, config) -> NamedTuple

Return the reusable `MFWorkspace` capacity needed to run `operation` at `shape`
with `config` on scalar type `MF`, so the warm path does not allocate. `shape`
is a `NamedTuple` with the operation's dimensions (e.g. `(n=)` for a
factorization, `(m=,k=,n=)` for GEMM, `(m=,n=)` for RRQR).

Returns `(factor=, ldlt_block=, gemm_workers=, gemm_capacity=)`, matching
`ensure_workspace_capacity!`. GEMM capacities are derived from `gemm_plan`
(actual packed elements per worker) rather than a guessed panel width.

Supported `operation` symbols: `:gemm`, `:syrk`, `:gemmt`, `:lu`, `:ldlt`,
`:rrqr`, `:cholesky`, `:solve`. Unsupported operations throw.
"""
function workspace_requirements end

@inline function _require_shape(shape, name::Symbol)
    hasproperty(shape, name) ||
        throw(ArgumentError("shape must include $name"))
    return getproperty(shape, name)
end

@inline function _require_nonnegative(value::Int, name::Symbol)
    value >= 0 || throw(ArgumentError("$name must be nonnegative"))
    return value
end

function workspace_requirements(
    ::Type{MF},
    operation::Symbol,
    shape::NamedTuple,
    config::KernelConfig,
) where {MF<:MultiFloat}
    if operation === :cholesky || operation === :lu
        n = _require_nonnegative(_require_shape(shape, :n), :n)
        return (factor=n, ldlt_block=0, gemm_workers=0, gemm_capacity=0)
    elseif operation === :ldlt
        n = _require_nonnegative(_require_shape(shape, :n), :n)
        return (factor=n, ldlt_block=0, gemm_workers=0, gemm_capacity=0)
    elseif operation === :rrqr
        m = _require_nonnegative(_require_shape(shape, :m), :m)
        n = _require_nonnegative(_require_shape(shape, :n), :n)
        return (factor=max(m, n), ldlt_block=0, gemm_workers=0, gemm_capacity=0)
    elseif operation === :gemm
        m = _require_nonnegative(_require_shape(shape, :m), :m)
        k = _require_nonnegative(_require_shape(shape, :k), :k)
        n = _require_nonnegative(_require_shape(shape, :n), :n)
        plan = gemm_plan(MF, m, k, n, config)
        return (factor=0, ldlt_block=0,
                gemm_workers=plan.workers,
                gemm_capacity=plan.packed_elements_per_worker)
    elseif operation === :syrk || operation === :gemmt
        # SYRK/GEMMT lower output is n x n; the packed panel holds k x panel_cols.
        k = _require_nonnegative(_require_shape(shape, :k), :k)
        n = _require_nonnegative(_require_shape(shape, :n), :n)
        plan = gemm_plan(MF, n, k, n, config)
        return (factor=0, ldlt_block=0,
                gemm_workers=plan.workers,
                gemm_capacity=plan.packed_elements_per_worker)
    elseif operation === :solve
        return (factor=0, ldlt_block=0, gemm_workers=0, gemm_capacity=0)
    end
    throw(ArgumentError("unsupported workspace_requirements operation: $operation"))
end

################################################################################
# 2. factor_cache_requirements(::Type{MF}, kind, shape, config)
################################################################################

"""
    factor_cache_requirements(::Type{MF}, kind, shape, config) -> NamedTuple

Exact, type- and config-dependent storage requirements for a factor cache of
`kind` (`:cholesky`, `:lu`, `:ldlt`, or `:rrqr`) at `shape` with the frozen
`config`. `shape` must contain `n` (and `m`, `n` for `:rrqr`). Every owned
buffer the cache's `prepare!` allocates is reported with its exact capacity, so
a caller can reserve once and then guarantee 0-byte warm `factorize!`/`solve!`.

The result always includes:

- `kind`, `matrix` (`(m=, n=)`);
- `factor_matrix_elements`, `pivots`, `blocks`, `dsub`;
- `weighted_panel` `(rows=, cols=)` (0x0 for the unblocked LDLT route);
- `tau`, `permutation`, `cycle_leaders`, `cycle_leaders_capacity`;
- `norm_scale`, `norm_sum`, `norm_dirty`;
- `ftranspose` `(rows=, cols=)`, `auxiliary`;
- `gemm_workers`, `packed_elements_per_worker`.

All dimensions are derived from `gemm_plan`/`ldlt_plan`/the RRQR blocked-panel
policy and the frozen `config`, so they match what `prepare!` will allocate.
"""
function factor_cache_requirements end

@inline function _gemm_packed_requirements(::Type{MF}, k, n, config) where {MF<:MultiFloat}
    plan = gemm_plan(MF, k, k, n, config)
    return (workers=plan.workers, packed=plan.packed_elements_per_worker)
end

# Checked element count so a pathological m*n cannot silently wrap to a negative
# capacity in the pure requirements query.
@inline function _checked_elements(m::Int, n::Int)
    m >= 0 && n >= 0 || throw(ArgumentError("matrix dimensions must be nonnegative"))
    elements = Base.checked_mul(m, n)
    return elements
end

function factor_cache_requirements(
    ::Type{MF},
    kind::Symbol,
    shape::NamedTuple,
    config::KernelConfig,
) where {MF<:MultiFloat}
    kind in (:cholesky, :lu, :ldlt, :rrqr) ||
        throw(ArgumentError("unsupported factor cache kind: $kind"))
    n = _require_nonnegative(_require_shape(shape, :n), :n)
    if kind === :rrqr
        m = _require_nonnegative(_require_shape(shape, :m), :m)
        rc = min(m, n)
        # prepare! always allocates the blocked-panel scratch at max(min(16, rc), 1)
        # rows (the max(...,1) keeps a well-defined 1-row buffer even at rc=0).
        block = max(min(_QR_BLOCK_SIZE, rc), 1)
        return (
            kind=kind,
            matrix=(m=m, n=n),
            factor_matrix_elements=_checked_elements(m, n),
            pivots=0,                       # RRQR uses permutation, not pivots
            blocks=0,
            dsub=0,
            weighted_panel=(rows=0, cols=0),
            tau=rc,
            permutation=n,
            cycle_leaders=n,                # capacity = n (matches prepare!)
            cycle_leaders_capacity=n,
            norm_scale=n, norm_sum=n, norm_dirty=n,
            ftranspose=(rows=block, cols=n),
            auxiliary=block,
            gemm_workers=0,
            gemm_packed_elements_per_worker=0,
        )
    end
    if kind === :ldlt
        plan = ldlt_plan(MF, n, config)
        block_capacity = plan.strategy === :blocked ? plan.block_size : 0
        weighted_rows = block_capacity > 0 ? n : 0
        weighted_cols = block_capacity > 0 ? block_capacity + 1 : 0
        return (
            kind=kind,
            matrix=(m=n, n=n),
            factor_matrix_elements=_checked_elements(n, n),
            pivots=n, blocks=n, dsub=n,
            weighted_panel=(rows=weighted_rows, cols=weighted_cols),
            tau=0, permutation=0,
            cycle_leaders=0, cycle_leaders_capacity=0,
            norm_scale=0, norm_sum=0, norm_dirty=0,
            ftranspose=(rows=0, cols=0), auxiliary=0,
            gemm_workers=0,
            gemm_packed_elements_per_worker=0,
        )
    end
    # cholesky / lu: factor matrix + pivots (LU only)
    n_pivots = kind === :lu ? n : 0
    return (
        kind=kind,
        matrix=(m=n, n=n),
        factor_matrix_elements=_checked_elements(n, n),
        pivots=n_pivots, blocks=0, dsub=0,
        weighted_panel=(rows=0, cols=0),
        tau=0, permutation=0,
        cycle_leaders=0, cycle_leaders_capacity=0,
        norm_scale=0, norm_sum=0, norm_dirty=0,
        ftranspose=(rows=0, cols=0), auxiliary=0,
        gemm_workers=0,
        gemm_packed_elements_per_worker=0,
    )
end

# 3. factor_cache_capacity(cache) — reflect the actual allocated storage.
############################################################################

"""
    factor_cache_capacity(cache) -> NamedTuple

Return the currently allocated owned-buffer capacities of a factor cache
(including its GEMM workspace). This is a pure, allocation-free query that
reports the same fields as [`factor_cache_requirements`](@ref) so a consistency
test can assert `capacity satisfies requirements`.
"""
function factor_cache_capacity(cache::MFCholeskyCache{MF}) where {MF<:MultiFloat}
    return (
        kind=:cholesky,
        matrix=(m=size(cache.factors,1), n=size(cache.factors,2)),
        factor_matrix_elements=length(cache.factors),
        pivots=0, blocks=0, dsub=0,
        weighted_panel=(rows=0, cols=0),
        tau=0, permutation=0,
        cycle_leaders=0, cycle_leaders_capacity=0,
        norm_scale=0, norm_sum=0, norm_dirty=0,
        ftranspose=(rows=0, cols=0), auxiliary=0,
        gemm_workers=0, gemm_packed_elements_per_worker=0,
    )
end

function factor_cache_capacity(cache::MFLUCache{MF}) where {MF<:MultiFloat}
    return (
        kind=:lu,
        matrix=(m=size(cache.factors,1), n=size(cache.factors,2)),
        factor_matrix_elements=length(cache.factors),
        pivots=length(cache.ipiv), blocks=0, dsub=0,
        weighted_panel=(rows=0, cols=0),
        tau=0, permutation=0,
        cycle_leaders=0, cycle_leaders_capacity=0,
        norm_scale=0, norm_sum=0, norm_dirty=0,
        ftranspose=(rows=0, cols=0), auxiliary=0,
        gemm_workers=0, gemm_packed_elements_per_worker=0,
    )
end

function factor_cache_capacity(cache::MFLDLTCache{MF}) where {MF<:MultiFloat}
    return (
        kind=:ldlt,
        matrix=(m=size(cache.factors,1), n=size(cache.factors,2)),
        factor_matrix_elements=length(cache.factors),
        pivots=length(cache.pivots), blocks=length(cache.blocks), dsub=length(cache.dsub),
        weighted_panel=(rows=size(cache.weighted,1), cols=size(cache.weighted,2)),
        tau=0, permutation=0,
        cycle_leaders=0, cycle_leaders_capacity=0,
        norm_scale=0, norm_sum=0, norm_dirty=0,
        ftranspose=(rows=0, cols=0), auxiliary=0,
        gemm_workers=0, gemm_packed_elements_per_worker=0,
    )
end

function factor_cache_capacity(cache::MFRRQRCache{MF}) where {MF<:MultiFloat}
    rc = min(size(cache.factors)...)
    return (
        kind=:rrqr,
        matrix=(m=size(cache.factors,1), n=size(cache.factors,2)),
        factor_matrix_elements=length(cache.factors),
        pivots=0, blocks=0, dsub=0,
        weighted_panel=(rows=0, cols=0),
        tau=length(cache.tau), permutation=length(cache.permutation),
        cycle_leaders=length(cache.cycle_leaders),
        cycle_leaders_capacity=length(cache.cycle_leaders),
        norm_scale=length(cache.norm_scale), norm_sum=length(cache.norm_sum),
        norm_dirty=length(cache.norm_dirty),
        ftranspose=(rows=size(cache.ftranspose,1), cols=size(cache.ftranspose,2)), auxiliary=length(cache.auxiliary),
        gemm_workers=0, gemm_packed_elements_per_worker=0,
    )
end
