"""
Pure, pre-execution capacity queries. These never benchmark, calibrate, mutate
storage, or read solver state; they return an exact description of the storage a
caller must reserve (via `MFWorkspace` / `ensure_workspace_capacity!`, or
`prepare!` on a factor cache) so that the subsequent warm single-threaded core
path performs no allocation.
"""

"""
    workspace_requirements(operation, shape, config) -> NamedTuple

Return the reusable `MFWorkspace` capacity needed to run `operation` at
`shape` without allocating on the warm path. `shape` is a `NamedTuple` with the
operation's dimensions (e.g. `(m=, n=)` for factorization, `(m=, k=, n=)` for
GEMM). Returns `(factor=, ldlt_block=, gemm_workers=, gemm_capacity=)` matching
`ensure_workspace_capacity!`, plus `allocating=false` when no reusable
workspace is required for that operation.

Supported `operation` symbols: `:gemm`, `:syrk`, `:gemmt`, `:lu`, `:ldlt`,
`:rrqr`, `:cholesky`, `:solve`. Unsupported operations throw.
"""
function workspace_requirements end

@inline function _require_shape(shape, name::Symbol)
    hasproperty(shape, name) ||
        throw(ArgumentError("shape must include $name"))
    return getproperty(shape, name)
end

function workspace_requirements(operation::Symbol, shape::NamedTuple, config::KernelConfig)
    if operation === :cholesky || operation === :lu
        n = _require_shape(shape, :n)
        return (factor=n, ldlt_block=0, gemm_workers=1, gemm_capacity=0)
    elseif operation === :ldlt
        n = _require_shape(shape, :n)
        return (factor=n, ldlt_block=0, gemm_workers=1, gemm_capacity=0)
    elseif operation === :rrqr
        m = _require_shape(shape, :m)
        n = _require_shape(shape, :n)
        return (factor=max(m, n), ldlt_block=0, gemm_workers=1, gemm_capacity=0)
    elseif operation === :gemm
        m = _require_shape(shape, :m)
        k = _require_shape(shape, :k)
        n = _require_shape(shape, :n)
        panel = max(config.gemm_panel_columns, 1)
        packed_elements = k * panel
        return (factor=0, ldlt_block=0, gemm_workers=1, gemm_capacity=packed_elements)
    elseif operation === :syrk || operation === :gemmt
        k = _require_shape(shape, :k)
        n = _require_shape(shape, :n)
        panel = max(config.gemm_panel_columns, 1)
        return (factor=0, ldlt_block=0, gemm_workers=1, gemm_capacity=k * panel)
    elseif operation === :solve
        return (factor=0, ldlt_block=0, gemm_workers=1, gemm_capacity=0)
    end
    throw(ArgumentError("unsupported workspace_requirements operation: $operation"))
end

"""
    factor_cache_requirements(kind::Symbol, shape::NamedTuple, config) -> NamedTuple

Return the storage a caller must reserve for a factor cache of `kind`
(`:cholesky`, `:lu`, `:ldlt`, or `:rrqr`) before `factorize!`, so that the warm
`factorize!` and `solve!` paths allocate zero bytes. Pure and allocation-free.

For `:rrqr`, `shape` must contain `m` and `n` (rectangular supported); for the
others it must contain `n`. `config` may be `nothing` (defaults apply) or a
`KernelConfig`.
"""
function factor_cache_requirements end

function factor_cache_requirements(kind::Symbol, shape::NamedTuple, config::Union{Nothing,KernelConfig}=nothing)
    kind in (:cholesky, :lu, :ldlt, :rrqr) ||
        throw(ArgumentError("unsupported factor cache kind: $kind"))
    if kind === :rrqr
        m = _require_shape(shape, :m)
        n = _require_shape(shape, :n)
        return (
            kind=kind,
            matrix=(m=m, n=n),
            factor_matrix_elements=m * n,
            metadata_elements=min(m, n) + n + n + n,  # tau + perm + norm_scale + norm_sum
            scratch=2 * min(m, n),                   # ftranspose/auxiliary
        )
    end
    n = _require_shape(shape, :n)
    return (
        kind=kind,
        matrix=(m=n, n=n),
        factor_matrix_elements=n * n,
        metadata_elements=kind === :ldlt ? 3 * n : n,
        scratch=kind === :ldlt ? n : 0,
    )
end
