"""
    KernelConfig(; ...)

Explicit cache, threading, and route configuration for MultiFloat kernels.
The package never runs hidden machine calibration: `:auto` resolves from the
stored thresholds and type-specific defaults, while `calibrate_gemm` returns a
reproducible profile that callers may apply with `with_gemm_profile`.

Uncalibrated GEMM remains on the established direct route. LDLT uses the
unblocked route below its documented crossover and the measured blocked route
at or above it.
"""
Base.@kwdef struct KernelConfig
    reduction_tile::Int = 64
    column_tile::Int = 16
    cholesky_block::Int = 16
    lu_block::Int = 16
    ldlt_block::Int = 0
    ldlt_strategy::Symbol = :auto
    ldlt_blocked_crossover::Int = 512
    thread_count::Int = Threads.nthreads()
    gemm_strategy::Symbol = :auto
    gemm_packed_crossover::Int = typemax(Int)
    gemm_panel_columns::Int = 0
    gemm_micro_columns::Int = 0
    gemm_profile_scalar_type::Union{Nothing,DataType} = nothing
end

"""
    GemmWorkspace(T; thread_count=Threads.nthreads(), capacity=0)

Reusable caller-owned packed-panel buffers. One buffer is reserved for each
compute worker within a call. Calls that share one workspace are serialized by
an object-local lock so packing storage is never concurrently overwritten.
"""
mutable struct GemmWorkspace{MF<:MultiFloat}
    buffers::Vector{Vector{MF}}
    lock::ReentrantLock
end

function GemmWorkspace(
    ::Type{MF};
    thread_count::Int=Threads.nthreads(),
    capacity::Int=0,
) where {MF<:MultiFloat}
    workers = max(thread_count, 1)
    initial_capacity = max(capacity, 0)
    buffers = Vector{Vector{MF}}(undef, workers)
    @inbounds for worker in 1:workers
        buffers[worker] = Vector{MF}(undef, initial_capacity)
    end
    return GemmWorkspace{MF}(buffers, ReentrantLock())
end

"""
    MFWorkspace(T; factor_capacity=0, ldlt_block_capacity=0,
                thread_count=Threads.nthreads(), gemm_capacity=0)

Caller-owned reusable storage for material repeated allocations. The workspace
contains packed GEMM buffers and metadata/scratch for LU, LDLT, and RRQR.
Residual and correction arrays are deliberately not stored because their
mutating APIs already accept caller-owned destinations.

Factorization metadata stored here is transient scratch. A factor returned by
`lu!`, `ldlt!`, or `rrqr!` owns a snapshot of the metadata it needs, so later
workspace reuse or growth does not invalidate live factors. Concurrent packed
GEMM calls may share this workspace; concurrent factorization calls must use
distinct `MFWorkspace` objects. No calibration or process-global state is
involved.
"""
mutable struct MFWorkspace{MF<:MultiFloat}
    gemm::GemmWorkspace{MF}
    lu_pivots::Vector{Int}
    ldlt_dsub::Vector{MF}
    ldlt_pivots::Vector{Int}
    ldlt_blocks::Vector{UInt8}
    ldlt_weighted::Matrix{MF}
    qr_tau::Vector{MF}
    qr_permutation::Vector{Int}
    qr_norm_scale::Vector{MF}
    qr_norm_sum::Vector{MF}
    qr_norm_dirty::Vector{Bool}
    qr_ftranspose::Matrix{MF}
    qr_aux::Vector{MF}
    factor_capacity::Int
    ldlt_block_capacity::Int
end

function MFWorkspace(
    ::Type{MF};
    factor_capacity::Int=0,
    ldlt_block_capacity::Int=0,
    thread_count::Int=Threads.nthreads(),
    gemm_capacity::Int=0,
) where {MF<:MultiFloat}
    _check_supported(MF)
    factor_capacity >= 0 || throw(ArgumentError("factor_capacity must be nonnegative"))
    ldlt_block_capacity >= 0 ||
        throw(ArgumentError("ldlt_block_capacity must be nonnegative"))
    gemm_capacity >= 0 || throw(ArgumentError("gemm_capacity must be nonnegative"))
    weighted_columns = ldlt_block_capacity > 0 ? ldlt_block_capacity + 1 : 0
    return MFWorkspace{MF}(
        GemmWorkspace(
            MF;
            thread_count=thread_count,
            capacity=gemm_capacity,
        ),
        Vector{Int}(undef, factor_capacity),
        Vector{MF}(undef, factor_capacity),
        Vector{Int}(undef, factor_capacity),
        Vector{UInt8}(undef, factor_capacity),
        Matrix{MF}(undef, factor_capacity, weighted_columns),
        Vector{MF}(undef, factor_capacity),
        Vector{Int}(undef, factor_capacity),
        Vector{MF}(undef, factor_capacity),
        Vector{MF}(undef, factor_capacity),
        Vector{Bool}(undef, factor_capacity),
        Matrix{MF}(undef, 0, 0),
        Vector{MF}(undef, 0),
        factor_capacity,
        ldlt_block_capacity,
    )
end

"""
    workspace_capacity(workspace::MFWorkspace) -> NamedTuple

Return the currently allocated factor, LDLT-panel, and per-worker GEMM
capacities. This query is pure and does not grow or otherwise mutate storage.
"""
function workspace_capacity(workspace::MFWorkspace)
    lock(workspace.gemm.lock)
    try
        gemm_capacity = isempty(workspace.gemm.buffers) ? 0 :
                        minimum(length, workspace.gemm.buffers)
        return (
            factor=workspace.factor_capacity,
            ldlt_block=workspace.ldlt_block_capacity,
            gemm_workers=length(workspace.gemm.buffers),
            gemm_elements_per_worker=gemm_capacity,
        )
    finally
        unlock(workspace.gemm.lock)
    end
end

"""
    ensure_workspace_capacity!(workspace; factor_capacity=0,
                               ldlt_block_capacity=0,
                               gemm_workers=1, gemm_capacity=0)

Grow selected workspace capacities without shrinking existing storage. Live
factors own their required metadata and remain valid after every growth path.
Capacity growth must not overlap factorization using the same `MFWorkspace`;
growth of the packed GEMM subworkspace is internally serialized with its users.
"""
function ensure_workspace_capacity!(
    workspace::MFWorkspace{MF};
    factor_capacity::Int=0,
    ldlt_block_capacity::Int=0,
    gemm_workers::Int=1,
    gemm_capacity::Int=0,
) where {MF<:MultiFloat}
    minimum((factor_capacity, ldlt_block_capacity, gemm_capacity)) >= 0 ||
        throw(ArgumentError("workspace capacities must be nonnegative"))
    gemm_workers >= 1 || throw(ArgumentError("gemm_workers must be positive"))

    new_factor_capacity = max(workspace.factor_capacity, factor_capacity)
    new_block_capacity = max(workspace.ldlt_block_capacity, ldlt_block_capacity)
    factor_grew = new_factor_capacity > workspace.factor_capacity
    block_grew = new_block_capacity > workspace.ldlt_block_capacity
    if factor_grew
        resize!(workspace.lu_pivots, new_factor_capacity)
        resize!(workspace.ldlt_dsub, new_factor_capacity)
        resize!(workspace.ldlt_pivots, new_factor_capacity)
        resize!(workspace.ldlt_blocks, new_factor_capacity)
        resize!(workspace.qr_tau, new_factor_capacity)
        resize!(workspace.qr_permutation, new_factor_capacity)
        resize!(workspace.qr_norm_scale, new_factor_capacity)
        resize!(workspace.qr_norm_sum, new_factor_capacity)
        resize!(workspace.qr_norm_dirty, new_factor_capacity)
        workspace.factor_capacity = new_factor_capacity
    end
    if factor_grew || block_grew
        weighted_columns = new_block_capacity > 0 ? new_block_capacity + 1 : 0
        workspace.ldlt_weighted =
            Matrix{MF}(undef, new_factor_capacity, weighted_columns)
        workspace.ldlt_block_capacity = new_block_capacity
    end
    _prepare_gemm_workspace!(workspace.gemm, gemm_workers, gemm_capacity)
    return workspace
end

function _acquire_factor_workspace!(
    workspace::MFWorkspace{MF},
    factor_capacity::Int,
    ldlt_block_capacity::Int=0,
) where {MF<:MultiFloat}
    ensure_workspace_capacity!(
        workspace;
        factor_capacity=factor_capacity,
        ldlt_block_capacity=ldlt_block_capacity,
    )
    return workspace
end

@inline _gemm_workspace(workspace::GemmWorkspace) = workspace
@inline _gemm_workspace(workspace::MFWorkspace) = workspace.gemm

struct GemmPlan
    strategy::Symbol
    reason::Symbol
    panel_columns::Int
    micro_columns::Int
    workers::Int
    packed_elements_per_worker::Int
end

struct LDLTPlan
    strategy::Symbol
    reason::Symbol
    block_size::Int
end

struct GemmProfile{MF<:MultiFloat}
    strategy::Symbol
    panel_columns::Int
    micro_columns::Int
    packed_crossover::Int
    thread_count::Int
    source::Symbol
    fingerprint::NamedTuple
end

struct GemmMeasurement
    size::Int
    strategy::Symbol
    panel_columns::Int
    micro_columns::Int
    seconds::Float64
end

struct GemmCalibration{MF<:MultiFloat}
    profile::GemmProfile{MF}
    measurements::Vector{GemmMeasurement}
    minimum_speedup::Float64
end

@inline _workers(config::KernelConfig, jobs::Int) =
    max(1, min(config.thread_count, Threads.nthreads(), max(jobs, 1)))

# `mightalias` is conservatively true for a transpose wrapper even when its
# parent view is disjoint from the destination. Unwrap only these storage-free
# wrappers so factorization block views retain an accurate no-alias check.
@inline _alias_storage(array) = array
@inline _alias_storage(array::LinearAlgebra.Transpose) =
    _alias_storage(parent(array))
@inline _alias_storage(array::LinearAlgebra.Adjoint) =
    _alias_storage(parent(array))

@inline function _require_no_output_alias(operation, destination, source)
    Base.mightalias(destination, _alias_storage(source)) &&
        throw(ArgumentError(
            "$operation does not support output/input aliasing",
        ))
    return nothing
end

@inline _check_supported(::Type{MultiFloat{T,N}}) where {T,N} = begin
    1 <= N <= 4 || throw(ArgumentError("only 1-4 limb MultiFloats are supported"))
    nothing
end

@inline _vec4_type(::Type{MultiFloat{T,N}}) where {T,N} =
    MultiFloatVec{4,T,N}

@inline _limb_count(::Type{MultiFloat{T,N}}) where {T,N} = N

# Task creation is only worthwhile once a GEMV call contains enough limb-level
# arithmetic to amortize Julia scheduling. This static threshold was chosen
# conservatively from x2/x3/x4 crossover measurements; it does not calibrate or
# branch inside a numerical reduction.
@inline function _vector_thread_work_worthwhile(
    ::Type{MF},
    outputs::Int,
    reduction::Int,
) where {MF<:MultiFloat}
    return outputs * reduction * _limb_count(MF) >= 100_000
end

@inline function _symv_thread_work_worthwhile(
    ::Type{MF},
    dimension::Int,
) where {MF<:MultiFloat}
    return dimension >= (_limb_count(MF) == 2 ? 128 : 64)
end

@inline function _all_finite(A::AbstractArray)
    @inbounds for index in eachindex(A)
        isfinite(A[index]) || return false
    end
    return true
end

@inline function _lower_triangle_finite(A::AbstractMatrix)
    rows = axes(A, 1)
    columns = axes(A, 2)
    @inbounds for column in columns
        for row in rows
            row >= column || continue
            isfinite(A[row, column]) || return false
        end
    end
    return true
end

function _maximum_abs(A::AbstractArray{MF}) where {MF<:MultiFloat}
    maximum_value = zero(MF)
    @inbounds for index in eachindex(A)
        maximum_value = max(maximum_value, abs(A[index]))
    end
    return maximum_value
end

function _lower_maximum_abs(A::AbstractMatrix{MF}) where {MF<:MultiFloat}
    maximum_value = zero(MF)
    @inbounds for column in axes(A, 2), row in column:size(A, 1)
        maximum_value = max(maximum_value, abs(A[row, column]))
    end
    return maximum_value
end

"""
    AbstractMFFactorization{MF}

Supertype for [`MFCholesky`](@ref), [`MFLU`](@ref), [`MFLDLT`](@ref), and
[`MFQR`](@ref).
Callers should interact with a factorization through the public accessors
`factor_status`, `factor_state`, `factor_kind`, `factor_matrix`,
`factor_precision`, `factor_provider`, factor-kind-specific metadata accessors,
`factor_diagnostics`, `issuccess`, `ldiv!`, and `solve` rather than reading its
concrete fields, so the internal storage can evolve without breaking solver
packages.
"""
abstract type AbstractMFFactorization{MF<:MultiFloat} end

"""
    factor_status(F::AbstractMFFactorization) -> Int
    factor_kind(F::AbstractMFFactorization) -> Symbol
    factor_matrix(F::AbstractMFFactorization) -> AbstractMatrix

The public factorization interface. Each concrete factorization implements
these three accessors instead of exposing its storage layout, so the internal
representation can change without breaking solver packages.

`factor_status` returns the status code: zero means success, while a nonzero
value describes why the factorization stopped. `issuccess(F)` is shorthand for
`iszero(factor_status(F))`.

`factor_matrix` returns the matrix holding the factored data (`L`/`U`/`D` plus
pivot structure). The returned matrix is borrowed internal storage; callers
must not mutate it.
"""
function factor_status end
function factor_kind end
function factor_matrix end

@inline _check_factor_valid(::AbstractMFFactorization) = nothing

issuccess(F::AbstractMFFactorization) =
    (_check_factor_valid(F); iszero(factor_status(F)))

Base.size(F::AbstractMFFactorization) =
    (_check_factor_valid(F); size(factor_matrix(F)))
Base.size(F::AbstractMFFactorization, dimension::Integer) =
    (_check_factor_valid(F); size(factor_matrix(F), dimension))

Base.eltype(::AbstractMFFactorization{MF}) where {MF} = MF

"""
    factor_precision(F::AbstractMFFactorization) -> Type
    factor_provider(F::AbstractMFFactorization) -> Symbol
    factor_state(F::AbstractMFFactorization) -> Symbol

Return the exact scalar type, provider identity, and stable symbolic execution
state of an opaque factor. `factor_status` retains its integer compatibility
contract; `factor_state` maps that code to `:success`, `:nonfinite_input`,
`:not_posdef`, `:singular`, or `:numerical_breakdown`. RRQR rank is a
caller-threshold diagnostic and is not a factorization failure state.
"""
factor_precision(::AbstractMFFactorization{MF}) where {MF} = MF
factor_provider(::AbstractMFFactorization) = :mfla

function factor_state(F::AbstractMFFactorization)
    status = factor_status(F)
    iszero(status) && return :success
    status == -1 && return :nonfinite_input
    status < 0 && return :numerical_breakdown
    kind = factor_kind(F)
    kind === :cholesky && return :not_posdef
    kind in (:lu, :ldlt) && return :singular
    return :numerical_breakdown
end

function _prepare_gemm_workspace!(
    workspace::GemmWorkspace{MF},
    workers::Int,
    capacity::Int,
) where {MF<:MultiFloat}
    lock(workspace.lock)
    try
        required_workers = max(workers, 1)
        while length(workspace.buffers) < required_workers
            push!(workspace.buffers, Vector{MF}(undef, max(capacity, 0)))
        end
        @inbounds for worker in 1:required_workers
            buffer = workspace.buffers[worker]
            length(buffer) >= capacity || resize!(buffer, capacity)
        end
    finally
        unlock(workspace.lock)
    end
    return workspace
end
