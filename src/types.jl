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
end

"""
    GemmWorkspace(T; thread_count=Threads.nthreads(), capacity=0)

Reusable caller-owned packed-panel buffers. One buffer is reserved for each
possible compute worker so concurrent output-column owners never share mutable
packing storage.
"""
mutable struct GemmWorkspace{MF<:MultiFloat}
    buffers::Vector{Vector{MF}}
    function GemmWorkspace{MF}(buffers::Vector{Vector{MF}}) where {MF<:MultiFloat}
        return new{MF}(buffers)
    end
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
    return GemmWorkspace{MF}(buffers)
end

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

@inline _check_supported(::Type{MultiFloat{T,N}}) where {T,N} = begin
    1 <= N <= 4 || throw(ArgumentError("only 1-4 limb MultiFloats are supported"))
    nothing
end

@inline _vec4_type(::Type{MultiFloat{T,N}}) where {T,N} =
    MultiFloatVec{4,T,N}

@inline _limb_count(::Type{MultiFloat{T,N}}) where {T,N} = N

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

"""
    AbstractMFFactorization{MF}

Supertype for [`MFCholesky`](@ref), [`MFLU`](@ref), and [`MFLDLT`](@ref).
Callers should interact with a factorization through the public accessors
`factor_status`, `factor_kind`, `factor_matrix`, `issuccess`, `ldiv!`, and
`solve` rather than reading its concrete fields, so the internal storage can
evolve without breaking solver packages.
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

issuccess(F::AbstractMFFactorization) = iszero(factor_status(F))

Base.size(F::AbstractMFFactorization) = size(factor_matrix(F))

Base.eltype(::AbstractMFFactorization{MF}) where {MF} = MF

function _prepare_gemm_workspace!(
    workspace::GemmWorkspace{MF},
    workers::Int,
    capacity::Int,
) where {MF<:MultiFloat}
    required_workers = max(workers, 1)
    while length(workspace.buffers) < required_workers
        push!(workspace.buffers, Vector{MF}(undef, max(capacity, 0)))
    end
    @inbounds for worker in 1:required_workers
        buffer = workspace.buffers[worker]
        length(buffer) >= capacity || resize!(buffer, capacity)
    end
    return workspace
end
