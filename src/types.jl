"""
    KernelConfig(; reduction_tile=64, column_tile=16, cholesky_block=16,
                 thread_count=Threads.nthreads())

Cache/thread geometry for MultiFloat kernels.

The defaults are conservative and intentionally independent of any solver.
Applications may keep one configuration per machine or benchmark-tune these
values without changing the numerical API.
"""
Base.@kwdef struct KernelConfig
    reduction_tile::Int = 64
    column_tile::Int = 16
    cholesky_block::Int = 16
    thread_count::Int = Threads.nthreads()
end

@inline _workers(config::KernelConfig, jobs::Int) =
    max(1, min(config.thread_count, Threads.nthreads(), max(jobs, 1)))

@inline _check_supported(::Type{MultiFloat{T,N}}) where {T,N} = begin
    1 <= N <= 4 || throw(ArgumentError("only 1-4 limb MultiFloats are supported"))
    nothing
end

@inline _vec4_type(::Type{MultiFloat{T,N}}) where {T,N} =
    MultiFloatVec{4,T,N}
