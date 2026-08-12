function _cpu_info_snapshot()
    info = try
        Sys.cpu_info()
    catch
        return nothing
    end
    isempty(info) && return nothing
    first_cpu = first(info)
    return (
        model=string(first_cpu.model),
        speed_mhz=Int(first_cpu.speed),
        cores=length(info),
    )
end

function machine_fingerprint(; thread_count::Int=Threads.nthreads())
    cpu = _cpu_info_snapshot()
    return (
        arch=Sys.ARCH,
        kernel=Sys.KERNEL,
        word_size=Sys.WORD_SIZE,
        cpu_model=cpu === nothing ? "unknown" : cpu.model,
        cpu_speed_mhz=cpu === nothing ? 0 : cpu.speed_mhz,
        cpu_cores=cpu === nothing ? 0 : cpu.cores,
        julia=VERSION,
        julia_threads=max(thread_count, 1),
    )
end

"""
    default_gemm_profile(T; thread_count=Threads.nthreads())

Return the built-in x1/x2/x3/x4 panel geometry without enabling the packed
route. Built-in profiles use an infinite crossover so the established direct
kernel remains authoritative until `calibrate_gemm` supplies measured evidence.
"""
function default_gemm_profile(
    ::Type{MF};
    thread_count::Int=Threads.nthreads(),
) where {MF<:MultiFloat}
    _check_supported(MF)
    return GemmProfile{MF}(
        :auto,
        _default_gemm_panel_columns(MF, thread_count),
        _default_gemm_micro_columns(MF),
        typemax(Int),
        max(thread_count, 1),
        :builtin,
        machine_fingerprint(; thread_count=thread_count),
    )
end

"""
    with_gemm_profile(config, profile; thread_count=profile.thread_count)

Copy a machine profile into an otherwise independent `KernelConfig`. The
returned configuration records all resolved GEMM values and is therefore
queryable and reproducible by `gemm_plan`.
"""
function with_gemm_profile(
    config::KernelConfig,
    profile::GemmProfile;
    thread_count::Int=profile.thread_count,
)
    return KernelConfig(
        reduction_tile=config.reduction_tile,
        column_tile=config.column_tile,
        cholesky_block=config.cholesky_block,
        lu_block=config.lu_block,
        ldlt_block=config.ldlt_block,
        ldlt_strategy=config.ldlt_strategy,
        ldlt_blocked_crossover=config.ldlt_blocked_crossover,
        thread_count=max(thread_count, 1),
        gemm_strategy=profile.strategy,
        gemm_packed_crossover=profile.packed_crossover,
        gemm_panel_columns=profile.panel_columns,
        gemm_micro_columns=profile.micro_columns,
    )
end

function _gemm_calibration_candidates(::Type{MF}) where {MF<:MultiFloat}
    limbs = _limb_count(MF)
    if limbs <= 2
        return [(24, 2), (24, 4), (32, 4), (48, 4)]
    elseif limbs == 3
        return [(16, 2), (24, 2), (24, 4), (32, 2)]
    end
    return [(8, 4), (12, 2), (16, 2), (24, 2)]
end

function _calibration_matrix(::Type{MF}, n::Int, phase::Float64) where {MF<:MultiFloat}
    matrix = Matrix{MF}(undef, n, n)
    @inbounds for column in 1:n
        for row in 1:n
            value =
                sin(phase + 0.013 * row + 0.021 * column) +
                0.25 * cos(0.017 * row - 0.011 * column)
            matrix[row, column] = MF(value)
        end
    end
    return matrix
end

function _median_elapsed_seconds(function_call, samples::Int)
    count = max(samples, 1)
    function_call()
    elapsed = Vector{Float64}(undef, count)
    for sample in 1:count
        GC.gc()
        start = time_ns()
        function_call()
        elapsed[sample] = (time_ns() - start) / 1.0e9
    end
    sort!(elapsed)
    return elapsed[cld(count, 2)]
end

function _calibration_seconds(
    measurements::Vector{GemmMeasurement},
    size::Int,
    strategy::Symbol,
    panel_columns::Int=0,
    micro_columns::Int=0,
)
    for measurement in measurements
        if measurement.size == size &&
           measurement.strategy === strategy &&
           (strategy === :direct ||
            (measurement.panel_columns == panel_columns &&
             measurement.micro_columns == micro_columns))
            return measurement.seconds
        end
    end
    throw(ArgumentError("missing GEMM calibration measurement"))
end

"""
    calibrate_gemm(T; sizes=(512, 1024), samples=3,
                   thread_count=Threads.nthreads(), candidates=nothing,
                   minimum_speedup=1.05)

Benchmark the direct route and a deterministic packed-panel candidate set. The
function returns every measurement plus a `GemmProfile`; it never installs
global state. Calibration is intentionally explicit and moderately expensive:
the defaults target the 512/1024 dense regime where panel packing can change
the memory-traffic balance. Callers working at other scales should pass their
own representative `sizes`.

The candidate is chosen by performance at the largest tested size. Packed mode
is accepted only when that largest point beats direct by `minimum_speedup`, and
the crossover is the earliest tested size from which every larger tested point
also clears the same margin. This suffix rule prevents a noisy smaller matrix
from enabling a route that regresses at the intended production scale.

The direct and packed outputs must also be exactly equal under the package's
ascending-reduction contract. If no candidate passes both numerical and timing
gates, the returned profile explicitly selects `:direct`.
"""
function calibrate_gemm(
    ::Type{MF};
    sizes=(512, 1024),
    samples::Int=3,
    thread_count::Int=Threads.nthreads(),
    candidates=nothing,
    minimum_speedup::Real=1.05,
) where {MF<:MultiFloat}
    _check_supported(MF)
    required_speedup = Float64(minimum_speedup)
    isfinite(required_speedup) && required_speedup > 1.0 ||
        throw(ArgumentError("minimum_speedup must be finite and greater than one"))

    tested_sizes = sort!(unique!(Int[max(Int(size), 1) for size in sizes]))
    isempty(tested_sizes) && throw(ArgumentError("calibration requires at least one size"))
    candidate_list = candidates === nothing ?
                     _gemm_calibration_candidates(MF) :
                     Tuple{Int,Int}[(Int(candidate[1]), Int(candidate[2])) for candidate in candidates]
    isempty(candidate_list) && throw(ArgumentError("calibration requires at least one packed candidate"))
    for (panel_columns, micro_columns) in candidate_list
        panel_columns > 0 || throw(ArgumentError("panel columns must be positive"))
        micro_columns in (1, 2, 4) ||
            throw(ArgumentError("micro columns must be 1, 2, or 4"))
    end

    measurements = GemmMeasurement[]
    workers = max(thread_count, 1)
    for n in tested_sizes
        A = _calibration_matrix(MF, n, 0.1)
        B = _calibration_matrix(MF, n, 0.7)
        direct_output = zeros(MF, n, n)

        direct_config = KernelConfig(
            thread_count=workers,
            gemm_strategy=:direct,
        )
        direct_seconds = _median_elapsed_seconds(samples) do
            gemm!(direct_output, A, B; config=direct_config)
        end
        push!(
            measurements,
            GemmMeasurement(n, :direct, 0, 0, direct_seconds),
        )

        for (panel_columns, micro_columns) in candidate_list
            packed_config = KernelConfig(
                thread_count=workers,
                gemm_strategy=:packed,
                gemm_panel_columns=panel_columns,
                gemm_micro_columns=micro_columns,
            )
            workspace = GemmWorkspace(
                MF;
                thread_count=workers,
                capacity=n * panel_columns,
            )
            packed_output = zeros(MF, n, n)
            packed_seconds = _median_elapsed_seconds(samples) do
                gemm!(
                    packed_output,
                    A,
                    B;
                    config=packed_config,
                    workspace=workspace,
                )
            end
            packed_output == direct_output ||
                throw(ErrorException("packed GEMM changed the direct reduction result"))
            push!(
                measurements,
                GemmMeasurement(
                    n,
                    :packed,
                    panel_columns,
                    micro_columns,
                    packed_seconds,
                ),
            )
        end
    end

    largest_size = last(tested_sizes)
    largest_candidate_seconds = Float64[
        _calibration_seconds(
            measurements,
            largest_size,
            :packed,
            panel_columns,
            micro_columns,
        )
        for (panel_columns, micro_columns) in candidate_list
    ]
    best_index = argmin(largest_candidate_seconds)
    best_panel, best_micro = candidate_list[best_index]

    function wins_at(size::Int)
        direct_seconds = _calibration_seconds(
            measurements, size, :direct,
        )
        packed_seconds = _calibration_seconds(
            measurements, size, :packed, best_panel, best_micro,
        )
        return packed_seconds * required_speedup <= direct_seconds
    end

    crossover = typemax(Int)
    if wins_at(largest_size)
        for first_index in eachindex(tested_sizes)
            stable_suffix = true
            for index in first_index:lastindex(tested_sizes)
                if !wins_at(tested_sizes[index])
                    stable_suffix = false
                    break
                end
            end
            if stable_suffix
                crossover = tested_sizes[first_index]
                break
            end
        end
    end

    profile = GemmProfile{MF}(
        crossover == typemax(Int) ? :direct : :auto,
        best_panel,
        best_micro,
        crossover,
        workers,
        :calibrated,
        machine_fingerprint(; thread_count=workers),
    )
    return GemmCalibration{MF}(
        profile,
        measurements,
        required_speedup,
    )
end
