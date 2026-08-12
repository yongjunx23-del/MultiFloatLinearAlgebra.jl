function machine_fingerprint(; thread_count::Int=Threads.nthreads())
    return (
        arch=Sys.ARCH,
        kernel=Sys.KERNEL,
        word_size=Sys.WORD_SIZE,
        julia=VERSION,
        julia_threads=max(thread_count, 1),
    )
end

"""
    default_gemm_profile(T; thread_count=Threads.nthreads())

Return the built-in, deterministic x1/x2/x3/x4 packing geometry. This does not
run a benchmark and is suitable for reproducible defaults or as a calibration
starting point.
"""
function default_gemm_profile(
    ::Type{MF};
    thread_count::Int=Threads.nthreads(),
) where {MF<:MultiFloat}
    _check_supported(MF)
    limbs = _limb_count(MF)
    crossover = limbs <= 2 ? 160 : limbs == 3 ? 176 : 192
    return GemmProfile{MF}(
        :auto,
        _default_gemm_panel_columns(MF, thread_count),
        _default_gemm_micro_columns(MF),
        crossover,
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
    return [(12, 2), (16, 1), (16, 2), (24, 2)]
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

"""
    calibrate_gemm(T; sizes=(128, 256), samples=3,
                   thread_count=Threads.nthreads(), candidates=nothing)

Benchmark the direct route and a small, deterministic packed-panel candidate
set. The function returns every measurement plus a `GemmProfile`; it never
installs global state. The crossover is the smallest tested size where the
winning packed geometry is at least two percent faster than the direct route.
If no packed candidate wins, the returned profile explicitly selects `:direct`.
"""
function calibrate_gemm(
    ::Type{MF};
    sizes=(128, 256),
    samples::Int=3,
    thread_count::Int=Threads.nthreads(),
    candidates=nothing,
) where {MF<:MultiFloat}
    _check_supported(MF)
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
        C = zeros(MF, n, n)

        direct_config = KernelConfig(
            thread_count=workers,
            gemm_strategy=:direct,
        )
        direct_seconds = _median_elapsed_seconds(samples) do
            gemm!(C, A, B; config=direct_config)
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
            packed_seconds = _median_elapsed_seconds(samples) do
                gemm!(
                    C,
                    A,
                    B;
                    config=packed_config,
                    workspace=workspace,
                )
            end
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

    scores = fill(0.0, length(candidate_list))
    for (candidate_index, (panel_columns, micro_columns)) in pairs(candidate_list)
        for measurement in measurements
            if measurement.strategy === :packed &&
               measurement.panel_columns == panel_columns &&
               measurement.micro_columns == micro_columns
                scores[candidate_index] += log(measurement.seconds)
            end
        end
    end
    best_index = argmin(scores)
    best_panel, best_micro = candidate_list[best_index]

    crossover = typemax(Int)
    for n in tested_sizes
        direct_seconds = only(
            measurement.seconds for measurement in measurements
            if measurement.size == n && measurement.strategy === :direct
        )
        packed_seconds = only(
            measurement.seconds for measurement in measurements
            if measurement.size == n &&
               measurement.strategy === :packed &&
               measurement.panel_columns == best_panel &&
               measurement.micro_columns == best_micro
        )
        if packed_seconds <= 0.98 * direct_seconds
            crossover = n
            break
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
    return GemmCalibration{MF}(profile, measurements)
end
