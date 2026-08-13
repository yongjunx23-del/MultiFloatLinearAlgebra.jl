using MultiFloats
using MultiFloatLinearAlgebra
using Printf
using Random

Random.seed!(0x737472756374)

const T = Float64x3
const V4 = MultiFloatVec{4,Float64,3}

@inline _benchmark_mulacc(acc::V4, x::V4, y::V4, ::Val{:standard}) =
    acc + x * y
@inline _benchmark_mulacc(acc::V4, x::V4, y::V4, ::Val{:fused}) =
    MultiFloatLinearAlgebra.mulacc_x3(acc, x, y)

function _gemmt_column!(output, left, right, column, mode)
    rows, reduction = size(left)
    row = column
    @inbounds while row + 3 <= rows
        accumulator = zero(V4)
        for k in 1:reduction
            values = V4(
                left[row, k],
                left[row + 1, k],
                left[row + 2, k],
                left[row + 3, k],
            )
            accumulator = _benchmark_mulacc(
                accumulator, values, V4(right[column, k]), mode,
            )
        end
        for lane in 1:4
            output[row + lane - 1, column] = accumulator[lane]
        end
        row += 4
    end
    while row <= rows
        accumulator = zero(T)
        for k in 1:reduction
            accumulator += left[row, k] * right[column, k]
        end
        output[row, column] = accumulator
        row += 1
    end
    return nothing
end

function benchmark_gemmt!(output, left, right, threads, mode)
    rows = size(left, 1)
    if threads == 1
        @inbounds for column in 1:rows
            _gemmt_column!(output, left, right, column, mode)
        end
        return output
    end
    @sync for worker in 1:threads
        Threads.@spawn begin
            @inbounds for column in worker:threads:rows
                _gemmt_column!(output, left, right, column, mode)
            end
        end
    end
    return output
end

function _syrk_column!(output, panel, column, mode)
    reduction, columns = size(panel)
    row = column
    @inbounds while row + 3 <= columns
        accumulator = zero(V4)
        for k in 1:reduction
            values = V4(
                panel[k, row],
                panel[k, row + 1],
                panel[k, row + 2],
                panel[k, row + 3],
            )
            accumulator = _benchmark_mulacc(
                accumulator, values, V4(panel[k, column]), mode,
            )
        end
        for lane in 1:4
            output[row + lane - 1, column] = accumulator[lane]
        end
        row += 4
    end
    while row <= columns
        accumulator = zero(T)
        for k in 1:reduction
            accumulator += panel[k, row] * panel[k, column]
        end
        output[row, column] = accumulator
        row += 1
    end
    return nothing
end

function benchmark_syrk!(output, panel, threads, mode)
    columns = size(panel, 2)
    if threads == 1
        @inbounds for column in 1:columns
            _syrk_column!(output, panel, column, mode)
        end
        return output
    end
    @sync for worker in 1:threads
        Threads.@spawn begin
            @inbounds for column in worker:threads:columns
                _syrk_column!(output, panel, column, mode)
            end
        end
    end
    return output
end

function median_seconds(f, samples)
    f()
    elapsed = Vector{Float64}(undef, samples)
    for sample in 1:samples
        GC.gc()
        start = time_ns()
        f()
        elapsed[sample] = (time_ns() - start) / 1.0e9
    end
    sort!(elapsed)
    return elapsed[cld(samples, 2)]
end

function bitwise_equal(left, right)
    size(left) == size(right) || return false
    for index in eachindex(left, right), limb in 1:3
        reinterpret(UInt64, left[index]._limbs[limb]) ==
            reinterpret(UInt64, right[index]._limbs[limb]) || return false
    end
    return true
end

function report(operation, rows, reduction, threads, standard, fused)
    @printf(
        "%-5s rows=%4d k=%2d threads=%d standard=%9.6fs fused=%9.6fs speedup=%6.3fx\n",
        operation,
        rows,
        reduction,
        threads,
        standard,
        fused,
        standard / fused,
    )
end

function main()
    samples = isempty(ARGS) ? 7 : parse(Int, ARGS[1])
    samples > 0 || throw(ArgumentError("sample count must be positive"))
    thread_counts = unique((1, min(4, Threads.nthreads())))
    println("Benchmark-local Float64x3 structured mulacc comparison")
    println(
        "Julia $(VERSION), architecture=$(Sys.ARCH), kernel=$(Sys.KERNEL), " *
        "Julia threads=$(Threads.nthreads()), samples=$samples",
    )
    println("Speedup > 1 favors fused; outputs are checked limb-by-limb.")
    println()
    for threads in thread_counts, rows in (128, 256), reduction in (8, 16, 32)
        left = T.(randn(rows, reduction))
        right = T.(randn(rows, reduction))
        standard = zeros(T, rows, rows)
        fused = zeros(T, rows, rows)
        standard_call = () -> benchmark_gemmt!(
            standard, left, right, threads, Val(:standard),
        )
        fused_call = () -> benchmark_gemmt!(
            fused, left, right, threads, Val(:fused),
        )
        standard_call()
        fused_call()
        bitwise_equal(standard, fused) ||
            error("GEMMT fused output changed limb bits")
        report(
            "gemmt",
            rows,
            reduction,
            threads,
            median_seconds(standard_call, samples),
            median_seconds(fused_call, samples),
        )

        panel = T.(randn(reduction, rows))
        standard_syrk = zeros(T, rows, rows)
        fused_syrk = zeros(T, rows, rows)
        standard_syrk_call = () -> benchmark_syrk!(
            standard_syrk, panel, threads, Val(:standard),
        )
        fused_syrk_call = () -> benchmark_syrk!(
            fused_syrk, panel, threads, Val(:fused),
        )
        standard_syrk_call()
        fused_syrk_call()
        bitwise_equal(standard_syrk, fused_syrk) ||
            error("SYRK fused output changed limb bits")
        report(
            "syrk",
            rows,
            reduction,
            threads,
            median_seconds(standard_syrk_call, samples),
            median_seconds(fused_syrk_call, samples),
        )
    end
end

main()
