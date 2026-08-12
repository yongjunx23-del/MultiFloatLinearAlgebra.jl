using MultiFloats
using MultiFloatLinearAlgebra
using Printf

const TYPES = (Float64x2, Float64x3, Float64x4)

function print_calibration(calibration)
    profile = calibration.profile
    println(
        "profile: strategy=$(profile.strategy), panel=$(profile.panel_columns), " *
        "micro=$(profile.micro_columns), crossover=$(profile.packed_crossover), " *
        "source=$(profile.source), minimum_speedup=$(calibration.minimum_speedup)",
    )
    println("fingerprint: $(profile.fingerprint)")
    println("measurements:")
    for measurement in calibration.measurements
        @printf(
            "  n=%4d route=%-7s panel=%3d micro=%d time=%10.6f s\n",
            measurement.size,
            string(measurement.strategy),
            measurement.panel_columns,
            measurement.micro_columns,
            measurement.seconds,
        )
    end
end

function main()
    sizes = length(ARGS) >= 2 ?
            (parse(Int, ARGS[1]), parse(Int, ARGS[2])) :
            (512, 1024)
    samples = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
    minimum_speedup = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.05
    threads = Threads.nthreads()

    println("MultiFloat GEMM machine calibration")
    println(
        "sizes=$sizes, samples=$samples, minimum_speedup=$minimum_speedup, " *
        "threads=$threads",
    )
    println()

    for T in TYPES
        println("== $T ==")
        calibration = calibrate_gemm(
            T;
            sizes=sizes,
            samples=samples,
            thread_count=threads,
            minimum_speedup=minimum_speedup,
        )
        print_calibration(calibration)
        println()
    end
end

main()
