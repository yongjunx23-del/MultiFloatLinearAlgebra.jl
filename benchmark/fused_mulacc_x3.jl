using InteractiveUtils
using MultiFloats
using Printf
using Random

const T3 = Float64x3
const V3 = MultiFloatVec{4,Float64,3}
const SINK = Ref{Any}(nothing)

# Reuse the arithmetic-network IR that ships with MultiFloats.jl.  This file is
# benchmark-only and does not add MFIR to the package runtime dependency graph.
const _mf_root = dirname(dirname(pathof(MultiFloats)))
const _mfir_path = joinpath(_mf_root, "scripts", "MFIR.jl")
isfile(_mfir_path) || error("MultiFloats MFIR script not found at $_mfir_path")
include(_mfir_path)
using .MFIR

# -----------------------------------------------------------------------------
# MFIR tail models
# -----------------------------------------------------------------------------

mutable struct IRBuilder
    instructions::Vector{MFIR.MFIRInstruction}
    next_register::Int
end

IRBuilder(num_inputs::Int) = IRBuilder(MFIR.MFIRInstruction[], num_inputs)

@inline function emit1!(builder::IRBuilder, op, a::Int)
    push!(builder.instructions, MFIR.MFIRInstruction(op, a))
    builder.next_register += 1
    return builder.next_register
end

@inline function emit1!(builder::IRBuilder, op, a::Int, b::Int)
    push!(builder.instructions, MFIR.MFIRInstruction(op, a, b))
    builder.next_register += 1
    return builder.next_register
end

@inline function emit2!(builder::IRBuilder, op, a::Int, b::Int)
    push!(builder.instructions, MFIR.MFIRInstruction(op, a, b))
    first = builder.next_register + 1
    builder.next_register += 2
    return first, first + 1
end

function append_mfadd3!(builder::IRBuilder, p0::Int, p1::Int, p2::Int)
    # Inputs 4:6 are accumulator limbs.  This is exactly the MultiFloats x3
    # mfadd network, represented in MFIR after the product-side fusion cut.
    a, b = emit2!(builder, MFIR.MFIR_TWO_SUM, 4, p0)
    c, d = emit2!(builder, MFIR.MFIR_TWO_SUM, 5, p1)
    e, f = emit2!(builder, MFIR.MFIR_TWO_SUM, 6, p2)
    a, c = emit2!(builder, MFIR.MFIR_FAST_TWO_SUM, a, c)
    b = emit1!(builder, MFIR.MFIR_ADD, b, f)
    d, e = emit2!(builder, MFIR.MFIR_TWO_SUM, d, e)
    a, d = emit2!(builder, MFIR.MFIR_FAST_TWO_SUM, a, d)
    b, c = emit2!(builder, MFIR.MFIR_TWO_SUM, b, c)
    c = emit1!(builder, MFIR.MFIR_ADD, c, e)
    c, d = emit2!(builder, MFIR.MFIR_TWO_SUM, c, d)
    b, c = emit2!(builder, MFIR.MFIR_TWO_SUM, b, c)
    a, b = emit2!(builder, MFIR.MFIR_FAST_TWO_SUM, a, b)
    c = emit1!(builder, MFIR.MFIR_ADD, c, d)
    b, c = emit2!(builder, MFIR.MFIR_FAST_TWO_SUM, b, c)
    a, b = emit2!(builder, MFIR.MFIR_FAST_TWO_SUM, a, b)
    b, c = emit2!(builder, MFIR.MFIR_FAST_TWO_SUM, b, c)
    return a, b, c
end

function build_tail_program(stage::Symbol)
    # Inputs:
    # 1=p00, 2=e00, 3=p01 from the common x3 multiplication prefix,
    # 4:6=accumulator limbs.
    builder = IRBuilder(6)
    p0, p1, p2 = 1, 2, 3
    if stage in (:mid, :conservative, :current)
        p1, p2 = emit2!(builder, MFIR.MFIR_TWO_SUM, p1, p2)
    end
    if stage in (:conservative, :current)
        p0, p1 = emit2!(builder, MFIR.MFIR_FAST_TWO_SUM, p0, p1)
    end
    if stage === :current
        p1, p2 = emit2!(builder, MFIR.MFIR_FAST_TWO_SUM, p1, p2)
        p0, p1 = emit2!(builder, MFIR.MFIR_FAST_TWO_SUM, p0, p1)
    elseif !(stage in (:aggressive, :mid, :conservative))
        throw(ArgumentError("unknown MFIR fusion stage $stage"))
    end
    a, b, c = append_mfadd3!(builder, p0, p1, p2)
    return MFIR.MFIRProgram(6, builder.instructions, UInt16[a, b, c])
end

function peak_live_registers(program::MFIR.MFIRProgram)
    uses = MFIR.use_counts(program)
    live = count(>(0), @view uses[1:program.num_inputs])
    peak = live
    @inbounds for (instruction, range) in zip(program.instructions, program.result_ranges)
        for j in 1:MFIR.arity(instruction)
            reg = Int(instruction.args[j])
            uses[reg] -= 1
            uses[reg] == 0 && (live -= 1)
        end
        for reg in range
            uses[Int(reg)] > 0 && (live += 1)
        end
        peak = max(peak, live)
    end
    return peak
end

function macro_critical_depth(program::MFIR.MFIRProgram)
    depth = zeros(Int, MFIR.num_registers(program))
    @inbounds for (instruction, range) in zip(program.instructions, program.result_ranges)
        input_depth = 0
        for j in 1:MFIR.arity(instruction)
            input_depth = max(input_depth, depth[Int(instruction.args[j])])
        end
        out_depth = input_depth + 1
        for reg in range
            depth[Int(reg)] = out_depth
        end
    end
    return maximum(depth[Int(i)] for i in program.output_indices)
end

function print_mfir_summary()
    println("MFIR fusion-tail summary (common multiplication prefix excluded)")
    @printf("%-14s %8s %10s %12s\n", "stage", "instr", "peak-live", "macro-depth")
    for stage in (:current, :conservative, :mid, :aggressive)
        program = build_tail_program(stage)
        @printf(
            "%-14s %8d %10d %12d\n",
            String(stage),
            length(program.instructions),
            peak_live_registers(program),
            macro_critical_depth(program),
        )
    end
    println()
end

# -----------------------------------------------------------------------------
# Fused arithmetic candidates
# -----------------------------------------------------------------------------

@inline function raw_mfadd3(
    x0, x1, x2,
    y0, y1, y2,
)
    a, b = MultiFloats.two_sum(x0, y0)
    c, d = MultiFloats.two_sum(x1, y1)
    e, f = MultiFloats.two_sum(x2, y2)
    a, c = MultiFloats.fast_two_sum(a, c)
    b += f
    d, e = MultiFloats.two_sum(d, e)
    a, d = MultiFloats.fast_two_sum(a, d)
    b, c = MultiFloats.two_sum(b, c)
    c += e
    c, d = MultiFloats.two_sum(c, d)
    b, c = MultiFloats.two_sum(b, c)
    a, b = MultiFloats.fast_two_sum(a, b)
    c += d
    b, c = MultiFloats.fast_two_sum(b, c)
    a, b = MultiFloats.fast_two_sum(a, b)
    b, c = MultiFloats.fast_two_sum(b, c)
    return V3((a, b, c))
end

@inline function product_prefix_x3(x::V3, y::V3)
    x0, x1, x2 = x._limbs
    y0, y1, y2 = y._limbs
    p00, e00 = MultiFloats.two_prod(x0, y0)
    p01, e01 = MultiFloats.two_prod(x0, y1)
    p10, e10 = MultiFloats.two_prod(x1, y0)
    p02 = MultiFloats.one_prod(x0, y2)
    p11 = MultiFloats.one_prod(x1, y1)
    p20 = MultiFloats.one_prod(x2, y0)
    p01, p10 = MultiFloats.two_sum(p01, p10)
    e01 += e10
    p02 += p20
    e00, p01 = MultiFloats.two_sum(e00, p01)
    p02 += p11
    p00, e00 = MultiFloats.fast_two_sum(p00, e00)
    p01 += p10
    e01 += p02
    p01 += e01
    return p00, e00, p01
end

@inline current_mulacc_x3(acc::V3, x::V3, y::V3) = acc + x * y

@inline function fused_mulacc_aggressive(acc::V3, x::V3, y::V3)
    p0, p1, p2 = product_prefix_x3(x, y)
    a0, a1, a2 = acc._limbs
    return raw_mfadd3(a0, a1, a2, p0, p1, p2)
end

@inline function fused_mulacc_mid(acc::V3, x::V3, y::V3)
    p0, p1, p2 = product_prefix_x3(x, y)
    p1, p2 = MultiFloats.two_sum(p1, p2)
    a0, a1, a2 = acc._limbs
    return raw_mfadd3(a0, a1, a2, p0, p1, p2)
end

@inline function fused_mulacc_conservative(acc::V3, x::V3, y::V3)
    p0, p1, p2 = product_prefix_x3(x, y)
    p1, p2 = MultiFloats.two_sum(p1, p2)
    p0, p1 = MultiFloats.fast_two_sum(p0, p1)
    a0, a1, a2 = acc._limbs
    return raw_mfadd3(a0, a1, a2, p0, p1, p2)
end

const CANDIDATES = (
    aggressive=fused_mulacc_aggressive,
    mid=fused_mulacc_mid,
    conservative=fused_mulacc_conservative,
)

# -----------------------------------------------------------------------------
# Data generation and BigFloat differential validation
# -----------------------------------------------------------------------------

function rich_big(rng::AbstractRNG, exponent::Int; sign::Int=1)
    base = ldexp(BigFloat(0.5 + rand(rng)), exponent)
    correction1 = ldexp(BigFloat(randn(rng)), exponent - 70)
    correction2 = ldexp(BigFloat(randn(rng)), exponent - 135)
    return sign * (base + correction1 + correction2)
end

function one_step_case(rng::AbstractRNG, mode::Symbol, i::Int, lane::Int)
    if mode === :random
        ex = rand(rng, -12:12)
        ey = rand(rng, -12:12)
        x = rich_big(rng, ex; sign=rand(rng, Bool) ? 1 : -1)
        y = rich_big(rng, ey; sign=rand(rng, Bool) ? 1 : -1)
        a = rich_big(rng, rand(rng, -12:12); sign=rand(rng, Bool) ? 1 : -1)
    elseif mode === :wide
        ex = rand(rng, -300:300)
        ey = rand(rng, -300:300)
        x = rich_big(rng, ex; sign=rand(rng, Bool) ? 1 : -1)
        y = rich_big(rng, ey; sign=rand(rng, Bool) ? 1 : -1)
        a = rich_big(rng, rand(rng, -500:500); sign=rand(rng, Bool) ? 1 : -1)
    elseif mode === :cancellation
        ex = rand(rng, -180:180)
        ey = rand(rng, -180:180)
        x = rich_big(rng, ex; sign=rand(rng, Bool) ? 1 : -1)
        y = rich_big(rng, ey; sign=rand(rng, Bool) ? 1 : -1)
        product = x * y
        perturb = ldexp(abs(product) + one(BigFloat), -150)
        a = -product + (isodd(i + lane) ? perturb : -perturb)
    elseif mode === :alternating
        ex = mod(17i + 11lane, 121) - 60
        ey = mod(13i + 7lane, 121) - 60
        x = rich_big(rng, ex; sign=isodd(i + lane) ? 1 : -1)
        y = rich_big(rng, ey; sign=isodd(i) ? -1 : 1)
        a = rich_big(rng, mod(5i + 3lane, 81) - 40; sign=isodd(lane) ? 1 : -1)
    elseif mode === :edge
        high = isodd(i + lane)
        ex = high ? 445 : -445
        ey = high ? 440 : -440
        x = rich_big(rng, ex; sign=isodd(i) ? 1 : -1)
        y = rich_big(rng, ey; sign=isodd(lane) ? -1 : 1)
        a = rich_big(rng, high ? 870 : -870; sign=isodd(i + lane) ? 1 : -1)
    else
        error("unknown mode $mode")
    end
    return T3(a), T3(x), T3(y)
end

function make_one_step_data(mode::Symbol, n::Int)
    rng = MersenneTwister(0x31f0 + Int(findfirst(==(mode), (:random, :wide, :cancellation, :alternating, :edge))))
    acc = Vector{V3}(undef, n)
    xs = Vector{V3}(undef, n)
    ys = Vector{V3}(undef, n)
    setprecision(BigFloat, 512) do
        for i in 1:n
            av = ntuple(lane -> one_step_case(rng, mode, i, lane)[1], Val(4))
            xv = ntuple(lane -> one_step_case(rng, mode, i, lane)[2], Val(4))
            yv = ntuple(lane -> one_step_case(rng, mode, i, lane)[3], Val(4))
            # Regenerate each triple coherently so cancellation mode keeps a/x/y related.
            triples = ntuple(lane -> one_step_case(rng, mode, i, lane), Val(4))
            acc[i] = V3(ntuple(lane -> triples[lane][1], Val(4))...)
            xs[i] = V3(ntuple(lane -> triples[lane][2], Val(4))...)
            ys[i] = V3(ntuple(lane -> triples[lane][3], Val(4))...)
        end
    end
    return acc, xs, ys
end

@inline scalar_lane(v::V3, lane::Int) = v[lane]

function normalized_vec(v::V3)
    @inbounds for lane in 1:4
        MultiFloats.isnormalized(v[lane]) || return false
    end
    return true
end

function validate_one_step(candidate, mode::Symbol; n::Int=384)
    acc, xs, ys = make_one_step_data(mode, n)
    max_current_scaled = BigFloat(0)
    max_candidate_scaled = BigFloat(0)
    worse = 0
    nonnormalized = 0
    setprecision(BigFloat, 512) do
        @inbounds for i in eachindex(acc, xs, ys)
            current = current_mulacc_x3(acc[i], xs[i], ys[i])
            fused = candidate(acc[i], xs[i], ys[i])
            normalized_vec(fused) || (nonnormalized += 1)
            for lane in 1:4
                a = BigFloat(acc[i][lane])
                x = BigFloat(xs[i][lane])
                y = BigFloat(ys[i][lane])
                ref = a + x * y
                scale = max(abs(a) + abs(x * y), ldexp(one(BigFloat), -1000))
                ec = abs(BigFloat(current[lane]) - ref) / scale
                ef = abs(BigFloat(fused[lane]) - ref) / scale
                max_current_scaled = max(max_current_scaled, ec)
                max_candidate_scaled = max(max_candidate_scaled, ef)
                ef > ec && (worse += 1)
            end
        end
    end
    ratio = iszero(max_current_scaled) ?
        (iszero(max_candidate_scaled) ? 1.0 : Inf) :
        Float64(max_candidate_scaled / max_current_scaled)
    return (
        current=max_current_scaled,
        fused=max_candidate_scaled,
        ratio=ratio,
        worse=worse,
        nonnormalized=nonnormalized,
        total=4n,
    )
end

function make_dot_data(mode::Symbol, n::Int)
    rng = MersenneTwister(0xd07 + Int(findfirst(==(mode), (:random, :wide, :cancellation, :alternating, :edge))))
    xs = Vector{V3}(undef, n)
    ys = Vector{V3}(undef, n)
    setprecision(BigFloat, 512) do
        i = 1
        while i <= n
            xv = Vector{T3}(undef, 4)
            yv = Vector{T3}(undef, 4)
            for lane in 1:4
                if mode === :wide
                    ex, ey = rand(rng, -280:280), rand(rng, -280:280)
                elseif mode === :edge
                    ex = isodd(i + lane) ? 430 : -430
                    ey = isodd(i + lane) ? 430 : -430
                else
                    ex, ey = rand(rng, -40:40), rand(rng, -40:40)
                end
                bx = rich_big(rng, ex; sign=rand(rng, Bool) ? 1 : -1)
                by = rich_big(rng, ey; sign=rand(rng, Bool) ? 1 : -1)
                if mode === :alternating
                    by *= isodd(i) ? one(BigFloat) : -one(BigFloat)
                end
                xv[lane] = T3(bx)
                yv[lane] = T3(by)
            end
            xs[i] = V3(xv...)
            ys[i] = V3(yv...)
            if mode === :cancellation && i < n
                xs[i + 1] = xs[i]
                partner = ntuple(Val(4)) do lane
                    base = -BigFloat(ys[i][lane])
                    perturb = ldexp(abs(base) + one(BigFloat), -145)
                    T3(base + (isodd(lane) ? perturb : -perturb))
                end
                ys[i + 1] = V3(partner...)
                i += 2
            else
                i += 1
            end
        end
    end
    return xs, ys
end

@noinline function current_dot(xs::Vector{V3}, ys::Vector{V3})
    acc = zero(V3)
    @inbounds for i in eachindex(xs, ys)
        acc = current_mulacc_x3(acc, xs[i], ys[i])
    end
    return acc
end

@noinline function fused_dot(candidate, xs::Vector{V3}, ys::Vector{V3})
    acc = zero(V3)
    @inbounds for i in eachindex(xs, ys)
        acc = candidate(acc, xs[i], ys[i])
    end
    return acc
end

function validate_dot(candidate, mode::Symbol; n::Int=512)
    xs, ys = make_dot_data(mode, n)
    current = current_dot(xs, ys)
    fused = fused_dot(candidate, xs, ys)
    normalized = normalized_vec(fused)
    current_max = BigFloat(0)
    fused_max = BigFloat(0)
    setprecision(BigFloat, 512) do
        for lane in 1:4
            ref = BigFloat(0)
            sumabs = BigFloat(0)
            @inbounds for i in eachindex(xs, ys)
                product = BigFloat(xs[i][lane]) * BigFloat(ys[i][lane])
                ref += product
                sumabs += abs(product)
            end
            scale = max(sumabs, ldexp(one(BigFloat), -1000))
            current_max = max(current_max, abs(BigFloat(current[lane]) - ref) / scale)
            fused_max = max(fused_max, abs(BigFloat(fused[lane]) - ref) / scale)
        end
    end
    ratio = iszero(current_max) ? (iszero(fused_max) ? 1.0 : Inf) : Float64(fused_max / current_max)
    return (current=current_max, fused=fused_max, ratio=ratio, normalized=normalized)
end

# -----------------------------------------------------------------------------
# Throughput and codegen
# -----------------------------------------------------------------------------

function median_seconds(f, samples::Int)
    f()
    times = Vector{Float64}(undef, samples)
    for sample in 1:samples
        GC.gc()
        t0 = time_ns()
        value = f()
        times[sample] = (time_ns() - t0) / 1e9
        SINK[] = value
    end
    sort!(times)
    return times[cld(samples, 2)]
end

function benchmark_dot(candidate; n::Int=8192, samples::Int=9)
    xs, ys = make_dot_data(:random, n)
    tc = median_seconds(() -> current_dot(xs, ys), samples)
    tf = median_seconds(() -> fused_dot(candidate, xs, ys), samples)
    scalar_terms = 4n
    return (
        current_ns=tc * 1e9 / scalar_terms,
        fused_ns=tf * 1e9 / scalar_terms,
        speedup=tc / tf,
    )
end

function native_stats(f)
    text = sprint(io -> code_native(io, f, (V3, V3, V3); debuginfo=:none))
    lines = split(text, '\n')
    instructions = count(line -> occursin('\t', line) && !startswith(strip(line), "."), lines)
    spill = count(line -> occursin("Spill", line) || occursin("Reload", line), lines)
    calls = count(line -> occursin(r"\bcall", lowercase(line)), lines)
    ymm = Set{Int}()
    for m in eachmatch(r"\bymm(\d+)\b", lowercase(text))
        push!(ymm, parse(Int, m.captures[1]))
    end
    return (instructions=instructions, spill=spill, calls=calls, ymm=length(ymm))
end

function main()
    println("Float64x3 MFIR fused mulacc prototype")
    println("Julia=$(VERSION) arch=$(Sys.ARCH) threads=$(Threads.nthreads()) MultiFloats=3.2.6")
    println()
    print_mfir_summary()

    modes = (:random, :wide, :cancellation, :alternating, :edge)
    accepted_accuracy = Dict{Symbol,Bool}()

    for (name, candidate) in pairs(CANDIDATES)
        println("== candidate: $name ==")
        accuracy_ok = true
        @printf("%-14s %12s %12s %9s %10s %10s\n", "mode", "current", "fused", "ratio", "worse", "non-norm")
        for mode in modes
            stats = validate_one_step(candidate, mode)
            @printf(
                "%-14s %12.3e %12.3e %9.3f %10d %10d\n",
                String(mode), Float64(stats.current), Float64(stats.fused),
                stats.ratio, stats.worse, stats.nonnormalized,
            )
            accuracy_ok &= stats.nonnormalized == 0
            accuracy_ok &= stats.ratio <= 1.05
        end

        println("dot differential:")
        @printf("%-14s %12s %12s %9s %10s\n", "mode", "current", "fused", "ratio", "normalized")
        for mode in modes
            stats = validate_dot(candidate, mode)
            @printf(
                "%-14s %12.3e %12.3e %9.3f %10s\n",
                String(mode), Float64(stats.current), Float64(stats.fused),
                stats.ratio, string(stats.normalized),
            )
            accuracy_ok &= stats.normalized
            accuracy_ok &= stats.ratio <= 1.05
        end
        accepted_accuracy[name] = accuracy_ok

        perf = benchmark_dot(candidate)
        @printf(
            "Vec4 dot: current=%8.3f ns/scalar-term fused=%8.3f ns/scalar-term speedup=%6.3fx\n",
            perf.current_ns, perf.fused_ns, perf.speedup,
        )
        code = native_stats(candidate)
        println("codegen: instructions=$(code.instructions) spill/reload=$(code.spill) calls=$(code.calls) ymm=$(code.ymm)")
        println("accuracy gate: $(accuracy_ok ? "PASS" : "FAIL")")
        println("production gate: $(accuracy_ok && perf.speedup > 1.05 ? "PASS" : "FAIL")")
        println()
    end

    current_code = native_stats(current_mulacc_x3)
    println("current codegen: instructions=$(current_code.instructions) spill/reload=$(current_code.spill) calls=$(current_code.calls) ymm=$(current_code.ymm)")
    println("No production arithmetic is modified by this experiment.")
end

main()
