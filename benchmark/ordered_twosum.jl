using InteractiveUtils
using MultiFloats
using Printf
using Random

Random.seed!(20260812)
const TYPES = (Float64x3, Float64x4)
const SINK = Ref{Any}(nothing)

@inline function local_fast_two_sum(a, b)
    s = a + b
    bprime = s - a
    e = b - bprime
    return s, e
end

@inline function ordered_two_sum(a, b)
    choose_a = abs(a) >= abs(b)
    large = MultiFloats.vifelse(choose_a, a, b)
    small = MultiFloats.vifelse(choose_a, b, a)
    return local_fast_two_sum(large, small)
end

@inline function ordered_add3(x::MultiFloatVec{4,Float64,3}, y::MultiFloatVec{4,Float64,3})
    x1, x2, x3 = x._limbs
    y1, y2, y3 = y._limbs
    a, b = ordered_two_sum(x1, y1)
    c, d = ordered_two_sum(x2, y2)
    e, f = ordered_two_sum(x3, y3)
    a, c = local_fast_two_sum(a, c)
    b += f
    d, e = ordered_two_sum(d, e)
    a, d = local_fast_two_sum(a, d)
    b, c = ordered_two_sum(b, c)
    c += e
    c, d = ordered_two_sum(c, d)
    b, c = ordered_two_sum(b, c)
    a, b = local_fast_two_sum(a, b)
    c += d
    b, c = local_fast_two_sum(b, c)
    a, b = local_fast_two_sum(a, b)
    b, c = local_fast_two_sum(b, c)
    return MultiFloatVec{4,Float64,3}((a, b, c))
end

@inline function ordered_add4(x::MultiFloatVec{4,Float64,4}, y::MultiFloatVec{4,Float64,4})
    x1, x2, x3, x4 = x._limbs
    y1, y2, y3, y4 = y._limbs
    a, b = ordered_two_sum(x1, y1)
    c, d = ordered_two_sum(x2, y2)
    e, f = ordered_two_sum(x3, y3)
    g, h = ordered_two_sum(x4, y4)
    a, c = local_fast_two_sum(a, c)
    b += h
    d, e = ordered_two_sum(d, e)
    f, g = ordered_two_sum(f, g)
    b, g = ordered_two_sum(b, g)
    c, d = local_fast_two_sum(c, d)
    e, f = ordered_two_sum(e, f)
    a, c = local_fast_two_sum(a, c)
    d, e = local_fast_two_sum(d, e)
    b, d = ordered_two_sum(b, d)
    c, g = local_fast_two_sum(c, g)
    e += f
    b, c = ordered_two_sum(b, c)
    d, e = ordered_two_sum(d, e)
    a, b = local_fast_two_sum(a, b)
    c, d = ordered_two_sum(c, d)
    e += g
    b, c = local_fast_two_sum(b, c)
    d, e = ordered_two_sum(d, e)
    a, b = local_fast_two_sum(a, b)
    c, d = local_fast_two_sum(c, d)
    b, c = local_fast_two_sum(b, c)
    d += e
    a, b = local_fast_two_sum(a, b)
    c, d = local_fast_two_sum(c, d)
    b, c = local_fast_two_sum(b, c)
    c, d = local_fast_two_sum(c, d)
    return MultiFloatVec{4,Float64,4}((a, b, c, d))
end

@inline ordered_add(x::MultiFloatVec{4,Float64,3}, y::MultiFloatVec{4,Float64,3}) = ordered_add3(x, y)
@inline ordered_add(x::MultiFloatVec{4,Float64,4}, y::MultiFloatVec{4,Float64,4}) = ordered_add4(x, y)

@inline function vec4_bitwise_equal(x::MultiFloatVec{4}, y::MultiFloatVec{4})
    @inbounds for lane in 1:4
        x[lane] === y[lane] || return false
    end
    return true
end

function array_bitwise_equal(x, y)
    length(x) == length(y) || return false
    @inbounds for i in eachindex(x, y)
        vec4_bitwise_equal(x[i], y[i]) || return false
    end
    return true
end

@inline function vec4_isnormalized(x::MultiFloatVec{4})
    @inbounds for lane in 1:4
        MultiFloats.isnormalized(x[lane]) || return false
    end
    return true
end

function array_isnormalized(x)
    @inbounds for value in x
        vec4_isnormalized(value) || return false
    end
    return true
end

function make_data(::Type{T}, n::Int; cancellation=false) where {T<:MultiFloat}
    V = MultiFloatVec{4,Float64,T.parameters[2]}
    a = Vector{V}(undef, n)
    b = Vector{V}(undef, n)
    for i in 1:n
        xs = ntuple(Val(4)) do lane
            scale = ldexp(1.0, mod(17 * i + 11 * lane, 80) - 40)
            T(scale * (0.75 + 0.2 * sin(0.013 * i + lane)))
        end
        ys = if cancellation
            ntuple(Val(4)) do lane
                # Near-opposite leading values with a smaller perturbation.
                -xs[lane] + T(ldexp(0.3 + 0.1 * cos(0.017 * i + lane), -40))
            end
        else
            ntuple(Val(4)) do lane
                scale = ldexp(1.0, mod(13 * i + 7 * lane, 80) - 40)
                T(scale * (0.65 + 0.2 * cos(0.019 * i + lane)))
            end
        end
        a[i] = V(xs...)
        b[i] = V(ys...)
    end
    return a, b
end

function median_seconds(f, samples)
    f()
    elapsed = Vector{Float64}(undef, samples)
    for i in eachindex(elapsed)
        GC.gc()
        t0 = time_ns()
        v = f()
        elapsed[i] = (time_ns() - t0) / 1e9
        SINK[] = v
    end
    sort!(elapsed)
    return elapsed[cld(samples, 2)]
end

@noinline function standard_kernel!(out, a, b, repeats)
    @inbounds for _ in 1:repeats
        for i in eachindex(out, a, b)
            out[i] = a[i] + b[i]
        end
    end
    return out[1]
end

@noinline function ordered_kernel!(out, a, b, repeats)
    @inbounds for _ in 1:repeats
        for i in eachindex(out, a, b)
            out[i] = ordered_add(a[i], b[i])
        end
    end
    return out[1]
end

function native_stats(f, signature)
    text = sprint(io -> code_native(io, f, signature; debuginfo=:none))
    spill = count(eachline(IOBuffer(text))) do line
        occursin("Spill", line) || occursin("Reload", line)
    end
    ymm = Set{Int}()
    for m in eachmatch(r"\bymm(\d+)\b", lowercase(text))
        push!(ymm, parse(Int, m.captures[1]))
    end
    return length(split(text, '\n')), spill, length(ymm), text
end

function check_twosum_primitive(::Type{T}) where {T<:MultiFloat}
    a, b = make_data(T, 4096)
    ac, bc = make_data(T, 4096; cancellation=true)
    for (xs, ys) in ((a, b), (ac, bc))
        for i in eachindex(xs, ys)
            for limb in 1:T.parameters[2]
                s1, e1 = MultiFloats.two_sum(xs[i]._limbs[limb], ys[i]._limbs[limb])
                s2, e2 = ordered_two_sum(xs[i]._limbs[limb], ys[i]._limbs[limb])
                (s1 === s2 && e1 === e2) || return false
            end
        end
    end
    return true
end

function profile_type(::Type{T}; n=4096, repeats=16, samples=5) where {T<:MultiFloat}
    V = MultiFloatVec{4,Float64,T.parameters[2]}
    println("== $T / $V ==")
    primitive_equal = check_twosum_primitive(T)
    println("ordered TwoSum primitive bitwise equal on stress set: $primitive_equal")
    primitive_equal || error("ordered TwoSum changed exact residual")

    for cancellation in (false, true)
        a, b = make_data(T, n; cancellation=cancellation)
        standard = similar(a)
        ordered = similar(a)
        standard_kernel!(standard, a, b, 1)
        ordered_kernel!(ordered, a, b, 1)
        equal = array_bitwise_equal(standard, ordered)
        normalized = array_isnormalized(ordered)
        println("mode=$(cancellation ? "cancellation" : "scaled-random") bitwise_equal=$equal normalized=$normalized")
        equal || error("ordered mfadd changed MultiFloat result")
        normalized || error("ordered mfadd returned non-normalized output")

        logical = n * 4 * repeats
        ts = median_seconds(() -> standard_kernel!(standard, a, b, repeats), samples)
        to = median_seconds(() -> ordered_kernel!(ordered, a, b, repeats), samples)
        ns_s = ts * 1e9 / logical
        ns_o = to * 1e9 / logical
        @printf("  standard %9.3f ns/scalar-add\n", ns_s)
        @printf("  ordered  %9.3f ns/scalar-add  speedup=%6.3fx\n", ns_o, ns_s / ns_o)
    end

    _, spill_s, regs_s, native_s = native_stats(+, (V, V))
    _, spill_o, regs_o, native_o = native_stats(ordered_add, (V, V))
    println("codegen standard spill/reload=$spill_s ymm=$regs_s")
    println("codegen ordered  spill/reload=$spill_o ymm=$regs_o")
    if !isempty(get(ENV, "MFLA_ORDERED_OUT", ""))
        out = ENV["MFLA_ORDERED_OUT"]
        mkpath(out)
        open(joinpath(out, "x$(T.parameters[2])_standard.native.txt"), "w") do io
            write(io, native_s)
        end
        open(joinpath(out, "x$(T.parameters[2])_ordered.native.txt"), "w") do io
            write(io, native_o)
        end
    end
    println()
end

function main()
    println("Magnitude-ordered FastTwoSum mfadd experiment")
    println("Julia=$(VERSION) arch=$(Sys.ARCH) threads=$(Threads.nthreads())")
    println()
    for T in TYPES
        profile_type(T)
    end
end

main()
