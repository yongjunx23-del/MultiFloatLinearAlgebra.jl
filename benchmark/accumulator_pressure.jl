using InteractiveUtils
using MultiFloats
using Printf

const TYPES = (Float64x2, Float64x3, Float64x4)
const SINK = Ref{Any}(nothing)

function median_seconds(f, samples::Int)
    f()
    elapsed = Vector{Float64}(undef, max(samples, 1))
    for sample in eachindex(elapsed)
        GC.gc()
        start = time_ns()
        value = f()
        elapsed[sample] = (time_ns() - start) / 1.0e9
        SINK[] = value
    end
    sort!(elapsed)
    return elapsed[cld(length(elapsed), 2)]
end

vec4_type(::Type{MultiFloat{Float64,N}}) where {N} = MultiFloatVec{4,Float64,N}

function make_inputs(::Type{T}, n::Int) where {T<:MultiFloat}
    V = vec4_type(T)
    a = Vector{V}(undef, n)
    b1 = Vector{T}(undef, n)
    b2 = Vector{T}(undef, n)
    b3 = Vector{T}(undef, n)
    b4 = Vector{T}(undef, n)
    @inbounds for i in 1:n
        a[i] = V(
            T(0.8 + 0.05 * sin(0.013 * i)),
            T(0.7 + 0.04 * cos(0.017 * i)),
            T(0.9 + 0.03 * sin(0.019 * i)),
            T(0.6 + 0.02 * cos(0.023 * i)),
        )
        b1[i] = T(0.7 + 0.03 * sin(0.029 * i))
        b2[i] = T(0.8 + 0.03 * cos(0.031 * i))
        b3[i] = T(0.9 + 0.02 * sin(0.037 * i))
        b4[i] = T(0.6 + 0.02 * cos(0.041 * i))
    end
    return a, b1, b2, b3, b4
end

@noinline function dot1(a::Vector{V}, b1::Vector{T}, repeats::Int) where {V,T}
    total = zero(V)
    @inbounds for _ in 1:repeats
        acc1 = zero(V)
        for k in eachindex(a, b1)
            acc1 += a[k] * V(b1[k])
        end
        total += acc1
    end
    return total
end

@noinline function dot2_interleaved(
    a::Vector{V}, b1::Vector{T}, b2::Vector{T}, repeats::Int,
) where {V,T}
    total = zero(V)
    @inbounds for _ in 1:repeats
        acc1 = zero(V)
        acc2 = zero(V)
        for k in eachindex(a, b1, b2)
            values = a[k]
            acc1 += values * V(b1[k])
            acc2 += values * V(b2[k])
        end
        total += acc1 + acc2
    end
    return total
end

@noinline function dot2_split(
    a::Vector{V}, b1::Vector{T}, b2::Vector{T}, repeats::Int,
) where {V,T}
    total = zero(V)
    @inbounds for _ in 1:repeats
        acc1 = zero(V)
        for k in eachindex(a, b1)
            acc1 += a[k] * V(b1[k])
        end
        acc2 = zero(V)
        for k in eachindex(a, b2)
            acc2 += a[k] * V(b2[k])
        end
        total += acc1 + acc2
    end
    return total
end

@noinline function dot4_interleaved(
    a::Vector{V}, b1::Vector{T}, b2::Vector{T}, b3::Vector{T}, b4::Vector{T}, repeats::Int,
) where {V,T}
    total = zero(V)
    @inbounds for _ in 1:repeats
        acc1 = zero(V)
        acc2 = zero(V)
        acc3 = zero(V)
        acc4 = zero(V)
        for k in eachindex(a, b1, b2, b3, b4)
            values = a[k]
            acc1 += values * V(b1[k])
            acc2 += values * V(b2[k])
            acc3 += values * V(b3[k])
            acc4 += values * V(b4[k])
        end
        total += (acc1 + acc2) + (acc3 + acc4)
    end
    return total
end

function native_text(f, signature)
    sprint() do io
        code_native(io, f, signature; debuginfo=:none)
    end
end

function actual_instructions(native::String)
    result = String[]
    for line in eachline(IOBuffer(native))
        s = strip(line)
        isempty(s) && continue
        (startswith(s, ".") || startswith(s, "#") || startswith(s, ";")) && continue
        endswith(s, ":") && continue
        occursin(r"^[A-Za-z][A-Za-z0-9.]*\s", s) || continue
        push!(result, s)
    end
    return result
end

function assembly_pressure(native::String)
    instructions = actual_instructions(native)
    stack_mem = count(instructions) do line
        occursin(r"\[[^\]]*\b(rsp|rbp)\b[^\]]*\]", lowercase(line))
    end
    calls = count(line -> startswith(lowercase(line), "call"), instructions)
    ymm = Int[]
    for m in eachmatch(r"\bymm(\d+)\b", lowercase(native))
        push!(ymm, parse(Int, m.captures[1]))
    end
    unique!(sort!(ymm))
    return (
        instructions=length(instructions),
        stack_mem=stack_mem,
        calls=calls,
        ymm_count=length(ymm),
        max_ymm=isempty(ymm) ? -1 : maximum(ymm),
    )
end

function write_native(outdir, label, native)
    isempty(outdir) && return
    mkpath(outdir)
    open(joinpath(outdir, label * ".native.txt"), "w") do io
        write(io, native)
    end
end

function report(label, seconds, products, reference)
    ns = seconds * 1e9 / products
    @printf("%-22s %9.3f ns/product  relative=%6.3fx\n", label, ns, ns / reference)
    return ns
end

function report_assembly(label, pressure)
    @printf(
        "  %-20s inst=%4d stack-mem=%3d calls=%2d ymm-count=%2d max-ymm=%2d\n",
        label,
        pressure.instructions,
        pressure.stack_mem,
        pressure.calls,
        pressure.ymm_count,
        pressure.max_ymm,
    )
end

function profile_type(::Type{T}; n=2048, repeats=16, samples=5, outdir="") where {T<:MultiFloat}
    V = vec4_type(T)
    a, b1, b2, b3, b4 = make_inputs(T, n)
    scalar_products = n * 4 * repeats

    println("== $T / $V ==")
    t1 = median_seconds(() -> dot1(a, b1, repeats), samples)
    ns1 = report("one accumulator", t1, scalar_products, t1 * 1e9 / scalar_products)

    t2i = median_seconds(() -> dot2_interleaved(a, b1, b2, repeats), samples)
    report("two interleaved", t2i, 2 * scalar_products, ns1)

    t2s = median_seconds(() -> dot2_split(a, b1, b2, repeats), samples)
    report("two split loops", t2s, 2 * scalar_products, ns1)

    t4 = median_seconds(() -> dot4_interleaved(a, b1, b2, b3, b4, repeats), samples)
    report("four interleaved", t4, 4 * scalar_products, ns1)

    probes = (
        ("dot1", dot1, (Vector{V}, Vector{T}, Int)),
        ("dot2_interleaved", dot2_interleaved, (Vector{V}, Vector{T}, Vector{T}, Int)),
        ("dot2_split", dot2_split, (Vector{V}, Vector{T}, Vector{T}, Int)),
        ("dot4_interleaved", dot4_interleaved, (Vector{V}, Vector{T}, Vector{T}, Vector{T}, Vector{T}, Int)),
    )
    println("assembly pressure:")
    limb_count = T.parameters[2]
    for (label, f, signature) in probes
        native = native_text(f, signature)
        pressure = assembly_pressure(native)
        report_assembly(label, pressure)
        write_native(outdir, "x$(limb_count)_" * label, native)
    end
    println()
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2048
    repeats = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 16
    samples = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5
    outdir = get(ENV, "MFLA_PRESSURE_OUT", "")
    println("MultiFloat GEMM accumulator pressure probe")
    println("Julia=$(VERSION) arch=$(Sys.ARCH) threads=$(Threads.nthreads()) n=$n repeats=$repeats samples=$samples")
    println()
    for T in TYPES
        profile_type(T; n=n, repeats=repeats, samples=samples, outdir=outdir)
    end
end

main()
