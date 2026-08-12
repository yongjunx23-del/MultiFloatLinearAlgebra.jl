using InteractiveUtils
using MultiFloats
using MultiFloatLinearAlgebra
using Printf

const TYPES = (Float64x2, Float64x3, Float64x4)
const SINK = Ref{Any}(nothing)

@inline lanes(::Type{<:MultiFloat}) = 1
@inline lanes(::Type{<:MultiFloatVec{M}}) where {M} = M

function median_seconds(f, samples::Int)
    count = max(samples, 1)
    f()
    elapsed = Vector{Float64}(undef, count)
    for sample in 1:count
        GC.gc()
        start = time_ns()
        value = f()
        elapsed[sample] = (time_ns() - start) / 1.0e9
        SINK[] = value
    end
    sort!(elapsed)
    return elapsed[cld(count, 2)]
end

function make_scalar_data(::Type{T}, n::Int, phase::Float64) where {T<:MultiFloat}
    data = Vector{T}(undef, n)
    @inbounds for i in 1:n
        value = 0.75 + 0.1 * sin(phase + 0.017 * i) + 0.05 * cos(0.013 * i)
        data[i] = T(value)
    end
    return data
end

function dirty_scalar(::Type{MultiFloat{Float64,N}}, i::Int, phase::Float64) where {N}
    head = 1.0 + 1.0e-3 * sin(phase + 0.11 * i)
    limbs = ntuple(Val(N)) do limb
        if limb == 1
            head
        else
            # Deliberately overlap adjacent limbs so renormalize must do work.
            ldexp(0.5 + 0.1 * cos(phase + 0.07 * i + limb), -40 * (limb - 1))
        end
    end
    return MultiFloat{Float64,N}(limbs)
end

function make_dirty_data(::Type{T}, n::Int, phase::Float64) where {T<:MultiFloat}
    data = Vector{T}(undef, n)
    @inbounds for i in 1:n
        data[i] = dirty_scalar(T, i, phase)
    end
    return data
end

function vec4_type(::Type{MultiFloat{Float64,N}}) where {N}
    return MultiFloatVec{4,Float64,N}
end

function pack_vec4(::Type{V}, data::Vector{T}) where {T<:MultiFloat,V<:MultiFloatVec{4}}
    length(data) % 4 == 0 || throw(ArgumentError("scalar length must be divisible by four"))
    result = Vector{V}(undef, length(data) ÷ 4)
    @inbounds for block in eachindex(result)
        first = 4 * (block - 1) + 1
        result[block] = V(data[first], data[first + 1], data[first + 2], data[first + 3])
    end
    return result
end

@noinline function add_kernel!(out, a, b, repeats::Int)
    @inbounds for _ in 1:repeats
        for i in eachindex(out, a, b)
            out[i] = a[i] + b[i]
        end
    end
    return out[1]
end

@noinline function mul_kernel!(out, a, b, repeats::Int)
    @inbounds for _ in 1:repeats
        for i in eachindex(out, a, b)
            out[i] = a[i] * b[i]
        end
    end
    return out[1]
end

@noinline function muladd_kernel!(out, a, b, c, repeats::Int)
    @inbounds for _ in 1:repeats
        for i in eachindex(out, a, b, c)
            out[i] = a[i] * b[i] + c[i]
        end
    end
    return out[1]
end

@noinline function dot_kernel(a, b, repeats::Int)
    total = zero(eltype(a))
    @inbounds for _ in 1:repeats
        acc = zero(eltype(a))
        for i in eachindex(a, b)
            acc += a[i] * b[i]
        end
        total += acc
    end
    return total
end

@noinline function renorm_kernel!(out, a, repeats::Int)
    @inbounds for _ in 1:repeats
        for i in eachindex(out, a)
            out[i] = MultiFloats.renormalize(a[i])
        end
    end
    return out[1]
end

@noinline op_add(a, b) = a + b
@noinline op_mul(a, b) = a * b
@noinline op_muladd(a, b, c) = a * b + c
@noinline op_renorm(a) = MultiFloats.renormalize(a)

function asm_text(f, signature)
    return sprint() do io
        code_native(io, f, signature; debuginfo=:none)
    end
end

function llvm_text(f, signature)
    return sprint() do io
        code_llvm(io, f, signature; debuginfo=:none, optimize=true)
    end
end

function count_pattern(text::AbstractString, regex::Regex)
    return count(line -> occursin(regex, lowercase(line)), eachline(IOBuffer(text)))
end

function code_stats(native::String, llvm::String)
    native_lines = collect(eachline(IOBuffer(native)))
    instruction_lines = count(native_lines) do line
        s = strip(line)
        !isempty(s) &&
        !startswith(s, ".") &&
        !endswith(s, ":") &&
        !startswith(s, ";") &&
        !startswith(s, "#") &&
        occursin(r"^[a-zA-Z]", s)
    end
    return (
        instructions=instruction_lines,
        calls=count_pattern(native, r"\bcall"),
        branches=count_pattern(native, r"\b(j[a-z]+|br)\b"),
        fma=count_pattern(native, r"(v?fm(add|sub)|v?fnm(add|sub))"),
        mul=count_pattern(native, r"(v?mul(sd|ss|pd|ps)?|fmul)"),
        add=count_pattern(native, r"(v?add(sd|ss|pd|ps)?|fadd)"),
        stack_refs=count_pattern(native, r"(rsp|rbp|sp)"),
        vector_regs=count_pattern(native, r"(xmm|ymm|zmm|v[0-9]+\.)"),
        llvm_calls=count_pattern(llvm, r"\bcall\b"),
        llvm_fmul=count_pattern(llvm, r"\bfmul\b"),
        llvm_fadd=count_pattern(llvm, r"\bfadd\b"),
    )
end

function maybe_write_codegen(outdir::String, label::String, native::String, llvm::String)
    isempty(outdir) && return
    mkpath(outdir)
    open(joinpath(outdir, label * ".native.txt"), "w") do io
        write(io, native)
    end
    open(joinpath(outdir, label * ".llvm.txt"), "w") do io
        write(io, llvm)
    end
end

function report_runtime(label, seconds, logical_terms, reference=nothing)
    ns = seconds * 1.0e9 / logical_terms
    @printf("%-28s %10.3f ns/scalar-term", label, ns)
    reference === nothing || @printf("  ratio=%6.2fx", ns / reference)
    println()
    return ns
end

function report_codegen(label, stats)
    @printf(
        "%-28s inst=%4d call=%3d br=%3d fma=%3d mul=%3d add=%3d stack=%4d vreg=%4d llvm(call/fmul/fadd)=%d/%d/%d\n",
        label,
        stats.instructions,
        stats.calls,
        stats.branches,
        stats.fma,
        stats.mul,
        stats.add,
        stats.stack_refs,
        stats.vector_regs,
        stats.llvm_calls,
        stats.llvm_fmul,
        stats.llvm_fadd,
    )
end

function profile_type(::Type{T}; scalar_terms=4096, repeats=16, samples=5, outdir="") where {T<:MultiFloat}
    V = vec4_type(T)
    a = make_scalar_data(T, scalar_terms, 0.1)
    b = make_scalar_data(T, scalar_terms, 0.7)
    c = make_scalar_data(T, scalar_terms, 1.3)
    dirty = make_dirty_data(T, scalar_terms, 0.4)
    clean = copy(a)

    av = pack_vec4(V, a)
    bv = pack_vec4(V, b)
    cv = pack_vec4(V, c)
    dirtyv = pack_vec4(V, dirty)
    cleanv = pack_vec4(V, clean)

    out = similar(a)
    outv = similar(av)
    logical_terms = scalar_terms * repeats

    println("== $T / $V ==")
    scalar_add = report_runtime(
        "scalar add",
        median_seconds(() -> add_kernel!(out, a, b, repeats), samples),
        logical_terms,
    )
    scalar_mul = report_runtime(
        "scalar mul",
        median_seconds(() -> mul_kernel!(out, a, b, repeats), samples),
        logical_terms,
    )
    scalar_muladd = report_runtime(
        "scalar mul+add",
        median_seconds(() -> muladd_kernel!(out, a, b, c, repeats), samples),
        logical_terms,
    )
    scalar_dot = report_runtime(
        "scalar dot accumulate",
        median_seconds(() -> dot_kernel(a, b, repeats), samples),
        logical_terms,
    )
    scalar_renorm_clean = report_runtime(
        "scalar renorm clean",
        median_seconds(() -> renorm_kernel!(out, clean, repeats), samples),
        logical_terms,
    )
    scalar_renorm_dirty = report_runtime(
        "scalar renorm dirty",
        median_seconds(() -> renorm_kernel!(out, dirty, repeats), samples),
        logical_terms,
    )

    report_runtime(
        "vec4 add",
        median_seconds(() -> add_kernel!(outv, av, bv, repeats), samples),
        logical_terms,
        scalar_add,
    )
    report_runtime(
        "vec4 mul",
        median_seconds(() -> mul_kernel!(outv, av, bv, repeats), samples),
        logical_terms,
        scalar_mul,
    )
    report_runtime(
        "vec4 mul+add",
        median_seconds(() -> muladd_kernel!(outv, av, bv, cv, repeats), samples),
        logical_terms,
        scalar_muladd,
    )
    report_runtime(
        "vec4 dot accumulate",
        median_seconds(() -> dot_kernel(av, bv, repeats), samples),
        logical_terms,
        scalar_dot,
    )
    report_runtime(
        "vec4 renorm clean",
        median_seconds(() -> renorm_kernel!(outv, cleanv, repeats), samples),
        logical_terms,
        scalar_renorm_clean,
    )
    report_runtime(
        "vec4 renorm dirty",
        median_seconds(() -> renorm_kernel!(outv, dirtyv, repeats), samples),
        logical_terms,
        scalar_renorm_dirty,
    )

    println("codegen:")
    probes = (
        ("scalar_add", op_add, (T, T)),
        ("scalar_mul", op_mul, (T, T)),
        ("scalar_muladd", op_muladd, (T, T, T)),
        ("scalar_renorm", op_renorm, (T,)),
        ("vec4_add", op_add, (V, V)),
        ("vec4_mul", op_mul, (V, V)),
        ("vec4_muladd", op_muladd, (V, V, V)),
        ("vec4_renorm", op_renorm, (V,)),
    )
    for (name, f, signature) in probes
        native = asm_text(f, signature)
        llvm = llvm_text(f, signature)
        stats = code_stats(native, llvm)
        label = string(T) * "_" * name
        report_codegen(name, stats)
        maybe_write_codegen(outdir, replace(label, r"[^A-Za-z0-9_.-]" => "_"), native, llvm)
    end
    println()
end

function main()
    scalar_terms = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4096
    repeats = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 16
    samples = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5
    scalar_terms % 4 == 0 || error("scalar_terms must be divisible by four")
    outdir = get(ENV, "MFLA_PROFILE_OUT", "")

    println("Raw MultiFloat arithmetic profile")
    println("Julia=$(VERSION) arch=$(Sys.ARCH) threads=$(Threads.nthreads())")
    println("scalar_terms=$scalar_terms repeats=$repeats samples=$samples")
    println("MultiFloats=$(Base.pkgversion(MultiFloats))")
    println()

    for T in TYPES
        profile_type(
            T;
            scalar_terms=scalar_terms,
            repeats=repeats,
            samples=samples,
            outdir=outdir,
        )
    end
end

main()
