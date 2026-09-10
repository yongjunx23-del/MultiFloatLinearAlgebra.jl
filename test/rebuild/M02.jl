# =============================================================================
# M02 — MFLA shape-oriented packing and SIMD microkernels: executable evidence
# =============================================================================
#
# Run:
#   julia --project="$REBUILD_ENV" -t1 MultiFloatLinearAlgebra.jl/test/rebuild/M02.jl
#
# This file is the executable evidence for the M02 candidate in
#   src/planning/gemm_plan.jl
#   src/kernels/packing.jl
#   src/kernels/gemm_microkernels.jl
#   src/kernels/gemm_schedule.jl
#
# DUAL MODE, AND WHY
#
# Those four files are NOT wired into `src/MultiFloatLinearAlgebra.jl` — wiring
# is the integration role's authority, not this task's. So the driver has two
# modes and runs the SAME assertions in both:
#
#   sandbox (current) — the four files are `include`d into a private module that
#                       imports the package's name surface, exactly as M01 did
#                       for the contract layer before I01 wired it. This proves
#                       the files are self-contained and change no existing
#                       MFLA definition.
#   wired   (later)   — if the package already defines `shape_packing_plan`, the
#                       package's OWN definitions are used and the sandbox is
#                       never created. The driver then also asserts that the
#                       names resolve inside `MultiFloatLinearAlgebra`, so the
#                       wired run cannot silently exercise a private copy.
#
# The bindings are made once, through `M02API`, so there is exactly one
# assertion path and the two modes cannot drift into asserting different things
# (the S06 defect the batch-5 kit records: an integrated branch silently skipped
# two assertions).
#
# THE ORACLE IS EXACT AND INDEPENDENT
#
# Every arithmetic claim below is checked against `Rational{BigInt}` evaluation
# of the SAME Float64 inputs the fixture was built from. The oracle
#   * never uses MultiFloat arithmetic, so it cannot share a defect with the
#     thing under test;
#   * never uses MPFR/BigFloat, so this process stays MF-only (the two-process
#     rule) and no BFLA code is loaded;
#   * reports `floor-bound on log2(relative error)`, an integer, so an x4 error
#     of 2^-200 is representable even though Float64 cannot hold it;
#   * is shown to be non-vacuous: the same oracle applied to an
#     Float64-truncated result must report a much worse exponent.
#
# A driver that only ever prints "test passed" is not evidence. Every claim this
# file makes is either a `@test` with a named control or a `MEASURE` line whose
# value is printed raw.

using Test
using Random
using LinearAlgebra
using SparseArrays
using MultiFloats
using MultiFloatLinearAlgebra

import MultiFloatLinearAlgebra
import MultiFloats: MultiFloat, MultiFloatVec

# ---------------------------------------------------------------------------
# Optional sparse provider (QDLDL), reached the same way M01 reached it: MFLA's
# `MultiFloatQDLDLExt` is a package extension, so QDLDL must be loaded first.
# Absence is INFRASTRUCTURE, not a numeric failure, and the sparse leg below
# records `unsupported` with the reason instead of failing the suite.
# ---------------------------------------------------------------------------
const QDLDL_UUID = Base.UUID("bfc457fd-c171-5ab7-bd9e-d5dbfc242d63")
const QDLDL_MODULE = try
    Base.require(Base.PkgId(QDLDL_UUID, "QDLDL"))
catch
    nothing
end
const QDLDL_PRESENT = QDLDL_MODULE !== nothing

const INTERACTIVE_UTILS_PRESENT = try
    @eval using InteractiveUtils
    true
catch
    false
end

const M02_REPO = normpath(joinpath(@__DIR__, "..", ".."))

# ---------------------------------------------------------------------------
# The name surface. One binding table, two modes.
# ---------------------------------------------------------------------------
const M02_API_NAMES = (
    :GEMM_SHAPE_CLASSES, :PACKING_LAYOUTS, :KERNEL_ORDER_MODES,
    :MICROKERNEL_MAX_LANES, :MICROKERNEL_MAX_COLUMNS,
    :GemmShapeClass, :ShapePackingPlan, :PanelPackLayout, :PackedPanel,
    :PackedGemmWorkspace, :GemmTimingBreakdown,
    :classify_gemm_shape, :shape_packing_plan, :supported_packing_variants,
    :shape_calibration_table, :format_shape_calibration, :select_certified_variant,
    :panel_pack_layout, :panel_buffer_size, :allocate_panel, :packing_elements,
    :pack_a_panel!, :pack_b_panel!, :unpack_panel!, :panel_element, :panel_limb,
    :microkernel_block!, :microkernel_tail!, :microkernel_dispatch!,
    :microkernel_variant_tag,
    :packed_gemm_workspace, :ensure_packed_gemm_capacity!,
    :packed_gemm_scheduled!, :packed_gemm_reference!, :total_seconds,
)

const M02_WIRED = isdefined(MultiFloatLinearAlgebra, :shape_packing_plan)

if !M02_WIRED
    @eval module M02Sandbox
        using MultiFloats
        using MultiFloatLinearAlgebra
        using LinearAlgebra
        import MultiFloatLinearAlgebra: _check_supported, _default_gemm_panel_columns
        import MultiFloats: MultiFloat, MultiFloatVec
        const M02_SRC = normpath(joinpath(@__DIR__, "..", "..", "src"))
        include(joinpath(M02_SRC, "planning", "gemm_plan.jl"))
        include(joinpath(M02_SRC, "kernels", "packing.jl"))
        include(joinpath(M02_SRC, "kernels", "gemm_microkernels.jl"))
        include(joinpath(M02_SRC, "kernels", "gemm_schedule.jl"))
    end
end

const M02API = M02_WIRED ? MultiFloatLinearAlgebra : M02Sandbox

for name in M02_API_NAMES
    @eval const $(name) = $(M02API).$(name)
end

const MEASUREMENTS = String[]
function measure(label, value)
    line = "MEASURE $(label) = $(value)"
    push!(MEASUREMENTS, line)
    println(line)
    return value
end

# ---------------------------------------------------------------------------
# Fixtures. The Float64 originals are kept, because the oracle evaluates THEM.
# ---------------------------------------------------------------------------
function fixture_random(rng, m, k, n)
    return randn(rng, m, k), randn(rng, k, n), randn(rng, m, n)
end

"""
Strong-cancellation fixture: the odd reduction indices contribute `+1e8` and
the even ones `-1e8` (each with an independent ~1e-9 relative perturbation), and
every B entry is `1e-8` with its own perturbation. The products are therefore
order 1 while their sum is order 1e-9 — about nine digits cancel — and the sum
is not identically zero, so the oracle is judging a real value rather than a
special case.

The first draft of this fixture used `(isodd(r + i), isodd(r + j))` signs, whose
product is `(-1)^(i+j)` — independent of `r`, so nothing cancelled at all and the
"cancellation" testset was silently an easy case. That is recorded here because
the failure was in the instrument, not in the code under test.
"""
function fixture_cancellation(rng, m, k, n)
    iseven(k) || throw(ArgumentError("the cancellation fixture needs even k"))
    A = Matrix{Float64}(undef, m, k)
    B = Matrix{Float64}(undef, k, n)
    for r in 1:k, i in 1:m
        A[i, r] = (isodd(r) ? 1.0 : -1.0) * 1.0e8 * (1.0 + 1.0e-9 * randn(rng))
    end
    for j in 1:n, r in 1:k
        B[r, j] = 1.0e-8 * (1.0 + 1.0e-9 * randn(rng))
    end
    return A, B, randn(rng, m, n)
end

"""
    cancellation_depth(A64, B64) -> Float64

`max |exact output| / max sum of |terms|`: how much the fixture actually
cancels. A measured property of the instrument, so "strong cancellation" is not
taken on trust.
"""
function cancellation_depth(A64, B64)
    m, k = size(A64)
    n = size(B64, 2)
    worst = 0.0
    for j in 1:n, i in 1:m
        expected = sum(exact_value(A64[i, r]) * exact_value(B64[r, j]) for r in 1:k)
        scale = sum(abs(exact_value(A64[i, r]) * exact_value(B64[r, j])) for r in 1:k)
        ratio = Float64(abs(expected) / scale)
        ratio > worst && (worst = ratio)
    end
    return worst
end

"""
Exponent-extreme fixture: operands near the Float64 range edges whose products
are order 1. A kernel that pre-multiplies or pre-scales in the wrong order
overflows or flushes to zero here, and the exact oracle reports it.
"""
function fixture_extremes(rng, m, k, n)
    A = Matrix{Float64}(undef, m, k)
    B = Matrix{Float64}(undef, k, n)
    for r in 1:k, i in 1:m
        A[i, r] = (isodd(r) ? 1.0e300 : 1.0e-300) * (1.0 + 0.5 * randn(rng))
    end
    for j in 1:n, r in 1:k
        B[r, j] = (isodd(r) ? 1.0e-300 : 1.0e300) * (1.0 + 0.5 * randn(rng))
    end
    return A, B, randn(rng, m, n)
end

function fixture_subnormal(rng, m, k, n)
    # Two subnormal magnitudes, assigned without any further arithmetic: the
    # first draft multiplied by a random factor near 1, which rounds 5.0e-324
    # down to 0.0 and turned the subnormal fixture into a zero fixture.
    A = Matrix{Float64}(undef, m, k)
    for r in 1:k, i in 1:m
        A[i, r] = isodd(i + r) ? 5.0e-324 : 1.0e-320
    end
    return A, fill(1.0e300, k, n), randn(rng, m, n)
end

# ---------------------------------------------------------------------------
# The exact oracle.
# ---------------------------------------------------------------------------
exact_value(x::Float64) = Rational{BigInt}(x)
exact_value(x::MultiFloat{Float64,N}) where {N} =
    sum(Rational{BigInt}(limb) for limb in x._limbs)

"""
    log2_upper(r) -> Int

An integer upper bound on `log2(r)` for a positive rational: within 1 of the
true value. Integer, because an x4 relative error of 2^-200 has no Float64
representation and must not be rounded to 0.0 (that would turn "very accurate"
into "exactly zero", i.e. an unmeasured value masquerading as a measurement).
"""
function log2_upper(r::Rational{BigInt})
    r > 0 || return typemin(Int)
    return ndigits(numerator(r), base=2) - ndigits(denominator(r), base=2) + 1
end

"""
    gemm_error_profile(C, A64, B64, alpha64, beta64, C064)

Exact forward-error profile of `C` against `alpha*A64*B64 + beta*C064`, all
evaluated in `Rational{BigInt}`. Two exponents are reported, because one is not
enough:

  * `worst` (normwise): `|actual - expected| / max(|expected|, sum|terms|)`. The
    mixed absolute/relative denominator keeps it well defined when the exact
    output is 0.
  * `worst_relative` (componentwise): `|actual - expected| / |expected|` over the
    elements whose exact value is non-zero, together with `condition`, the
    largest `sum|terms| / |expected|` in the block.

The componentwise form is the one that can see a precision loss under
cancellation. A purely normwise bound cannot: when the answer is 1e-9 and the
terms are order 1, an Float64-precision result has a normwise error near
2^-53/16 — small — while its error relative to the answer it claims to compute is
2^-53 x 1e9. The first draft of this oracle had only the normwise form and its
cancellation control passed for that reason.

Returns `(worst, at, worst_relative, at_relative, condition, exact, total)`.
"""
function gemm_error_profile(C, A64, B64, alpha64, beta64, C064)
    m, k = size(A64)
    n = size(B64, 2)
    exact_alpha = exact_value(alpha64)
    exact_beta = beta64 === nothing ? zero(Rational{BigInt}) : exact_value(beta64)
    worst = typemin(Int)
    at = (0, 0)
    worst_relative = typemin(Int)
    at_relative = (0, 0)
    condition = 1.0
    exact_count = 0
    for j in 1:n, i in 1:m
        expected = zero(Rational{BigInt})
        scale = zero(Rational{BigInt})
        for r in 1:k
            term = exact_value(A64[i, r]) * exact_value(B64[r, j])
            expected += term
            scale += abs(term)
        end
        expected *= exact_alpha
        if C064 !== nothing
            carry = exact_beta * exact_value(C064[i, j])
            expected += carry
            scale += abs(carry)
        end
        actual = exact_value(C[i, j])
        # `condition` is measured for EVERY element, before the exactness
        # shortcut. The first draft updated it only for elements that were not
        # exactly right, so a run in which the candidate was exact everywhere
        # reported the initializer 1.0 as though it had been measured — an
        # unmeasured value wearing a measurement's clothes.
        if expected != 0
            ratio = scale / abs(expected)
            local_condition = Float64(min(ratio, Rational{BigInt}(10)^30))
            local_condition > condition && (condition = local_condition)
        end
        if actual == expected
            exact_count += 1
            continue
        end
        denominator = max(abs(expected), scale)
        exponent = log2_upper(abs(actual - expected) / denominator)
        if exponent > worst
            worst = exponent
            at = (i, j)
        end
        if expected != 0
            relative = log2_upper(abs(actual - expected) / abs(expected))
            if relative > worst_relative
                worst_relative = relative
                at_relative = (i, j)
            end
        end
    end
    return (
        worst=worst,
        at=at,
        worst_relative=worst_relative,
        at_relative=at_relative,
        condition=condition,
        exact=exact_count,
        total=m * n,
    )
end

"""
    error_budget(limbs, k; condition=1.0) -> Int

The exponent the oracle demands: `-52*limbs + ceil(log2(k)) + slack +
ceil(log2(condition))`. `-52*limbs` is `log2(eps(MultiFloat{Float64,limbs}))`,
which the driver measures separately rather than assuming; the `log2(k)` term is
the summation growth of a length-`k` dot product; `slack` covers the alpha
multiplication, the beta*C term and the k-block partial sums; and the condition
term is the *measured* `sum|terms|/|expected|` of the fixture, so a cancelling
fixture is allowed exactly the precision its conditioning costs and no more. It
is an integer bound on the error, not a tuned tolerance: nothing here is relaxed
to make a case pass.
"""
function error_budget(limbs::Int, k::Int; condition::Float64=1.0)
    condition_term = condition > 1.0 ? ceil(Int, log2(condition)) : 0
    return -52 * limbs + ceil(Int, log2(max(k, 1))) + 8 + condition_term
end

"""
    float64_truncated(C, MF)

`C` rounded to Float64 and back — an informational line, NOT the precision
control. Rounding an already-accurate result introduces an error of order
`2^-53` relative to the value itself, whatever the fixture's conditioning, so it
says nothing about whether the oracle can see a precision loss under
cancellation.
"""
function float64_truncated(C::AbstractMatrix{MF}, ::Type{MF}) where {MF}
    return MF.(Float64.(C))
end

"""
    float64_precision_result(A64, B64, alpha64, beta64, C064, MF)

The same operation *carried out in Float64 working precision* — the arithmetic a
single-limb accumulator gives — with the same ascending accumulation order. This
is the oracle's non-vacuity control: on a fixture whose condition number is
`2^c`, a 53-bit accumulation must land about `2^(c-53)` away in relative terms,
so a budget of `2^(-52N+c+...)` has to reject it whenever the fixture actually
cancels. Truncating an accurate x2 result cannot show this; recomputing at x1
precision can.
"""
function float64_precision_result(A64, B64, alpha64, beta64, C064, ::Type{MF}) where {MF}
    m, k = size(A64)
    n = size(B64, 2)
    C = Matrix{Float64}(undef, m, n)
    for j in 1:n, i in 1:m
        accumulator = 0.0
        for r in 1:k
            accumulator += A64[i, r] * B64[r, j]
        end
        value = alpha64 * accumulator
        if beta64 != 0.0 && C064 !== nothing
            value += beta64 * C064[i, j]
        end
        C[i, j] = value
    end
    return MF.(C)
end

function bitsame(X::AbstractMatrix, Y::AbstractMatrix)
    size(X) == size(Y) || return false
    return all(X[i, j]._limbs === Y[i, j]._limbs for i in axes(X, 1), j in axes(X, 2))
end

"""
    render_exponent(value) -> String

`typemin(Int)` is what the error profile reports when a result was correct to
the last bit in every element. Printing it as `-9223372036854775808` reads like a
broken number, so it is rendered as the fact it is — and a sentinel that means
"exactly right" must never be confused with a large measured magnitude.
"""
render_exponent(value::Int) =
    value == typemin(Int) ? "exact_all_elements" : string(value)

function measure_exponent(label, value::Int)
    return measure(label, render_exponent(value))
end

function max_abs_difference(X::AbstractMatrix, Y::AbstractMatrix)
    return maximum(abs(Float64(X[i, j] - Y[i, j])) for i in axes(X, 1), j in axes(X, 2))
end

# ---------------------------------------------------------------------------
# Subprocess probe: the raw QDLDL matrix entry point.
#
# WHY THIS IS OUT OF PROCESS. A01b-F2 records that `QDLDL.solve(factor, ::Matrix)`
# dies with `ReadOnlyMemoryError`. Measured here, that description is too benign:
# the call performs an out-of-bounds write, and its failure mode is not
# deterministic. It USUALLY raises a catchable `ReadOnlyMemoryError` — 10
# catchable observations across two independent parties (4/4 the parent's
# isolated control, 6/6 this driver's child processes: 2 in-driver plus 4 in the
# churn sweep, all exit 0) — and it has ONCE killed a process: the parent's
# in-process run of an earlier revision of this driver, executing concurrently
# with another worker, died exit 139 / signal 11 inside `ipermute!` at
# QDLDL/src/QDLDL.jl:619 with no Test Summary. That is n=1 in the fatal
# condition.
#
# THE MECHANISM IS NOT ESTABLISHED, AND MUST NOT BE INVENTED HERE. An earlier
# version of this comment asserted that the fatal mode "depends on the heap
# history". The experiment that tested it — child-side allocation churn at 0,
# 1e6, 1e7 and 2e7 iterations, the last being 23 s of deliberate allocation —
# came back NEGATIVE 4/4 (see rebuild-reports/M02/M02_qdldl_probe_sweep.log), so
# that mechanism is WITHDRAWN. Concurrent memory pressure and address-space
# state, allocator state after a long mixed workload, and address-space layout
# are UNTESTED candidates, not findings. The call is memory-unsafe with an
# unreliably-triggered fatal mode, NOT a deterministic exception.
#
# A SIGSEGV is not catchable, which is why this cannot be probed in-process: the
# first version of this driver wrapped the call in `try/catch`, and an
# independent re-run died at that line with exit 139 and no Test Summary. The
# observation was real; the instrument was the defect.
#
# So the call runs in a child process that does nothing else, and the child's
# exit code, termination signal and output are the measurement. A child killed by
# a signal is then an observed result rather than a killed run, which is strictly
# more informative than the caught exception it replaces.
# ---------------------------------------------------------------------------

const QDLDL_PROBE_SCRIPT = raw"""
const QDLDL_UUID = Base.UUID("bfc457fd-c171-5ab7-bd9e-d5dbfc242d63")
const CHURN = isempty(ARGS) ? 0 : parse(Int, ARGS[1])
const ORDER = isempty(ARGS) ? 24 : (length(ARGS) > 1 ? parse(Int, ARGS[2]) : 24)
const QDLDL = try
    Base.require(Base.PkgId(QDLDL_UUID, "QDLDL"))
catch error
    println("PROBE_SETUP_FAILED ", sprint(showerror, error))
    flush(stdout)
    exit(3)
end
using LinearAlgebra, SparseArrays, MultiFloats, MultiFloatLinearAlgebra
MF = MultiFloat{Float64,2}
n = ORDER
tridiagonal = spdiagm(0 => 4.0 * ones(n), 1 => -1.0 * ones(n - 1))
pattern = SparseMatrixCSC{MF,Int}(
    n, n, copy(tridiagonal.colptr), copy(tridiagonal.rowval),
    MF.(nonzeros(tridiagonal)),
)
cache = sparse_ldlt_cache(MF, pattern; dsigns=ones(Int, n), nrhs=7)
factorize!(cache, pattern)
println("PROBE_FACTORIZED issuccess=", MultiFloatLinearAlgebra.issuccess(cache))
flush(stdout)
if CHURN > 0
    # Bounded heap churn before the call, to give the undefined behaviour a
    # chance to show its other failure mode. The churn level is recorded, so a
    # negative result is bounded rather than vague.
    started = time_ns()
    for index in 1:CHURN
        buffer = Matrix{MF}(undef, 8, 8)
        @inbounds buffer[1, 1] = MF(index)
    end
    println("PROBE_CHURN iterations=", CHURN, " seconds=", (time_ns() - started) / 1e9)
    flush(stdout)
end
println("PROBE_CALLING_RAW_MATRIX_SOLVE")
flush(stdout)
destination = zeros(MF, n, 2)
outcome = try
    QDLDL.solve(something(cache.factor), destination)
    "completed"
catch error
    string(typeof(error), ": ", sprint(showerror, error))
end
println("PROBE_OUTCOME ", outcome)
flush(stdout)
"""

"""
    run_qdldl_raw_matrix_probe(; churn::Int=0, order::Int=24) -> NamedTuple

Run the raw `QDLDL.solve(factor, ::Matrix)` call in a subprocess and report what
the subprocess did: `exit_code`, `term_signal` (non-zero means the child was
killed by a signal, which `run` cannot raise as an exception), the child's raw
output, and `outcome` — the `PROBE_OUTCOME` line if the child survived long
enough to print one, otherwise `"no outcome line: process died at or before the
call"`.

`churn` allocates `churn` small matrices in the child before the call, so that
how much allocation the child has done is an explicit, recorded parameter instead
of an accident of how much work the driver happened to do first. It is a probe of
one candidate condition, NOT an established trigger: at 0, 1e6, 1e7 and 2e7
iterations the outcome did not change (4/4 catchable), so churn alone does not
produce the fatal mode.
"""
function run_qdldl_raw_matrix_probe(; churn::Int=0, order::Int=24)
    directory = mktempdir()
    script = joinpath(directory, "qdldl_raw_matrix_probe.jl")
    log = joinpath(directory, "qdldl_raw_matrix_probe.log")
    write(script, QDLDL_PROBE_SCRIPT)
    active = Base.active_project()
    project_flag = active === nothing ? `` : `--project=$(active)`
    command = `$(Base.julia_cmd()) --startup-file=no -t1 $project_flag $script $churn $order`
    process = run(pipeline(ignorestatus(command), stdout=log, stderr=log); wait=true)
    output = isfile(log) ? read(log, String) : ""
    outcome = "no outcome line: process died at or before the call"
    for line in split(output, '\n')
        if startswith(line, "PROBE_OUTCOME ")
            outcome = String(line[length("PROBE_OUTCOME ")+1:end])
            break
        end
    end
    if startswith(output, "PROBE_SETUP_FAILED") || occursin("PROBE_SETUP_FAILED", output)
        outcome = "setup failed: " * output
    end
    return (
        churn=churn,
        order=order,
        exit_code=process.exitcode,
        term_signal=process.termsignal,
        died_on_signal=process.termsignal != 0,
        outcome=outcome,
        output=output,
    )
end

# ---------------------------------------------------------------------------
# Contention instrumentation. The host has 4 cores and other workers run on it,
# so a wall-clock number without a load record is not a measurement of the code.
# ---------------------------------------------------------------------------
const LOADAVG_SUPPORTED = try
    Sys.loadavg()
    true
catch
    false
end

load_snapshot() = LOADAVG_SUPPORTED ? Sys.loadavg() : (NaN, NaN, NaN)

"""
    idle_guard(; threshold=0.75, max_wait_s=15.0)

Wait, bounded, for the 1-minute load average to fall below `threshold`. Returns
`(quiet, waited_seconds, load)` where `quiet == false` means the host never got
quiet and every timing taken afterwards must be reported as contended.
"""
function idle_guard(; threshold::Float64=0.75, max_wait_s::Float64=15.0)
    start = time()
    load = load_snapshot()
    while load[1] > threshold && time() - start < max_wait_s
        sleep(0.5)
        load = load_snapshot()
    end
    return (quiet=load[1] <= threshold, waited=time() - start, load=load)
end

function measure_seconds(f; samples::Int=3, warmup::Int=1)
    for _ in 1:warmup
        f()
    end
    times = Float64[]
    for _ in 1:samples
        started = time_ns()
        f()
        push!(times, (time_ns() - started) / 1e9)
    end
    ordered = sort(times)
    return (
        minimum=ordered[1],
        median=ordered[cld(length(ordered), 2)],
        samples=times,
    )
end

host_identity() = (
    cpu_name=Sys.CPU_NAME,
    arch=string(Sys.ARCH),
    cpu_threads=Sys.CPU_THREADS,
    julia_threads=Threads.nthreads(),
    julia_version=string(VERSION),
    loadavg=load_snapshot(),
)

function git_revision(repo)
    return try
        strip(read(`git -C $(repo) rev-parse HEAD`, String))
    catch
        "unavailable"
    end
end

# ---------------------------------------------------------------------------
# A plan-shaped helper: build plan + workspace + prefilled C in one place so
# every testset exercises the same call path.
# ---------------------------------------------------------------------------
function run_candidate(
    ::Type{MF},
    A64,
    B64,
    alpha64,
    beta64,
    C064;
    kwargs...,
) where {MF<:MultiFloat}
    A = MF.(A64)
    B = MF.(B64)
    m, k = size(A)
    n = size(B, 2)
    C = C064 === nothing ? fill(MF(NaN), m, n) : MF.(C064)
    plan = shape_packing_plan(MF, m, k, n; kwargs...)
    workspace = packed_gemm_workspace(MF, plan, m, k, n)
    timings = GemmTimingBreakdown()
    packed_gemm_scheduled!(
        C, A, B, MF(alpha64), MF(beta64), plan, workspace; timings=timings,
    )
    return C, plan, timings, workspace, A, B
end

# =============================================================================
println("="^78)
println("M02 — MFLA shape-oriented packing and SIMD microkernels")
println("="^78)

@testset "M02 driver" begin
# ---------------------------------------------------------------------------
@testset "T1 identity: revision, environment, mode, limb epsilon" begin
    println("M02 MODE = ", M02_WIRED ? "wired (package definitions)" :
            "sandbox (files included into a private module)")
    measure("SDPX_repo", git_revision(normpath(joinpath(M02_REPO, "..", "SDPX.jl"))))
    measure("MFLA_repo", git_revision(M02_REPO))
    measure("MFLA_worktree_status",
        strip(read(`git -C $(M02_REPO) status --porcelain`, String)))
    host = host_identity()
    for (name, value) in pairs(host)
        measure("host_$(name)", value)
    end
    measure("QDLDL_present", QDLDL_PRESENT)
    measure("InteractiveUtils_present", INTERACTIVE_UTILS_PRESENT)

    # eps(MF) is measured, not assumed, and it is the basis of the oracle budget.
    for N in (2, 3, 4)
        MF = MultiFloat{Float64,N}
        measure("eps_log2_x$(N)", log2(Float64(eps(MF))))
    end
    @test log2(Float64(eps(MultiFloat{Float64,2}))) == -104.0
    @test log2(Float64(eps(MultiFloat{Float64,3}))) == -156.0
    @test log2(Float64(eps(MultiFloat{Float64,4}))) == -208.0

    # Wired mode must name the package's own definitions. In sandbox mode the
    # control is the opposite assertion: the package does NOT define them (so
    # the include is genuinely additive), which is also the load-reachability
    # fact the integration patch will change.
    if M02_WIRED
        @test M02API === MultiFloatLinearAlgebra
        @test isdefined(MultiFloatLinearAlgebra, :packed_gemm_scheduled!)
        @test MultiFloatLinearAlgebra.packed_gemm_scheduled! === packed_gemm_scheduled!
    else
        @test M02API !== MultiFloatLinearAlgebra
        @test !isdefined(MultiFloatLinearAlgebra, :packed_gemm_scheduled!)
        @test !isdefined(MultiFloatLinearAlgebra, :shape_packing_plan)
    end
end

# ---------------------------------------------------------------------------
@testset "T2 control: the candidate changes no existing default route" begin
    # A reachability claim needs a control. The claim is "these files enable
    # nothing"; the control is that `gemm_plan` — the production entry point —
    # returns exactly what an untouched MFLA returns, because the candidate
    # never touches it. The expected values below were read from an untouched
    # tree in the same process before the include (T2a) and are re-derived here
    # from the documented policy (T2b), so the test does not compare a value to
    # itself.
    for N in (2, 3, 4)
        MF = MultiFloat{Float64,N}
        plan = gemm_plan(MF, 64, 64, 64)
        measure("gemm_plan_64_default_x$(N)", plan)
    end
    MF = MultiFloat{Float64,2}
    # Policy controls, each forced through the production entry point's own
    # documented switch rather than through a threshold this driver guessed.
    # (`config` is positional on `gemm_plan`, which the first draft of this
    # driver got wrong; the MethodError was the driver's bug, not MFLA's.)
    @test gemm_plan(MF, 64, 64, 64).strategy === :direct
    production_reason = gemm_plan(MF, 64, 64, 64).reason
    measure("gemm_plan_64_reason", production_reason)
    @test production_reason in (:auto_below_crossover, :auto_reduction_too_small,
        :auto_outside_calibrated_shape)
    @test gemm_plan(MF, 256, 64, 8).reason === :auto_outside_calibrated_shape
    @test gemm_plan(MF, 8, 8, 8, KernelConfig(gemm_strategy=:direct)).strategy === :direct
    @test gemm_plan(MF, 8, 8, 8, KernelConfig(gemm_strategy=:packed)).strategy === :packed
    @test gemm_plan(MF, 8, 8, 8, KernelConfig(gemm_strategy=:direct)).reason === :forced_direct
    measure("gemm_plan_packed_64", gemm_plan(MF, 64, 64, 64,
        KernelConfig(gemm_strategy=:packed)).strategy)

    # The candidate's own planner is a different entry point and never the
    # default: `default_path` is false for every plan it can produce.
    candidate_plan = shape_packing_plan(MF, 64, 64, 64)
    @test candidate_plan.default_path === false
    @test candidate_plan.evidence === :policy_default
    forced_plan = shape_packing_plan(MF, 64, 64, 64; lanes=2, layout=:soa_limbs)
    @test forced_plan.default_path === false
    @test forced_plan.evidence === :caller_forced
    @test forced_plan.layout === :soa_limbs
    # ...and SoA is not reachable without naming it.
    @test shape_packing_plan(MF, 64, 64, 64).layout === :aos
    @test panel_pack_layout(MF, 4, 4).mode === :aos
    @test_throws ArgumentError panel_pack_layout(MF, 4, 4; mode=:soa)
end

# ---------------------------------------------------------------------------
@testset "T3 shape classification by trace" begin
    MF = MultiFloat{Float64,2}
    cases = (
        (64, 64, 64, :near_square),
        (128, 128, 96, :near_square),
        (256, 32, 4, :panel),
        (512, 512, 8, :panel),
        (512, 512, 1, :vector),
        (1000, 200, 8, :panel),
        (4096, 128, 128, :tall),
        (128, 128, 4096, :wide),
        (0, 8, 8, :degenerate),
        (8, 0, 8, :degenerate),
        (8, 8, -1, :degenerate),
    )
    for (m, k, n, expected) in cases
        classified = classify_gemm_shape(MF, m, k, n)
        measure("classify_$(m)x$(k)x$(n)", classified.shape)
        @test classified.shape === expected
        @test classified.limb_class === :x2
        @test classified.limbs == 2
    end
    # Every declared class is reachable except `:degenerate`'s no-route case,
    # which the planner must refuse.
    reached = unique(measure("classify_reached",
        sort(unique(string(classify_gemm_shape(MF, m, k, n).shape)
                    for (m, k, n, _) in cases))))
    @test Set(String.(reached)) ⊇ Set(["near_square", "panel", "vector", "tall", "wide", "degenerate"])
    @test_throws ArgumentError shape_packing_plan(MF, 0, 8, 8)

    # Control for the ratio threshold: the same shape flips class when the
    # ratio moves, so the classification is not a constant.
    @test classify_gemm_shape(MF, 32, 32, 8).shape === :panel
    @test classify_gemm_shape(MF, 400, 400, 100; panel_width=1, ratio=4).shape === :near_square
    @test classify_gemm_shape(MF, 4096, 512, 512; ratio=4).shape === :tall
    @test classify_gemm_shape(MF, 4096, 512, 512; ratio=64).shape === :near_square
    # The planner's tails are reported, and a zero tail is a measured zero.
    tailed = shape_packing_plan(MF, 17, 5, 7; lanes=4, micro_columns=4, k_block=2)
    @test (tailed.tail_rows, tailed.tail_reduction, tailed.tail_columns) == (1, 1, 3)
    exact_fit = shape_packing_plan(MF, 16, 8, 8; lanes=4, micro_columns=4, k_block=4)
    @test (exact_fit.tail_rows, exact_fit.tail_reduction, exact_fit.tail_columns) == (0, 0, 0)
    @test exact_fit.packing_elements == 8 * (16 + 8)
    @test tailed.microkernel_calls > 0
end

# ---------------------------------------------------------------------------
@testset "T4 packing is bit-exact, and the comparison can fail" begin
    rng = MersenneTwister(0x4d3032)
    for N in (2, 3, 4)
        MF = MultiFloat{Float64,N}
        for mode in PACKING_LAYOUTS
            for (A64, _, _) in (fixture_random(rng, 9, 5, 3), fixture_extremes(rng, 8, 4, 2))
                A = MF.(A64)
                m, k = size(A)
                layout = panel_pack_layout(MF, k, m; mode=mode)
                buffer = allocate_panel(MF, layout)
                pack_a_panel!(buffer, A, 1, 1, layout)
                out = Matrix{MF}(undef, k, m)
                unpack_panel!(out, buffer, layout)
                @test all(out[kk, i]._limbs === A[i, kk]._limbs
                          for kk in 1:k, i in 1:m)
                # Control: the same comparison must be able to report a
                # difference. Corrupting one value in the packed buffer must
                # make the check fail, otherwise "bit-exact" is unfalsifiable.
                # The corruption is applied to the packed storage, not through
                # the MultiFloat constructor: a renormalising constructor could
                # legitimately absorb a perturbed low limb and make this control
                # inconclusive for a reason that has nothing to do with packing.
                corrupt = allocate_panel(MF, layout)
                pack_a_panel!(corrupt, A, 1, 1, layout)
                if mode === :aos
                    corrupt[1, 1] = corrupt[1, 1] + MultiFloat{Float64,N}(0.5)
                else
                    corrupt[(N - 1) * k + 1, 1] += 0.5
                end
                corrupted = Matrix{MF}(undef, k, m)
                unpack_panel!(corrupted, corrupt, layout)
                @test !all(corrupted[kk, i]._limbs === A[i, kk]._limbs
                           for kk in 1:k, i in 1:m)
                measure("packing_roundtrip_bits_x$(N)_$(mode)",
                    "exact; corruption detected")
            end
        end
    end
end

# ---------------------------------------------------------------------------
@testset "T5 packing handles transposes, adjoints and strides" begin
    rng = MersenneTwister(0x7305)
    MF = MultiFloat{Float64,2}
    A64, B64, _ = fixture_random(rng, 7, 4, 5)
    A = MF.(A64)
    B = MF.(B64)
    transposed = Matrix(transpose(A))            # k × m, to be wrapped back
    for mode in PACKING_LAYOUTS
        layout = panel_pack_layout(MF, 4, 7; mode=mode)
        buffer = allocate_panel(MF, layout)
        # A as a Transpose wrapper: exercise generic indexing, not a fast path.
        pack_a_panel!(buffer, transpose(transposed), 1, 1, layout)
        for i in 1:7, kk in 1:4
            @test panel_element(buffer, layout, kk, i)._limbs === A[i, kk]._limbs
        end
        # Non-unit row stride and a column offset.
        strided = view(A, 1:2:7, 2:4)            # 4 × 3
        strided_layout = panel_pack_layout(MF, 3, 4; mode=mode)
        strided_buffer = allocate_panel(MF, strided_layout)
        pack_a_panel!(strided_buffer, strided, 1, 1, strided_layout)
        for i in 1:4, kk in 1:3
            @test panel_element(strided_buffer, strided_layout, kk, i)._limbs ===
                  strided[i, kk]._limbs
        end
        # B with an Adjoint wrapper: `Bt` is n × k so `adjoint(Bt)` is the
        # k × n operand. Rows walk with the reduction.
        Bt = Matrix(adjoint(B))
        b_layout = panel_pack_layout(MF, 4, 5; mode=mode)
        b_buffer = allocate_panel(MF, b_layout)
        pack_b_panel!(b_buffer, adjoint(Bt), 1, 1, b_layout)
        for j in 1:5, kk in 1:4
            @test panel_element(b_buffer, b_layout, kk, j)._limbs === B[kk, j]._limbs
        end
        # Control: a wrong offset must produce a wrong panel, so the checks
        # above are not passing on a constant.
        wrong = Matrix{MF}(undef, 4, 5)
        unpack_panel!(wrong, b_buffer, b_layout)
        @test wrong[1, 1]._limbs === B[1, 1]._limbs
        @test wrong[1, 1]._limbs !== B[1, 2]._limbs
        measure("packing_transpose_stride_$(mode)", "exact for A', A strided, B'")
    end
end

# ---------------------------------------------------------------------------
@testset "T6 exact oracle: beta=0, beta=1, general beta, alpha=0" begin
    rng = MersenneTwister(0x0606)
    for N in (2, 3, 4)
        MF = MultiFloat{Float64,N}
        for (A64, B64, C064) in (fixture_random(rng, 7, 5, 6),
                                 fixture_random(rng, 4, 4, 4))
            m, k = size(A64)
            n = size(B64, 2)
            budget = error_budget(N, k)
            # beta = 0: C is NOT read. Prefill with NaN so a kernel that reads C
            # contaminates the result visibly instead of silently agreeing.
            nan_c = fill(NaN, m, n)
            C0, plan0, timings0, _, _, _ = run_candidate(
                MF, A64, B64, 1.5, 0.0, nan_c;
                lanes=2, micro_columns=2, k_block=2,
            )
            @test !any(isnan, Float64.(C0))
            profile0 = gemm_error_profile(C0, A64, B64, 1.5, 0.0, nothing)
            measure("oracle_beta0_x$(N)_$(m)x$(k)x$(n)_worst_log2", profile0.worst)
            @test profile0.worst <= budget
            @test profile0.total == m * n

            # beta = 1
            C1, _, _, _, _, _ = run_candidate(MF, A64, B64, -0.75, 1.0, C064;
                lanes=4, micro_columns=1, k_block=3)
            profile1 = gemm_error_profile(C1, A64, B64, -0.75, 1.0, C064)
            measure("oracle_beta1_x$(N)_$(m)x$(k)x$(n)_worst_log2", profile1.worst)
            @test profile1.worst <= budget

            # general beta
            C2, _, _, _, _, _ = run_candidate(MF, A64, B64, 0.5, -2.25, C064;
                lanes=2, micro_columns=4, k_block=4)
            profile2 = gemm_error_profile(C2, A64, B64, 0.5, -2.25, C064)
            measure("oracle_betageneral_x$(N)_$(m)x$(k)x$(n)_worst_log2", profile2.worst)
            @test profile2.worst <= budget

            # The oracle's non-vacuity control on the same fixture: an
            # Float64-truncated result must be rejected by the same budget.
            truncated = float64_truncated(C2, MF)
            profile_t = gemm_error_profile(truncated, A64, B64, 0.5, -2.25, C064)
            measure("oracle_x1control_x$(N)_$(m)x$(k)x$(n)_worst_log2", profile_t.worst)
            @test profile_t.worst > budget
            @test profile_t.worst >= -53 + ceil(Int, log2(k)) - 8

            # alpha = 0: neither operand is read. Put NaN in both operands; an
            # implementation that computes 0*(A*B) would return NaN.
            nan_operands = fill(NaN, m, k), fill(NaN, k, n)
            C3, _, _, _, _, _ = run_candidate(MF, nan_operands[1], nan_operands[2],
                0.0, 2.0, C064; lanes=4, micro_columns=4, k_block=2)
            @test !any(isnan, Float64.(C3))
            profile3 = gemm_error_profile(C3, A64, B64, 0.0, 2.0, C064)
            @test profile3.worst <= budget
            measure("oracle_alpha0_x$(N)_$(m)x$(k)x$(n)_worst_log2", profile3.worst)

            # beta = 0 AND alpha = 0: C is exactly zero, not NaN.
            C4, _, _, _, _, _ = run_candidate(MF, nan_operands[1], nan_operands[2],
                0.0, 0.0, nan_c; lanes=2, micro_columns=2, k_block=2)
            @test all(iszero, C4)
            @test plan0.shape.shape isa Symbol
            @test timings0.packed_elements == k * (m + n)
        end
    end
end

# ---------------------------------------------------------------------------
@testset "T7 transposes and strides end to end" begin
    rng = MersenneTwister(0x0707)
    MF = MultiFloat{Float64,2}
    A64, B64, C064 = fixture_random(rng, 6, 5, 4)
    m, k = size(A64)
    n = size(B64, 2)
    budget = error_budget(2, k)
    A = MF.(A64)
    B = MF.(B64)
    plan = shape_packing_plan(MF, m, k, n; lanes=2, micro_columns=2, k_block=2)
    workspace = packed_gemm_workspace(MF, plan, m, k, n)

    # A as a Transpose wrapper of its transpose; B likewise. Both must agree
    # with the oracle, and with each other bitwise.
    C_plain = fill(MF(NaN), m, n)
    packed_gemm_scheduled!(C_plain, A, B, MF(1), MF(0), plan, workspace)
    C_wrapped = fill(MF(NaN), m, n)
    packed_gemm_scheduled!(C_wrapped, transpose(Matrix(transpose(A))),
        adjoint(Matrix(adjoint(B))), MF(1), MF(0), plan, workspace)
    @test bitsame(C_plain, C_wrapped)
    @test gemm_error_profile(C_wrapped, A64, B64, 1.0, 0.0, nothing).worst <= budget
    measure("transpose_wrapped_worst_log2",
        gemm_error_profile(C_wrapped, A64, B64, 1.0, 0.0, nothing).worst)

    # Operands with non-unit strides and offsets: embed A and B in larger
    # matrices so the strided views are not contiguous.
    A_padded = zeros(Float64, 2m, 2k + 1)
    B_padded = zeros(Float64, 2k + 1, 2n)
    for i in 1:m, r in 1:k
        A_padded[2i - 1, 2r] = A64[i, r]
    end
    for r in 1:k, j in 1:n
        B_padded[2r, 2j - 1] = B64[r, j]
    end
    A_view = view(MF.(A_padded), 1:2:2m, 2:2:2k)
    # B was stored on the EVEN rows, so the strided view must take those; the
    # first draft took the odd rows and the oracle reported a completely wrong
    # result, which is the oracle behaving correctly on a broken fixture.
    B_view = view(MF.(B_padded), 2:2:2k, 1:2:2n)
    C_strided = fill(MF(NaN), m, n)
    packed_gemm_scheduled!(C_strided, A_view, B_view, MF(1), MF(0), plan, workspace)
    @test gemm_error_profile(C_strided, A64, B64, 1.0, 0.0, nothing).worst <= budget
    measure("strided_operands_worst_log2",
        gemm_error_profile(C_strided, A64, B64, 1.0, 0.0, nothing).worst)
    # Control: the padded matrices must NOT equal the views (otherwise the
    # stride test would be exercising a contiguous copy).
    @test !(size(A_padded) == size(A64))
end

# ---------------------------------------------------------------------------
@testset "T8 tails are reachable, reported, and correct" begin
    rng = MersenneTwister(0x0808)
    MF = MultiFloat{Float64,2}
    # 17 = 4*4 + 1 rows, k = 5 with k_block 4 -> reduction tail 1,
    # n = 7 with micro_columns 4 -> column tail 3.
    A64, B64, C064 = fixture_random(rng, 17, 5, 7)
    m, k = size(A64)
    n = size(B64, 2)
    budget = error_budget(2, k)
    C, plan, timings, _, _, _ = run_candidate(MF, A64, B64, 1.0, 0.0, nothing;
        lanes=4, micro_columns=4, k_block=4)
    measure("tail_plan", (plan.tail_rows, plan.tail_reduction, plan.tail_columns))
    measure("tail_calls", timings.tail_calls)
    measure("tail_total_calls", timings.microkernel_calls)
    @test (plan.tail_rows, plan.tail_reduction, plan.tail_columns) == (1, 1, 3)
    @test timings.tail_calls > 0            # measured, not assumed
    @test timings.microkernel_calls > timings.tail_calls
    profile = gemm_error_profile(C, A64, B64, 1.0, 0.0, nothing)
    measure("tail_worst_log2", profile.worst)
    @test profile.worst <= budget

    # Control: a shape with no tail must report zero tail calls, so "the tail
    # path runs" above is not a constant.
    A64b, B64b, _ = fixture_random(rng, 16, 8, 8)
    _, plan_b, timings_b, _, _, _ = run_candidate(MF, A64b, B64b, 1.0, 0.0, nothing;
        lanes=4, micro_columns=4, k_block=4)
    measure("no_tail_plan", (plan_b.tail_rows, plan_b.tail_reduction, plan_b.tail_columns))
    measure("no_tail_calls", timings_b.tail_calls)
    @test (plan_b.tail_rows, plan_b.tail_reduction, plan_b.tail_columns) == (0, 0, 0)
    @test timings_b.tail_calls == 0

    # Every tail width 1..3 in rows and columns must be correct, not just one.
    for rows in (1, 2, 3, 5), columns in (1, 2, 3, 5)
        a, b, _ = fixture_random(rng, rows, 3, columns)
        c, p, t, _, _, _ = run_candidate(MF, a, b, 1.0, 0.0, nothing;
            lanes=4, micro_columns=4, k_block=4)
        @test gemm_error_profile(c, a, b, 1.0, 0.0, nothing).worst <= error_budget(2, 3)
    end
end

# ---------------------------------------------------------------------------
@testset "T9 strong cancellation against the exact oracle" begin
    rng = MersenneTwister(0x0909)
    for N in (2, 3, 4)
        MF = MultiFloat{Float64,N}
        A64, B64, C064 = fixture_cancellation(rng, 6, 16, 5)
        m, k = size(A64)
        n = size(B64, 2)
        C, _, _, _, _, _ = run_candidate(MF, A64, B64, 1.0, 0.0, nothing;
            lanes=4, micro_columns=2, k_block=4)
        profile = gemm_error_profile(C, A64, B64, 1.0, 0.0, nothing)
        # The budget carries the fixture's own measured conditioning, so the
        # componentwise check is not a disguised absolute tolerance.
        budget = error_budget(N, k; condition=profile.condition)
        measure_exponent("cancellation_x$(N)_normwise_worst_log2", profile.worst)
        measure_exponent("cancellation_x$(N)_relative_worst_log2", profile.worst_relative)
        measure("cancellation_x$(N)_condition", profile.condition)
        measure("cancellation_x$(N)_budget_with_condition", budget)
        measure("cancellation_x$(N)_exact_outputs", "$(profile.exact)/$(profile.total)")
        @test profile.worst <= error_budget(N, k)
        @test profile.worst_relative <= budget
        @test profile.condition > 1.0e6      # the fixture's conditioning, measured
        # Non-vacuity: the same operation carried out in Float64 working
        # precision must be rejected by the same budget. Under cancellation the
        # componentwise form is the one that can see this; the normwise form
        # cannot, and that is exactly why both are reported.
        x1_result = float64_precision_result(A64, B64, 1.0, 0.0, nothing, MF)
        profile_x1 = gemm_error_profile(x1_result, A64, B64, 1.0, 0.0, nothing)
        measure_exponent("cancellation_x$(N)_x1control_normwise_log2", profile_x1.worst)
        measure_exponent("cancellation_x$(N)_x1control_relative_log2",
            profile_x1.worst_relative)
        @test profile_x1.worst_relative > budget
        # Informational: rounding the accurate result to Float64 lands at the
        # 53-bit floor, which is *not* the same measurement as the control above.
        truncated = float64_truncated(C, MF)
        profile_t = gemm_error_profile(truncated, A64, B64, 1.0, 0.0, nothing)
        measure_exponent("cancellation_x$(N)_truncated_result_relative_log2",
            profile_t.worst_relative)
    end
    # The fixture really does cancel: report how much, measured exactly. This is
    # a property of the instrument, and without it "strong cancellation" would
    # be an unverified adjective.
    A64, B64, _ = fixture_cancellation(MersenneTwister(1), 6, 16, 5)
    depth = cancellation_depth(A64, B64)
    measure("cancellation_depth_max_output_over_term_scale", depth)
    @test depth < 1.0e-6
    @test depth > 0.0
end

# ---------------------------------------------------------------------------
@testset "T10 exponent extremes and subnormals" begin
    rng = MersenneTwister(0x1010)
    for N in (2, 3)
        MF = MultiFloat{Float64,N}
        A64, B64, _ = fixture_extremes(rng, 5, 6, 4)
        m, k = size(A64)
        n = size(B64, 2)
        C, _, _, _, _, _ = run_candidate(MF, A64, B64, 1.0, 0.0, nothing;
            lanes=2, micro_columns=2, k_block=3)
        @test all(isfinite, Float64.(C))
        profile = gemm_error_profile(C, A64, B64, 1.0, 0.0, nothing)
        measure("extremes_x$(N)_worst_log2", profile.worst)
        @test profile.worst <= error_budget(N, k)
        measure("extremes_x$(N)_finite", true)

        # Control: the fixture really spans the exponent range.
        @test maximum(abs, A64) > 1.0e100
        @test minimum(abs, A64) < 1.0e-100
    end
    # Subnormal operands: products are representable but an implementation that
    # scales before multiplying can flush them.
    MF = MultiFloat{Float64,2}
    A64, B64, _ = fixture_subnormal(rng, 3, 4, 3)
    C, _, _, _, _, _ = run_candidate(MF, A64, B64, 1.0, 0.0, nothing;
        lanes=2, micro_columns=2, k_block=2)
    @test all(!iszero, C)
    @test all(isfinite, Float64.(C))
    profile = gemm_error_profile(C, A64, B64, 1.0, 0.0, nothing)
    measure("subnormal_worst_log2", profile.worst)
    @test profile.worst <= error_budget(2, 4)
    @test minimum(abs, A64) == 5.0e-324
end

# ---------------------------------------------------------------------------
@testset "T11 fixed order: invariance, and the instrument can see a change" begin
    rng = MersenneTwister(0x1111)
    MF = MultiFloat{Float64,2}
    # Full-precision random operands, not the cancelling fixture. The first
    # draft used the cancelling fixture and the reassociation control FAILED:
    # with a well-conditioned sum the compensated arithmetic returned identical
    # bits under every association, so the control could not detect a change.
    # The well-conditioned case is recorded below as a measurement of its own,
    # because "reassociation changes the bits" is evidently not unconditional.
    A64, B64, _ = fixture_random(rng, 33, 12, 41)
    A = MF.(A64)
    B = MF.(B64)
    m, k = size(A)
    n = size(B, 2)

    function scheduled(; lanes, micro_columns, panel_columns, layout, k_block,
                       order=nothing)
        plan = shape_packing_plan(MF, m, k, n;
            lanes=lanes, micro_columns=micro_columns, panel_columns=panel_columns,
            layout=layout, k_block=k_block)
        workspace = packed_gemm_workspace(MF, plan, m, k, n)
        C = fill(MF(NaN), m, n)
        packed_gemm_scheduled!(C, A, B, MF(1), MF(0), plan, workspace;
            k_block_order=order)
        return C
    end

    base = scheduled(lanes=4, micro_columns=4, panel_columns=64,
        layout=:aos, k_block=4)
    invariant_variants = (
        (1, 1, 1, :aos), (2, 2, 4, :aos), (4, 1, 8, :aos), (1, 4, 64, :aos),
        (2, 4, 7, :aos), (4, 4, 41, :aos), (3, 3, 13, :aos),
        (4, 4, 64, :soa_limbs), (2, 2, 4, :soa_limbs), (1, 1, 1, :soa_limbs),
    )
    for (lanes, micro_columns, panel_columns, layout) in invariant_variants
        other = scheduled(lanes=lanes, micro_columns=micro_columns,
            panel_columns=panel_columns, layout=layout, k_block=4)
        equal = bitsame(other, base)
        measure("fixed_order_invariance_lanes$(lanes)_mc$(micro_columns)_panel$(panel_columns)_$(layout)",
            equal)
        @test equal
    end

    # Control: k_blocking changes the association, because the microkernel
    # forms a block partial sum and the panel accumulator adds it once per
    # block. The batch of assertions above would be unfalsifiable if nothing
    # could change the bits, so this is the control that the comparison can
    # detect a difference at all.
    blocked_2 = scheduled(lanes=4, micro_columns=4, panel_columns=64,
        layout=:aos, k_block=2)
    blocked_12 = scheduled(lanes=4, micro_columns=4, panel_columns=64,
        layout=:aos, k_block=12)
    measure("kblock_4_vs_2_bitsame", bitsame(base, blocked_2))
    measure("kblock_4_vs_12_bitsame", bitsame(base, blocked_12))
    @test !bitsame(base, blocked_12)
    # A single k-block is the association the scalar reference uses, so it
    # agrees with the reference bit for bit. This is the explanation of the
    # k-block sensitivity, measured rather than asserted.
    reference = fill(MF(NaN), m, n)
    packed_gemm_reference!(reference, A, B, MF(1), MF(0))
    measure("kblock_k_vs_scalar_reference_bitsame", bitsame(blocked_12, reference))
    measure("kblock_4_vs_scalar_reference_bitsame", bitsame(base, reference))
    @test bitsame(blocked_12, reference)
    @test !bitsame(base, reference)

    # Descending k-block order is the measured-order mode; it must differ too.
    descending = scheduled(lanes=4, micro_columns=4, panel_columns=64,
        layout=:aos, k_block=4, order=:descending)
    measure("ascending_vs_descending_bitsame", bitsame(base, descending))
    measure("ascending_vs_descending_max_abs", max_abs_difference(base, descending))
    @test !bitsame(base, descending)
    @test max_abs_difference(base, descending) > 0.0

    # The well-conditioned counterpart of the control above, on the cancelling
    # fixture: k-block reassociation is NOT automatically visible in the bits.
    # This bounds every "bitwise identical" claim in this file to the fixture it
    # was measured on.
    Ac64, Bc64, _ = fixture_cancellation(rng, 33, 12, 41)
    Ac = MF.(Ac64)
    Bc = MF.(Bc64)
    function scheduled_cancelling(; k_block)
        plan = shape_packing_plan(MF, m, k, n; lanes=4, micro_columns=4,
            panel_columns=64, layout=:aos, k_block=k_block)
        workspace = packed_gemm_workspace(MF, plan, m, k, n)
        C = fill(MF(NaN), m, n)
        packed_gemm_scheduled!(C, Ac, Bc, MF(1), MF(0), plan, workspace)
        return C
    end
    c4 = scheduled_cancelling(k_block=4)
    c12 = scheduled_cancelling(k_block=12)
    measure("cancelling_fixture_kblock_4_vs_12_bitsame", bitsame(c4, c12))
    measure("cancelling_fixture_kblock_4_vs_12_max_abs", max_abs_difference(c4, c12))
end

# ---------------------------------------------------------------------------
@testset "T12 packing is accounted for, and the warm path reuses buffers" begin
    rng = MersenneTwister(0x1212)
    MF = MultiFloat{Float64,2}
    A64, B64, _ = fixture_random(rng, 48, 32, 40)
    m, k = size(A64)
    n = size(B64, 2)
    A = MF.(A64)
    B = MF.(B64)
    plan = shape_packing_plan(MF, m, k, n; lanes=4, micro_columns=4, panel_columns=16)
    workspace = packed_gemm_workspace(MF, plan, m, k, n)
    reallocations_after_build = workspace.reallocations
    timings = GemmTimingBreakdown()
    C = fill(MF(NaN), m, n)
    packed_gemm_scheduled!(C, A, B, MF(1), MF(0), plan, workspace; timings=timings)
    measure("timings_first_call", timings)
    measure("packing_elements_expected", k * (m + n))
    measure("packing_elements_measured", timings.packed_elements)
    @test timings.packed_elements == k * (m + n)
    @test timings.panels == cld(n, 16)
    @test timings.microkernel_calls > 0
    @test timings.packing_seconds > 0.0
    @test timings.microkernel_seconds > 0.0
    @test total_seconds(timings) >= timings.packing_seconds
    @test total_seconds(timings) >= timings.microkernel_seconds
    # Packing is a real, non-trivial share of this schedule, which is why the
    # card requires it to be counted; the number is reported, not asserted.
    measure("packing_share_of_total", timings.packing_seconds / total_seconds(timings))

    # Warm reuse: a second call on the same shape must not reallocate.
    second = GemmTimingBreakdown()
    packed_gemm_scheduled!(C, A, B, MF(1), MF(0), plan, workspace; timings=second)
    measure("workspace_reallocations_after_warm_call", workspace.reallocations)
    measure("workspace_reallocations_after_build", reallocations_after_build)
    @test workspace.reallocations == reallocations_after_build

    # Control for the reuse claim: a larger shape MUST reallocate, so a zero
    # reallocation count is a measurement and not a counter that never moves.
    big_plan = shape_packing_plan(MF, 2m, k, 2n; lanes=4, micro_columns=4,
        panel_columns=16)
    ensure_packed_gemm_capacity!(workspace, big_plan, 2m, k, 2n)
    measure("workspace_reallocations_after_growth", workspace.reallocations)
    @test workspace.reallocations > reallocations_after_build

    # The plan refuses to be reused for a different shape (no silent reuse).
    # The first check is the dimensional one the schedule enforces against C;
    # the second reaches the plan check itself, where A and C agree with each
    # other but not with the plan's recorded shape.
    C_small = fill(MF(NaN), 4, 4)
    small_plan = shape_packing_plan(MF, 4, 4, 4)
    small_ws = packed_gemm_workspace(MF, small_plan, 4, 4, 4)
    @test_throws DimensionMismatch packed_gemm_scheduled!(
        C_small, MF.(zeros(5, 4)), MF.(zeros(4, 4)), MF(1), MF(0), small_plan, small_ws)
    @test_throws ArgumentError packed_gemm_scheduled!(
        fill(MF(NaN), 5, 4), MF.(zeros(5, 4)), MF.(zeros(4, 4)), MF(1), MF(0),
        small_plan, small_ws)
end

# ---------------------------------------------------------------------------
@testset "T13 per-limb timing on the target CPU, with packing counted" begin
    rng = MersenneTwister(0x1313)
    for N in (2, 3, 4)
        MF = MultiFloat{Float64,N}
        m = k = n = 128
        A64, B64, _ = fixture_random(rng, m, k, n)
        A = MF.(A64)
        B = MF.(B64)
        plan = shape_packing_plan(MF, m, k, n)
        workspace = packed_gemm_workspace(MF, plan, m, k, n)
        C = fill(MF(NaN), m, n)
        timings = GemmTimingBreakdown()
        gate = idle_guard()
        call = () -> packed_gemm_scheduled!(C, A, B, MF(1), MF(0), plan, workspace;
            timings=timings)
        # `timings` accumulates across samples; reset it once and time the last
        # sample separately so the reported breakdown is one call, not a sum.
        result = measure_seconds(call; samples=3, warmup=1)
        measure("timing_x$(N)_minimum_seconds", result.minimum)
        measure("timing_x$(N)_median_seconds", result.median)
        measure("timing_x$(N)_samples", result.samples)
        measure("timing_x$(N)_idle_guard_quiet", gate.quiet)
        measure("timing_x$(N)_idle_guard_waited_seconds", gate.waited)
        measure("timing_x$(N)_loadavg_after", load_snapshot())
        measure("timing_x$(N)_contended", !gate.quiet)
        @test result.minimum > 0.0
        @test isfinite(result.minimum)
        @test timings.packed_elements > 0
        @test timings.packing_seconds > 0.0
        measure("timing_x$(N)_plan_lanes_mc_kblock",
            (plan.lanes, plan.micro_columns, plan.k_block))
        measure("timing_x$(N)_plan_reason", plan.reason)
    end

    # Variant grid on one limb count: lanes and micro columns are measured, not
    # assumed to help. This is the "wider SIMD is not automatically faster"
    # question in the card, and the answer is whatever these numbers say.
    MF = MultiFloat{Float64,2}
    m = k = n = 96
    A64, B64, _ = fixture_random(rng, m, k, n)
    A = MF.(A64)
    B = MF.(B64)
    gate = idle_guard()
    for lanes in (1, 2, 4), micro_columns in (1, 2, 4)
        plan = shape_packing_plan(MF, m, k, n; lanes=lanes, micro_columns=micro_columns)
        workspace = packed_gemm_workspace(MF, plan, m, k, n)
        C = fill(MF(NaN), m, n)
        result = measure_seconds(
            () -> packed_gemm_scheduled!(C, A, B, MF(1), MF(0), plan, workspace);
            samples=3, warmup=1,
        )
        measure("variant_x2_lanes$(lanes)_mc$(micro_columns)_minimum_seconds", result.minimum)
        measure("variant_x2_lanes$(lanes)_mc$(micro_columns)_samples", result.samples)
        @test result.minimum > 0.0
    end

    # Head-to-head with the production entry point on the same shape and the
    # same operands. This is context for the candidate's maturity, not a claim
    # that it should be enabled; nothing here flips a default.
    MF = MultiFloat{Float64,2}
    m = k = n = 128
    A64, B64, _ = fixture_random(rng, m, k, n)
    A = MF.(A64)
    B = MF.(B64)
    C_prod = fill(MF(NaN), m, n)
    production = measure_seconds(() -> gemm!(C_prod, A, B, MF(1), MF(0)); samples=3, warmup=1)
    measure("production_gemm_x2_minimum_seconds", production.minimum)
    measure("production_gemm_x2_samples", production.samples)
    measure("contended_during_t13", !gate.quiet)
    measure("loadavg_after_t13", load_snapshot())
    @test production.minimum > 0.0
end

# ---------------------------------------------------------------------------
@testset "T14 register-pressure proxy: LLVM stack slots per variant" begin
    if !INTERACTIVE_UTILS_PRESENT
        measure("spill_proxy", "unsupported: InteractiveUtils unavailable")
        @test_skip true
    else
        MF = MultiFloat{Float64,2}
        n = 8
        acc = fill(MF(0), n, n)
        alayout = panel_pack_layout(MF, 4, n)
        blayout = panel_pack_layout(MF, 4, n)
        abuf = allocate_panel(MF, alayout)
        bbuf = allocate_panel(MF, blayout)
        apanel = PackedPanel(abuf, alayout, 0)
        bpanel = PackedPanel(bbuf, blayout, 0)
        counts = Dict{Tuple{Int,Int},Int}()
        for lanes in (1, 2, 4), micro_columns in (1, 2, 4)
            ir = try
                sprint(io -> InteractiveUtils.code_llvm(io, microkernel_block!,
                    Tuple{Matrix{MF},typeof(apanel),typeof(bpanel),Int,Int,
                          Val{lanes},Val{micro_columns}}))
            catch error
                measure("spill_proxy_error_lanes$(lanes)_mc$(micro_columns)", sprint(showerror, error))
                ""
            end
            allocas = length(collect(eachmatch(r"=\s*alloca", ir)))
            counts[(lanes, micro_columns)] = allocas
            measure("llvm_alloca_count_lanes$(lanes)_mc$(micro_columns)", allocas)
            measure("llvm_ir_chars_lanes$(lanes)_mc$(micro_columns)", length(ir))
            # Determinism control: the same query must produce the same count,
            # otherwise the number is an artifact of the query rather than of
            # the specialization.
            ir_again = sprint(io -> InteractiveUtils.code_llvm(io, microkernel_block!,
                Tuple{Matrix{MF},typeof(apanel),typeof(bpanel),Int,Int,
                      Val{lanes},Val{micro_columns}}))
            @test length(collect(eachmatch(r"=\s*alloca", ir_again))) == allocas
        end
        # Non-vacuity: the counter must be able to report different values for
        # different register demands, otherwise every count above is noise.
        measure("llvm_alloca_distinct_values", length(unique(values(counts))))
        @test length(unique(values(counts))) > 1
        # The widest variant (4 lanes x 4 columns = 16 vector accumulators) is
        # expected to need the most stack; that is reported, not asserted.
        widest = counts[(4, 4)]
        narrowest = counts[(1, 1)]
        measure("llvm_alloca_widest_minus_narrowest", widest - narrowest)
    end
    # Text measurement over the candidate sources: no unconditional fast math,
    # no forced vectorisation pragma. Comments are stripped first, because the
    # first draft counted the words "fastmath" and "@simd" in the very comments
    # that state their absence and reported a false positive on the candidate's
    # own documentation.
    function code_only(text)
        return join(
            (line for line in split(text, '\n') if !startswith(strip(line), "#")),
            '\n',
        )
    end
    for file in ("planning/gemm_plan.jl", "kernels/packing.jl",
                 "kernels/gemm_microkernels.jl", "kernels/gemm_schedule.jl")
        text = code_only(read(joinpath(M02_REPO, "src", file), String))
        fastmath = count("fastmath", text)
        simd_pragma = count("@simd", text)
        measure("source_fastmath_count_$(basename(file))", fastmath)
        measure("source_simd_pragma_count_$(basename(file))", simd_pragma)
        @test fastmath == 0
        @test simd_pragma == 0
    end
    # Control: the same counter reports non-zero on code that does contain the
    # tokens, so the zero above is a measurement and not a broken strip.
    control_text = code_only("# @fastmath is absent here\n@fastmath x + y\n@simd for i in 1:3\n")
    @test count("fastmath", control_text) == 1
    @test count("@simd", control_text) == 1
    @test count("fastmath", code_only("# @fastmath only in a comment\n")) == 0
end

# ---------------------------------------------------------------------------
@testset "T15 S05-F1: is multi-RHS genuinely supported, or per-column?" begin
    MF = MultiFloat{Float64,2}
    n = 24
    nrhs = 7
    rng = MersenneTwister(0x1515)
    dense = randn(rng, n, n)
    # Gram matrix plus a shift: symmetric positive definite by construction and
    # well conditioned enough that a Cholesky/LDLT leg is a real test of the
    # solve path rather than of a pivot fallback.
    symmetric = transpose(dense) * dense + n * I
    B64 = randn(rng, n, nrhs)
    SPD = MF.(symmetric)
    B = MF.(B64)

    # (1) Reachability: does a batched (matrix RHS) method exist at all, as a
    # DISTINCT method from the vector-RHS method?
    vector_method = which(solve!, Tuple{Vector{MF},MFLDLTCache{MF},Vector{MF}})
    matrix_method = which(solve!, Tuple{Matrix{MF},MFLDLTCache{MF},Matrix{MF}})
    measure("multirhs_ldlt_vector_method", string(vector_method))
    measure("multirhs_ldlt_matrix_method", string(matrix_method))
    @test vector_method != matrix_method
    # Control: a method that does not exist must not silently resolve. `which`
    # signals this with an ErrorException wrapping the MethodError, not with a
    # MethodError, which is itself worth recording.
    @test_throws ErrorException which(solve!, Tuple{Matrix{Float64},MFLDLTCache{MF},Matrix{MF}})

    for (name, system_matrix, make_cache) in (
        ("ldlt", symmetric, () -> begin
            cache = MFLDLTCache(MF; config=KernelConfig())
            prepare!(cache, n)
            factorize!(cache, SPD)
            cache
        end),
        ("cholesky", symmetric, () -> begin
            cache = MFCholeskyCache(MF; config=KernelConfig())
            prepare!(cache, n)
            factorize!(cache, SPD)
            cache
        end),
        # The LU leg factorizes the UNSYMMETRIC matrix, so its residual must be
        # checked against that matrix: the first draft checked every leg against
        # the symmetric one and reported a residual of 216 for LU, which was the
        # driver comparing two different systems.
        ("lu", dense, () -> begin
            cache = MFLUCache(MF; config=KernelConfig())
            prepare!(cache, n)
            factorize!(cache, MF.(dense))
            cache
        end),
    )
        cache = make_cache()
        @test MultiFloatLinearAlgebra.issuccess(cache)
        # (2) Correctness: the batched call must agree with the per-column loop
        # BIT FOR BIT, per column, including a trailing column that is not a
        # multiple of any tile width.
        X_batched = fill(MF(NaN), n, nrhs)
        solve!(X_batched, cache, B)
        X_loop = fill(MF(NaN), n, nrhs)
        for column in 1:nrhs
            solve!(view(X_loop, :, column), cache, view(B, :, column))
        end
        agreement = bitsame(X_batched, X_loop)
        measure("multirhs_$(name)_batched_equals_per_column_bitsame", agreement)
        @test agreement
        # Residual control: the batched result must actually solve the system,
        # otherwise "equal to the loop" could mean "both wrong the same way".
        system = MF.(system_matrix)
        residual = maximum(
            abs(Float64(sum(system[i, r] * X_batched[r, j] for r in 1:n) - B[i, j]))
            for i in 1:n, j in 1:nrhs
        )
        measure("multirhs_$(name)_max_abs_residual", residual)
        @test residual < 1.0e-25

        # (3) Cost profile: is the batched call distinguishable from the loop,
        # or is it the loop wearing a matrix signature? Timing is the only
        # instrument available at -t1; the numbers are reported raw either way.
        gate = idle_guard()
        batched_time = measure_seconds(() -> solve!(X_batched, cache, B);
            samples=3, warmup=1)
        loop_time = measure_seconds(() -> begin
            for column in 1:nrhs
                solve!(view(X_loop, :, column), cache, view(B, :, column))
            end
        end; samples=3, warmup=1)
        measure("multirhs_$(name)_batched_minimum_seconds", batched_time.minimum)
        measure("multirhs_$(name)_per_column_minimum_seconds", loop_time.minimum)
        measure("multirhs_$(name)_ratio_batched_over_loop",
            batched_time.minimum / loop_time.minimum)
        measure("multirhs_$(name)_idle_guard_quiet", gate.quiet)
        measure("multirhs_$(name)_loadavg", load_snapshot())
        @test batched_time.minimum > 0.0
        @test loop_time.minimum > 0.0
    end

    # (4) The kernel-level batching question, measured in the candidate this
    # task owns: does processing several RHS columns in ONE microkernel pass
    # pay, or is batching a signature-only generalisation? Same operands, same
    # total work, only the micro-column width changes.
    m = k = 64
    ncols = 32
    A64, Bwide64, _ = fixture_random(rng, m, k, ncols)
    A = MF.(A64)
    Bwide = MF.(Bwide64)
    for micro_columns in (1, 2, 4)
        plan = shape_packing_plan(MF, m, k, ncols; lanes=4,
            micro_columns=micro_columns)
        workspace = packed_gemm_workspace(MF, plan, m, k, ncols)
        C = fill(MF(NaN), m, ncols)
        result = measure_seconds(
            () -> packed_gemm_scheduled!(C, A, Bwide, MF(1), MF(0), plan, workspace);
            samples=3, warmup=1)
        measure("microbatch_mc$(micro_columns)_seconds", result.minimum)
        measure("microbatch_mc$(micro_columns)_per_column_seconds",
            result.minimum / ncols)
        @test result.minimum > 0.0
    end

    # (5) The sparse provider. A01b-F2 records that QDLDL multi-RHS is
    # per-column only. Re-measure it here rather than inheriting the claim.
    if !QDLDL_PRESENT || !sparse_ldlt_available(MF)
        measure("multirhs_sparse", "unsupported: QDLDL extension not loaded")
        @test_skip true
    else
        # A tridiagonal SPD matrix stored as the UPPER triangle, which is what
        # the sparse cache's frozen pattern requires.
        tridiagonal = spdiagm(0 => 4.0 * ones(n), 1 => -1.0 * ones(n - 1))
        sparse_pattern = SparseMatrixCSC{MF,Int}(
            n, n, copy(tridiagonal.colptr), copy(tridiagonal.rowval),
            MF.(nonzeros(tridiagonal)),
        )
        cache = sparse_ldlt_cache(MF, sparse_pattern;
            dsigns=ones(Int, n), nrhs=nrhs)
        factorize!(cache, sparse_pattern)
        @test MultiFloatLinearAlgebra.issuccess(cache)
        # The sparse cache's solve! argument order is (cache, destination, rhs),
        # reversed from every dense cache (M01-F2). Measure that asymmetry.
        sparse_cache_method = which(solve!,
            Tuple{typeof(cache),Matrix{MF},Matrix{MF}})
        measure("multirhs_sparse_matrix_method", string(sparse_cache_method))
        B_sparse = MF.(Matrix(tridiagonal) * B64)
        X_sparse = fill(MF(NaN), n, nrhs)
        outcome = try
            solve!(cache, X_sparse, B_sparse)
            "completed"
        catch error
            string(typeof(error), ": ", sprint(showerror, error))
        end
        measure("multirhs_sparse_matrix_rhs_outcome", outcome)
        # Per-column on the same cache is the control: if this also failed, the
        # failure above would be about the cache, not about multi-RHS.
        X_sparse_loop = fill(MF(NaN), n, nrhs)
        per_column_outcome = try
            for column in 1:nrhs
                solve!(cache, view(X_sparse_loop, :, column), view(B_sparse, :, column))
            end
            "completed"
        catch error
            string(typeof(error), ": ", sprint(showerror, error))
        end
        measure("multirhs_sparse_per_column_outcome", per_column_outcome)
        @test per_column_outcome == "completed"
        @test outcome == "completed"
        if outcome == "completed"
            measure("multirhs_sparse_batched_equals_per_column_bitsame",
                bitsame(X_sparse, X_sparse_loop))
            @test bitsame(X_sparse, X_sparse_loop)
            # The same timing instrument as for the dense caches. If the sparse
            # path is the per-column loop it is documented to be, its ratio
            # should sit at ~1 where the dense one does not; both numbers are
            # reported and neither is asserted to a threshold.
            sparse_gate = idle_guard()
            sparse_batched = measure_seconds(() -> solve!(cache, X_sparse, B_sparse);
                samples=3, warmup=1)
            sparse_loop = measure_seconds(() -> begin
                for column in 1:nrhs
                    solve!(cache, view(X_sparse_loop, :, column),
                        view(B_sparse, :, column))
                end
            end; samples=3, warmup=1)
            measure("multirhs_sparse_batched_minimum_seconds", sparse_batched.minimum)
            measure("multirhs_sparse_per_column_minimum_seconds", sparse_loop.minimum)
            measure("multirhs_sparse_ratio_batched_over_loop",
                sparse_batched.minimum / sparse_loop.minimum)
            measure("multirhs_sparse_idle_guard_quiet", sparse_gate.quiet)
            @test sparse_batched.minimum > 0.0
        end
        # A01b-F2's claim, re-measured OUT OF PROCESS. The raw QDLDL matrix entry
        # point performs an out-of-bounds write and is memory-unsafe: it usually
        # raises a catchable ReadOnlyMemoryError (10/10 across two independent
        # parties) and has once killed a process (the parent's in-process run,
        # under concurrent load: exit 139, signal 11 in ipermute!, n=1). The
        # trigger is NOT identified — a 0..2e7-iteration allocation-churn sweep
        # came back negative, so no mechanism is claimed. Running it here would
        # risk killing this run for no gain, so the child's exit code and signal
        # ARE the measurement.
        clean_probe = run_qdldl_raw_matrix_probe(churn=0, order=n)
        measure("multirhs_qdldl_raw_matrix_solve_child_exit_code", clean_probe.exit_code)
        measure("multirhs_qdldl_raw_matrix_solve_child_term_signal", clean_probe.term_signal)
        measure("multirhs_qdldl_raw_matrix_solve_child_died_on_signal", clean_probe.died_on_signal)
        measure("multirhs_qdldl_raw_matrix_solve_outcome", clean_probe.outcome)
        measure("multirhs_qdldl_raw_matrix_solve_child_reached_call",
            occursin("PROBE_CALLING_RAW_MATRIX_SOLVE", clean_probe.output))
        @test occursin("PROBE_CALLING_RAW_MATRIX_SOLVE", clean_probe.output)
        @test occursin("PROBE_FACTORIZED issuccess=true", clean_probe.output)

        # The same call after a bounded heap churn in the child. If the child
        # still raises a catchable exception, this driver did NOT reproduce the
        # parent's signal death at this churn level, and that negative is
        # recorded with the churn level that failed to reproduce it rather than
        # left as an unstated gap.
        churned_probe = run_qdldl_raw_matrix_probe(churn=1_000_000, order=n)
        measure("multirhs_qdldl_raw_matrix_solve_churned_exit_code", churned_probe.exit_code)
        measure("multirhs_qdldl_raw_matrix_solve_churned_term_signal", churned_probe.term_signal)
        measure("multirhs_qdldl_raw_matrix_solve_churned_died_on_signal",
            churned_probe.died_on_signal)
        measure("multirhs_qdldl_raw_matrix_solve_churned_outcome", churned_probe.outcome)
        measure("multirhs_qdldl_raw_matrix_solve_churned_reached_call",
            occursin("PROBE_CALLING_RAW_MATRIX_SOLVE", churned_probe.output))
        @test occursin("PROBE_CALLING_RAW_MATRIX_SOLVE", churned_probe.output)
        for line in split(churned_probe.output, '\n')
            startswith(line, "PROBE_CHURN ") && measure("multirhs_qdldl_raw_matrix_churn", line)
        end
    end

    # The provider-wide Bool, and what the measurements above license.
    caps = capabilities(MF)
    measure("capabilities_multi_rhs", caps.multi_rhs)
    measure("capabilities_vector_rhs", caps.vector_rhs)
    @test caps.multi_rhs === true      # MFLA reports the generalisation...
    @test caps.vector_rhs === true     # ...and the per-column form

    # The instrument's own limitation, stated as evidence: the dense legs
    # measured batched/loop ratios below 1, and so did the SPARSE leg — whose
    # in-tree implementation is an explicit per-column loop
    # (`ext/MultiFloatQDLDLExt.jl`, read-only inspection). A timing ratio below 1
    # is therefore NOT evidence of batching, and no batching claim in this report
    # rests on it. The conclusions rest on method identity, on bitwise agreement
    # with the per-column loop, and on the measured residual.
    measure("s05f1_timing_ratio_is_not_a_batching_instrument", true)
    println("S05-F1 VERDICT: measured — the dense caches expose a distinct " *
            "matrix-RHS `solve!` method that returns results bitwise identical " *
            "to the per-column loop, with residuals at the MF precision floor " *
            "(~1e-31 for x2). The sparse QDLDL cache also completes a matrix " *
            "RHS and agrees bitwise, but its in-tree implementation is an " *
            "explicit per-column loop. The provider-wide Bool `multi_rhs` " *
            "therefore conflates a dense matrix method with a sparse " *
            "per-column loop; `MultiRHSPerColumn` remains the accurate " *
            "provider-wide label. The wall-clock ratio could not settle " *
            "batched-vs-loop: the known per-column sparse leg also measured " *
            "below 1. The only batching measurement this driver can support is " *
            "the micro-kernel one (MEASURE microbatch_mc*).")
end

# ---------------------------------------------------------------------------
@testset "T16 calibration export and certified-variant separation" begin
    MF = MultiFloat{Float64,2}
    variants = supported_packing_variants(MF; lane_grid=(1, 4), micro_grid=(1, 2),
        k_block_grid=(0, 4))
    measure("variant_grid_size", length(variants))
    @test length(variants) == 8
    @test all(v -> v.lanes in (1, 4), variants)

    entries = [
        (m=64, k=64, n=64, limbs=2, lanes=4, micro_columns=2, k_block=64,
         layout=:aos, seconds=0.010, packing_seconds=0.001, contended=false,
         bitwise_token="0xabc"),
        (m=64, k=64, n=64, limbs=2, lanes=1, micro_columns=1, k_block=64,
         layout=:aos, seconds=0.030, packing_seconds=0.001, contended=false,
         bitwise_token="0xabc"),
        (m=64, k=64, n=64, limbs=2, lanes=2, micro_columns=2, k_block=64,
         layout=:aos, seconds=0.001, packing_seconds=0.001, contended=true,
         bitwise_token="0xabc"),
    ]
    table = shape_calibration_table(entries)
    rendered = format_shape_calibration(table)
    measure("calibration_lines", count("\n", rendered))
    @test count("CALIB[", rendered) == 3
    @test occursin("seconds=0.01", rendered)

    plan, verdict = select_certified_variant(MF, 64, 64, 64, table)
    measure("certified_verdict", verdict)
    measure("certified_plan_lanes_mc", (plan.lanes, plan.micro_columns))
    @test verdict === :calibrated
    @test plan.evidence === :calibrated          # named as calibrated...
    @test plan.default_path === false            # ...and still not a default
    @test plan.lanes == 4                        # the fastest UNCONTENDED entry
    @test plan.micro_columns == 2

    # Control: the contended entry is faster but must NOT be selected, and a
    # table with no matching shape must fall back with a stated reason.
    empty_plan, empty_verdict = select_certified_variant(MF, 96, 96, 96, table)
    measure("certified_empty_verdict", empty_verdict)
    @test empty_verdict === :no_calibration_entry
    @test empty_plan.evidence === :policy_default
    @test empty_plan.default_path === false
end

# ---------------------------------------------------------------------------
@testset "T17 packing layout comparison is a measurement, not an assumption" begin
    # The card asks for a small-range comparison of panel-local SoA against the
    # interleaved layout. Both are correct; which is faster is reported.
    rng = MersenneTwister(0x1717)
    MF = MultiFloat{Float64,2}
    m = k = n = 96
    A64, B64, _ = fixture_random(rng, m, k, n)
    A = MF.(A64)
    B = MF.(B64)
    budget = error_budget(2, k)
    gate = idle_guard()
    results = Dict{Symbol,Float64}()
    for layout in (:aos, :soa_limbs)
        plan = shape_packing_plan(MF, m, k, n; lanes=4, micro_columns=4,
            layout=layout)
        workspace = packed_gemm_workspace(MF, plan, m, k, n)
        C = fill(MF(NaN), m, n)
        # One call with its own breakdown, so the counters describe exactly one
        # call. (`measure_seconds` runs four, and an earlier draft asserted a
        # per-call count against a four-call accumulator.)
        counted = GemmTimingBreakdown()
        packed_gemm_scheduled!(C, A, B, MF(1), MF(0), plan, workspace;
            timings=counted)
        profile = gemm_error_profile(C, A64, B64, 1.0, 0.0, nothing)
        result = measure_seconds(
            () -> packed_gemm_scheduled!(C, A, B, MF(1), MF(0), plan, workspace);
            samples=3, warmup=1)
        results[layout] = result.minimum
        measure("layout_$(layout)_minimum_seconds", result.minimum)
        measure_exponent("layout_$(layout)_worst_log2", profile.worst)
        measure("layout_$(layout)_packing_seconds", counted.packing_seconds)
        measure("layout_$(layout)_packed_elements_one_call", counted.packed_elements)
        @test profile.worst <= budget
        @test counted.packed_elements == k * (m + n)
    end
    measure("layout_soa_over_aos_ratio", results[:soa_limbs] / results[:aos])
    measure("layout_comparison_idle_guard_quiet", gate.quiet)
    measure("layout_comparison_loadavg", load_snapshot())
end
end # @testset "M02 driver"

println()
println("M02 MEASUREMENT LINES: ", length(MEASUREMENTS))
println("M02 DRIVER COMPLETE")
