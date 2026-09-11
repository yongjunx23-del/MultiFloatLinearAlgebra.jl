module MultiFloatLinearAlgebra

import LinearAlgebra
import Base.Threads
using MultiFloats

import MultiFloats: MultiFloat, MultiFloatVec

include("types.jl")
include("factor_cache_defs.jl")
include("kernels/dot.jl")
include("kernels/gemv.jl")
include("kernels/gemm.jl")
include("kernels/packed_gemm.jl")
include("kernels/mulacc_x2.jl")
include("kernels/mulacc_x3.jl")
include("kernels/mulacc_x4.jl")
include("calibration.jl")
include("kernels/syrk.jl")
include("kernels/gemmt.jl")
include("kernels/trsm.jl")
include("kernels/trsv.jl")
include("kernels/trmm.jl")
include("kernels/symv.jl")
include("kernels/block_updates.jl")
include("factorizations/cholesky.jl")
include("factorizations/cholesky_pivoted.jl")
include("factorizations/lu.jl")
include("factorizations/ldlt.jl")
include("factorizations/qr.jl")
include("factorizations/ldlt_tuning.jl")
include("factorizations/ldlt_gemmt.jl")
include("diagnostics.jl")
include("solve.jl")
include("residual.jl")
include("factor_caches.jl")
include("factor_cache_requirements.jl")

# --- rebuild packet: M02 shape-oriented packing and SIMD microkernels --------
#
# CANDIDATE ONLY. These four files add a shape-oriented planner, panel-local
# packing, SIMD microkernels and a packed schedule. They are ADDITIVE: they
# touch no existing definition, they export nothing, and every plan they build
# carries `default_path = false`, so no route becomes reachable by their
# presence. Wiring them makes M02's driver run in its `wired` mode.
#
# ORDER: gemm_plan.jl needs _default_gemm_panel_columns (kernels/gemm.jl) and
# _check_supported (types.jl); packing.jl needs _m02_limb_count, defined in
# gemm_plan.jl. Do not reorder.
include("planning/gemm_plan.jl")
include("kernels/packing.jl")
include("kernels/gemm_microkernels.jl")
include("kernels/gemm_schedule.jl")

# --- rebuild packet: the M01 factor/cache/summary contract -------------------
#
# @integration/core-cutover. The contract layer the packet's M01 task built,
# wired so that M02/M03/P02 and the SDPX adapter can rely on it. These three
# files were INERT until this include: nothing referenced them, so no MFLA user
# was protected by the contract.
#
# The ORDER IS MEASURED, NOT STYLISTIC (M01-F8). `factors.jl` and `workspace.jl`
# are each standalone-includable; `summary.jl` is not, and fails alone with
# `UndefVarError: FactorLease` because it needs `FactorLease` from factors.jl AND
# `BlockGrammar` from workspace.jl. M01's testset 10 asserts this coupling
# executably, including that the partial orders fail, rather than leaving it as a
# comment. An integrator who moves summary.jl earlier gets a load-time failure,
# not a degraded contract.
#
# ADDITIVE ONLY. This wires the API surface and nothing else: it does not change
# any factorize!/solve! behaviour, does not enable a new GEMM/QR/sparse default,
# and does not apply M01's IP-2 (which would call `record_factor_summary!` at
# factorize commit points). IP-2 is a behaviour change and belongs to I02, whose
# card is 逐candidate引入 new kernels and caches rather than switching defaults.
#
# CONSEQUENCE, recorded rather than glossed: this changes MFLA's include graph,
# which EXPIRES the loaded-code equivalence ADR-002 §10 records against
# `50e6e0b`. Provider measurements taken at `3ddf8ed` remain valid for the
# numeric kernels — those are untouched and nothing new is on an execution path —
# but "identical loaded code" is no longer true and must not be claimed.
include("contracts/factors.jl")
include("contracts/workspace.jl")
include("contracts/summary.jl")

include("capabilities.jl")

function _linearsolve_extension()
    extension = Base.get_extension(@__MODULE__, :MultiFloatLinearSolveExt)
    extension === nothing && throw(ArgumentError(
        "load LinearSolve before constructing a MultiFloat LinearSolve algorithm",
    ))
    return extension
end

"""
    MultiFloatLU(; config=KernelConfig())

Construct the optional LinearSolve.jl algorithm backed by MFLA's dense
partial-pivoting LU factorization. Load `LinearSolve` before calling this
constructor.
"""
function MultiFloatLU(; config::KernelConfig=KernelConfig())
    return getproperty(_linearsolve_extension(), :MultiFloatLU)(config)
end

"""
    MultiFloatCholesky(; config=KernelConfig())

Construct the optional LinearSolve.jl algorithm backed by MFLA's dense lower
Cholesky factorization. Load `LinearSolve` before calling this constructor.
"""
function MultiFloatCholesky(; config::KernelConfig=KernelConfig())
    return getproperty(_linearsolve_extension(), :MultiFloatCholesky)(config)
end

"""Whether the optional QDLDL sparse-LDL extension is loaded for `MF`."""
sparse_ldlt_available(::Type{MF}) where {MF<:MultiFloat} =
    Base.get_extension(@__MODULE__, :MultiFloatQDLDLExt) !== nothing

"""
    sparse_ldlt_cache(MF, pattern; kwargs...)

Construct the optional QDLDL-backed sparse LDL cache for a frozen upper-
triangular CSC `pattern`. QDLDL owns symbolic/numeric LDL; MFLA owns the
MultiFloat type and cache contract. Loading QDLDL is explicit and absence
fails closed.
"""
function sparse_ldlt_cache(::Type{MF}, pattern; kwargs...) where {MF<:MultiFloat}
    extension = Base.get_extension(@__MODULE__, :MultiFloatQDLDLExt)
    extension === nothing && throw(ArgumentError(
        "load QDLDL before constructing a MultiFloat sparse LDL cache",
    ))
    return getproperty(extension, :MFSparseLDLCache)(MF, pattern; kwargs...)
end

export KernelConfig, GemmWorkspace, MFWorkspace
export workspace_capacity, ensure_workspace_capacity!
export GemmPlan, GemmProfile, GemmMeasurement, GemmCalibration, LDLTPlan
export machine_fingerprint, default_gemm_profile, calibrate_gemm
export with_gemm_profile, profile_compatible, gemm_plan, ldlt_plan
export mfdot, gemv!, gemm!, syrk!, syrk_packed!, gemmt!, trsm!, trsv!, trmm!, symv!
export AbstractMFFactorization, factor_status, factor_kind, factor_matrix
export factor_state, factor_precision, factor_provider, factor_diagnostics
export factor_pivots, factor_blocks, factor_permutation, factor_inertia
export factor_rdiag, numerical_rank
export MFCholesky, MFCholeskyPivoted, MFLU, MFLDLT, MFQR
export cholesky!, cholesky_pivoted!, lu!, ldlt!, qr!, rrqr!, issuccess
export apply_q!, solve_r!
export ldiv!, solve
export residual!, residual_mixed!, normwise_backward_error, refinement_correction!
export capabilities
export MultiFloatLU, MultiFloatCholesky
export sparse_ldlt_available, sparse_ldlt_cache
export AbstractMFFactorCache, MFCholeskyCache, MFLUCache, MFLDLTCache, MFRRQRCache
export prepare!, factorize!, solve!, invalidate!, reconfigure!, refresh!
export workspace_requirements, factor_cache_requirements, factor_cache_capacity

end
