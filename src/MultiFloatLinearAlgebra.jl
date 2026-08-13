module MultiFloatLinearAlgebra

import LinearAlgebra
import Base.Threads
using MultiFloats

import MultiFloats: MultiFloat, MultiFloatVec

include("types.jl")
include("kernels/dot.jl")
include("kernels/gemv.jl")
include("kernels/gemm.jl")
include("kernels/packed_gemm.jl")
include("kernels/mulacc_x3.jl")
include("calibration.jl")
include("kernels/syrk.jl")
include("kernels/gemmt.jl")
include("kernels/trsm.jl")
include("kernels/trsv.jl")
include("kernels/trmm.jl")
include("kernels/symv.jl")
include("factorizations/cholesky.jl")
include("factorizations/lu.jl")
include("factorizations/ldlt.jl")
include("factorizations/qr.jl")
include("factorizations/ldlt_tuning.jl")
include("factorizations/ldlt_gemmt.jl")
include("diagnostics.jl")
include("solve.jl")
include("residual.jl")
include("capabilities.jl")

export KernelConfig, GemmWorkspace, MFWorkspace
export workspace_capacity, ensure_workspace_capacity!
export GemmPlan, GemmProfile, GemmMeasurement, GemmCalibration, LDLTPlan
export machine_fingerprint, default_gemm_profile, calibrate_gemm
export with_gemm_profile, profile_compatible, gemm_plan, ldlt_plan
export mfdot, gemv!, gemm!, syrk!, syrk_packed!, gemmt!, trsm!, trsv!, trmm!, symv!
export AbstractMFFactorization, factor_status, factor_kind, factor_matrix
export factor_state, factor_precision, factor_provider, factor_diagnostics
export factor_permutation, factor_rdiag, numerical_rank
export MFCholesky, MFLU, MFLDLT, MFQR, cholesky!, lu!, ldlt!, rrqr!, issuccess
export apply_q!, solve_r!
export ldiv!, solve
export residual!, residual_mixed!, normwise_backward_error, refinement_correction!
export capabilities

end
