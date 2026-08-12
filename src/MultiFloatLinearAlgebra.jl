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
include("calibration.jl")
include("kernels/syrk.jl")
include("kernels/gemmt.jl")
include("kernels/trsm.jl")
include("factorizations/cholesky.jl")
include("factorizations/lu.jl")
include("factorizations/ldlt.jl")
include("factorizations/ldlt_tuning.jl")
include("factorizations/ldlt_gemmt.jl")
include("solve.jl")

export KernelConfig, GemmWorkspace
export GemmPlan, GemmProfile, GemmMeasurement, GemmCalibration, LDLTPlan
export machine_fingerprint, default_gemm_profile, calibrate_gemm
export with_gemm_profile, gemm_plan, ldlt_plan
export mfdot, gemv!, gemm!, syrk!, gemmt!, trsm!
export MFCholesky, MFLU, MFLDLT, cholesky!, lu!, ldlt!, issuccess
export ldiv!, solve

end
