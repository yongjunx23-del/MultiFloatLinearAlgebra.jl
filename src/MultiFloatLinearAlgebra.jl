module MultiFloatLinearAlgebra

using LinearAlgebra
using Base.Threads
using MultiFloats

import MultiFloats: MultiFloat, MultiFloatVec

include("types.jl")
include("kernels/dot.jl")
include("kernels/gemv.jl")
include("kernels/gemm.jl")
include("kernels/syrk.jl")
include("factorizations/cholesky.jl")
include("factorizations/lu.jl")
include("solve.jl")

export KernelConfig
export mfdot, gemv!, gemm!, syrk!
export MFCholesky, MFLU, cholesky!, lu!, issuccess
export ldiv!, solve

end
