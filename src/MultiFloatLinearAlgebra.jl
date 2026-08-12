module MultiFloatLinearAlgebra

import LinearAlgebra
import Base.Threads
using MultiFloats

import MultiFloats: MultiFloat, MultiFloatVec

include("types.jl")
include("kernels/dot.jl")
include("kernels/gemv.jl")
include("kernels/gemm.jl")
include("kernels/syrk.jl")
include("kernels/trsm.jl")
include("factorizations/cholesky.jl")
include("factorizations/lu.jl")
include("factorizations/ldlt.jl")
include("solve.jl")

export KernelConfig
export mfdot, gemv!, gemm!, syrk!, trsm!
export MFCholesky, MFLU, MFLDLT, cholesky!, lu!, ldlt!, issuccess
export ldiv!, solve

end
