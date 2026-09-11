# Fused x2 multiply-accumulate primitive.
#
# PROVENANCE. The limb network is ported verbatim from
# `MultiFloatArithmetic.fma_fast_limbs(::NTuple{2}, ::NTuple{2}, ::NTuple{2})`,
# reordering operands to the MFLA convention `mulacc(acc, x, y) == acc + x*y`.
# The source network carries a pinned FPAN structural proof for its explicit
# FastTwoSum precondition and output non-overlap relation; see
# MultiFloatArithmetic.jl `docs/NUMERICAL_CONTRACT.md`.
#
# NUMERICAL CONTRACT — DIFFERENT FROM mulacc_x3/mulacc_x4. The x3/x4 networks
# are empirically bitwise-identical to `acc + x * y`. This x2 network is NOT:
# it satisfies the operand-relative bound
#     |z - (x*y + c)| <= C_2 * u^2 * (|x*y| + |c|),   C_2 = 34,
# i.e. it may differ from `acc + x*y` in the last limb under destructive
# cancellation. That bound is acceptable for accumulation inside
# GEMM/SYRK/QR trailing updates (the error stays proportional to the operands,
# far below the x2 working precision), but this kernel must not be used where
# bitwise agreement with `acc + x*y` is asserted.

@inline function _mulacc_x2_limbs(x, y, c)
    x0, x1 = x
    y0, y1 = y
    c0, c1 = c
    p00, e00 = MultiFloats.two_prod(x0, y0)
    p01 = MultiFloats.one_prod(x0, y1)
    p10 = MultiFloats.one_prod(x1, y0)
    cross = p01 + p10
    low = e00 + c1
    low += cross
    high, carry = MultiFloats.two_sum(p00, c0)
    carry += low
    return MultiFloats.fast_two_sum(high, carry)
end

@inline function mulacc_x2(
    acc::MultiFloatVec{4,Float64,2},
    x::MultiFloatVec{4,Float64,2},
    y::MultiFloatVec{4,Float64,2},
)
    return MultiFloatVec{4,Float64,2}(
        _mulacc_x2_limbs(x._limbs, y._limbs, acc._limbs),
    )
end

@inline function mulacc_x2(
    acc::MultiFloat{Float64,2},
    x::MultiFloat{Float64,2},
    y::MultiFloat{Float64,2},
)
    return MultiFloat{Float64,2}(
        _mulacc_x2_limbs(x._limbs, y._limbs, acc._limbs),
    )
end

@inline function _gemm_mulacc(
    acc::MultiFloatVec{4,Float64,2},
    x::MultiFloatVec{4,Float64,2},
    y::MultiFloatVec{4,Float64,2},
)
    return mulacc_x2(acc, x, y)
end

# Deliberately no `_structured_mulacc` x2 method: the default structured
# kernels (SYRK/GEMMT) certify bitwise-identical reduction order, and this
# network is only operand-relative. x2 fusion is reachable exclusively through
# the explicit `gemm_strategy=:fused` opt-in on measured platforms.
