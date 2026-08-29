# Fused x4 multiply-accumulate primitive.
#
# Mirrors `mulacc_x3`: the standard accumulation `acc += x * y` first fully
# normalizes the x4 product into four limbs and then feeds those limbs
# through the x4 add network.  `mulacc_x4` skips the product-only
# renormalization chain (the steps that operate only on the accumulated
# quadruple after all cross-weight mass has been absorbed) and feeds the
# partially compressed product prefix straight into the accumulator add
# network, shortening the dependency chain and removing redundant rounding.
#
# This changes intermediate rounding relative to `acc + x * y` in principle;
# adoption is gated on the adversarial differential validation (bitwise
# identity over random / wide-range / cancellation / alternating-sign /
# near-underflow / near-overflow / zero suites) plus the 512-bit BigFloat
# differential showing no accuracy degradation.  As with the x3 kernel this
# is a deterministic optimization, not a universal identity.

@inline function _product_prefix_x4(
    x::MultiFloatVec{4,Float64,4}, y::MultiFloatVec{4,Float64,4},
)
    x0, x1, x2, x3 = x._limbs
    y0, y1, y2, y3 = y._limbs
    p00, e00 = MultiFloats.two_prod(x0, y0)
    p01, e01 = MultiFloats.two_prod(x0, y1)
    p10, e10 = MultiFloats.two_prod(x1, y0)
    p02, e02 = MultiFloats.two_prod(x0, y2)
    p11, e11 = MultiFloats.two_prod(x1, y1)
    p20, e20 = MultiFloats.two_prod(x2, y0)
    p03 = MultiFloats.one_prod(x0, y3)
    p12 = MultiFloats.one_prod(x1, y2)
    p21 = MultiFloats.one_prod(x2, y1)
    p30 = MultiFloats.one_prod(x3, y0)
    p01, p10 = MultiFloats.two_sum(p01, p10)
    e01, e10 = MultiFloats.two_sum(e01, e10)
    p02, p20 = MultiFloats.two_sum(p02, p20)
    e02 += e20
    p03 += p30
    p12 += p21
    e00, p01 = MultiFloats.two_sum(e00, p01)
    e01, p11 = MultiFloats.two_sum(e01, p11)
    e10 += e02
    p20 += e11
    p03 += p12
    p00, e00 = MultiFloats.fast_two_sum(p00, e00)
    p01, p10 = MultiFloats.fast_two_sum(p01, p10)
    e01, p02 = MultiFloats.two_sum(e01, p02)
    e10 += p03
    p11 += p20
    p01, e01 = MultiFloats.two_sum(p01, e01)
    p10 += p11
    e10 += p02
    p10 += e01
    p01, p10 = MultiFloats.two_sum(p01, p10)
    e00, p01 = MultiFloats.two_sum(e00, p01)
    # ---- fusion cut: everything below operates only on the accumulated
    # quadruple after the last cross-weight absorption. ----
    p10 += e10
    return p00, e00, p01, p10
end

@inline function _raw_mfadd4(x0, x1, x2, x3, y0, y1, y2, y3)
    a, b = MultiFloats.two_sum(x0, y0)
    c, d = MultiFloats.two_sum(x1, y1)
    e, f = MultiFloats.two_sum(x2, y2)
    g, h = MultiFloats.two_sum(x3, y3)
    a, c = MultiFloats.fast_two_sum(a, c)
    b += h
    d, e = MultiFloats.two_sum(d, e)
    f, g = MultiFloats.two_sum(f, g)
    b, g = MultiFloats.two_sum(b, g)
    c, d = MultiFloats.fast_two_sum(c, d)
    e, f = MultiFloats.two_sum(e, f)
    a, c = MultiFloats.fast_two_sum(a, c)
    d, e = MultiFloats.fast_two_sum(d, e)
    b, d = MultiFloats.two_sum(b, d)
    c, g = MultiFloats.fast_two_sum(c, g)
    e += f
    b, c = MultiFloats.two_sum(b, c)
    d, e = MultiFloats.two_sum(d, e)
    a, b = MultiFloats.fast_two_sum(a, b)
    c, d = MultiFloats.two_sum(c, d)
    e += g
    b, c = MultiFloats.fast_two_sum(b, c)
    d, e = MultiFloats.two_sum(d, e)
    a, b = MultiFloats.fast_two_sum(a, b)
    c, d = MultiFloats.fast_two_sum(c, d)
    b, c = MultiFloats.fast_two_sum(b, c)
    d += e
    a, b = MultiFloats.fast_two_sum(a, b)
    c, d = MultiFloats.fast_two_sum(c, d)
    b, c = MultiFloats.fast_two_sum(b, c)
    c, d = MultiFloats.fast_two_sum(c, d)
    return MultiFloatVec{4,Float64,4}((a, b, c, d))
end

@inline function mulacc_x4(
    acc::MultiFloatVec{4,Float64,4},
    x::MultiFloatVec{4,Float64,4},
    y::MultiFloatVec{4,Float64,4},
)
    p0, p1, p2, p3 = _product_prefix_x4(x, y)
    p2, p3 = MultiFloats.two_sum(p2, p3)
    a0, a1, a2, a3 = acc._limbs
    return _raw_mfadd4(a0, a1, a2, a3, p0, p1, p2, p3)
end

