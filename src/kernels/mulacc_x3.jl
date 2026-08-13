# Fused x3 multiply-accumulate primitive.
#
# The standard accumulation `acc += x * y` first fully normalizes the x3
# product into three limbs and then feeds those limbs through the x3 add
# network. `mulacc_x3` skips the final product-only compression EFTs and feeds
# the partially compressed product prefix straight into the accumulator add
# network, shortening the dependency chain and removing redundant rounding.
#
# This changes intermediate rounding relative to `acc + x * y` in principle,
# but the fused network has been empirically bitwise-identical to the reference
# over the committed adversarial validation suite (random / wide-range /
# cancellation / alternating-sign / near-underflow / near-overflow / zero), and
# a 512-bit BigFloat differential shows no degradation (error ratio 1.000).
# This is a deterministic optimization, not a universal identity: a formal
# proof of bitwise equivalence across the EFT network would be required before
# claiming mathematical identity.

@inline function _product_prefix_x3(x::MultiFloatVec{4,Float64,3}, y::MultiFloatVec{4,Float64,3})
    x0, x1, x2 = x._limbs
    y0, y1, y2 = y._limbs
    p00, e00 = MultiFloats.two_prod(x0, y0)
    p01, e01 = MultiFloats.two_prod(x0, y1)
    p10, e10 = MultiFloats.two_prod(x1, y0)
    p02 = MultiFloats.one_prod(x0, y2)
    p11 = MultiFloats.one_prod(x1, y1)
    p20 = MultiFloats.one_prod(x2, y0)
    p01, p10 = MultiFloats.two_sum(p01, p10)
    e01 += e10
    p02 += p20
    e00, p01 = MultiFloats.two_sum(e00, p01)
    p02 += p11
    p00, e00 = MultiFloats.fast_two_sum(p00, e00)
    p01 += p10
    e01 += p02
    p01 += e01
    return p00, e00, p01
end

@inline function _raw_mfadd3(x0, x1, x2, y0, y1, y2)
    a, b = MultiFloats.two_sum(x0, y0)
    c, d = MultiFloats.two_sum(x1, y1)
    e, f = MultiFloats.two_sum(x2, y2)
    a, c = MultiFloats.fast_two_sum(a, c)
    b += f
    d, e = MultiFloats.two_sum(d, e)
    a, d = MultiFloats.fast_two_sum(a, d)
    b, c = MultiFloats.two_sum(b, c)
    c += e
    c, d = MultiFloats.two_sum(c, d)
    b, c = MultiFloats.two_sum(b, c)
    a, b = MultiFloats.fast_two_sum(a, b)
    c += d
    b, c = MultiFloats.fast_two_sum(b, c)
    a, b = MultiFloats.fast_two_sum(a, b)
    b, c = MultiFloats.fast_two_sum(b, c)
    return MultiFloatVec{4,Float64,3}((a, b, c))
end

@inline function mulacc_x3(acc::MultiFloatVec{4,Float64,3}, x::MultiFloatVec{4,Float64,3}, y::MultiFloatVec{4,Float64,3})
    p0, p1, p2 = _product_prefix_x3(x, y)
    p1, p2 = MultiFloats.two_sum(p1, p2)
    a0, a1, a2 = acc._limbs
    return _raw_mfadd3(a0, a1, a2, p0, p1, p2)
end

_supports_fused_mulacc(::Type{MultiFloat{Float64,3}}) = true
_supports_fused_mulacc(::Type{<:MultiFloat}) = false
