const MULACC_X3_PROOF_VECTORS = (
    (
        name=:ordinary,
        acc=(0x3ff0000000000000, 0x3c90000000000000, 0x3930000000000000),
        x=(0x3fe8000000000000, 0xbc80000000000000, 0x3910000000000000),
        y=(0xbff4000000000000, 0x3c80000000000000, 0xb910000000000000),
        output=(0x3fb0000000000008, 0x3910000000000000, 0x0000000000000000),
    ),
    (
        name=:exact_cancel,
        acc=(0xbff0000000000000, 0xbc90000000000000, 0xb930000000000000),
        x=(0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000),
        y=(0x3ff0000000000000, 0x3c90000000000000, 0x3930000000000000),
        output=(0x0000000000000000, 0x0000000000000000, 0x0000000000000000),
    ),
    (
        name=:wide_cancel,
        acc=(0xc310000000000000, 0x3f50000000000000, 0x3be0000000000000),
        x=(0x58f0000000000000, 0x5530000000000000, 0x5170000000000000),
        y=(0x2a10000000000000, 0xa650000000000000, 0x2290000000000000),
        output=(0x3f50000000000000, 0x3be0800000000000, 0x0000000000000000),
    ),
    (
        name=:near_underflow,
        acc=(0x8170000000000000, 0x0000000000004000, 0x0000000000000000),
        x=(0x20b0000000000000, 0x1cf0000000000000, 0x1930000000000000),
        y=(0x20b0000000000000, 0x9cf0000000000000, 0x1930000000000000),
        output=(0x0000000000004000, 0x0000000000000000, 0x0000000000000000),
    ),
    (
        name=:near_overflow,
        acc=(0xfe70000000000000, 0x7ab0000000000000, 0x76f0000000000000),
        x=(0x5f30000000000000, 0x5b70000000000000, 0x57b0000000000000),
        y=(0x5f30000000000000, 0xdb70000000000000, 0x57b0000000000000),
        output=(0x7ab0000000000000, 0x7700000000000000, 0x0000000000000000),
    ),
    (
        name=:alternating,
        acc=(0xc008000000000000, 0x3cafffffffffffff, 0x0000000000000000),
        x=(0xbff8000000000000, 0x3c90000000000000, 0x3920000000000000),
        y=(0x4000000000000000, 0xbca0000000000000, 0x3940000000000000),
        output=(0xc017ffffffffffff, 0xbcbc000000000001, 0x3950000000000000),
    ),
    (
        name=:tiny_accumulator,
        acc=(0x0030000000000004, 0x0000000000000000, 0x0000000000000000),
        x=(0x3370000000000000, 0x2fb0000000000000, 0x2bf0000000000000),
        y=(0x8df0000000000000, 0x0a30000000000000, 0x8670000000000000),
        output=(0x816ffffe00000000, 0x0000000000000010, 0x0000000000000000),
    ),
    (
        name=:zero_product,
        acc=(0x0000000000000000, 0x0000000000000000, 0x0000000000000000),
        x=(0x0000000000000000, 0x0000000000000000, 0x0000000000000000),
        y=(0x4090000000000000, 0x0000000000000000, 0x0000000000000000),
        output=(0x0000000000000000, 0x0000000000000000, 0x0000000000000000),
    ),
)

@testset "mulacc_x3 fixed proof vectors" begin
    T = Float64x3
    V = MultiFloatVec{4,Float64,3}
    from_bits(bits) = T(ntuple(index -> reinterpret(Float64, bits[index]), 3))
    limb_bits(value) = ntuple(
        index -> reinterpret(UInt64, value._limbs[index]), 3,
    )

    @test length(MULACC_X3_PROOF_VECTORS) % 4 == 0
    for first_index in 1:4:length(MULACC_X3_PROOF_VECTORS)
        cases = MULACC_X3_PROOF_VECTORS[first_index:(first_index + 3)]
        for permutation in ((1, 2, 3, 4), (4, 2, 1, 3))
            ordered = ntuple(index -> cases[permutation[index]], 4)
            scalar_acc = map(case -> from_bits(case.acc), ordered)
            scalar_x = map(case -> from_bits(case.x), ordered)
            scalar_y = map(case -> from_bits(case.y), ordered)

            @test all(MultiFloats.isnormalized, scalar_acc)
            @test all(MultiFloats.isnormalized, scalar_x)
            @test all(MultiFloats.isnormalized, scalar_y)

            acc = V(scalar_acc...)
            x = V(scalar_x...)
            y = V(scalar_y...)
            fused = MultiFloatLinearAlgebra.mulacc_x3(acc, x, y)
            standard = acc + x * y

            for lane in 1:4
                @test limb_bits(fused[lane]) == ordered[lane].output
                @test limb_bits(fused[lane]) == limb_bits(standard[lane])
                @test MultiFloats.isnormalized(fused[lane])
                setprecision(BigFloat, 512) do
                    exact = BigFloat(scalar_acc[lane]) +
                        BigFloat(scalar_x[lane]) * BigFloat(scalar_y[lane])
                    @test abs(BigFloat(fused[lane]) - exact) ==
                        abs(BigFloat(standard[lane]) - exact)
                end
            end
        end
    end
end
