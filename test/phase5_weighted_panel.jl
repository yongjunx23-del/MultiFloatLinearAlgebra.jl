@testset "LDLT weighted-panel independent-row bit identity" begin
    T = Float64x4
    n = 1000
    panel_first, panel_last = 1, 64
    A = Matrix{T}(undef, n, n)
    @inbounds for column in 1:n, row in column:n
        value = T(sin(0.011 * row + 0.007 * column))
        A[row, column] = value
        A[column, row] = value
    end
    dsub = zeros(T, n)
    blocks = zeros(UInt8, n)
    @inbounds for index in panel_first:panel_last
        blocks[index] = UInt8(1)
        dsub[index] = T(0.25 + 0.001 * index)
    end
    serial = zeros(T, n - panel_last, panel_last + 1)
    threaded = similar(serial)
    MultiFloatLinearAlgebra._build_ldlt_weighted_panel!(
        serial, A, panel_first, panel_last, dsub, blocks,
        KernelConfig(thread_count=1),
    )
    MultiFloatLinearAlgebra._build_ldlt_weighted_panel!(
        threaded, A, panel_first, panel_last, dsub, blocks,
        KernelConfig(thread_count=Threads.nthreads()),
    )
    @test serial == threaded
end
