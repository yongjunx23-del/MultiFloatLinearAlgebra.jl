@testset "TRSM independent-RHS bit identity" begin
    T = Float64x4
    n = 96
    nrhs = 64
    lower = zeros(T, n, n)
    @inbounds for column in 1:n, row in column:n
        lower[row, column] = row == column ? T(2.0 + 0.001 * row) :
            T(0.002 * sin(0.013 * row + 0.017 * column))
    end
    rhs = Matrix{T}(undef, n, nrhs)
    @inbounds for column in 1:nrhs, row in 1:n
        rhs[row, column] = T(sin(0.021 * row + 0.11 * column))
    end

    serial = copy(rhs)
    threaded = copy(rhs)
    serial_config = KernelConfig(thread_count=1, column_tile=16)
    threaded_config = KernelConfig(thread_count=Threads.nthreads(), column_tile=16)
    trsm!(serial, lower; side=:left, uplo=:lower, trans=:N,
        diag=:nonunit, config=serial_config)
    trsm!(threaded, lower; side=:left, uplo=:lower, trans=:N,
        diag=:nonunit, config=threaded_config)
    @test serial == threaded

    rhs_right = transpose(rhs)
    serial_right = Matrix(rhs_right)
    threaded_right = Matrix(rhs_right)
    trsm!(serial_right, lower; side=:right, uplo=:lower, trans=:N,
        diag=:nonunit, config=serial_config)
    trsm!(threaded_right, lower; side=:right, uplo=:lower, trans=:N,
        diag=:nonunit, config=threaded_config)
    @test serial_right == threaded_right
end
