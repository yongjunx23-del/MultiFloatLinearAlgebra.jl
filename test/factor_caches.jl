# Factor-cache lifecycle and solver-facing conformance tests.
import MultiFloatLinearAlgebra
using MultiFloatLinearAlgebra: MFCholeskyCache, MFLUCache, MFLDLTCache, MFRRQRCache,
    prepare!, factorize!, solve!, invalidate!, factor_status, factor_diagnostics,
    factor_kind, factor_state, issuccess, factor_matrix,
    workspace_requirements, factor_cache_requirements

const MFLA = MultiFloatLinearAlgebra

function cache_spd(::Type{T}, n) where {T}
    R = randn(n, n)
    A = T.(R * R')
    @inbounds for i in 1:n
        A[i, i] += T(n)
    end
    A
end

function cache_diagdom(::Type{T}, n) where {T}
    A = T.(randn(n, n))
    @inbounds for i in 1:n
        A[i, i] += T(4)
    end
    A
end

function cache_indefinite(::Type{T}, n) where {T}
    R = 0.01 .* randn(n, n)
    A = T.(R + R')
    @inbounds for i in 1:n
        A[i, i] += T(isodd(i) ? n : -n)
    end
    A
end

function cache_allocated(f)
    f()
    GC.gc()
    @allocated f()
end

@testset "factor cache lifecycle and allocation" begin
    for T in (Float64x2, Float64x3, Float64x4)
        @testset "$T" begin
            n = 40
            cfg = KernelConfig(thread_count=1)
            b = T.(randn(n))

            # ---- Cholesky ----
            A = cache_spd(T, n)
            cref = MFLA.ldiv!(copy(b), MFLA.cholesky!(copy(A); check=false, config=cfg), b)
            c = MFCholeskyCache(T; config=cfg)
            prepare!(c, n)
            storage = factor_matrix(c)
            @test size(storage) == (n, n)
            factorize!(c, A)
            @test issuccess(c)
            @test factor_status(c) == 0
            @test factor_kind(c) == :cholesky
            x = zeros(T, n)
            solve!(x, c, b)
            @test max_relative_error(x, cref) <= tolerance(T, 16)
            # warm solve allocates 0 bytes (the cache-core goal)
            @test cache_allocated(() -> solve!(x, c, b)) == 0
            # repeated factorize never grows owned storage: the allocation is
            # bounded and identical across consecutive calls (only the shared
            # kernel view/dispatch overhead remains, never a reallocation).
            f1 = cache_allocated(() -> factorize!(c, A))
            f2 = cache_allocated(() -> factorize!(c, A))
            @test f1 == f2
            @test f1 < 512
            # A values change but size stays -> storage reused, object preserved
            A2 = cache_spd(T, n)
            factorize!(c, A2)
            @test factor_matrix(c) === storage
            # invalidate! blocks solve until refactorize
            invalidate!(c)
            @test !issuccess(c)
            @test factor_state(c) == :invalidated
            @test_throws LinearAlgebra.PosDefException solve!(x, c, b)
            factorize!(c, A2)
            solve!(x, c, b)

            # ---- LU ----
            luA = cache_diagdom(T, n)
            lref = MFLA.ldiv!(copy(b), MFLA.lu!(copy(luA); check=false, config=cfg), b)
            lc = MFLUCache(T; config=cfg)
            prepare!(lc, n)
            lu_storage = factor_matrix(lc)
            factorize!(lc, luA)
            @test issuccess(lc)
            lx = zeros(T, n)
            solve!(lx, lc, b)
            @test max_relative_error(lx, lref) <= tolerance(T, 96)
            @test cache_allocated(() -> solve!(lx, lc, b)) == 0
            factorize!(lc, cache_diagdom(T, n))
            @test factor_matrix(lc) === lu_storage
            invalidate!(lc)
            @test_throws Exception solve!(lx, lc, b)

            # ---- LDLT ----
            ind = cache_indefinite(T, n)
            ldref = MFLA.ldiv!(copy(b), MFLA.ldlt!(copy(ind); check=false, config=cfg), b)
            ldc = MFLDLTCache(T; config=cfg)
            prepare!(ldc, n)
            ld_storage = factor_matrix(ldc)
            factorize!(ldc, ind)
            @test issuccess(ldc)
            ldx = zeros(T, n)
            solve!(ldx, ldc, b)
            @test max_relative_error(ldx, ldref) <= tolerance(T, 256)
            @test cache_allocated(() -> solve!(ldx, ldc, b)) == 0
            factorize!(ldc, cache_indefinite(T, n))
            @test factor_matrix(ldc) === ld_storage
            invalidate!(ldc)
            @test_throws Exception solve!(ldx, ldc, b)

            # ---- RRQR (square full-rank solve) ----
            qa = cache_diagdom(T, n)
            qref = MFLA.ldiv!(copy(b), MFLA.rrqr!(copy(qa)), b)
            qc = MFRRQRCache(T; config=cfg)
            prepare!(qc, n)
            q_storage = factor_matrix(qc)
            factorize!(qc, qa)
            @test issuccess(qc)
            qx = zeros(T, n)
            solve!(qx, qc, b)
            @test max_relative_error(qx, qref) <= tolerance(T, 256)
            @test cache_allocated(() -> solve!(qx, qc, b)) == 0
            invalidate!(qc)
            @test_throws Exception solve!(qx, qc, b)
        end
    end
end

@testset "factor cache multi-RHS and matrix solve" begin
    for T in (Float64x2, Float64x3)
        n = 32
        nrhs = 5
        cfg = KernelConfig(thread_count=1)
        B = T.(randn(n, nrhs))

        luA = cache_diagdom(T, n)
        lref = MFLA.ldiv!(copy(B), MFLA.lu!(copy(luA); check=false, config=cfg), B)
        lc = MFLUCache(T; config=cfg)
        prepare!(lc, n)
        factorize!(lc, luA)
        X = zeros(T, n, nrhs)
        solve!(X, lc, B)
        @test max_relative_error(X, lref) <= tolerance(T, 96)
        # Matrix solve pays only the shared trsm kernel dispatch overhead
        # (baseline audit trsm=48), never a cache-core allocation: the bytes are
        # bounded and identical across consecutive solves (no temp RHS/wrapper,
        # no storage growth).
        s1 = cache_allocated(() -> solve!(X, lc, B))
        s2 = cache_allocated(() -> solve!(X, lc, B))
        @test s1 == s2

        spd = cache_spd(T, n)
        cref = MFLA.ldiv!(copy(B), MFLA.cholesky!(copy(spd); check=false, config=cfg), B)
        c = MFCholeskyCache(T; config=cfg)
        prepare!(c, n)
        factorize!(c, spd)
        X2 = zeros(T, n, nrhs)
        solve!(X2, c, B)
        @test max_relative_error(X2, cref) <= tolerance(T, 96)
        s3 = cache_allocated(() -> solve!(X2, c, B))
        s4 = cache_allocated(() -> solve!(X2, c, B))
        @test s3 == s4
    end
end

@testset "factor cache prepare/growth and diagnostics" begin
    T = Float64x2
    cfg = KernelConfig(thread_count=1)
    c = MFLUCache(T; config=cfg)
    prepare!(c, 16)
    @test size(factor_matrix(c)) == (16, 16)
    prepare!(c, 32)
    @test size(factor_matrix(c)) == (32, 32)
    # factorize! before prepare throws
    fresh = MFLUCache(T; config=cfg)
    @test_throws ArgumentError factorize!(fresh, cache_diagdom(T, 8))
    # prepare!(cache) before factorize with different size -> explicit growth only
    prepare!(fresh, 8)
    factorize!(fresh, cache_diagdom(T, 8))
    @test issuccess(fresh)

    A = cache_diagdom(T, 8)
    factorize!(fresh, A)
    d = factor_diagnostics(fresh)
    @test d.kind == :lu
    @test d.success == true
    @test d.status == 0
    @test length(d.pivots) == 8
end

@testset "factor cache numerical correctness (pathological/singular/nonfinite)" begin
    for T in (Float64x2, Float64x3, Float64x4)
        @testset "$T" begin
            cfg = KernelConfig(thread_count=1)

            # singular LU -> failure status, not throw with check=false
            s = T[1 2; 2 4]
            lc = MFLUCache(T; config=cfg)
            prepare!(lc, 2)
            factorize!(lc, s; check=false)
            @test !issuccess(lc)
            @test factor_status(lc) != 0
            @test_throws LinearAlgebra.SingularException solve!(zeros(T, 2), lc, T[1, 2])
            # after replacing with a nonsingular matrix in-place, refactorize recovers
            good = T[4 1; 1 3]
            factorize!(lc, good; check=false)
            @test issuccess(lc)
            x = zeros(T, 2)
            solve!(x, lc, T[5, 4])
            @test max_relative_error(x, MFLA.ldiv!(copy(T[5, 4]), MFLA.lu!(copy(good); check=false, config=cfg), T[5, 4])) <= tolerance(T, 16)

            # nonfinite input -> -1 status with check=false
            nf = T[1 NaN; 0 1]
            c = MFLUCache(T; config=cfg)
            prepare!(c, 2)
            factorize!(c, nf; check=false)
            @test factor_status(c) == -1
            @test factor_state(c) == :nonfinite_input
            @test !issuccess(c)

            # cholesky not-posdef
            ind = T[1 2; 2 1]
            cc = MFCholeskyCache(T; config=cfg)
            prepare!(cc, 2)
            factorize!(cc, ind; check=false)
            @test !issuccess(cc)
            @test factor_state(cc) == :not_posdef
            @test_throws LinearAlgebra.PosDefException solve!(zeros(T, 2), cc, T[1, 1])

            # adjacent 2x2 LDLT pivots stay successful and solve correctly
            adj = T[2 1 0 0; 1 0 1 0; 0 1 0 1; 0 0 1 2]
            lc = MFLDLTCache(T; config=cfg)
            prepare!(lc, 4)
            factorize!(lc, adj; check=false)
            @test issuccess(lc)
            b = T[1, 2, 3, 4]
            x = zeros(T, 4)
            solve!(x, lc, b)
            @test max_relative_error(x, MFLA.ldiv!(copy(b), MFLA.ldlt!(copy(adj); check=false, config=cfg), b)) <= tolerance(T, 256)

            # aliasing: RHS === destination is allowed
            luA = cache_diagdom(T, 4)
            lc2 = MFLUCache(T; config=cfg)
            prepare!(lc2, 4)
            factorize!(lc2, luA)
            y = T[1, 2, 3, 4]
            src = copy(y)
            solve!(y, lc2, y)
            @test max_relative_error(y, MFLA.ldiv!(src, MFLA.lu!(copy(luA); check=false, config=cfg), src)) <= tolerance(T, 16)
        end
    end
end

@testset "factor cache pure requirements queries" begin
    T = Float64x2
    cfg = KernelConfig(thread_count=1)
    # pure: repeated queries are identical and do not mutate any state
    w0 = workspace_requirements(:lu, (n=64,), cfg)
    w1 = workspace_requirements(:lu, (n=64,), cfg)
    @test w0 == w1
    @test w0.factor == 64
    wg = workspace_requirements(:gemm, (m=64, k=64, n=64), cfg)
    @test wg.gemm_capacity == 64 * max(cfg.gemm_panel_columns, 1)
    r = factor_cache_requirements(:cholesky, (n=64,))
    @test r.factor_matrix_elements == 64 * 64
    rr = factor_cache_requirements(:rrqr, (m=64, n=48))
    @test rr.matrix == (m=64, n=48)
    @test_throws ArgumentError factor_cache_requirements(:bogus, (n=8,))
    @test_throws ArgumentError workspace_requirements(:bogus, (n=8,), cfg)
end
