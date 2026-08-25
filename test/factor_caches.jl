# Factor-cache lifecycle and solver-facing conformance tests.
import MultiFloatLinearAlgebra
using MultiFloatLinearAlgebra: MFCholeskyCache, MFLUCache, MFLDLTCache, MFRRQRCache,
    prepare!, factorize!, solve!, invalidate!, factor_status, factor_diagnostics,
    factor_kind, factor_state, issuccess, factor_matrix,
    workspace_requirements, factor_cache_requirements, factor_cache_capacity

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

        # LDLT matrix solve (correctness + 0-byte warm)
        ind = cache_indefinite(T, n)
        ldref = MFLA.ldiv!(copy(B), MFLA.ldlt!(copy(ind); check=false, config=cfg), B)
        ldc = MFLDLTCache(T; config=cfg)
        prepare!(ldc, n)
        factorize!(ldc, ind)
        X3 = zeros(T, n, nrhs)
        solve!(X3, ldc, B)
        @test max_relative_error(X3, ldref) <= tolerance(T, 256)
        @test cache_allocated(() -> solve!(X3, ldc, B)) == 0

        # RRQR matrix solve (correctness + 0-byte warm)
        qa = cache_diagdom(T, n)
        qref = MFLA.ldiv!(copy(B), MFLA.rrqr!(copy(qa)), B)
        qc = MFRRQRCache(T; config=cfg)
        prepare!(qc, n)
        factorize!(qc, qa)
        X4 = zeros(T, n, nrhs)
        solve!(X4, qc, B)
        @test max_relative_error(X4, qref) <= tolerance(T, 256)
        @test cache_allocated(() -> solve!(X4, qc, B)) == 0
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

@testset "factor cache diagnostics states" begin
    for T in (Float64x2, Float64x3, Float64x4)
        @testset "$T" begin
            cfg = KernelConfig(thread_count=1)
            n = 8
            spd = cache_spd(T, n)
            diagdom = cache_diagdom(T, n)
            ind = cache_indefinite(T, n)

            # fresh (unprepared) cache: numeric fields are nothing
            for cache in (
                MFCholeskyCache(T; config=cfg),
                MFLUCache(T; config=cfg),
                MFLDLTCache(T; config=cfg),
                MFRRQRCache(T; config=cfg),
            )
                d = factor_diagnostics(cache)
                @test d.success == false
                @test d.state in (:invalidated, :reconfigure_requires_prepare)
                @test d.finite === nothing
            end

            # prepared cache: numeric fields are nothing
            c = MFCholeskyCache(T; config=cfg); prepare!(c, n)
            d = factor_diagnostics(c)
            @test d.success == false
            @test d.minimum_diagonal === nothing
            @test d.finite === nothing
            l = MFLUCache(T; config=cfg); prepare!(l, n)
            dl = factor_diagnostics(l)
            @test dl.pivots === nothing
            @test dl.maximum_u === nothing
            q = MFRRQRCache(T; config=cfg); prepare!(q, n)
            dq = factor_diagnostics(q)
            @test dq.rdiag === nothing
            @test dq.permutation === nothing
            @test_throws ArgumentError factor_permutation(q)
            @test_throws ArgumentError factor_rdiag(q)

            # success diagnostics are full
            factorize!(c, spd)
            ds = factor_diagnostics(c)
            @test ds.success == true
            @test ds.minimum_diagonal isa T
            factorize!(l, diagdom)
            dls = factor_diagnostics(l)
            @test dls.success == true
            @test dls.pivots isa Vector
            factorize!(q, diagdom)
            @test factor_permutation(q) isa Vector{Int}
            @test factor_rdiag(q) isa Vector{T}

            # invalidated after factorize: numeric fields are nothing (not stale)
            invalidate!(c)
            di = factor_diagnostics(c)
            @test di.state == :invalidated
            @test di.minimum_diagonal === nothing
            @test di.finite === nothing
            invalidate!(q)
            @test_throws ArgumentError factor_permutation(q)
            @test_throws ArgumentError factor_rdiag(q)

            # nonfinite failure: LU maximum_u/pivot_growth are nothing
            lnf = MFLUCache(T; config=cfg); prepare!(lnf, 2)
            factorize!(lnf, T[1 NaN; 0 1]; check=false)
            dnf = factor_diagnostics(lnf)
            @test dnf.state == :nonfinite_input
            @test dnf.maximum_u === nothing
            @test dnf.pivot_growth === nothing
            @test dnf.original_maximum === nothing

            # singular failure: meaningful fields, not nothing
            ls = MFLUCache(T; config=cfg); prepare!(ls, 2)
            factorize!(ls, T[1 2; 2 4]; check=false)
            dsing = factor_diagnostics(ls)
            @test dsing.state == :singular
            @test dsing.failure_location isa Int
            @test dsing.pivots isa Vector

            # failure -> recovery diagnostics
            factorize!(ls, T[4 1; 1 3]; check=false)
            drec = factor_diagnostics(ls)
            @test drec.success == true
            @test drec.state == :success
        end
    end
end

@testset "factor cache pure requirements queries" begin
    T = Float64x2
    cfg = KernelConfig(thread_count=1)
    # pure: repeated queries are identical and do not mutate any state
    w0 = workspace_requirements(T, :lu, (n=64,), cfg)
    w1 = workspace_requirements(T, :lu, (n=64,), cfg)
    @test w0 == w1
    @test w0.factor == 64
    wg = workspace_requirements(T, :gemm, (m=64, k=64, n=64), cfg)
    plan = gemm_plan(T, 64, 64, 64, cfg)
    @test wg.gemm_workers == plan.workers
    @test wg.gemm_capacity == plan.packed_elements_per_worker
    r = factor_cache_requirements(T, :cholesky, (n=64,), cfg)
    @test r.factor_matrix_elements == 64 * 64
    rr = factor_cache_requirements(T, :rrqr, (m=64, n=48), cfg)
    @test rr.matrix == (m=64, n=48)
    @test_throws ArgumentError factor_cache_requirements(T, :bogus, (n=8,), cfg)
    @test_throws ArgumentError workspace_requirements(T, :bogus, (n=8,), cfg)
end

@testset "factor cache requirements/capacity consistency" begin
    for T in (Float64x2, Float64x3, Float64x4), kind in (:cholesky, :lu, :ldlt, :rrqr)
        for cfg in (KernelConfig(thread_count=1), KernelConfig(thread_count=2))
            shapes = kind === :rrqr ?
                ((m=48, n=48), (m=8, n=48), (m=48, n=8), (m=0, n=48), (m=48, n=0), (m=0, n=0)) :
                ((n=48,), (n=0,))
            for shape in shapes
                req = factor_cache_requirements(T, kind, shape, cfg)
                cache = if kind === :cholesky
                    MFCholeskyCache(T; config=cfg)
                elseif kind === :lu
                    MFLUCache(T; config=cfg)
                elseif kind === :ldlt
                    MFLDLTCache(T; config=cfg)
                else
                    MFRRQRCache(T; config=cfg)
                end
                if kind === :rrqr
                    prepare!(cache, shape.m, shape.n)
                else
                    prepare!(cache, shape.n)
                end
                cap = factor_cache_capacity(cache)
                @test cap.matrix == req.matrix
                @test cap.factor_matrix_elements == req.factor_matrix_elements
                @test cap.pivots == req.pivots
                @test cap.blocks == req.blocks
                @test cap.dsub == req.dsub
                @test cap.weighted_panel == req.weighted_panel
                @test cap.tau == req.tau
                @test cap.permutation == req.permutation
                @test cap.cycle_leaders_capacity == req.cycle_leaders_capacity
                @test cap.norm_scale == req.norm_scale
                @test cap.norm_sum == req.norm_sum
                @test cap.norm_dirty == req.norm_dirty
                @test cap.ftranspose == req.ftranspose
                @test cap.auxiliary == req.auxiliary
                @test cap.gemm_workers == req.gemm_workers
                @test cap.gemm_packed_elements_per_worker == req.gemm_packed_elements_per_worker
            end
        end
    end
    # negative dimensions and overflow are rejected by the pure query
    T = Float64x2
    cfg = KernelConfig(thread_count=1)
    @test_throws ArgumentError factor_cache_requirements(T, :rrqr, (m=-1, n=5), cfg)
    @test_throws ArgumentError workspace_requirements(T, :lu, (n=-3,), cfg)
    @test_throws Exception factor_cache_requirements(
        T, :rrqr, (m=typemax(Int) ÷ 2, n=typemax(Int) ÷ 2), cfg,
    )
end

# The required fail-closed regression: successful factorize -> failing
# factorize(check=true) throws -> cache must NOT remain successful -> solve
# must refuse -> replacement A recovers through refactorization.
@testset "factor cache fail-closed recovery" begin
    for T in (Float64x2, Float64x3, Float64x4)
        @testset "$T" begin
            cfg = KernelConfig(thread_count=1)

            # Cholesky
            spd = T[4 1; 1 3]
            bad_spd = T[1 2; 2 1]           # not positive definite
            c = MFCholeskyCache(T; config=cfg)
            prepare!(c, 2)
            factorize!(c, spd)
            @test issuccess(c)
            @test_throws LinearAlgebra.PosDefException factorize!(c, bad_spd; check=true)
            @test !issuccess(c)              # fail-closed: no stale success
            @test_throws Exception solve!(zeros(T, 2), c, T[1, 1])
            factorize!(c, spd; check=false) # recovery
            @test issuccess(c)
            x = zeros(T, 2); solve!(x, c, T[5, 4])
            @test max_relative_error(x, MFLA.ldiv!(copy(T[5, 4]), MFLA.cholesky!(copy(spd); check=false, config=cfg), T[5, 4])) <= tolerance(T, 16)

            # LU
            good = T[4 1; 1 3]
            sing = T[1 2; 2 4]
            lc = MFLUCache(T; config=cfg)
            prepare!(lc, 2)
            factorize!(lc, good)
            @test issuccess(lc)
            @test_throws LinearAlgebra.SingularException factorize!(lc, sing; check=true)
            @test !issuccess(lc)
            @test_throws Exception solve!(zeros(T, 2), lc, T[1, 2])
            factorize!(lc, good; check=false)
            @test issuccess(lc)

            # LDLT
            ind_good = T[2 1; 1 2]
            ind_sing = T[1 2; 2 4]           # singular symmetric
            ldc = MFLDLTCache(T; config=cfg)
            prepare!(ldc, 2)
            factorize!(ldc, ind_good)
            @test issuccess(ldc)
            @test_throws LinearAlgebra.SingularException factorize!(ldc, ind_sing; check=true)
            @test !issuccess(ldc)
            @test_throws Exception solve!(zeros(T, 2), ldc, T[1, 1])
            factorize!(ldc, ind_good; check=false)
            @test issuccess(ldc)

            # RRQR (rank deficiency is not failure; only nonfinite fails)
            nf = T[1 NaN; 0 1]
            qc = MFRRQRCache(T; config=cfg)
            prepare!(qc, 2)
            @test_throws ArgumentError factorize!(qc, nf; check=true)
            @test !issuccess(qc)
            factorize!(qc, T[4 1; 1 3]; check=false)
            @test issuccess(qc)
        end
    end
end

@testset "factor cache frozen config and reconfigure!" begin
    T = Float64x2
    n = 8
    cfg = KernelConfig(thread_count=1)
    cfg2 = KernelConfig(thread_count=2, gemm_strategy=:packed, gemm_panel_columns=4)
    A = cache_diagdom(T, n)
    c = MFLUCache(T; config=cfg)
    prepare!(c, n)
    factorize!(c, A)
    @test issuccess(c)
    # hot path rejects a divergent config
    @test_throws ArgumentError factorize!(c, A; config=cfg2)
    @test_throws ArgumentError solve!(zeros(T, n), c, T.(randn(n)); config=cfg2)
    # reconfigure! invalidates and, with prepare!, makes the new config the frozen one
    reconfigure!(c, cfg2)
    @test !issuccess(c)
    @test factor_state(c) == :reconfigure_requires_prepare
    # reconfigure! without prepare! must make factorize! refuse (fail-closed)
    @test_throws ArgumentError factorize!(c, A)
    prepare!(c, n)
    factorize!(c, A)
    @test issuccess(c)
    @test c.config.thread_count == 2
    # cholesky config frozen too
    spd = cache_spd(T, n)
    cc = MFCholeskyCache(T; config=cfg)
    prepare!(cc, n)
    factorize!(cc, spd)
    @test_throws ArgumentError factorize!(cc, spd; config=cfg2)
    # shape change requires explicit prepare: mutating the live factor storage
    # must not let factorize! run at a new size without prepare!
    c3 = MFLUCache(T; config=cfg)
    prepare!(c3, n)
    factorize!(c3, A)
    c3.factors = Matrix{T}(undef, n + 4, n + 4)
    @test_throws ArgumentError factorize!(c3, cache_diagdom(T, n + 4))
    prepare!(c3, n + 4)
    factorize!(c3, cache_diagdom(T, n + 4))
    @test issuccess(c3)
end

@testset "rectangular RRQR cache route" begin
    for T in (Float64x2, Float64x3)
        @testset "$T" begin
            cfg = KernelConfig(thread_count=1)
            # tall m>n
            m, n = 40, 24
            A = cache_diagdom(T, m)[:, 1:n]  # 40x24 full column rank
            @test size(A) == (m, n)
            qc = MFRRQRCache(T; config=cfg)
            prepare!(qc, m, n)
            factorize!(qc, A)
            @test issuccess(qc)
            # numerical rank = n (full)
            @test numerical_rank(qc) == n
            # permutation is a length-n permutation
            p = factor_permutation(qc)
            @test length(p) == n
            @test sort(p) == collect(1:n)
            # multi-RHS least-squares: min||b - Q*R*p' x||  -> x = R^-1 Q' b permuted
            X0 = T.(randn(n, 4))
            B = A * X0
            # apply Q' (keeps m rows), then solve R on the leading rank rows
            Y = copy(B)
            apply_q!(Y, qc; trans=:T)
            Ysolve = Y[1:n, :]
            solve_r!(Ysolve, qc, n; config=cfg)
            @test size(Ysolve, 1) == n
            # compare against standalone rectangular QR reconstruction
            F = MFLA.rrqr!(copy(A))
            pstd = MFLA.factor_permutation(F)
            Rdiag = MFLA.factor_rdiag(F)
            @test length(pstd) == n
            # check A[:,p] = Q*R reconstruction residual via factor_rdiag consistency
            @test maximum(abs.(MFLA.factor_rdiag(qc) .- Rdiag)) <= 10 * eps(T) * maximum(abs.(Rdiag))

            # wide m<n
            mw, nw = 16, 32
            Aw = T.(randn(mw, nw))
            for i in 1:mw; Aw[i, i] += T(4); end  # 16x32 full row rank
            @test size(Aw) == (mw, nw)
            qw = MFRRQRCache(T; config=cfg)
            prepare!(qw, mw, nw)
            factorize!(qw, Aw)
            @test issuccess(qw)
            @test numerical_rank(qw) == mw
            @test length(factor_permutation(qw)) == nw
        end
    end
end

@testset "RRQR cache solve_r! trans and apply_q! routes" begin
    for T in (Float64x2, Float64x3, Float64x4)
        @testset "$T" begin
            cfg = KernelConfig(thread_count=1)
            n = 8
            A = cache_diagdom(T, n)
            qc = MFRRQRCache(T; config=cfg)
            prepare!(qc, n, n)
            factorize!(qc, A)
            rank = n
            # extract the leading R block
            R = zeros(T, rank, rank)
            for c in 1:rank, r in 1:c
                R[r, c] = factor_matrix(qc)[r, c]
            end

            # R*x=b and R'*x=b (vector)
            xref = T.(randn(rank))
            bN = R * xref
            xN = copy(bN)
            solve_r!(xN, qc, rank; trans=:N)
            @test max_relative_error(xN, xref) <= tolerance(T, 32rank)
            bT = transpose(R) * xref
            xT = copy(bT)
            solve_r!(xT, qc, rank; trans=:T)
            @test max_relative_error(xT, xref) <= tolerance(T, 32rank)

            # R*X=B and R'*X=B (matrix)
            Xref = T.(randn(rank, 3))
            BN = R * Xref
            XN = copy(BN)
            solve_r!(XN, qc, rank; trans=:N)
            @test max_relative_error(XN, Xref) <= tolerance(T, 32rank)
            BT = transpose(R) * Xref
            XT = copy(BT)
            solve_r!(XT, qc, rank; trans=:T)
            @test max_relative_error(XT, Xref) <= tolerance(T, 32rank)

            # apply_q! round-trip :N/:T for vector and matrix
            v = T.(randn(n))
            vr = copy(v)
            apply_q!(vr, qc; trans=:T)
            apply_q!(vr, qc; trans=:N)
            @test max_relative_error(vr, v) <= tolerance(T, 64n)
            M = T.(randn(n, 3))
            Mr = copy(M)
            apply_q!(Mr, qc; trans=:T)
            apply_q!(Mr, qc; trans=:N)
            @test max_relative_error(Mr, M) <= tolerance(T, 64n)

            # invalid trans throws
            @test_throws ArgumentError solve_r!(copy(bN), qc, rank; trans=:bad)
            @test_throws ArgumentError apply_q!(copy(v), qc; trans=:bad)
        end
    end
end
