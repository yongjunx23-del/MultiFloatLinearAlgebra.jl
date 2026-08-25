# independent_accuracy.jl
# -----------------------------------------------------------------------------
# Independent numerical validation of the MFLA factor-cache layer against a
# 512-bit BigFloat reference.
#
# Design goal: do NOT merely re-check the cache against the standalone factor
# (which shares the same numerical core). Every accuracy quantity is measured
# against an independent high-precision BigFloat reference solve / reconstruction
# computed directly in this validation script.
#
# Scope: Float64x2 / Float64x3 / Float64x4 x {Cholesky, LU, LDLT, RRQR}.
#
# Checks per type x kind:
#   * solution max relative error vs BigFloat reference (well-conditioned SPD
#     / diagonally-dominant matrices), vector and multi-RHS;
#   * factor reconstruction residual
#       Cholesky: A ≈ L*L'
#       LU:       P*A ≈ L*U
#       LDLT:     A[p,p] ≈ L*D*L'
#       RRQR:     A[:,p] ≈ Q*R
#   * normwise backward error of the solved system;
#   * permutation correctness (LU ipiv valid row permutation; LDLT/RRQR
#     permutations valid; LDLT inertia computed);
#   * LU pivot growth (max|U| / max|A|);
#   * RRQR rank / reconstruction;
#   * pathological cases: singular LU/LDLT must report failure status;
#     rank-deficient RRQR (rank<n, still successful); badly scaled matrix;
#     nonfinite input (status -1);
#   * success -> failure (check=true throws) -> recovery;
#   * serial (1 thread) vs threaded (4 threads) bit-identical results.
#
# Prints PASS/FAIL per check and exits nonzero if any check fails.
# -----------------------------------------------------------------------------

using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra
using Printf
using Random

const MFLA = MultiFloatLinearAlgebra
const TYPES = (Float64x2, Float64x3, Float64x4)

# 512-bit independent reference arithmetic.
setprecision(BigFloat, 512)
const BF = BigFloat

Random.seed!(0xc0ffee9d)

# -----------------------------------------------------------------------------
# Reporting harness
# -----------------------------------------------------------------------------
const _FAILURES = Ref(0)
const _CHECKS = Ref(0)

function check(name::String, ok::Bool, detail::AbstractString="")
    _CHECKS[] += 1
    ok || (_FAILURES[] += 1)
    line = @sprintf("[%s] %s", ok ? "PASS" : "FAIL", name)
    isempty(detail) || (line *= "  (" * detail * ")")
    println(line)
    return ok
end

limb(::Type{MultiFloat{Float64,N}}) where {N} = N

# ---------------------------------------------------------------------------
# Tolerances. Relative quantities for well-conditioned problems come in near
# machine eps of the working type; we allow a large constant factor so the
# checks discriminate genuine numerical bugs while remaining robust.
# ---------------------------------------------------------------------------
tol_rel(::Type{T}) where {T} = 1.0e6 * eps(T)    # relative error / reconstruction
tol_back(::Type{T}) where {T} = 1.0e7 * eps(T)    # normwise backward error

# ---------------------------------------------------------------------------
# Matrix generators producing a BigFloat reference and its MF rounding.
# ---------------------------------------------------------------------------
# Well-conditioned SPD (Cholesky): A = G*G' + n*I.
function spd_matrices(::Type{T}, n) where {T}
    G = BF.(randn(BigFloat, n, n))
    Abig = G * G' + BF(n) * Matrix{BF}(I, n, n)
    return Abig, T.(Abig)
end

# Well-conditioned diagonally dominant (LU / RRQR): R + n*I.
function diagdom_matrices(::Type{T}, n) where {T}
    R = BF.(randn(BigFloat, n, n))
    Abig = R + BF(n) * Matrix{BF}(I, n, n)
    return Abig, T.(Abig)
end

# Well-conditioned symmetric indefinite with known inertia (LDLT).
# A = G' * Diagonal(lam) * G, with npos positive / nneg negative entries.
function indefinite_matrices(::Type{T}, n) where {T}
    npos = cld(n, 2)
    G = BF.(randn(BigFloat, n, n))
    lam = [BF(i <= npos ? 0.5 + rand() : -(0.5 + rand())) for i in 1:n]
    Abig = G' * Diagonal(lam) * G
    return Abig, T.(Abig), lam
end

# Badly scaled symmetric diagonally dominant matrix (LU / LDLT / RRQR
# robustness). Entries span ~10^0 .. 10^(4*(n-1)) in the diagonal and up to
# ~10^(8(n-1)) off the diagonal, staying inside the Float64 exponent range for
# n<=30. The matrix is ill-conditioned (diagonal range ~1e92), so the
# meaningful stability check is the normwise BACKWARD error, which must stay
# near eps regardless of conditioning.
function badly_scaled_matrices(::Type{T}, n) where {T}
    R = BF.(randn(BigFloat, n, n))
    Rs = (R + R') / 2
    D = Diagonal([BF(10.0)^(4 * (i - 1)) for i in 1:n])
    Abig = D * (Rs + BF(n) * Matrix{BF}(I, n, n)) * D
    return Abig, T.(Abig)
end

# ---------------------------------------------------------------------------
# Reference systems / error measures in BigFloat.
# ---------------------------------------------------------------------------
function reference_rhs(Abig, n, nrhs)
    xbig = BF.(randn(BigFloat, n))
    bbig = Abig * xbig
    nrhs == 1 && return xbig, bbig
    Xbig = BF.(randn(BigFloat, n, nrhs))
    Bbig = Abig * Xbig
    return Xbig, Bbig
end

_rel_err(x, xbig) = norm(x - xbig, Inf) / max(norm(xbig, Inf), eps(BF))

function _rel_err_vector(x, xbig)
    return _rel_err(BigFloat.(x), xbig)
end
function _rel_err_matrix(X, Xbig)
    return maximum(_rel_err(BigFloat.(X)[:, c], Xbig[:, c]) for c in axes(Xbig, 2))
end

# Normwise backward error: ||b - A x||_inf / (||A||_inf ||x||_inf + ||b||_inf).
function _backward_error(Abig, xbig::AbstractVector, bbig::AbstractVector)
    r = bbig - Abig * xbig
    den = norm(Abig, Inf) * norm(xbig, Inf) + norm(bbig, Inf)
    iszero(den) && return norm(r, Inf)
    return norm(r, Inf) / den
end
function _backward_error(Abig, Xbig::AbstractMatrix, Bbig::AbstractMatrix)
    best = 0.0
    for c in axes(Xbig, 2)
        best = max(best, _backward_error(Abig, Xbig[:, c], Bbig[:, c]))
    end
    return best
end

_is_permutation(p) = (length(unique(p)) == length(p)) && (sort(p) == collect(1:length(p)))

# Attempt a matrix (multi-RHS) cache solve. Returns (ok::Bool, result).
# On failure returns (false, the thrown exception) so the suite can continue.
function _safe_matrix_solve(cache, ::Type{T}, Bbig, n, nrhs) where {T}
    X = zeros(T, n, nrhs)
    try
        solve!(X, cache, T.(Bbig))
        return (true, X)
    catch e
        return (false, e)
    end
end


_rel_recon(rec, Abig) = norm(rec - Abig, Inf) / norm(Abig, Inf)

# ---------------------------------------------------------------------------
# Factor reconstruction in BigFloat.
# ---------------------------------------------------------------------------
# Cholesky: lower triangle (with diagonal) -> L*L'.
function cholesky_reconstruction(factors)
    L = BigFloat.(tril(factors))
    return L * L'
end

# LU: unit-lower L + upper U, and the row-permutation from ipiv.
function lu_reconstruction(factors)
    n = size(factors, 1)
    L = BigFloat.(tril(factors, -1)) + Matrix{BF}(I, n, n)
    U = BigFloat.(triu(factors))
    return L * U
end
function lu_row_permutation(ipiv)
    n = length(ipiv)
    p = collect(1:n)
    @inbounds for k in 1:n
        p[k], p[ipiv[k]] = p[ipiv[k]], p[k]
    end
    return p
end

# LDLT: assemble unit-lower L and block-diagonal D from the factor, plus the
# final symmetric permutation p with A[p,p] = L*D*L'.
function ldlt_reconstruct(factors, dsub, pivots, blocks)
    n = size(factors, 1)
    F = BigFloat.(factors)
    Ds = BigFloat.(dsub)
    L = Matrix{BF}(I, n, n)
    @inbounds for j in 1:n, i in (j + 1):n
        L[i, j] = F[i, j]
    end
    D = zeros(BF, n, n)
    k = 1
    @inbounds while k <= n
        if blocks[k] == UInt8(1)
            D[k, k] = F[k, k]
            k += 1
        elseif blocks[k] == UInt8(2)
            D[k, k] = F[k, k]
            D[k, k + 1] = Ds[k]
            D[k + 1, k] = Ds[k]
            D[k + 1, k + 1] = F[k + 1, k + 1]
            k += 2
        else
            break
        end
    end
    p = collect(1:n)
    k = 1
    @inbounds while k <= n
        block = blocks[k]
        if block == UInt8(1)
            pivot = pivots[k]
            p[k], p[pivot] = p[pivot], p[k]
            k += 1
        elseif block == UInt8(2) && k < n
            pivot = pivots[k]
            p[k + 1], p[pivot] = p[pivot], p[k + 1]
            k += 2
        else
            break
        end
    end
    return L, D, p
end

# RRQR: reconstruct Q from Householder vectors and R from the upper triangle.
function rrqr_reconstruction(factors, tau, permutation)
    m, n = size(factors)
    F = BigFloat.(factors)
    t = BigFloat.(tau)
    Q = Matrix{BF}(I, m, m)
    r = min(m, n)
    @inbounds for step in 1:r
        v = zeros(BF, m)
        v[step] = 1
        for i in (step + 1):m
            v[i] = F[i, step]
        end
        H = Matrix{BF}(I, m, m) - t[step] * (v * v')
        Q = Q * H
    end
    R = zeros(BF, m, n)
    @inbounds for j in 1:n, i in 1:min(j, m)
        R[i, j] = F[i, j]
    end
    return Q, R
end

# ---------------------------------------------------------------------------
# Per-kind well-conditioned accuracy tests.
# ---------------------------------------------------------------------------
function check_cholesky(::Type{T}, n) where {T}
    tag = "x$(limb(T))-cholesky"
    Abig, A = spd_matrices(T, n)
    xbig, bbig = reference_rhs(Abig, n, 1)
    Xbig, Bbig = reference_rhs(Abig, n, 3)
    cache = MFCholeskyCache(T)
    prepare!(cache, n)
    factorize!(cache, A)
    check("$tag factorize success", MFLA.issuccess(cache))

    x = zeros(T, n)
    solve!(x, cache, T.(bbig))
    e1 = _rel_err_vector(x, xbig)
    check("$tag vector solve rel err", e1 <= tol_rel(T), @sprintf("%.3e", e1))

    X = zeros(T, n, 3)
    solve!(X, cache, T.(Bbig))
    em = _rel_err_matrix(X, Xbig)
    check("$tag multi-RHS rel err", em <= tol_rel(T), @sprintf("%.3e", em))

    r = _rel_recon(cholesky_reconstruction(factor_matrix(cache)), Abig)
    check("$tag reconstruction A≈L*L'", r <= tol_rel(T), @sprintf("%.3e", r))

    b = max(_backward_error(Abig, BigFloat.(x), bbig),
            _backward_error(Abig, BigFloat.(X), Bbig))
    check("$tag backward error", b <= tol_back(T), @sprintf("%.3e", b))
end

function check_lu(::Type{T}, n) where {T}
    tag = "x$(limb(T))-lu"
    Abig, A = diagdom_matrices(T, n)
    xbig, bbig = reference_rhs(Abig, n, 1)
    Xbig, Bbig = reference_rhs(Abig, n, 3)
    cache = MFLUCache(T)
    prepare!(cache, n)
    factorize!(cache, A)
    check("$tag factorize success", MFLA.issuccess(cache))

    x = zeros(T, n)
    solve!(x, cache, T.(bbig))
    e1 = _rel_err_vector(x, xbig)
    check("$tag vector solve rel err", e1 <= tol_rel(T), @sprintf("%.3e", e1))

    X = zeros(T, n, 3)
    solve!(X, cache, T.(Bbig))
    em = _rel_err_matrix(X, Xbig)
    check("$tag multi-RHS rel err", em <= tol_rel(T), @sprintf("%.3e", em))

    p = lu_row_permutation(cache.ipiv)
    check("$tag ipiv valid row permutation", _is_permutation(p))
    r = _rel_recon(lu_reconstruction(factor_matrix(cache)), Abig[p, :])
    check("$tag reconstruction P*A≈L*U", r <= tol_rel(T), @sprintf("%.3e", r))

    b = max(_backward_error(Abig, BigFloat.(x), bbig),
            _backward_error(Abig, BigFloat.(X), Bbig))
    check("$tag backward error", b <= tol_back(T), @sprintf("%.3e", b))

    maxu = maximum(abs, triu(factor_matrix(cache)))
    maxa = maximum(abs, Abig)
    growth = maxu / maxa
    # Pivot growth rho = max|U|/max|A|. rho is finite and, for a diagonally
    # dominant well-conditioned matrix, must be small (near or below 1; it can
    # legitimately dip below 1 when the max element of A is eliminated).
    check("$tag pivot growth finite & no blow-up", isfinite(growth) && growth < 1.0e3,
          @sprintf("growth=%.3e", growth))
end

function check_ldlt(::Type{T}, n) where {T}
    tag = "x$(limb(T))-ldlt"
    Abig, A, lam = indefinite_matrices(T, n)
    xbig, bbig = reference_rhs(Abig, n, 1)
    Xbig, Bbig = reference_rhs(Abig, n, 3)
    cache = MFLDLTCache(T)
    prepare!(cache, n)
    factorize!(cache, A)
    check("$tag factorize success", MFLA.issuccess(cache))

    x = zeros(T, n)
    solve!(x, cache, T.(bbig))
    e1 = _rel_err_vector(x, xbig)
    check("$tag vector solve rel err", e1 <= tol_rel(T), @sprintf("%.3e", e1))

    # multi-RHS matrix solve (guarded: LDLT cache matrix solve is currently
    # broken in the package, see report)
    Xok, X = _safe_matrix_solve(cache, T, Bbig, n, 3)
    if Xok
        em = _rel_err_matrix(X, Xbig)
        check("$tag multi-RHS rel err", em <= tol_rel(T), @sprintf("%.3e", em))
    else
        check("$tag multi-RHS rel err", false,
              "LDLT cache matrix solve! threw $(typeof(X)) (trsv! on a Matrix)")
    end

    L, D, p = ldlt_reconstruct(factor_matrix(cache), cache.dsub, cache.pivots, cache.blocks)
    check("$tag LDLT permutation valid", _is_permutation(p))
    r = _rel_recon(L * D * L', Abig[p, p])
    check("$tag reconstruction A[p,p]≈L*D*L'", r <= tol_rel(T), @sprintf("%.3e", r))

    b = _backward_error(Abig, BigFloat.(x), bbig)
    Xok && (b = max(b, _backward_error(Abig, BigFloat.(X), Bbig)))
    check("$tag backward error", b <= tol_back(T), @sprintf("%.3e", b))

    inertia = factor_diagnostics(cache).inertia
    npos = count(>(0), lam)
    nneg = count(<(0), lam)
    got = (inertia.positive, inertia.negative, inertia.zero)
    check("$tag inertia computed (+,−,0)", got == (npos, nneg, 0),
          "got=$got expect=($npos,$nneg,0)")
end

function check_rrqr(::Type{T}, n) where {T}
    tag = "x$(limb(T))-rrqr"
    Abig, A = diagdom_matrices(T, n)
    xbig, bbig = reference_rhs(Abig, n, 1)
    Xbig, Bbig = reference_rhs(Abig, n, 3)
    cache = MFRRQRCache(T)
    prepare!(cache, n)
    factorize!(cache, A)
    check("$tag factorize success", MFLA.issuccess(cache))

    x = zeros(T, n)
    solve!(x, cache, T.(bbig))
    e1 = _rel_err_vector(x, xbig)
    check("$tag vector solve rel err", e1 <= tol_rel(T), @sprintf("%.3e", e1))

    # multi-RHS matrix solve (validated: the matrix-destination permutation
    # iterates columns, so multi-column solutions are correct)
    Xok, X = _safe_matrix_solve(cache, T, Bbig, n, 3)
    if Xok
        em = _rel_err_matrix(X, Xbig)
        check("$tag multi-RHS rel err", em <= tol_rel(T), @sprintf("%.3e", em))
    else
        check("$tag multi-RHS rel err", false,
              "RRQR cache matrix solve! threw $(typeof(X))")
    end

    p = factor_permutation(cache)
    check("$tag permutation valid", _is_permutation(p))
    Q, R = rrqr_reconstruction(factor_matrix(cache), cache.tau, cache.permutation)
    r = _rel_recon(Q * R, Abig[:, p])
    check("$tag reconstruction A[:,p]≈Q*R", r <= tol_rel(T), @sprintf("%.3e", r))
    orth = norm(Q' * Q - Matrix{BF}(I, n, n), Inf)
    check("$tag Q orthogonal", orth <= tol_rel(T), @sprintf("%.3e", orth))

    b = max(_backward_error(Abig, BigFloat.(x), bbig),
            _backward_error(Abig, BigFloat.(X), Bbig))
    check("$tag backward error", b <= tol_back(T), @sprintf("%.3e", b))
end

# Rectangular RRQR cache route validated against the 512-bit BigFloat reference:
# tall least-squares, wide reconstruction, and solve_r! trans=:T for vector and
# matrix RHS. This closes the coverage gap that previously hid the trans=:T bug.
function check_rrqr_rectangular(::Type{T}, m, n) where {T}
    tag = "x$(limb(T))-rrqr-rect"
    Abig = Matrix{BF}(undef, m, n)
    for j in 1:n, i in 1:m
        Abig[i, j] = BF(sin(0.1 * i + 0.2 * j) + 0.3 * cos(0.05 * i * j))
    end
    for j in 1:min(m, n)
        Abig[j, j] += BF(4)   # diagonal dominance -> full column rank
    end
    A = T.(Abig)
    cache = MFRRQRCache(T)
    prepare!(cache, m, n)
    factorize!(cache, A)
    check("$tag factorize success", MFLA.issuccess(cache))

    p = factor_permutation(cache)
    # tall least-squares: x = R^-1 Q' b permuted (only meaningful when m >= n)
    if m >= n
        xbig = Matrix{BF}(undef, n, 1)
        for i in 1:n
            xbig[i, 1] = BF(sin(0.3 * i))
        end
        bbig = Abig * xbig
        b = T.(bbig[:, 1])
        y = copy(b)
        apply_q!(y, cache; trans=:T)
        ylead = y[1:n]
        solve_r!(ylead, cache, n; trans=:N)
        x = zeros(T, n)
        for i in 1:n
            x[p[i]] = ylead[i]
        end
        e = _rel_err_vector(x, xbig[:, 1])
        check("$tag tall least-squares rel err", e <= tol_rel(T), @sprintf("%.3e", e))
    end

    # wide reconstruction A[:,p] ≈ Q*R
    Q, R = rrqr_reconstruction(factor_matrix(cache), cache.tau, cache.permutation)
    r = _rel_recon(Q * R, Abig[:, p])
    check("$tag wide reconstruction A[:,p]≈Q*R", r <= tol_rel(T), @sprintf("%.3e", r))

    # solve_r! trans=:T for vector and matrix RHS (leading rank = min(m,n))
    rank = min(m, n)
    Rlead = zeros(T, rank, rank)
    for c in 1:rank, r in 1:c
        Rlead[r, c] = factor_matrix(cache)[r, c]
    end
    xref = T.(randn(rank))
    bT = transpose(Rlead) * xref
    xT = copy(bT)
    solve_r!(xT, cache, rank; trans=:T)
    eT = _rel_err_vector(xT, xref)
    check("$tag solve_r! trans=:T rel err", eT <= tol_rel(T), @sprintf("%.3e", eT))
    Xref = T.(randn(rank, 3))
    BT = transpose(Rlead) * Xref
    XT = copy(BT)
    solve_r!(XT, cache, rank; trans=:T)
    eTm = _rel_err_matrix(XT, Xref)
    check("$tag solve_r! mat trans=:T rel err", eTm <= tol_rel(T), @sprintf("%.3e", eTm))
end

# ---------------------------------------------------------------------------
# Pathological cases
# ---------------------------------------------------------------------------
function pathological_cholesky(::Type{T}, n) where {T}
    tag = "x$(limb(T))-cholesky"
    # nonfinite input -> status -1
    A = T.(randn(n, n)); A[1, 1] = T(NaN)
    c = MFCholeskyCache(T); prepare!(c, n)
    factorize!(c, A; check=false)
    ok = (factor_status(c) == -1) && (factor_state(c) == :nonfinite_input) && !MFLA.issuccess(c)
    check("$tag nonfinite input status -1", ok, "status=$(factor_status(c))")
end

function Singular_lu(::Type{T}, n) where {T}
    tag = "x$(limb(T))-lu"
    # singular: first column zero -> zero pivot
    A = T.(randn(n, n)); A[1, :] .= zero(T)
    c = MFLUCache(T); prepare!(c, n)
    factorize!(c, A; check=false)
    ok = (factor_status(c) > 0) && (factor_state(c) == :singular) && !MFLA.issuccess(c)
    check("$tag singular failure status", ok, "status=$(factor_status(c))")
    # check=true throws
    threw = try
        factorize!(c, A; check=true); false
    catch e
        e isa LinearAlgebra.SingularException
    end
    check("$tag singular check=true throws SingularException", threw)
end

function Singular_ldlt(::Type{T}, n) where {T}
    tag = "singular-ldlt x$(limb(T))"
    # symmetric with a zero leading row/column => zero pivot
    A = T.(randn(n, n)); A = A + A'; A[1, :] .= zero(T); A[:, 1] .= zero(T)
    c = MFLDLTCache(T); prepare!(c, n)
    factorize!(c, A; check=false)
    ok = (factor_status(c) > 0) && (factor_state(c) == :singular) && !MFLA.issuccess(c)
    check("$tag singular failure status", ok, "status=$(factor_status(c))")
    threw = try
        factorize!(c, A; check=true); false
    catch e
        e isa LinearAlgebra.SingularException
    end
    check("$tag singular check throws SingularException", threw)
end

function RankDeficient_rrqr(::Type{T}, m, n, rank_true) where {T}
    tag = "x$(limb(T))-rrqr rank-def"
    G = BF.(randn(BigFloat, m, rank_true))
    W = BF.(randn(BigFloat, rank_true, n))
    Abig = G * W
    A = T.(Abig)
    c = MFRRQRCache(T); prepare!(c, m, n)
    factorize!(c, A)
    ok_succ = MFLA.issuccess(c)
    check("$tag rank-deficient still success", ok_succ)
    r = numerical_rank(c; atol=T(1e-40), rtol=T(1e-6))
    check("$tag numerical rank < n", ok_succ && r == rank_true && r < n,
          "rank=$r expect=$rank_true")
    # reconstruction on rectangular case
    p = factor_permutation(c)
    Q, R = rrqr_reconstruction(factor_matrix(c), c.tau, c.permutation)
    rec = _rel_recon(Q * R, Abig[:, p])
    check("$tag reconstruction A[:,p]≈Q*R", rec <= tol_rel(T), @sprintf("%.3e", rec))
end

function BadlyScaled(::Type{T}, n) where {T}
    tag = "x$(limb(T))-badly-scaled"
    Abig, A = badly_scaled_matrices(T, n)
    xbig, bbig = reference_rhs(Abig, n, 1)
    for (kind, cache) in [
        (:lu, MFLUCache(T)),
        (:ldlt, MFLDLTCache(T)),
        (:rrqr, MFRRQRCache(T)),
    ]
        prepare!(cache, n)
        factorize!(cache, A)
        okf = MFLA.issuccess(cache)
        check("$tag $kind factorize", okf)
        x = zeros(T, n)
        solve!(x, cache, T.(bbig))
        e = _rel_err_vector(x, xbig)
        b = _backward_error(Abig, BigFloat.(x), bbig)
        # forward error grows with the (large) condition number; the stability
        # criterion is a small normwise backward error.
        check("$tag $kind backward error (stability)",
              okf && b <= tol_back(T),
              "fwrel=$(@sprintf("%.2e", e)) be=$(@sprintf("%.2e", b))")
    end
end

# ---------------------------------------------------------------------------
# success -> failure -> recovery
# ---------------------------------------------------------------------------
function Recovery_lu(::Type{T}, n) where {T}
    tag = "x$(limb(T))-lu-recovery"
    _, A_good = diagdom_matrices(T, n)
    c = MFLUCache(T); prepare!(c, n)
    factorize!(c, A_good)
    check("$tag initial success", MFLA.issuccess(c))

    A_sing = T.(randn(n, n)); A_sing[1, :] .= zero(T)
    threw = try
        factorize!(c, A_sing; check=true); false
    catch e
        e isa LinearAlgebra.SingularException
    end
    check("$tag failure throws", threw)
    check("$tag failed cache not success", !MFLA.issuccess(c))

    solve_throws = try
        x = zeros(T, n); b = T.(randn(n))
        solve!(x, c, b); false
    catch e
        true
    end
    check("$tag solve refused after failure", solve_throws)

    # replacement good matrix -> recovery
    Abig2, A_good2 = diagdom_matrices(T, n)
    xbig2, bbig2 = reference_rhs(Abig2, n, 1)
    factorize!(c, A_good2)
    check("$tag recovery success", MFLA.issuccess(c))
    x = zeros(T, n)
    solve!(x, c, T.(bbig2))
    e = _rel_err_vector(x, xbig2)
    check("$tag recovered solve", e <= tol_rel(T), @sprintf("%.3e", e))
end

# ---------------------------------------------------------------------------
# Serial vs threaded bit-identical results
# ---------------------------------------------------------------------------
function check_threading(::Type{T}, n, kind) where {T}
    tag = "x$(limb(T))-$(kind)-threading"
    Abig, A = if kind == :cholesky
        spd_matrices(T, n)
    elseif kind == :ldlt
        indefinite_matrices(T, n)[1:2]
    else
        diagdom_matrices(T, n)
    end
    xbig, bbig = reference_rhs(Abig, n, 1)
    b = T.(bbig)

    c1 = _make_cache(T, kind; config=KernelConfig(thread_count=1))
    c4 = _make_cache(T, kind; config=KernelConfig(thread_count=4))
    prepare!(c1, n); prepare!(c4, n)
    factorize!(c1, A); factorize!(c4, A)
    same_factor = factor_matrix(c1) == factor_matrix(c4)
    # pivot / permutation metadata equality
    meta_eq = _cache_metadata_equal(c1, c4)
    check("$tag factor bit-identical 1t vs 4t", same_factor && meta_eq)

    x1 = zeros(T, n); x4 = zeros(T, n)
    solve!(x1, c1, b); solve!(x4, c4, b)
    check("$tag solve bit-identical 1t vs 4t", x1 == x4)
    return
end

function _make_cache(::Type{T}, kind; config::KernelConfig=KernelConfig()) where {T}
    if kind == :cholesky
        return MFCholeskyCache(T; config=config)
    elseif kind == :lu
        return MFLUCache(T; config=config)
    elseif kind == :ldlt
        return MFLDLTCache(T; config=config)
    else
        return MFRRQRCache(T; config=config)
    end
end

# Bit-identical metadata for a kind (pivots/permutation/blocks/tau).
function _cache_metadata_equal(c1, c2)
    factor_status(c1) == factor_status(c2) || return false
    if c1 isa MFLUCache
        return c1.ipiv == c2.ipiv
    elseif c1 isa MFLDLTCache
        return c1.pivots == c2.pivots && c1.blocks == c2.blocks && c1.dsub == c2.dsub
    elseif c1 isa MFRRQRCache
        return c1.permutation == c2.permutation && c1.tau == c2.tau &&
               c1.cycle_count == c2.cycle_count
    end
    return true   # cholesky: no pivot metadata
end

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
function main()
    println("===============================================================")
    println("Independent factor-cache accuracy validation vs 512-bit BigFloat")
    println("Julia $(VERSION), threads=$(Threads.nthreads()), 512-bit reference")
    println("===============================================================")

    n_well = 24      # well-conditioned accuracy tests
    n_thread = 72    # threading consistency (engages threaded kernels)

    for T in TYPES
        check_cholesky(T, n_well)
        check_lu(T, n_well)
        check_ldlt(T, n_well)
        check_rrqr(T, n_well)
        check_rrqr_rectangular(T, 40, 24)   # tall
        check_rrqr_rectangular(T, 16, 32)   # wide
    end

    for T in TYPES
        pathological_cholesky(T, 8)
        Singular_lu(T, 8)
        Singular_ldlt(T, 8)
        RankDeficient_rrqr(T, 12, 8, 6)
        BadlyScaled(T, 12)
        Recovery_lu(T, 8)
    end

    for T in TYPES, kind in (:cholesky, :lu, :ldlt, :rrqr)
        check_threading(T, n_thread, kind)
    end

    println()
    println("SUMMARY: $(_CHECKS[]) checks, $(_FAILURES[]) failures")
    exit(_FAILURES[] == 0 ? 0 : 1)
end

main()
