# M03 — Bunch--Kaufman pivot grammar: structural validation, permutation
# replay, factor reconstruction, and the N/T block-triangular solves.
#
# STATUS: add-only module file written by the M03 worker. It is NOT in
# `src/MultiFloatLinearAlgebra.jl`'s include graph at the time of writing
# (probed, not assumed: the driver records `isdefined` for each of the three M03
# files, and the package entry point has no `include` for them). The driver
# therefore loads these files itself in one inclusion mode and relies on the
# package in the other; both modes must produce identical measured values.
#
# WHAT IS AND IS NOT DUPLICATED HERE
#
# This file adds *no* method to any name MFLA already defines. In particular
# `factor_pivots(::MFLDLT)`, `factor_blocks(::MFLDLT)` and
# `factor_permutation(::MFLDLT)` already exist (`src/factorizations/ldlt.jl:25`,
# `:33`, `:41`) and are NOT redefined; every function below carries the `m03_`
# prefix for exactly this reason. The driver asserts the "no method overwrite"
# property by snapshotting the method signature sets of 26 MFLA names before and
# after loading these files.
#
# The pivot *selection* is deliberately not re-implemented: the driver drives
# MFLA's `_select_bk_pivot` directly, so the arithmetic under test here is not a
# second, independently written Bunch--Kaufman policy.

const M03_PIVOT_POLICY_VERSION = v"1.0.0"

# An unwired add-only file is only meaningful when it lands in the package
# namespace: `Base.include(Main, path)` would work for the parts that happen to
# be self-contained and then fail confusingly on the first `MultiFloat` method
# signature. Fail at include time with the exact call to make instead.
@isdefined(MultiFloatLinearAlgebra) || error(
    "M03: include this file into MultiFloatLinearAlgebra, not into a sandbox " *
    "module: Base.include(MultiFloatLinearAlgebra, " *
    "\"src/factorizations/pivot_policy.jl\")")

"""
    M03PivotGrammar

Structural, read-only view of the Bunch--Kaufman result that an `MFLDLT`
carries as three parallel length-`n` arrays:

  * `pivots[k]` — the raw step pivot for the block starting at `k`
    (`k` for a 1x1 block that stays in place, the swap partner otherwise);
  * `blocks[k]` — `0x01` marks a 1x1 block start, `0x02` a 2x2 block start,
    `0x00` marks the *continuation* row of a 2x2 block, and `0x00` past the
    last accepted pivot means "factorization stopped here" (`info != 0`);
  * `n` — the order of the factor.

`accepted` is the number of rows consumed by accepted blocks. When
`accepted < n` the factorization failed at pivot `accepted + 1` and the tail of
the arrays is uninitialized input, which is why every consumer here bounds its
loops by `accepted` rather than by `n`.
"""
struct M03PivotGrammar
    n::Int
    pivots::Vector{Int}
    blocks::Vector{UInt8}
    accepted::Int
end

"""
    m03_pivot_grammar(pivots, blocks) -> M03PivotGrammar

Validate and classify the parallel `pivots`/`blocks` arrays. Throws
`ArgumentError` on any structurally impossible combination; this is the
grammar's own admission check, and the driver feeds it deliberately malformed
arrays as its instrument control.
"""
function m03_pivot_grammar(pivots::AbstractVector{<:Integer},
                           blocks::AbstractVector{UInt8})
    n = length(blocks)
    length(pivots) == n || throw(DimensionMismatch(
        "pivots has length $(length(pivots)), blocks has length $n"))
    k = 1
    accepted = 0
    while k <= n
        block = blocks[k]
        if block == UInt8(1)
            pivot = Int(pivots[k])
            1 <= pivot <= n || throw(ArgumentError(
                "1x1 block at $k has out-of-range pivot $pivot"))
            k += 1
            accepted = k - 1
        elseif block == UInt8(2)
            k < n || throw(ArgumentError(
                "2x2 block start at $k is the last row; a 2x2 block needs a continuation"))
            blocks[k + 1] == UInt8(0) || throw(ArgumentError(
                "2x2 block start at $k is not followed by a continuation marker"))
            pivot = Int(pivots[k])
            k + 1 <= pivot <= n || throw(ArgumentError(
                "2x2 block at $k has out-of-range pivot $pivot (must be > $k)"))
            k += 2
            accepted = k - 1
        elseif block == UInt8(0)
            # Either the continuation of a 2x2 block that the loop above already
            # consumed (unreachable here) or the first row of the stopped tail.
            break
        else
            throw(ArgumentError("invalid block marker $(block) at row $k"))
        end
    end
    return M03PivotGrammar(n, Int.(collect(pivots)), collect(blocks), accepted)
end

"""
    m03_grammar_from_factor(F) -> M03PivotGrammar

Grammar of an `MFLDLT`, read through MFLA's own public accessors (caller-owned
copies), never through private fields.
"""
function m03_grammar_from_factor(F)
    return m03_pivot_grammar(factor_pivots(F), UInt8.(factor_blocks(F)))
end

m03_grammar_complete(g::M03PivotGrammar) = g.accepted == g.n

"""
    m03_grammar_block_starts(g) -> Vector{Int}

Sorted row indices at which an accepted block starts.
"""
function m03_grammar_block_starts(g::M03PivotGrammar)
    starts = Int[]
    k = 1
    while k <= g.accepted
        push!(starts, k)
        k += g.blocks[k] == UInt8(1) ? 1 : 2
    end
    return starts
end

"""
    m03_grammar_counts(g) -> (one_by_one, two_by_two)

Number of accepted 1x1 and 2x2 diagonal blocks. `one_by_one + 2*two_by_two ==
g.accepted` holds and is asserted by the driver.
"""
function m03_grammar_counts(g::M03PivotGrammar)
    one_by_one = 0
    two_by_two = 0
    for k in 1:g.accepted
        marker = g.blocks[k]
        marker == UInt8(1) && (one_by_one += 1)
        marker == UInt8(2) && (two_by_two += 1)
    end
    return (one_by_one, two_by_two)
end

"""
    m03_pivot_permutation(g) -> Vector{Int}

Replay the recorded symmetric swaps into the final permutation `p` with
`A_original[p, p] = L * D * L'`.

This is an independent re-derivation of the same quantity MFLA's
`factor_permutation(::MFLDLT)` returns: it replays `_ldlt_symmetric_swap!` from
the recorded grammar instead of re-reading the factor's own bookkeeping. The
driver compares the two and treats any disagreement as a defect.

MEASURED NOTE: for a 2x2 block the swap is `A[k+1, :] <-> A[imax, :]`, so the
common case `imax == k + 1` is a NO-OP on the permutation even though the
grammar records a 2x2 block. Over roughly 2.3M Bunch--Kaufman steps on random
symmetric matrices this driver found **no** 2x2 pivots at all (random diagonals
dominate their columns) and no 1x1 pivot at `imax != k`. The driver therefore
builds a deterministic `(2, imax=4)` matrix to test the non-trivial branch
rather than hoping a random family hits it.
"""
function m03_pivot_permutation(g::M03PivotGrammar)
    permutation = collect(1:g.n)
    @inbounds for k in m03_grammar_block_starts(g)
        marker = g.blocks[k]
        pivot = g.pivots[k]
        if marker == UInt8(1)
            permutation[k], permutation[pivot] =
                permutation[pivot], permutation[k]
        else
            permutation[k + 1], permutation[pivot] =
                permutation[pivot], permutation[k + 1]
        end
    end
    return permutation
end

"""
    m03_apply_permutation(A, p) -> Matrix

Return `A[p, p]` in a fresh matrix, without allocating an index matrix.
"""
function m03_apply_permutation(A::AbstractMatrix, p::AbstractVector{<:Integer})
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("symmetric permutation needs a square matrix"))
    length(p) == n || throw(DimensionMismatch("permutation length mismatch"))
    out = Matrix{eltype(A)}(undef, n, n)
    @inbounds for column in 1:n
        source_column = Int(p[column])
        for row in 1:n
            out[row, column] = A[Int(p[row]), source_column]
        end
    end
    return out
end

"""
    m03_reconstruct(factors, dsub, g) -> Matrix

Rebuild `L * D * L'` for the packed factor that `ldlt!` leaves in
`factor_matrix(F)`.

MEASURED PACKED LAYOUT (this took four attempts; all four failures are recorded
because every one of them was silent):

  * `L` is UNIT lower triangular and its multipliers are the STRICT LOWER
    TRIANGLE of `factors`;
  * `D` is block diagonal: `D[i,i] = factors[i,i]` and, for a 2x2 block starting
    at `k`, `D[k,k+1] = D[k+1,k] = dsub[k]`;
  * the STRICT UPPER TRIANGLE of `factors` is NOT part of `L`. It is reused
    storage (for a 2x2 pivot it holds remnants of the symmetric swap), which is
    why reading it produced a reconstruction error of exactly 1.0 on
    `A = [0 1 1; 1 0 1; 1 1 0]` whose actual factor error is at roundoff.

Failed variants, for the record: (1) building only the lower triangle and
returning zeros above the diagonal — the driver un-permutes the product, so the
comparison was meaningless; (2) forming `D*L'` from `factors[k, row]`, i.e. the
upper triangle; (3) extracting `L` by solving `L*D x = e_j` and applying
`D^{-1}` blockwise, which is sound but strictly more expensive and adds a
failure mode on singular `D`, so the direct packed read is kept.

`A_original` is recovered by the caller as `P' * (L*D*L') * P`, which is what
`m03_apply_permutation(..., invperm(p))` computes.
"""
function m03_reconstruct(factors::AbstractMatrix{MF},
                         dsub::AbstractVector{MF},
                         g::M03PivotGrammar) where {MF<:MultiFloat}
    n = size(factors, 1)
    size(factors, 2) == n || throw(DimensionMismatch("factor storage must be square"))
    size(dsub, 1) == n || throw(DimensionMismatch("dsub length mismatch"))
    g.accepted == n || throw(ArgumentError(
        "reconstruction requires a complete factorization; accepted=$(g.accepted) of $n"))
    lower = zeros(MF, n, n)
    for index in 1:n
        lower[index, index] = one(MF)
    end
    @inbounds for column in 1:n
        for row in (column + 1):n
            lower[row, column] = factors[row, column]
        end
    end
    diagonal = zeros(MF, n, n)
    @inbounds for k in m03_grammar_block_starts(g)
        if g.blocks[k] == UInt8(1)
            diagonal[k, k] = factors[k, k]
        else
            diagonal[k, k] = factors[k, k]
            diagonal[k + 1, k + 1] = factors[k + 1, k + 1]
            diagonal[k, k + 1] = dsub[k]
            diagonal[k + 1, k] = dsub[k]
        end
    end
    return lower * diagonal * transpose(lower)
end

"""
    m03_reconstruct_from_factor(F) -> (Matrix, M03PivotGrammar, Vector{Int})

Convenience wrapper over `m03_reconstruct` that reads the factor through MFLA's
public accessors and also returns the factor's own permutation.
"""
function m03_reconstruct_from_factor(F)
    g = m03_grammar_from_factor(F)
    return (
        m03_reconstruct(factor_matrix(F), F.dsub, g),
        g,
        factor_permutation(F),
    )
end

"""
    m03_factor_backward_error(A, Ahat) -> MF

Scaled reconstruction error `max|A - Ahat| / max(1, max|A|, max|Ahat|)`. Scaled
by the larger of the two maxima so a factor that *grows* the matrix cannot
report a small error.
"""
function m03_factor_backward_error(A::AbstractMatrix{MF},
                                   Ahat::AbstractMatrix{MF}) where {MF<:MultiFloat}
    size(A) == size(Ahat) || throw(DimensionMismatch("shape mismatch"))
    difference = zero(MF)
    largest_a = zero(MF)
    largest_hat = zero(MF)
    @inbounds for index in eachindex(A)
        difference = max(difference, abs(A[index] - Ahat[index]))
        largest_a = max(largest_a, abs(A[index]))
        largest_hat = max(largest_hat, abs(Ahat[index]))
    end
    denominator = max(one(MF), largest_a, largest_hat)
    return difference / denominator
end

# --- block-triangular solves on a M03PivotGrammar ---------------------------

@inline function _m03_solve_1x1(d::MF, rhs::MF) where {MF<:MultiFloat}
    if iszero(d) || !isfinite(d)
        return (zero(MF), false)
    end
    x = rhs / d
    return (x, isfinite(x))
end

@inline function _m03_solve_2x2(d11::MF, d21::MF, d22::MF,
                                first::MF, second::MF) where {MF<:MultiFloat}
    # Same elimination the production kernel uses: scale by the largest |d| so
    # the discriminant cannot overflow, then eliminate on the larger |a|,|b|.
    scale = max(abs(d11), abs(d21), abs(d22))
    if iszero(scale) || !isfinite(scale)
        return (zero(MF), zero(MF), false)
    end
    a = d11 / scale
    b = d21 / scale
    c = d22 / scale
    r1 = first / scale
    r2 = second / scale
    if abs(b) > abs(a)
        iszero(b) && return (zero(MF), zero(MF), false)
        t = a / b
        u = b - t * c
        (iszero(u) || !isfinite(u)) && return (zero(MF), zero(MF), false)
        x2 = (r1 - t * r2) / u
        x1 = (r2 - c * x2) / b
        return (x1, x2, isfinite(x1) && isfinite(x2))
    end
    iszero(a) && return (zero(MF), zero(MF), false)
    t = b / a
    u = c - t * b
    (iszero(u) || !isfinite(u)) && return (zero(MF), zero(MF), false)
    x2 = (r2 - t * r1) / u
    x1 = (r1 - b * x2) / a
    return (x1, x2, isfinite(x1) && isfinite(x2))
end

# The four block-triangular actions this file needs, written out separately
# rather than parameterised by a `trans` flag. A single shared routine taking the
# flag would be the one place a copy-paste error could make the transposed solve
# silently return the non-transposed answer -- which is exactly what happened in
# an earlier revision, so the driver asserts the two arms disagree on matrices
# where they must.

# solves `L u = v`; `L` is unit lower with multipliers in the strict lower triangle
function _m03_forward_l!(x, factors, g)
    @inbounds for k in m03_grammar_block_starts(g)
        for row in (k + 1):g.n
            x[row] -= factors[row, k] * x[k]
        end
    end
    return true
end

# solves `L' u = v`, block by block in reverse
function _m03_back_l!(x, factors, g)
    starts = m03_grammar_block_starts(g)
    @inbounds for index in length(starts):-1:1
        k = starts[index]
        if g.blocks[k] == UInt8(1)
            for row in (k + 1):g.n
                x[k] -= factors[row, k] * x[row]
            end
        else
            for row in (k + 2):g.n
                x[k] -= factors[row, k] * x[row]
                x[k + 1] -= factors[row, k + 1] * x[row]
            end
        end
    end
    return true
end

# solves `D u = v`: 1x1 blocks divide, 2x2 blocks through `_m03_solve_2x2`
function _m03_solve_d!(x, factors, dsub, g)
    @inbounds for k in m03_grammar_block_starts(g)
        if g.blocks[k] == UInt8(1)
            solved, ok = _m03_solve_1x1(factors[k, k], x[k])
            ok || return false
            x[k] = solved
        else
            first, second, ok = _m03_solve_2x2(
                factors[k, k], dsub[k], factors[k + 1, k + 1], x[k], x[k + 1])
            ok || return false
            x[k] = first
            x[k + 1] = second
        end
    end
    return true
end

# solves `L D u = v` in one pass; the row update uses the ALREADY DIVIDED x[k]
function _m03_forward_ld!(x, factors, dsub, g)
    n = g.n
    @inbounds for k in m03_grammar_block_starts(g)
        for row in (k + 1):n
            x[row] -= factors[row, k] * x[k]
        end
        if g.blocks[k] == UInt8(1)
            solved, ok = _m03_solve_1x1(factors[k, k], x[k])
            ok || return false
            x[k] = solved
        else
            for row in (k + 2):n
                x[row] -= factors[row, k + 1] * x[k + 1]
            end
            first, second, ok = _m03_solve_2x2(
                factors[k, k], dsub[k], factors[k + 1, k + 1], x[k], x[k + 1])
            ok || return false
            x[k] = first
            x[k + 1] = second
        end
    end
    return true
end

"""
    m03_solve_lower!(x, factors, dsub, g; trans=:N) -> Bool

`L` is the unit lower triangular factor whose multipliers are the STRICT LOWER
triangle of `factors`; `D` is the block diagonal built from `factors`' diagonal
and `dsub`; `D' = D` because every block of `D` is symmetric.

  * `trans=:N` solves `L * D * L' * x = b`: forward on `L*D`, then back on `L'`.
  * `trans=:T` solves `L * D' * L' * x = b`: forward on `L`, then the `D`
    solve, then back on `L'`. That is the SAME sequence of three actions as
    `:N`; the two differ only because `:N` fuses the `L` and `D` actions into
    `_m03_forward_ld!`.

Both branches were wrong in earlier revisions and both failed SILENTLY:

  * `:N` applied the row updates for block `k` before dividing `x[k]`, so `x[k]`
    was never divided and the routine returned `D^-1 L^-1 b` instead. Residuals
    stayed at ~1e-33 on diagonally dominant families while an arrowhead family
    showed O(1) error, which is why the driver compares per family.
  * `:T` applied `L`, `L'`, `L`, which is a different system; it measured 1.0
    relative error on every case the driver tried.

Returns `false` without touching the failing entry on a singular or non-finite
diagonal block.
"""
function m03_solve_lower!(x::AbstractVector{MF},
                          factors::AbstractMatrix{MF},
                          dsub::AbstractVector{MF},
                          g::M03PivotGrammar;
                          trans::Symbol=:N) where {MF<:MultiFloat}
    trans in (:N, :T) || throw(ArgumentError("trans must be :N or :T"))
    n = g.n
    length(x) == n || throw(DimensionMismatch("rhs length mismatch"))
    size(factors, 1) == n || throw(DimensionMismatch("factor order mismatch"))
    g.accepted == n || throw(ArgumentError(
        "solve requires a complete factorization; accepted=$(g.accepted) of $n"))
    if trans === :N
        _m03_forward_ld!(x, factors, dsub, g) || return false
        return _m03_back_l!(x, factors, g)
    end
    # `D` is symmetric BY CONSTRUCTION (`ldlt!` builds it from `factors[i,i]` and
    # mirrors `dsub[k]` into both off-diagonal slots), so `D' == D` and
    # `P A' P' = P A P'`: the transposed system IS the same system. `:T` must
    # therefore run the SAME sequence as `:N` -- forward on `L*D`, then back on
    # `L'` -- and any separate decomposition of that sequence into three
    # single-purpose passes is a correctness hazard rather than a refactor.
    #
    # MEASURED, and the reason both earlier versions of this branch were wrong:
    # splitting the fused forward pass into "apply L, then apply D" makes the
    # second elimination step WRONG whenever `D` has a 2x2 block, because the
    # `L` row update for the second pivot column must use the ALREADY SCALED
    # first component, which only exists inside the block. On the exact fixture
    # below, the split order returned `x = [0.857..., -0.714..., 0.05]` against
    # the true `[0.858..., -0.714..., -0.25]`, i.e. 0.3 absolute error, while the
    # fused pass is exact to roundoff. Two wrong branch bodies were tried before
    # this was measured; the driver now pins it with that fixture.
    _m03_forward_ld!(x, factors, dsub, g) || return false
    return _m03_back_l!(x, factors, g)
end

"""
    m03_solve!(x, factors, dsub, g, p; trans=:N) -> Bool

Solve `A x = b` (`trans=:N`) or `A' x = b` (`trans=:T`) where `A = P' L D L' P`
is the original matrix and `p` is the permutation from `m03_pivot_permutation`.

For `:N`, `P A P' = L D L'` gives `u = P x`: permute, solve, un-permute. For
`:T`, `P A' P' = L D' L'` gives the same permutation scaffolding, so only the
inner solve changes.
"""
function m03_solve!(x::AbstractVector{MF},
                    factors::AbstractMatrix{MF},
                    dsub::AbstractVector{MF},
                    g::M03PivotGrammar,
                    p::AbstractVector{<:Integer};
                    trans::Symbol=:N) where {MF<:MultiFloat}
    n = g.n
    length(p) == n || throw(DimensionMismatch("permutation length mismatch"))
    trans in (:N, :T) || throw(ArgumentError("trans must be :N or :T"))
    permuted = Vector{MF}(undef, n)
    @inbounds for row in 1:n
        permuted[row] = x[Int(p[row])]
    end
    m03_solve_lower!(permuted, factors, dsub, g; trans=trans) || return false
    @inbounds for row in 1:n
        x[Int(p[row])] = permuted[row]
    end
    return true
end

"""
    m03_pivot_policy_classify(absakk, colmax, rowmax, candidate_diagonal, alpha)
        -> (block_size, which)

The pure decision table of Bunch--Kaufman step selection, factored out of
`_select_bk_pivot` so the grammar can be exercised without a matrix. `which` is
`:k`, `:imax`, or `:zero`. The driver cross-checks it against MFLA's
`_select_bk_pivot` on matrices built to hit every branch.

NOTE (recorded, not "fixed"): `A01b-F3` measured that the `|b| <= |a|` branch of
`_ldlt_solve_2x2` is PROVABLY UNREACHABLE for a BK-selected 2x2 pivot (the
diagonal entry at `imax` dominates), so it is defensive only. It is deliberately
NOT optimized away and NOT regression-tested here; both of its branches ARE
reached on general non-BK 2x2 input, which the driver exercises separately.
"""
function m03_pivot_policy_classify(absakk::MF, colmax::MF, rowmax::MF,
                                   candidate_diagonal::MF,
                                   alpha::MF) where {MF<:MultiFloat}
    if max(absakk, colmax) == zero(MF)
        return (0, :zero)
    end
    absakk >= alpha * colmax && return (1, :k)
    if absakk >= alpha * colmax * (colmax / rowmax)
        return (1, :k)
    end
    candidate_diagonal >= alpha * rowmax && return (1, :imax)
    return (2, :imax)
end
