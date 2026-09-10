# =============================================================================
# M01 contract layer — operator snapshot vs factor storage, block grammar
# =============================================================================
#
# New, inert contract layer (see `src/contracts/factors.jl` for the ownership
# rules). Nothing here is `include`d by the package bootstrap yet.
#
# Core distinction encoded here (ADR-002; M01 step 3):
#
#   factor storage    the arrays a factorization owns: the packed L/U/D factor,
#                     pivots, block grammar, permutation. `factor_matrix(F)`
#                     BORROWS this — mutating it is not a supported operation.
#   operator snapshot an explicit, expensive COPY of the caller's operator,
#                     produced by `copy_operator_snapshot`. Mutating the
#                     snapshot must never change the factor, and mutating the
#                     factor must never change the snapshot.
#
# Block grammar (the Bunch-Kaufman 1x1/2x2 pivot pattern of an MFLDLT) is
# recorded as its own independent, immutable description so a downstream
# adapter can convert it exactly once instead of re-deriving it per query.

# Self-contained dependency declaration (see `factors.jl`).
#
# @integration/core-cutover: the `const MultiFloat` declaration is GUARDED.
# Unguarded it is correct for the standalone case this file was developed in —
# included into its own module, nothing else has bound the name — but it is a
# hard error once the file is `include`d into `MultiFloatLinearAlgebra` itself,
# which already binds `MultiFloat` via `import MultiFloats: MultiFloat,
# MultiFloatVec` at src/MultiFloatLinearAlgebra.jl:7:
#
#   ERROR: cannot declare MultiFloatLinearAlgebra.MultiFloat constant;
#          it was already declared as an import
#
# The guard keeps the standalone path working and removes the collision. It is
# the same class of defect as S06's driver gap: a file exercised in only ONE of
# its two inclusion modes.
import MultiFloatLinearAlgebra
import MultiFloats
if !isdefined(@__MODULE__, :MultiFloat)
    const MultiFloat = MultiFloats.MultiFloat
end

# This file is independently includable, so it does not rely on a hash helper
# defined in a sibling contract file. `_grammar_mix` is a pure integer mix with
# no numeric effect.
@inline function _grammar_mix(value::UInt64)
    h = value
    h ⊻= h >> 33
    h *= 0xff51afd7ed558ccd
    h ⊻= h >> 33
    h *= 0xc4ceb9fe1a85ec53
    h ⊻= h >> 33
    return h
end

"""
    BlockGrammar

An independent, immutable description of a block-pivot grammar.

`blocks[k]` is the size of the block starting at position `k`; `counts` maps a
block size to how many blocks of that size occur; `complete` is `false` when
the recorded pattern ends before it covers `row_count` rows, which is exactly
what a mid-factorization breakdown produces. `fingerprint` is a pure hash of
the pattern, for cheap identity checks across the adapter boundary.
"""
struct BlockGrammar
    row_count::Int
    blocks::Vector{UInt8}
    counts::Tuple{Int,Int}
    complete::Bool
    fingerprint::UInt64
end

function Base.show(io::IO, grammar::BlockGrammar)
    print(io, "BlockGrammar(rows=", grammar.row_count,
          ", 1x1=", grammar.counts[1], ", 2x2=", grammar.counts[2],
          ", complete=", grammar.complete, ")")
    return nothing
end

Base.:(==)(a::BlockGrammar, b::BlockGrammar) =
    a.row_count == b.row_count && a.blocks == b.blocks &&
    a.counts == b.counts && a.complete == b.complete &&
    a.fingerprint == b.fingerprint

"""
    block_grammar(blocks, row_count) -> BlockGrammar

Build an independent grammar description from a block-size prefix vector.
Stops at the first entry that is neither `1` nor `2`, and marks the grammar
`complete` only if the accepted blocks cover `row_count` rows exactly.
"""
function block_grammar(blocks::AbstractVector{UInt8}, row_count::Integer)
    row_count >= 0 || throw(ArgumentError("row_count must be nonnegative"))
    accepted = UInt8[]
    covered = 0
    one_by_one = 0
    two_by_two = 0
    complete = true
    for block in blocks
        size = Int(block)
        if size == 1
            one_by_one += 1
        elseif size == 2
            two_by_two += 1
        else
            complete = false
            break
        end
        covered += size
        push!(accepted, block)
        if covered > row_count
            complete = false
            break
        end
    end
    complete &= covered == row_count
    return BlockGrammar(
        row_count, accepted, (one_by_one, two_by_two), complete,
        _grammar_fingerprint(accepted),
    )
end

function _grammar_fingerprint(blocks::Vector{UInt8})
    fingerprint = 0xcbf29ce484222325
    for block in blocks
        fingerprint ⊻= UInt64(block)
        fingerprint = _grammar_mix(fingerprint)
    end
    return _grammar_mix(fingerprint ⊻ UInt64(length(blocks)))
end

"""
    grammar_block_sizes(grammar) -> Vector{Int}

Convenience expansion of a grammar into per-block sizes, for adapters that
consume a size list rather than a size tag.
"""
grammar_block_sizes(grammar::BlockGrammar) = Int[Int(b) for b in grammar.blocks]

# ---------------------------------------------------------------------------
# Operator snapshots
# ---------------------------------------------------------------------------

"""
    OperatorSnapshot{MF}

An explicit owned copy of a caller's operator. This is deliberately a distinct
type from every factor type: it is not a factorization, it has no status, and
`same_factor_storage(snapshot, factor)` is always `false`. Constructing one is
an expensive, caller-visible copy; it must not be used on a hot path.
"""
struct OperatorSnapshot{MF<:MultiFloat}
    data::Matrix{MF}
    size::Tuple{Int,Int}
    fingerprint::UInt64
    source_objectid::UInt
end

"""
    copy_operator_snapshot(A::AbstractMatrix) -> OperatorSnapshot

Explicitly copy `A` into caller-owned snapshot storage. The snapshot shares no
storage with `A`, so later mutation of either side is invisible to the other.
"""
function copy_operator_snapshot(A::AbstractMatrix{MF}) where {MF<:MultiFloat}
    Base.require_one_based_indexing(A)
    data = Matrix{MF}(undef, size(A, 1), size(A, 2))
    copyto!(data, A)
    return OperatorSnapshot{MF}(
        data, size(A), _matrix_fingerprint(data), objectid(A),
    )
end

Base.size(snapshot::OperatorSnapshot) = snapshot.size
Base.size(snapshot::OperatorSnapshot, dimension::Integer) = snapshot.size[dimension]
Base.eltype(::OperatorSnapshot{MF}) where {MF} = MF

"""
    snapshot_matrix(snapshot) -> Matrix

The snapshot's own storage. Unlike `factor_matrix(F)`, this matrix is
caller-owned and supported for mutation.
"""
snapshot_matrix(snapshot::OperatorSnapshot) = snapshot.data

"""
    snapshot_fingerprint(snapshot) -> UInt64

Pure hash of the snapshot payload. `not_run`-safe: it is a real scan of the
snapshot's own storage, not of any factor.
"""
snapshot_fingerprint(snapshot::OperatorSnapshot) = _matrix_fingerprint(snapshot.data)

"""
    snapshot_matches(snapshot, A) -> Bool

`true` when the snapshot payload still equals `A`. Used to prove that mutating
a snapshot is visible only in the snapshot.
"""
function snapshot_matches(snapshot::OperatorSnapshot, A::AbstractMatrix)
    size(snapshot) == size(A) || return false
    return _matrix_fingerprint(A) == snapshot_fingerprint(snapshot)
end

function _matrix_fingerprint(A::AbstractMatrix)
    fingerprint = 0xcbf29ce484222325
    @inbounds for column in axes(A, 2), row in axes(A, 1)
        value = A[row, column]
        fingerprint ⊻= _grammar_mix(reinterpret(UInt64, UInt64(hash(value))))
        fingerprint = _grammar_mix(fingerprint)
    end
    return _grammar_mix(fingerprint ⊻ UInt64(length(A)))
end

# A snapshot is never factor storage, in either direction. There is
# deliberately no `(::Any, ::Any)` method here: in Julia `f(::Any, ::Any)` is
# more specific than the untyped `same_factor_storage(a, b)` fallback and would
# shadow it. A snapshot is not an `AbstractMFFactorization`, so the fallback is
# already the correct answer for every snapshot/factor pair.
same_factor_storage(::OperatorSnapshot, ::OperatorSnapshot) = false
same_factor_storage(::OperatorSnapshot, ::Any) = false
same_factor_storage(::Any, ::OperatorSnapshot) = false
