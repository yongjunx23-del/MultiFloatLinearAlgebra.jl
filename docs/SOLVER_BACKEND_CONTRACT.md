# Solver backend contract

This document freezes the solver-facing contract for the dense fixed-
MultiFloat provider. MFLA executes explicitly requested numerical operations;
the caller owns formulation, fallback, precision escalation, stopping, and
certification policy.

## Opaque factors

`MFCholesky`, `MFLU`, `MFLDLT`, and `MFQR` implement
`AbstractMFFactorization`. Solver code may use only:

```julia
issuccess(F)
factor_kind(F)
factor_status(F)       # compatibility integer
factor_state(F)        # stable symbolic state
factor_matrix(F)       # borrowed, read-only factor payload
factor_diagnostics(F)  # vectors in the result are defensive copies
factor_precision(F)    # exact MultiFloat scalar type
factor_provider(F)     # :mfla
size(F)
eltype(F)
```

LDLT callers that need only structural facts use the O(n) lightweight
accessors rather than the comprehensive diagnostic scan:

```julia
factor_pivots(F)       # raw Bunch-Kaufman step pivots; defensive copy
factor_blocks(F)       # UInt8 1/2 block-start markers; defensive copy
factor_permutation(F)  # p such that A_original[p, p] = L*D*L'
factor_inertia(F)      # (positive, negative, zero)
```

`factor_inertia` scans only the stored D blocks. It does not perform the full
factor-payload finiteness and block-quality scan supplied by
`factor_diagnostics`.

The stable states are `:success`, `:nonfinite_input`, `:not_posdef`,
`:singular`, and `:numerical_breakdown`. RRQR rank deficiency is not a failed
factorization: callers choose a rank threshold through `numerical_rank` or
`factor_diagnostics(F; atol, rtol)`.

Cholesky, LU, and LDLT solve APIs are:

```julia
ldiv!(x, F; config)       # overwrite RHS in place
ldiv!(X, F; config)
ldiv!(x, F, b; config)    # distinct caller-owned destination/source
ldiv!(X, F, B; config)
solve(F, b; config)       # allocating convenience wrapper
```

Vector solves use TRSV and matrix RHS solves use TRSM. Square RRQR factors
with an exactly nonzero R diagonal also implement `ldiv!` without selecting a
rank threshold. Rectangular, rank-selected, and least-squares routes use
`factor_permutation`, `apply_q!`, and `solve_r!` explicitly: MFLA does not
infer rank or formulation.

Concrete fields and Householder layout remain private. The accessors above and
the QR metadata accessors return caller-owned copies where applicable.

## Failure and storage

`cholesky!`, `lu!`, `ldlt!`, and `rrqr!` are destructive. Their input matrix
is the factor payload and may be partially overwritten after a numerical
failure. With `check=true`, invalid input or a failed pivot throws. With
`check=false`, a fully initialized failed factor is returned.

Every failed factor remains safe to query through all accessors and
`factor_diagnostics`. `factor_matrix` is allowed and exposes the possibly
partial destructive payload. A solve with a failed factor throws before
mutating its destination. A non-finite LDLT input is checked before any
lower-to-upper mirror write.

The factor matrix is borrowed from the destructive input and must not be
mutated by callers. Solve does not mutate factor payload or metadata.

## Workspace lifetime and concurrency

`MFWorkspace` owns reusable factorization scratch and packed GEMM buffers.
Returned LU, LDLT, and RRQR factors copy the metadata needed by solve and
diagnostics. Therefore:

- multiple factors created from one workspace may remain live;
- later workspace reuse does not invalidate a factor;
- workspace growth does not invalidate a factor;
- each factor continues to borrow only its destructive input matrix.

One `MFWorkspace` must not be used by concurrent factorization calls because
its factorization scratch is mutable. Packed GEMM calls that share one
`GemmWorkspace`, or the GEMM component of one `MFWorkspace`, are safe: an
object-local lock serializes each complete packed call. Separate workspaces
retain call-level concurrency.

No process-global mutable numerical workspace is used.

## Precision and capabilities

The production provider contract is exactly `MultiFloat{Float64,N}` for
`N=1:4`. Other base types and x5-x8 arithmetic are outside the public support
range even where internal code remains generic.

`capabilities(T)` is a pure descriptor for the exact Float64-based MultiFloat
type `T`. It includes `provider`, `scalar_type`, workspace/concurrency facts,
SYRK storage facts, `ldlt_lightweight_metadata`, exact
`mixed_residual_target_types`, and these ownership facts:

```text
factor_metadata_ownership = :factor_owned
factor_matrix_ownership = :borrowed_input
factorization_destructive = true
factor_solve_mutates_factor = false
```

A `KernelConfig` produced by `with_gemm_profile` is tagged with the profile's
exact arithmetic type. `gemm_plan` and `gemm!` reject use with another type;
plain configurations and uncalibrated `:auto` routing remain type-neutral.

The only promoted residual pairs are:

```text
Float64x2 -> Float64x3 or Float64x4
Float64x3 -> Float64x4
Float64x4 -> no higher MultiFloat target
```

`residual_mixed!` requires every source operand to have the same exact source
type and requires the output type to be one of that source type's listed
targets. There is no implicit conversion, precision escalation, BigFloat
fallback, or backend substitution.

## Symmetric storage

`syrk!(C, panel, alpha, beta; uplo=:lower, config)` accepts `:lower` or `:L`
and writes only the lower triangle. The upper triangle is inactive, is not
read, and is preserved. When `beta == 0`, the lower destination is not read.
Other `uplo` values are rejected. Cholesky consumes
only authoritative lower input and uses lower-only SYRK updates; the upper
triangle may contain stale, NaN, Inf, or otherwise invalid data.

`symv!`, `residual!`, and `residual_mixed!` similarly honor their explicit
lower/upper authoritative-triangle options. Mirroring is never implicit on a
solver hot path unless an internal factorization requires its own private
dense representation.
