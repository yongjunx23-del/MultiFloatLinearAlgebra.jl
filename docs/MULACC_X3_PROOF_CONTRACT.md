# Float64x3 fused multiply-accumulate proof contract

Status: proof boundary defined; formal proof not yet completed.

This document isolates the arithmetic claim behind the package-internal
`mulacc_x3` network. It is a soft integration boundary for an external proof
repository. MultiFloatLinearAlgebra has no dependency on a proof tool or IR,
and production kernels do not execute proof or calibration code.

## Operation boundary

The operation is lane-local:

```text
mulacc_x3(acc, x, y) -> result
```

Each argument is a `MultiFloatVec{4,Float64,3}`. For each of its four SIMD
lanes, the input is three binary64 limb bit patterns:

```text
acc = (a0, a1, a2)
x   = (x0, x1, x2)
y   = (y0, y1, y2)
```

The output is three binary64 limb bit patterns `(r0, r1, r2)`. There is no
cross-lane state or reduction. A proof can therefore reason about one scalar
lane and separately prove that SIMD operations preserve lane independence.

The package calls this network only with normalized MultiFloat inputs. The
target finite-input domain is:

- all input limbs are finite IEEE 754 binary64 values;
- every input triple satisfies `MultiFloats.isnormalized`;
- primitive arithmetic uses round-to-nearest, ties-to-even;
- `two_prod` uses an IEEE fused multiply-add for its error term;
- gradual underflow is enabled and floating-point reassociation is disabled;
- every intermediate required by the two compared networks is finite.

NaN, infinity, alternate rounding modes, flush-to-zero, and overflowed
intermediates are outside the current proof claim. Production behavior for
those cases is not advertised as a bitwise identity.

## Claim to prove

For every lane in the target domain, prove both:

1. `mulacc_x3(acc, x, y)` has exactly the same three output limb bit patterns
   as the current MultiFloats expression `acc + x * y`.
2. The output triple is normalized under the exact
   `MultiFloats.isnormalized` definition.

This is a network-equivalence claim, not a claim that either network is the
correctly rounded three-limb value of the real expression. It deliberately
tracks the established MultiFloats arithmetic semantics used as the reference
by MFLA.

The primitive network consists only of binary64 `+`, `-`, `*`, FMA,
`MultiFloats.two_sum`, `MultiFloats.fast_two_sum`, and
`MultiFloats.two_prod`. The implementation in
`src/kernels/mulacc_x3.jl` is the authoritative instruction order. A proof
artifact must pin the MultiFloats version or import the exact primitive
definitions used by the tested package environment.

## Reproducible vectors and invariants

`test/mulacc_x3_proof_vectors.jl` contains fixed UInt64 limb inputs and expected
UInt64 limb outputs. The cases cover:

- ordinary full-limb arithmetic;
- exact product/accumulator cancellation;
- cancellation across a wide exponent range;
- gradual-underflow intermediates;
- near-overflow finite intermediates;
- alternating signs;
- a subnormal-scale accumulator;
- an exact zero product.

The test reconstructs Float64 values from the UInt64 patterns and checks:

- every input is normalized;
- the fused output equals the committed expected bits;
- the fused and standard networks are limb-bit identical;
- every output is normalized;
- permuting SIMD lanes only permutes the corresponding outputs;
- the fused and standard outputs have identical 512-bit BigFloat error against
  `acc + x*y` evaluated from the represented input values.

These vectors are regression witnesses, not a finite substitute for the
universal proof. An external proof project can transcribe the constants
directly without importing MFLA at runtime.

## Integration rule

A future proof result should report:

- the exact MFLA and MultiFloats commits or versions;
- the floating-point semantics and excluded cases;
- whether both bit equivalence and normalization were established;
- a machine-checkable artifact and the command used to verify it;
- confirmation that every committed fixed vector is reproduced.

Until then, code and documentation must describe the identity as empirically
validated rather than formally proved. Any arithmetic-network change must
update the fixed outputs or demonstrate that they remain unchanged, rerun the
full adversarial suite, and invalidate or refresh the external proof artifact.
