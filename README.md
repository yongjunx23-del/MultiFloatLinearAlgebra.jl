# MultiFloatLinearAlgebra.jl

`MultiFloatLinearAlgebra.jl` (MFLA) is a CPU linear-algebra backend for
[MultiFloats.jl](https://github.com/dzhang314/MultiFloats.jl). It provides
specialized dense kernels and factorizations for `Float64x2`, `Float64x3`, and
`Float64x4` without converting through `BigFloat`.

MFLA is solver-independent. Optimization packages can use it as a numerical
backend while keeping model semantics, KKT assembly, precision policy,
refinement policy, and certification in their own layers.

## Installation

Until the package is registered in Julia General:

```julia
using Pkg
Pkg.add(url = "https://github.com/yongjunx23-del/MultiFloatLinearAlgebra.jl")
```

## Quick start

```julia
using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra

T = Float64x4
A = T.(randn(128, 128))
B = T.(randn(128, 128))
C = zeros(T, 128, 128)

gemm!(C, A, B)

R = T.(randn(128, 128))
S = R * transpose(R) + T(128) * I
F = MultiFloatLinearAlgebra.cholesky!(copy(Matrix(S)))
x = solve(F, T.(randn(128)))
```

## Main features

- BLAS-like dense kernels: `gemv!`, `gemm!`, `syrk!`, `gemmt!`, `trsm!`,
  `trsv!`, `trmm!`, and `symv!`;
- Cholesky, LU, symmetric-indefinite LDLT, and column-pivoted RRQR;
- vector and multi-RHS factor solves;
- factor status, rank, pivot, inertia, precision, provider, and diagnostic APIs;
- residual, backward-error, mixed-precision residual, and refinement-correction
  utilities;
- reusable caller-owned workspaces;
- deterministic reduction order in the supported production kernels;
- explicit threading and no solver-specific policy.

The supported production arithmetic types are `Float64x2`, `Float64x3`, and
`Float64x4`.

## Performance

The table below compares MFLA with Julia's generic `LinearAlgebra` algorithms
at `n = 256`. Measurements were collected on a GitHub-hosted Ubuntu 24.04
runner with Julia 1.12.6, four Julia threads, and one BLAS thread. Values are
speedups (`generic time / MFLA time`), so larger is better.

| Operation | `Float64x2` | `Float64x3` | `Float64x4` |
|---|---:|---:|---:|
| GEMM | 14.46x | 14.07x | 12.00x |
| TRSM | 4.44x | 4.88x | 7.88x |
| Cholesky | 2.25x | 5.15x | 6.11x |
| LU | 2.63x | 6.48x | 7.52x |

These numbers are machine- and workload-dependent, not portable performance
guarantees. Full benchmark methodology, timings, larger-size experiments, and
allocation measurements are in [`benchmark/RESULTS.md`](benchmark/RESULTS.md).

## Solver-facing factor API

The factorization types share a common public interface, including:

```text
issuccess
factor_matrix
factor_status
factor_kind
factor_precision
factor_provider
factor_diagnostics
numerical_rank
ldiv!
solve
```

Factor metadata required for later solves and diagnostics is factor-owned.
Reusable workspaces provide scratch storage without making solver policy part
of MFLA.

See [`docs/SOLVER_BACKEND_CONTRACT.md`](docs/SOLVER_BACKEND_CONTRACT.md) and
[`docs/SDPX_PROVIDER_READINESS.md`](docs/SDPX_PROVIDER_READINESS.md) for the
more detailed backend contract.

## Testing

```julia
using Pkg
Pkg.test("MultiFloatLinearAlgebra")
```

CI currently tests Julia 1.10, 1.11, and 1.12 on Linux and macOS.

## Contributors and AI disclosure

- **Yongjun Xu** — maintainer; design, implementation, numerical validation,
  benchmarking, and review.
- **OpenAI Codex** — substantial implementation, refactoring, testing, and
  documentation assistance under human review.

This repository is a fork of
[`dzhang314/MultiFloatLinearAlgebra.jl`](https://github.com/dzhang314/MultiFloatLinearAlgebra.jl),
originally created by **David K. Zhang**. Original MIT attribution is preserved.

## License

MIT. See [`LICENSE`](LICENSE).
