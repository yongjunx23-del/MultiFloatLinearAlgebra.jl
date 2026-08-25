# Factor-cache independent accuracy validation report

**Date:** 2025-08-25
**Branch:** `fix/factor-cache-contract-hardening`
**Script:** `benchmark/independent_accuracy.jl`
**Julia:** 1.12.6, `Threads.nthreads()` = 4 (threaded run), 512-bit `BigFloat` reference
**Result:** **162 checks, 9 failures** — every failure is one of two genuine
multi-RHS defects in the cache layer (see [Defects found](#defects-found)).
All single-RHS, reconstruction, backward-error, permutation, rank, pathological,
recovery, and serial/threaded-consistency checks **PASS**.

The script exits `1` on any failure (9 failures), as required.

---

## 1. Methodology

This is an *independent* validation. It does **not** compare the factor cache
against the standalone `cholesky!`/`lu!`/`ldlt!`/`rrqr!` factors, which share
the same numerical core. Instead every quantity is measured against a
**512-bit `BigFloat` reference** computed directly in the validation script:

* **Reference solves** — choose `x*` in `BigFloat`, form `b = A_big * x*`
  (512-bit), round `b` to the working `MultiFloat` type, solve with the cache,
  and compare against `x*`. Reported as the **max relative error**
  `‖x − x*‖∞ / ‖x*‖∞` (vector and multi-RHS).
* **Factor reconstruction** — rebuild the factors in `BigFloat` from the
  cache's owned storage and compare to the (BigFloat) input:
  * Cholesky: `A ≈ L·Lᵀ` (L from the lower triangle)
  * LU: `P·A ≈ L·U` (P derived from `ipiv`; L unit-lower, U upper)
  * LDLT: `A[p,p] ≈ L·D·Lᵀ` (L unit-lower, D block-diagonal from `factors`+`dsub`,
    p = final symmetric permutation from `pivots`/`blocks`)
  * RRQR: `A[:,p] ≈ Q·R` (Q rebuilt from Householder vectors, R from the upper
    triangle; also `QᵀQ ≈ I`)
* **Normwise backward error** of the *computed* solution:
  `‖b − A·x‖∞ / (‖A‖∞·‖x‖∞ + ‖b‖∞)`, computed in `BigFloat`.
* **Permutation correctness** — each `p` is asserted to be a bijection on
  `1:n`. LDLT inertia (`factor_diagnostics(cache).inertia`) is compared to the
  exact sign count of the generating diagonal `λ`.
* **LU pivot growth** `max|U|/max|A|` — finite and no blow-up (well-conditioned
  matrix ⇒ near 1 or below).
* **RRQR rank** via `numerical_rank(cache; atol, rtol)` on a rank-deficient
  rectangular matrix.

The caches are exercised through the uniform public API
(`prepare!`, `factorize!`, `solve!`, `factor_status`, `factor_state`,
`issuccess`, `factor_permutation`, `numerical_rank`, `factor_diagnostics`).

## 2. Matrices

| Kind | Matrix | Purpose |
|---|---|---|
| Cholesky | `G·Gᵀ + n·I`, `G∈ℝⁿˣⁿ` random | well-conditioned SPD |
| LU / RRQR | `R + n·I`, `R` random | well-conditioned diagonally dominant |
| LDLT | `Gᵀ·diag(λ)·G`, `λ` mixed ± sign | symmetric indefinite, known inertia |
| all | `D·(Rₛ + n·I)·D`, `D=diag(10^{4(i-1)})` | badly scaled (dynamic range ≳1e92) |
| RRQR | `G·W` (12×8, rank 6) | rank-deficient rectangular |
| singular | first row/column zero (LU), symmetric zero pivot (LDLT) | failure paths |

Sizes: n = 24 (accuracy), n = 12 (pathological / badly scaled), n = 72
(threading).

## 3. Tolerances

* Relative error / reconstruction / orthogonality: `tol_rel = 1e6·eps(T)`
* Normwise backward error: `tol_back = 1e7·eps(T)`

Measured `eps`: `Float64x2 ≈ 4.9e-32`, `Float64x3 ≈ 1.1e-47`,
`Float64x4 ≈ 2.4e-63`.

## 4. Results

### 4.1 Solution max relative error vs BigFloat reference (vector / multi-RHS)

| Type | Cholesky | LU | LDLT | RRQR |
|---|---|---|---|---|
| x2 | 4.86e-32 / 4.61e-32 | 4.64e-32 / 5.61e-32 | 7.35e-31 / **FAIL** | 7.63e-32 / **FAIL** |
| x3 | 8.91e-49 / 3.37e-48 | 2.78e-48 / 1.53e-48 | 4.59e-47 / **FAIL** | 3.86e-48 / **FAIL** |
| x4 | 3.67e-64 / 3.62e-64 | 4.57e-64 / 5.44e-64 | 1.10e-61 / **FAIL** | 1.50e-63 / **FAIL** |

Vector solves are at or below machine eps for every type and kind. The
multi-RHS columns marked **FAIL** are the two genuine defects (Section 6):
LDLT multi-RHS throws; RRQR multi-RHS returns a corrupt solution.

### 4. Factor reconstruction residual

| Type | Cholesky `A≈L·Lᵀ` | LU `P·A≈L·U` | LDLT `A[p,p]≈L·D·Lᵀ` | RRQR `A[:,p]≈Q·R` |
|---|---|---|---|---|
| x2 | 4.80e-32 | 2.27e-32 | 6.46e-32 | 1.13e-31 |
| x3 | 1.32e-48 | 8.26e-49 | 3.09e-48 | 3.23e-48 |
| x4 | 5.65e-64 | 5.45e-65 | 2.38e-64 | 3.73e-63 |

All at machine precision. `Q` also satisfies `QᵀQ ≈ I` to within `eps`.

### Normwise backward error (computed solution)

| Type | Cholesky | LU | LDLT (vector) | RRQR (multi-RHS FAIL) |
|---|---|---|---|---|
| x2 | 1.74e-32 | 2.71e-32 | 5.29e-32 | **0.65** |
| x3 | 8.30e-49 | 1.29e-48 | 1.25e-48 | **0.66** |
| x4 | 1.38e-64 | 2.46e-64 | 4.09e-64 | **0.76** |

The `RRQR` backward-error **FAIL** is a direct consequence of the multi-RHS
solve defect (it corrupts multi-column solutions; the single-vector value is
clean). Cholesky / LU / LDLT single-vector backward errors are at eps.

### Permutation correctness

* LU: `ipiv` is a valid row permutation on `1:n` — **PASS** (all types).
* LDLT / LU: final symmetric permutation `p` is a valid permutation — **PASS**.
* LDLT inertia: exact match to the generating diagonal sign count
  `(12,12,0)` for all types — **PASS**.

### LU pivot growth

| Type | growth = max|U|/max|A| |
|---|---|---|
| x2 | 1.00e+00 |
| x3 | 9.99e-01 |
| x4 | 9.96e-01 |

Finite and ≪ blow-up threshold (1e3) for the well-conditioned diagonally
dominant matrix. (Growth can legitimately dip below 1 when the max element of A
is eliminated.)

### RRQR rank / reconstruction (rank-deficient, m=12, n=8, rank=6)

`issuccess` true (rank deficiency is not a failure), `numerical_rank` = 6 with
`rank < n`, reconstruction `A[:,p]≈Q·R` = 2.57e-32 — **PASS**.

### Pathological cases (all types)

| Case | Result |
|---|---|
| Non-finite input → `factor_status == -1`, `factor_state == :nonfinite_input` | PASS |
| Singular LU → `factor_status > 0`, `:singular`, `check=true` throws `SingularException` | PASS |
| Singular LDLT → same failure contract | PASS |
| Badly scaled matrix (dynamic range ≳10⁹²) → factorize succeeds; normwise backward error stays at eps (`4.0e-49`/`6.3e-43`/`5.8e-78` for x2; similar x3/x4) | PASS |
| Success → failure (`check=true` throws) → cache no longer `issuccess` → `solve!` refused → replace `A`, re-factorize → full recovery | PASS |

### Serial (1 thread) vs threaded (4 threads) consistency

For every type × kind, `KernelConfig(thread_count=1)` vs
`KernelConfig(thread_count=4)` produces **bit-identical** factor matrices,
pivot/permutation metadata, and solve results (n = 72, exercised under
`-t 4` so the threaded kernels actually engage). **PASS (24/24 checks).**

## 5. Summary

| Category | PASS / FAIL |
|---|---|
| Solution relative error (vector) | 12 / 0 |
| Solution relative error (multi-RHS) | 6 / 6 (defects, §6) |
| Factor reconstruction | 12 / 0 |
| Backward error | 9 / 3 (defects, §6) |
| Permutation / inertia | 12 / 0 |
| LU pivot growth | 3 / 0 |
| RRQR rank / reconstruction | 3 / 0 |
| Pathological (non-finite, singular, rank-def, badly-scaled) | 36 / 0 |
| Success → failure → recovery | 18 / 0 |
| Serial vs threaded consistency | 24 / 0 |
| **Total** | **153 / 9** |

All failures stem from the two defects below; there are **no** false-positive
failures from the validation itself.

## 6. Defects found

These are genuine defects in the **factor-cache layer only** (the standalone
factors are correct). They affect the **multi-RHS (`AbstractMatrix`) `solve!`**
path of the LDLT and RRQR caches. Vector (`AbstractVector`) solves, all
factorizations, and every other check are correct.

### Defect 1 — LDLT cache multi-RHS `solve!` throws `MethodError`

`MFLDLTCache` matrix `solve!` routes into `_ldlt_cache_solve!`, which calls
`trsv!` on a `Matrix` destination (`src/factor_caches.jl`). `trsv!` only has a
`Vector` method, so any `solve!(X::Matrix, ldlt_cache, B::Matrix)` throws
`MethodError: no method matching trsv!(::Matrix, ::Matrix)`.
The standalone `ldlt!` + `ldiv!` matrix path works. Reproduced for all types.

2. **RRQR cache multi-RHS `solve!` returns a corrupt solution** — `src/factor_caches.jl`'s
   `_cache_permute_qr_solution!` restores the row permutation with linear
   indexing (`destination[start]`/`destination[target]`), which treats a matrix
   destination as a flat column-major vector. For a single RHS this is correct;
   for multiple RHS it permutes individual elements instead of whole rows,
   silently corrupting every column after the first. Measured residual for a
   well-conditioned 8×8 multi-RHS solve ≈ **2.2**, versus ≈ 8e-32 via the
   standalone `rrqr!` + `ldiv!`.

Both defects are flagged as FAIL by the script and cause a nonzero exit.

## 7. How to run

```bash
export JULIA_DEPOT_PATH="/Users/xuyongjun/Desktop/project/SDPX/.julia-depot"
julia --project=. -t 4 benchmark/independent_accuracy.jl   # threaded consistency
julia --project=. -t 1 benchmark/independent_accuracy.jl   # serial-only
```

Expected exit code: **1** (because of the two defects). After the defects are
fixed in `src/`, all 162 checks should PASS and the exit code becomes **0**.
