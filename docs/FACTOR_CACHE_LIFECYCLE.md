# Factor-cache and solver lifecycle diagrams

Three ASCII state diagrams: the standalone factor API, the reusable factor-cache
API (including the fail-closed paths), and the LinearSolve adapter. Each state is
labeled (e.g. `unprepared`, `prepared/invalidated`, `factorized/success`) and
each transition is labeled with the call that causes it.

The cache-specific states map to the symbolic `factor_state` values the cache
reports: `:invalidated`, `:nonfinite_input`, `:not_posdef`, `:singular`, and
`:success`.

## 1. Standalone factor API

Every `lu!` / `cholesky!` / `ldlt!` / `rrqr!` call builds a **fresh, ownable**
factor object. Factorization is destructive (`copy(A)` when the input must
survive), and with `check=false` a fully-initialized failed factor is returned
and remains queryable.

```
 A (caller-owned, possibly copied)

   |  lu!(A) / cholesky!(A) / ldlt!(A) / rrqr!(A)     (check=true by default)
   v
[ fresh factor: MFlu / MFCholesky / MFLDLT / MFQR ]
   |
   |  check=true and invalid input / failed pivot  -->  throws
   |  check=false + numerical failure               -->  [ factor/failed ]
   |
   +-----------------------------+------------------------------+
   |  success                    |  failed                     |
   v                             v                             |
[ factor/success ]          [ factor/failed ]                  |
   |                              |                            |
   |  ldiv!(x, F, b)              |  solve/ldiv! throws        |
   |  solve(F, b)                 |  (before mutating dest)    |
   v                              v                            |
[ solved x ]             [ query factor_status / factor_state / |
                           factor_diagnostics on the failure ] |
                                                            <---+
   Each new standalone call returns an independent factor; the
   caller owns its metadata. No caching or reuse between calls.
```

## 2. Factor-cache API lifecycle (with fail-closed paths)

A cache owns its storage and reuses it across calls. `prepare!` is the only
growth point. `factorize!` invalidates **before** touching storage (fail-closed),
so `issuccess` becomes true only after a complete success, and every `solve!`
refuses (throws, without mutating its destination) unless the cache is
successful.

```
[ unprepared ]  (fresh cache, status = -2, state = :invalidated)
   |
   |  prepare!(cache, n)                 (RRQR: prepare!(cache, m, n))
   v
[ prepared / invalidated ]               state = :invalidated
   |
   |  factorize!(cache, A)               (fail-closed: invalidates first;
   |                                      same-size storage only, never grows)
   v
[ factorized ]  -- full success? -->
   |        \                        yes                     no
   |         +-------------> [ factorized/success ]    [ factorized/failed ]
   |                             state = :success             state = :nonfinite_input /
   |                             issuccess == true             :not_posdef / :singular
   |                                                            (check=false)
   |                                                              |
   |  solve!(x, cache, b)  solve!(X, cache, B)                    | solve! throws
   |        (0-byte vector warm)                                 | (destination untouched)
   |           |                                                  |
   |           v                                                  v
   |      [ solved x ]                              [ replacement-A recovery ]
   |
   |  -------- refresh / reconfiguration / invalidation paths --------
   |
   |  invalidate!(cache)            (after caller mutates A in place)
   |  reconfigure!(cache,new_cfg)   (then prepare! again: config change
   |                                 may need larger workspace)
   |  prepare!(cache, n)            (explicit resize at a new size)
   |
   +--------> back to [ prepared!/invalidated ]
              (solve! now throws until factorize! runs again)

   Recovery after failure:
      [ factorized/failed ]  --factorize!(cache, A') (replacement A, same size)-->
                           [ factorized/success ]

   Guard rails (throw, cache unchanged):
   * factorize! at a size larger than prepared capacity  -> ArgumentError
   * solve! / apply_q! / solve_r! / numerical_rank when !issuccess -> throws
   * any hot-path call with a `config` != cache.config  -> ArgumentError
     (use reconfigure! then prepare!, never pass a different config)
```

## 3. LinearSolve adapter lifecycle

`MultiFloatLU` / `MultiFloatCholesky` are the optional LinearSolve.jl weak-dep
extension. `init` builds an **empty** cache (`init_cacheval`) to fix the
`LinearCache.cacheval` field type **without** running the O(n³) factorization.
Storage is grown on the first (or size-changing) factorization and reused
afterward. A failed factorization returns `ReturnCode.Failure` while leaving
`isfresh` true so the caller can replace `A` and retry.

```
            LinearProblem(A, b) ; alg = MultiFloatLU() / MultiFloatCholesky()
                     |
                     | LinearSolve.init / init_cacheval
                     v
         [ initialized ]  LinearCache holds an EMPTY cache (no factorization)
                     |
                     | first SciMLBase.solve!(cache, alg)
                     |   (cache.isfresh == true)
                     v
         [ factorize into cache ]  prepare!(n) if size changed; factorize!(A)
                     |
          success? yes                       no
                     |                         |
                     v                         |
              isfresh = false                 [failure] retcode = Failure
                     |                         |  isfresh stays TRUE
                     |                         |  (cache stays in failed/invalidated
                     |                         |   state, no silent fallback)
                     |                         v
                     v               caller replaces A and calls solve! again
              [ first solve ]        -> returns to [factorize into cache]
                     |
              +------+-------------+
              | RHS-only update    | A-update (LinearSolve marks cache fresh)
              | (isfresh stays     | (isfresh -> true)
              |  false)            |
              v                    v
      [reuse factor, 0-byte]   [re-factorize into existing storage]
      retcode = Success        then [solve]; success -> retcode Success,
                               failure -> retcode Failure, isfresh true, retry

   Key invariants:
   * init does not run the real factorization (no double numerical work).
   * an unchanged A with a new RHS reuses the factor (0-byte RHS-reuse solve).
   * an A update re-factorizes into the cache's existing storage.
   * a failed factorization reports Failure and keeps the cache fresh so a
     replacement A can be retried; MFLA never silently falls back.
```

## Reading the diagrams

- **States** are bracketed, e.g. `[prepared/invalidated]`; the labeled value in
  parentheses is the `factor_state` a cache reports for that state.
- **Transitions** are arrow labels naming the exact call that moves between
  states.
- **Fail-closed** means a transition to a failure state can only be escaped by
  re-running `factorize!` with a valid `A` (replacement-A recovery), never by a
  later `solve!`.
