# Changelog

## [Unreleased] — branch `feat-radial-map`

**Experiment, not an upstream proposal.** Stacked on `fix-alpha-beta-truncation` (unmerged,
PR #7). No issue filed and no PR opened — this branch exists to unblock a downstream
(`vimp-prediction`) trial balloon before it is known whether the balloon flies. No version
bump: the next unused number is `0.1.9` (0.1.6 = PR #9's anchor fix, 0.1.7 = PR #10's `gamma`
guard, 0.1.8 = PR #7), and claiming it would reserve a number the maintainers may want for
something else, for work that may never be proposed.

**Pushed to `origin` as of 2026-08-22, deliberately** — this branch is no longer local-only.
It is pushed so cluster runs of `vimp-prediction` can fetch the pinned commit directly
instead of rsyncing a working copy from a laptop. Pushing a branch is not proposing it: no
issue, no PR, no version claimed, and `stable` is untouched. If the balloon does not fly, the
branch is deleted from `origin` and nothing was spent upstream.

**Restacked twice on 2026-08-22**, as the branches below it merged. The anchor fix that used
to be this branch's third change landed upstream on its own as 0.1.6 (issue #8, PR #9), so
the first rebase dropped that commit as already applied and its entry moved to the 0.1.6
section below; the second followed PR #10 and PR #7's renumbering. What remains here is the
radial map and `slope_shifted` — both reachable only by passing a new keyword. The
Measurements below were re-taken against each new base on the same host rather than carried
over: the reference capture, the four example descriptor matrices and all 46 acceptance
criteria were re-run each time, and the bit-identity result came out stronger after the first
restack than before it (see Measurements).

### Features

- [x] `radial_map` selector on `expand_jacobi` and `JLGridFingerprints`, default `'cosine'`
  (bit-identical to the prior behavior — see Measurements), and a new `'log'` map,
  $x(r) = \gamma\,(1 - 2\ln(1+r/r_{\rm soft}) / \ln(1+r_{\rm cut}/r_{\rm soft}))$. Where the
  cosine map gives the near-nuclear region under one part in four hundred of the basis's
  dynamic range at every corner of the searched `rcut`/`rmin` box, the logarithmic map
  concentrates resolution there instead, governed by a new `rsoft` parameter (the softening
  length below which the map is linear in distance rather than logarithmic, keeping it
  finite at $r=0$). Motivated by a near-nuclear cusp in an all-electron potential that a full
  hyperparameter search over the existing cosine-map descriptor could not fit.
- [x] `slope_shifted` on both call sites, an additive keyword alongside `shifted` /
  `double_shifted`: constrains the radial basis to vanish at `rcut` in value *and* first
  derivative, rather than value alone. Needed because the cosine map supplies the derivative
  condition for free ($dx/dr \propto \sin\theta \to 0$ at `rcut`) while the logarithmic map,
  being strictly monotone, does not — so a basis merely `shifted` under the new map leaves a
  real kink at the seam (measured $39.7\,\text{Å}^{-1}$, flat under grid refinement, i.e. a
  genuine nonzero slope, not numerical noise). Built as
  $(\gamma+x)^2\,P_k^{(\alpha,\beta+2)}(x)$ rather than the algebraically equivalent
  subtractive form $\hat P_n = P_n - a_n - b_n P_1$: both span the same space (confirmed by
  rank), but $b_n$ grows like $n^2$ (from $-4.14$ at $n=2$ to $-5282$ at $n=24$) and swamps
  $P_n$ under any coefficient penalty. Consumes two orders, exactly as `double_shifted` does;
  mutually exclusive with it, and reaches the 1B call site that `double_shifted` does not.

### Measurements

Performed in the local podman dev container (`mldensity-dev`), same host for every
comparison — `setup.py` passes `-march=native`, so a comparison is only a comparison of two
code versions on one host. **46 of 46 acceptance criteria pass.** The harness and its raw
output are local development artifacts and are deliberately not published: this branch has no
issue and no pull request, so there is nowhere public that holds them, and citing a path would
point at a directory no other checkout has. Every number this section rests on is therefore
stated inline below rather than delegated to a link. Anyone reproducing it needs the branch,
the container and one command; ask the branch author for the harness.

- [x] **Bitwise neutral wherever no new keyword is passed, verified against a reference
  capture taken on the branch's own base.** All 109 swept arrays (`expand_jacobi` over
  exponent sets, `nmax`, `rmin`, `gamma`, and both shift flags, plus a full fcc-Al descriptor
  matrix) are `np.array_equal` — nothing changes at all. Re-measured on 2026-08-22 after the
  restack, and the result is stronger than before it: the pre-restack capture had exactly 24
  arrays differing, the ones combining `double_shifted` with $\gamma \neq 1$, because the
  anchor fix was then part of this branch. With that fix upstream in 0.1.6 and its commit
  dropped by the rebase, the changed set is empty, which is the claim this branch actually
  wants to make — every remaining change is reachable only by passing `radial_map`, `rsoft`
  or `slope_shifted`, and the reference sweep passes none of them. Criterion 1 of the
  verification harness was updated to expect the empty set rather than the 24; the
  pre-restack capture is kept alongside as `reference_prerestack.npz`. Re-captured again
  after the second restack, whose base added PR #10's `gamma <= 0` guard: the sweep runs at
  `gamma` ∈ {0.5, 1.0, 2.0}, all positive, so the guard is never reached and the empty
  changed set holds unchanged.
- [x] **The four example pipelines are unaffected**, checked more strongly than a stdout
  diff: their real settings dicts, run on their real tracked structures, produce
  bit-identical descriptor matrices and unchanged feature counts (`2d_mos2` 2346,
  `aluminium` 120, `benzene` 1572, `molybdenium` 812) against the base commit. (Their printed
  wall-clock timings were not compared — they do not reproduce run to run — and the
  pipelines cannot in any case run end-to-end from this checkout, since the `CHGCAR` files
  they read are not tracked.) Re-captured against the post-restack base on 2026-08-22, so the
  comparison is against the base this branch now sits on rather than the one it was written
  against; the pre-restack capture is kept as `examples_prerestack.npz`.
- [x] **The new behavior is real, not nominal.** The logarithmic map's endpoints land at
  $x=\pm\gamma$ to 15 digits and are strictly decreasing; its `rsoft`$\to\infty$ limit
  converges to a linear map (max deviation $2\times10^{-1} \to 5\times10^{-7}$ over three
  decades of `rsoft`); all five documented rejections (non-positive `rsoft`, `rsoft` set
  under `'cosine'`, nonzero `rmin` under `'log'`, an unknown map name, a negative distance)
  raise. `slope_shifted`'s vanishing point is a genuine double root — $|f|$ falls
  $100\times$ per decade of approach to `rcut`, against $10\times$ for `shifted` alone (a
  single root) — and its basis spans the same space as the algebraically-equivalent
  subtractive construction, confirmed by matching full rank on a sampling grid dense enough
  to resolve the logarithmic map's high orders (a coarser grid understated the rank of both
  constructions identically, which would have been a false pass).
- [x] `pytest` (59 tests) and `ruff check` / `ruff format --check` pass unmodified. The count
  rose from 48 across the two restacks without any test being added here: the base now carries
  the anchor fix's test module (8 cases) and PR #10's `gamma` guard cases (3), which this
  branch inherits instead of supplying.



## [0.1.8] - 2026-08-22 — branch `fix-alpha-beta-truncation`

Widens `expand_jacobi`'s alpha/beta exponents from `int` to `double`, fixing a silent
truncation-toward-zero that affected five of the eight parameter sets published across this
repo and its ecosystem. Filed as issue #6 before landing, per the project's rule that numerical
behavior changes only with the Sanvito group's review.

Numbered 0.1.8, not the 0.1.6 this branch originally claimed. Two PRs opened after it merged
before it, and each took the next unused patch version under the rule agreed in issue #8 —
*the PR merging first takes the next unused patch version*: PR #9 (the `double_shifted`
anchor fix) took 0.1.6, and PR #10 (the `gamma` guard) took 0.1.7. The rule keys on merge
order rather than on which branch wrote a number down first, so this branch renumbered twice
while under review. Nothing about the fix itself changed either time.

### Bug fixes

- [x] `expand_jacobi` (`polynomials.pyx`) declared its Jacobi weight exponents alpha/beta as `int`, while every worker beneath it (`calculate_jacobi`, `jacobi_eval_single`, `jacobi_eval`) already took `double`. Cython's int coercion truncated a non-integer argument toward zero instead of raising — no exception, no warning — so a fitted float exponent silently became the integer it truncated to (e.g. `7.875386069413652` evaluated as `7`). Widening the one signature to `double` is the whole fix; nothing downstream changes, and `fingerprints.py`'s docstring has always promised `list of float`.

### Behavior notes

- [x] `JLGridFingerprints.__init__` now raises `ValueError` for any alpha or beta component `<= -1`, naming the offending value. The Jacobi weight `(1-x)^alpha (1+x)^beta` is only defined for `alpha, beta > -1`; outside that domain the three-term recurrence's divisor can vanish, and `# cython: cdivision=True` turns that into a silent `inf`/`NaN` far downstream rather than a clear error at construction time. Truncation had been accidentally protecting this — every alpha/beta in `(-1, 0)` used to collapse to `0` — and widening the type removed that accident.
- [x] `aluminium`, `2d_mos2` and `molybdenium` example pipelines (`create_data.py`, `predict_chgcar.py`, `plot_diff_map.py`, `plot_diff_line_twinx.py`) and the README quickstart now state alpha/beta as the integers that were actually in force (e.g. aluminium's `[7.875386069413652, 5.875090883472657]` → `[7, 5]`), so a fixed library still reproduces the historical basis without relying on the old truncation to get there. `benzene` is untouched — its settings were already integral. Numerically a no-op given the type widening above; kept as its own commit so the Sanvito group can ask for it to be dropped without touching the fix.

### Measurements

Performed on FZJ PGI cluster iffSLURM, partition `th1-2020-32`, node `iffcluster1909`, AMD EPYC 7452, one job/one node for all three legs, example `aluminium` only.

- [x] **Byte-identical wherever settings were already integral, verified by hash.** With `aluminium`'s example settings rewritten as the integers in force, the fixed library's fitted train/test metrics table hashes identically (sha256) against `stable`'s. That is what distinguishes a pure type widening from a change to the recurrence itself, and it is what lets every historical model stay reconstructible from a fixed library.
- [x] **Honouring the published floats changes fitted accuracy by well under 0.1%.** On `aluminium`, test RMSE moves −0.06% and test MAE +0.01% relative to `stable` — opposite signs, both roughly three orders of magnitude inside the ±10% guard set before measuring. The published alpha/beta were selected while the search was scoring the truncated basis, so some change was expected; the reason it is this small is that 15+6 Jacobi orders span nearly the same function space regardless of the exact alpha/beta tilt, so a least-squares fit recovers almost the same result from re-tilted ingredients. Full write-up in `reports/2026-08-19-alpha-beta-truncation/delta-report.md` (untracked, per the `reports/` gitignore convention).

## [0.1.7] - 2026-08-22 — branch `fix-gamma-guard`

Follow-up to the review on PR #9, merged as `57db1b6`. Takes 0.1.7 by the merge-order rule
from issue #8 — *the PR merging first takes the next unused patch version* — which this PR
reached before PR #7. PR #7 had pre-claimed 0.1.7 in its branch while open; it moves to
0.1.8, since the rule is decided by merge order rather than by which branch wrote a number
down first. Applying it the other way would have left `stable` numbered non-monotonically,
with 0.1.7 appearing only after 0.1.8.

### Bug fixes

- [x] `expand_jacobi` (`polynomials.pyx`) and `JLGridFingerprints.__init__` now reject
  `gamma <= 0` with a `ValueError`, matching the existing treatment of `rcut <= 0`. `gamma`
  is the half-width of the interval $[-\gamma, +\gamma]$ the radial expansion runs over, so
  `gamma = 0` collapses that interval to the single point $x = 0$ and `gamma < 0` reverses its
  orientation; neither is a usable expansion. The `gamma = 0` case was the reason to add the
  check rather than document the constraint: the `double_shifted` normalisation divides by
  `2 * gamma`, and because the module compiles with `cdivision=True`, that division does not
  raise — **every returned order came back as a silent NaN**, measured. Before the 0.1.6 anchor
  fix the same call returned zeros, so 0.1.6 introduced the NaN path; this closes it. Raised at
  construction in the Python layer as well, so a bad setting fails before the grid loop rather
  than inside it.

  Behaviour change for a caller who passes `gamma <= 0`, which previously returned NaN (with
  `double_shifted`) or an all-zero basis (without). Nothing in this repository does — `gamma`
  appears in no example settings dict, no `scripts/` entry and no other test.

### Tests

- [x] `tests/test_polynomials_double_shifted.py` — three parametrized cases
  (`gamma` ∈ {0.0, −1.0, −0.5}) asserting the `ValueError`, each checked with `double_shifted`
  both on and off, since the guard is a property of `gamma` itself rather than of either shift
  mode.

- [x] The module docstring no longer asserts that `expand_jacobi` declares its exponents
  `int` — a claim that goes stale the moment PR #7 lands. It now states the intent instead:
  integral `alpha`/`beta` keep the test independent of how non-integer exponents are handled,
  which is true in either merge order.

## [0.1.6] - 2026-08-22 — branch `fix-double-shifted-anchor`

Anchors the `double_shifted` basis at its own interval end. Merged as `3abf58f`. The branch
carried no version number on purpose — it and PR #7 were open at the same time, independent of
each other, and either could have landed first — under the rule that *the PR merging first takes
the next unused patch version*. This one merged first, so it takes 0.1.6 and PR #7 moves to
0.1.7. The three version files are brought into agreement here rather than in the squashed
merge, which took the branch verbatim.

### Bug fixes

- [x] `calculate_jacobi` (`polynomials.pyx`) anchored the *upper* vanishing point of the
  `double_shifted` basis at a hard-coded `x = +1`, while the interval it works over is
  $[-\gamma, +\gamma]$ and the *lower* anchor was already correctly parameterised as
  `-1 * gamma`. The two ends were therefore anchored to two different intervals, and coincided
  only at the default `gamma = 1`. Away from that default the option's defining property is not
  merely weakened but inverted: at `gamma = 2` every returned order attains its *maximum*
  magnitude exactly at the endpoint where it is documented to vanish. And because `x = +1` lies
  inside the reachable range whenever `gamma > 1`, the vanishing point that belongs at the
  interval end instead appeared as a hard node at an interior radius,
  $r^\ast = r_{\min} + \frac{r_{\rm cut}-r_{\min}}{\pi}\arccos(1/\gamma)$ — so a caller
  sweeping `gamma` with `double_shifted` on was not sweeping a smooth family of bases, but
  moving a node inward from the cutoff. Fixed by anchoring both halves at `x = +gamma`: the
  point `pj1` is evaluated at, and the `gfac` denominator, which is the closed form of the same
  normalisation. Both must change together — changing either alone gives a basis vanishing at
  neither end.

  **Bit-identical for everything in this repository, measured.** Against a reference capture
  taken before the change on the same host and compiler (`setup.py` passes `-march=native`, so
  that matters): of 109 `expand_jacobi` arrays swept over exponent sets, `nmax`, `rmin`,
  `gamma` and both shift flags, exactly the 24 combining `double_shifted` with `gamma != 1`
  change; the other 84 are `np.array_equal`, including every `gamma = 1.0` array, every
  non-`double_shifted` array, and a full 4×4×4 fcc-Al descriptor matrix. `gamma` is set in no
  example settings dict, no `scripts/` entry and no test — `grep -rn gamma` over all four
  example pipelines and `scripts/` returns nothing — so every published artifact runs at the
  default where the two anchors agree. No published number, no example output and no existing
  test expectation changes.

  Fixed unconditionally, with no compatibility toggle: the prior behaviour has no defensible
  interpretation at `gamma != 1` — it neither vanishes at the interval end nor is documented as
  vanishing anywhere else — and a switch whose only purpose is to reproduce an internally
  inconsistent basis would be a permanent cost for a path nobody should choose.

### Tests

- [x] `tests/test_polynomials_double_shifted.py` — endpoint vanishing across
  `gamma` ∈ {0.5, 0.75, 1.0, 1.5, 2.0}, absence of the spurious interior node for `gamma > 1`,
  and a frozen `array_equal` regression at `gamma = 1`. Uses integral `alpha`/`beta`
  deliberately, since `expand_jacobi` still declares those exponents `int` (the subject of a
  separate issue) and float values would be silently truncated, making the test exercise
  different exponents than it names. **The module fails 6 of its 8 cases on unfixed code**; the
  2 that pass either way are the `gamma = 1` cases, which is the bit-identity claim expressed
  as a test.

## [0.1.5] - 2026-08-09 — branch `fast-fingerprints`

Moves parallelism to the right loop. `JLGridFingerprints.create()` gains an
opt-in process-parallel path, and the Cython kernels drop the OpenMP that was
parallelising a loop far too short to benefit from it.

### Features

- [x] New `fast_fingerprints.JLGridFingerprints`, a subclass of the serial descriptor whose `create()` accepts `batch_size` and `num_proc`. The centers are sliced into contiguous blocks, evaluated over a `multiprocessing.Pool` (`spawn`, module-level worker + `Pool(initializer=...)`), and concatenated back in order. Mirrors the `fast_predictor` design from 0.1.3, adapted for a `(batch, n_features)` payload: contiguous array slicing instead of a tuple chunker, and `np.concatenate(..., axis=0)` instead of `np.fromiter(chain.from_iterable(...))`, which would flatten the feature axis. Must be constructed with keyword arguments (the settings are stashed so workers can rebuild the descriptor under `spawn`).
- [x] `scripts/benchmark_fingerprints/` — a two-sweep benchmark (strong scaling, and the *crossover* point-set size), a plot script, and an archived SLURM job.

### Performance

- [x] Remove `prange` from `jlcontraction.pyx` and `utils.pyx`, drop the unused `prange` imports from `geometry.pyx` and `polynomials.pyx`, and stop passing `-fopenmp` in `setup.py`. Those loops sat inside a single grid point's ~150-element contraction, invoked once per center (~2.8M times for a 140³ grid), so OpenMP thread-team setup dominated the work they parallelised.

  **Bitwise neutral, verified two ways.** Every `prange` iterated over the *output* index (`jac[i]`, `prod[n,m]`, `vhat[n,j]`) while the accumulators were filled by serial inner `range` loops — there was never a cross-thread reduction in this package, so there is no summation order to change. Confirmed by (a) pre-change output being `np.array_equal` between `OMP_NUM_THREADS` unset and `=1`, and (b) post-change output matching a saved pre-change reference exactly. No `.so` links `libgomp` any more.

### Behavior notes

- [x] `OMP_NUM_THREADS=1` is no longer required for fingerprint work — the ~20–30× slowdown it used to guard against is structurally gone. It remains advisable for `fast_predictor`, where scikit-learn brings its own threaded BLAS into each worker.
- [x] `fast_fingerprints` and `fast_predictor` parallelise the *same* axis and must not be nested; `multiprocessing.Pool` workers are daemonic, so nesting raises rather than silently oversubscribing.

### Measured (iffSLURM `viti`/`iffcluster0806`, Xeon E5-2680 v2, 20 cores)

- [x] 2,744,000 centers: 355 s serial → 32 s at `num_proc=16` (11.20×, 70% efficiency); 1.92× / 3.60× / 6.39× at 2 / 4 / 8. Batching alone (`num_proc=1`) costs 0.03%, i.e. nothing.
- [x] Break-even is ~18,500 centers per `create()` call — below it the parallel path is slower. The `create_data.py` pipelines evaluate 13,720 centers per frame and therefore sit *below* it (measured 0.67×); they should parallelise over frames instead. Left to the owners of those example scripts. Full write-up in `reports/fingerprints-parallel/` (untracked, per the `reports/` gitignore convention).

## [0.1.4] - 2026-07-11 — branch `fixes/post-review-hygiene`

Addresses GitHub Copilot's automated review of PR #2, filed just after that PR
merged. Hygiene and input-validation only — no change to numerical behavior for
valid inputs. Two comments touching sampling math were deferred to TODO.md for
the Sanvito group; one signature-churn suggestion was declined.

### Bug fixes

- [x] `JLPredictor.__init__` now raises a clear `ValueError` when both `grid_size` and `encut` are `None`, instead of failing later with a cryptic `TypeError` inside `get_chgcar_grid` (`encut / Rydberg` on `None`).
- [x] `sample_charge` (`tools.py`) now raises a clear `ValueError` when `n_samples` exceeds the voxel count, instead of `numpy`'s opaque "cannot take a larger sample than population" from `rng.choice(replace=False)`.
- [x] `JLPredictor.__init__` now rejects a `grid_size` that is not an `(nx, ny, nz)` triple, instead of silently ignoring it and falling back to `encut` (which crashed when `encut` was `None`).
- [x] `predict_chgcar` now validates its inputs up front — `use_scaler=True` without a loaded scaler, and non-positive `batch_size` — mirroring the checks the `fast_predictor.JLPredictor` subclass already performs, so the two predictors fail the same way on the same bad input.

### Robustness / performance

- [x] Wrap the settings-JSON, model, and scaler loads in `JLPredictor.__init__` in `with open(...)` context managers (added explicit UTF-8 encoding for the JSON) so file handles close promptly even if parsing/unpickling raises.
- [x] Replace the per-chunk `np.append(..., axis=0)` in the batched `predict_chgcar` path (O(n²) reallocation) with list accumulation + a single `np.concatenate`. Output is bit-identical; only the batched large-grid path is affected.

## [0.1.3] - 2026-07-11 — branch `fast-predictor`

### Features

- [x] Integrate the parallel `fast_predictor.JLPredictor` from the Sanvito group as a subclass of the serial `predictor.JLPredictor`. Adds a `num_proc` argument that evaluates the FFT grid over a `multiprocessing.Pool` (per-batch fingerprint creation + prediction, streamed via `np.fromiter`), opt-in renormalisation (`normalize=False` — the serial predictor always normalises), and `predict_key_chgcar(...)` for predicting on an explicit point set + grid shape.

### Bug fixes (in the received `fast_predictor.py`)

- [x] `get_chgcar_grid` referenced `Bohr`/`Rydberg` with the import dropped (NameError) — fixed by inheriting the method from `predictor.JLPredictor`.
- [x] `use_scaler` transformed into an unused variable and then predicted on the unscaled features — fixed in the shared evaluation helper.
- [x] Removed a duplicate `normalize_nelect` definition (now inherited).
- [x] Replaced the unpicklable per-call closure (broken under the `spawn` start method on macOS/Windows) with a module-level worker + `Pool(initializer=...)`.
- [x] Forced the `spawn` start method for the worker pool: the default `fork` (Linux, Python ≤ 3.13) deadlocks because `JLGridFingerprints.create` links OpenMP and forking an initialised OpenMP runtime inherits locked thread state. Affected any `num_proc > 1` use on those interpreters (surfaced as a hung CI `test (3.10)` job).
- [x] Removed stray debug prints and a `np.save(...)` side-effect that wrote to the CWD.

### Tests / Docs

- [x] Add `tests/test_fast_predictor.py`: assert the fast predictor matches the serial one across serial, batched and multiprocessing paths, and that `predict_key_chgcar` matches `predict_chgcar`.
- [x] Add `scripts/benchmark_predictors/` (`benchmark_predictors.py` with `--json` output, and `plot_speedup.py`): serial vs parallel wall-time + speedup on a real aluminium frame.
- [x] Document the two predictor variants in the README, including the requirement to set `OMP_NUM_THREADS=1` when using `num_proc > 1` — `JLGridFingerprints.create` uses OpenMP, so otherwise each worker process oversubscribes the cores and the parallel path runs ~20–30× *slower* than serial (measured). The benchmark script warns when this is unset.
- [x] Measured speedup on iffSLURM `th1-2020-64` (32 cores, full 140³ grid, `OMP_NUM_THREADS=1`): the parallel predictor reaches 6.2× at `num_proc=8` and 11.1× at `num_proc=16`, with efficiency tailing off beyond ~16 processes.

## [0.1.2] - 2026-06-30 — branch `improve-package-standards`

### Tooling

- [x] Add ruff config (`[tool.ruff]`) and a ruff-based pre-commit hook; apply one repo-wide format pass over the maintained package surface.

### Packaging

- [x] Complete `pyproject.toml` metadata: `description`, `readme`, `requires-python` (>=3.10), `authors`, `keywords`, `classifiers`, `[project.urls]`, and a `test` optional-dependency; add lower bounds on numpy/ase/pymatgen.

### Docs

- [x] Add NumPy-style docstrings and signature type hints across the public API (`JLGridFingerprints`, `JLPredictor`, `tools.py`, and the public Cython functions), using the grid-centered JLCDM 1B/2B body-order convention.
- [x] Add a README quickstart, `CITATION.cff` (software + preferred-citation to the npj paper), and `CONTRIBUTING.md`.

### Tests / CI

- [x] Add pytest smoke tests (tools, fingerprint shape, predictor round-trip).
- [x] Add a GitHub Actions workflow: build the Cython extensions and run pytest on Python 3.10 and 3.14, plus a ruff lint job.

## [0.1.1] - 2026-04-21 — branch `improve-package-setup`

### Packaging / install

- [x] Modernize packaging: replace conda `environment.yml` + manual `jlgridfingerprints/setup.py` with `pyproject.toml` + root `setup.py`. Cython extensions now build automatically on `pip install -e .` (5572df1)
- [x] Fix Cython import: bare `from lib.utils import ...` → relative import in `polynomials.pyx`, broken by packaging refactor (5572df1)
- [x] Add `requirements.txt` mirroring `pyproject.toml` deps, for IDE environments where the package itself cannot be installed (8175433)
- [x] Remove conda lock file `environment.yml` (8f92d87)
- [x] Add `Dockerfile` (uv-based) and `containers/` with conda/venv alternatives (5572df1)

### Bug fixes

- [x] Fix `jlgridfingerprints/lib/` not present after `git clone`: bare `lib/` pattern in `.gitignore` (from Python template) silently ignored the directory. Added negation patterns and `.gitkeep`; `setup.py` also now creates the directory explicitly before build
- [x] Update sklearn API across all example scripts: replace deprecated `metrics.mean_squared_error(..., squared=False)` (removed in sklearn 1.6) with `metrics.root_mean_squared_error(...)`. Tested on sklearn 1.8 (3f68610)

### Misc

- [x] Update `.gitignore` with Cython build artifacts and Python standard entries (61a8edd)
- [x] Update `README.md` with new install instructions and package layout (5572df1, 10e1ca2, 39a379b)
