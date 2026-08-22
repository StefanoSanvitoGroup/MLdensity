# Changelog

## [Unreleased] — branch `fix-gamma-guard`

Follow-up to the review on PR #9. No version number claimed: PR #7 is still open and takes
0.1.7 under the merge-order rule from issue #8, so this waits rather than reserve 0.1.8 for
work that may land in either order.

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
