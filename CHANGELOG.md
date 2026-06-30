# Changelog

## [Unreleased] — branch `fast-predictor`

### Features

- [x] Integrate the parallel `fast_predictor.JLPredictor` from the Sanvito group as a subclass of the serial `predictor.JLPredictor`. Adds a `num_proc` argument that evaluates the FFT grid over a `multiprocessing.Pool` (per-batch fingerprint creation + prediction, streamed via `np.fromiter`), opt-in renormalisation (`normalize=False` — the serial predictor always normalises), and `predict_key_chgcar(...)` for predicting on an explicit point set + grid shape.

### Bug fixes (in the received `fast_predictor.py`)

- [x] `get_chgcar_grid` referenced `Bohr`/`Rydberg` with the import dropped (NameError) — fixed by inheriting the method from `predictor.JLPredictor`.
- [x] `use_scaler` transformed into an unused variable and then predicted on the unscaled features — fixed in the shared evaluation helper.
- [x] Removed a duplicate `normalize_nelect` definition (now inherited).
- [x] Replaced the unpicklable per-call closure (broken under the `spawn` start method on macOS/Windows) with a module-level worker + `Pool(initializer=...)`.
- [x] Removed stray debug prints and a `np.save(...)` side-effect that wrote to the CWD.

### Tests / Docs

- [x] Add `tests/test_fast_predictor.py`: assert the fast predictor matches the serial one across serial, batched and multiprocessing paths, and that `predict_key_chgcar` matches `predict_chgcar`.
- [x] Add `scripts/benchmark_predictors.py` (with `--json` output) and `scripts/plot_speedup.py`: serial vs parallel wall-time + speedup on a real aluminium frame.
- [x] Document the two predictor variants in the README, including the requirement to set `OMP_NUM_THREADS=1` when using `num_proc > 1` — `JLGridFingerprints.create` uses OpenMP, so otherwise each worker process oversubscribes the cores and the parallel path runs ~20–30× *slower* than serial (measured). The benchmark script warns when this is unset.
- [x] Add `reports/predictor-comparison.md` with the full comparison and a measured speedup plot. On iffSLURM `th1-2020-64` (32 cores, full 140³ grid, `OMP_NUM_THREADS=1`) the parallel predictor reaches 6.2× at `num_proc=8` and 11.1× at `num_proc=16` (efficiency tails off beyond ~16).

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
