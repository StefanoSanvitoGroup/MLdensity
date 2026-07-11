# TODO

Completed work is recorded in [CHANGELOG.md](CHANGELOG.md), by version.

## Package standards — remaining

- [ ] LICENSE + pyproject `license` field (pending Sanvito group's choice — can land last).
- [ ] Zenodo release→DOI: enable GitHub–Zenodo integration, cut a release (post-merge).
- [ ] Follow-up (found while adding docstrings): `expand_jacobi` in `src/polynomials.pyx` declares `int alpha, int beta`, but the example pipelines pass floats (e.g. `7.875`) and the internal `calculate_jacobi` uses `double` — likely α/β truncation bug. Numerical behavior → verify with the Sanvito group before any fix.
- [ ] Follow-up (Copilot review of PR #2): `sample_charge` in `tools.py` normalises with `prob_chg /= sum(prob_chg)`. If all Gaussian weights underflow to 0 (many near-zero `chg` voxels), this divides by zero → NaN probabilities → `rng.choice(p=...)` fails. Copilot suggests falling back to uniform probabilities when the normalisation is not finite/positive. That changes sampling behavior → verify with the Sanvito group before any fix.

## Integrate the `fast-predictor` module — DONE (branch `fast-predictor`, see CHANGELOG)

Faster `JLPredictor` variant from the Sanvito group (Luke; orig. Urvesh), integrated as a
subclass of `predictor.JLPredictor` in `jlgridfingerprints/fast_predictor.py`. What it adds over
`predictor.py`:

- `num_proc` arg → parallelizes per-batch fingerprint creation + prediction over a
  `multiprocessing.Pool`; streams results via a batched iterator + `np.fromiter`
  instead of the sequential `np.append` chunk loop.
- Normalization made opt-in (`normalize=False`); original always normalized.
- New `predict_key_chgcar(...)` for an explicit point set + grid size.

Bugs fixed on integration (in the received file):

- [x] Restore `from ase.units import Bohr, Rydberg` — fixed by inheriting `get_chgcar_grid`.
- [x] `use_scaler` scaled into an unused `X_`; now predicts on the scaled array.
- [x] Remove duplicate `normalize_nelect` definition — gone (inherited).
- [x] `calc_fn` closure isn't picklable under the spawn start method (macOS/Windows); replaced with a module-level worker + `Pool(initializer=...)`.
- [x] Rename to importable `fast_predictor.py` (hyphen breaks `import`).

## Performance: parallelize `JLGridFingerprints.create()` over centers, not inside the contraction kernel

- [ ] OpenMP `prange` in `calculate_3b_upper` / `calculate_3b` (`jlcontraction.pyx`) parallelizes the inner descriptor contraction (~147 elements for Al settings). This function is called once per grid point from a **Python loop** in `create()` (`fingerprints.py` lines 195–208), amounting to ~2.8M tiny parallel calls per structure at full-grid prediction scale. Thread overhead dominates; more threads = slower.

  The parallelism should be over the outer loop of centers (each grid point is independent). Options in order of increasing effort:

  - [ ] **Move center loop into Cython `prange`**: rewrite `create()` so the `for io in range(self._n_centers)` loop runs in a Cython `prange`, replacing per-element OpenMP inside the contraction kernels. Requires passing neighbor data as flat arrays rather than Python lists.
  - [ ] **Vectorize over the batch dimension**: restructure `create_2b_jl` / `create_3b_jl` to operate on all centers at once via NumPy broadcasting or `einsum`, eliminating the Python loop entirely.
  - [ ] **Quick partial fix**: at minimum, remove `prange` from the inner kernels to stop the thread-overhead regression when `OMP_NUM_THREADS > 1`.

  Context: observed on iffSLURM `th1-2020-64` partition (Intel x86, 128 cores / 128 GB). `OMP_NUM_THREADS=1` is currently the fastest setting. Mac M2 is ~1.5x faster than the cluster at serial execution due to higher single-core throughput and lower memory latency.
