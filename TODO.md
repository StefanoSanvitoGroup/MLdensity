# TODO

Completed work is recorded in [CHANGELOG.md](CHANGELOG.md), by version.

## Package standards — remaining

- [ ] LICENSE + pyproject `license` field (pending Sanvito group's choice — can land last).
- [ ] Zenodo release→DOI: enable GitHub–Zenodo integration, cut a release (post-merge).
- [x] Follow-up (found while adding docstrings): `expand_jacobi` in `src/polynomials.pyx` declared `int alpha, int beta`, but the example pipelines pass floats (e.g. `7.875`) and the internal `calculate_jacobi` uses `double` — the signature silently truncated α/β toward zero. Filed as issue #6 and fixed on branch `fix-alpha-beta-truncation`, see CHANGELOG (v0.1.8).
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

- [x] OpenMP `prange` in `calculate_3b_upper` / `calculate_3b` (`jlcontraction.pyx`) parallelizes the inner descriptor contraction (~147 elements for Al settings). This function is called once per grid point from a **Python loop** in `create()`, amounting to ~2.8M tiny parallel calls per structure at full-grid prediction scale. Thread overhead dominates; more threads = slower.

  The parallelism should be over the outer loop of centers (each grid point is independent). Options in order of increasing effort:

  - [ ] **Move center loop into Cython `prange`**: rewrite `create()` so the `for io in range(self._n_centers)` loop runs in a Cython `prange`, replacing per-element OpenMP inside the contraction kernels. Requires passing neighbor data as flat arrays rather than Python lists. **Still open, and now the natural next step** — it would remove the ~2 s process-spawn cost that creates the break-even below, and avoid pickling descriptors back to the parent. The inner-kernel `prange` that would have fought it is out of the way as of v0.1.5.
  - [ ] **Vectorize over the batch dimension**: restructure `create_2b_jl` / `create_3b_jl` to operate on all centers at once via NumPy broadcasting or `einsum`, eliminating the Python loop entirely.
  - [x] **Quick partial fix**: at minimum, remove `prange` from the inner kernels to stop the thread-overhead regression when `OMP_NUM_THREADS > 1`. **Done in v0.1.5** (`fast-fingerprints`, commit `c13727a`): `prange` removed from `jlcontraction.pyx` and `utils.pyx`, unused imports dropped from the other two, and `-fopenmp` removed from `setup.py`. Bitwise neutral — every `prange` was over the output index, never a reduction — verified against a saved reference. `OMP_NUM_THREADS` is no longer needed for fingerprint work.
  - [x] **Process parallelism over centers**, done in the meantime as `fast_fingerprints.JLGridFingerprints` (v0.1.5): 11.20× at `num_proc=16` on 2.744M centers, break-even ~18,500 centers per call. See the [v0.1.5 release notes](https://github.com/StefanoSanvitoGroup/MLdensity/releases/tag/v0.1.5).

  Context: originally observed on iffSLURM `th1-2020-64` (Intel x86, 128 cores / 128 GB), where `OMP_NUM_THREADS=1` was the fastest setting. The v0.1.5 measurements are from `viti`/`iffcluster0806` (Xeon E5-2680 v2, 20 cores) — a much older part, so absolute times differ; only the scaling shape is comparable. Mac M2 is ~1.5x faster than the cluster at serial execution due to higher single-core throughput and lower memory latency.

## Performance: parallelize the example pipelines over frames, not centers

- [ ] `*/ml_model/data_ml/create_data.py` (4 copies) call `create()` once per frame with 0.5% of the voxels — 13,720 centers for a 140³ grid. That is *below* the ~18,500-center break-even of `fast_fingerprints` (measured 0.67×, i.e. slower), because ~2 s of process spawn cannot be amortized over ~1.6 s of work. The frame loop is the right axis there: frames are independent and the fixed cost is then paid once per dataset rather than once per frame. Deliberately not done on `fast-fingerprints` — these are example pipelines owned by the Sanvito group, and the recommendation now rests on a measurement rather than an assumption. See the [v0.1.5 release notes](https://github.com/StefanoSanvitoGroup/MLdensity/releases/tag/v0.1.5).
