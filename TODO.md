# TODO

Open items only. Completed work is recorded in [CHANGELOG.md](CHANGELOG.md) and in the
[GitHub releases](https://github.com/StefanoSanvitoGroup/MLdensity/releases), by version.

## Open pull requests — under review by the Sanvito group

All three change behaviour. Reviewed by Luke, to be merged if the group's own testing agrees
and the PI signs off, as agreed at the code review meeting of 2026-09-15. The measured
evidence lives in the issue and the pull request; the patch version each takes is decided by
merge order, per the rule in issue #8. They are stacked in the order listed, each branch based
on the one above it, so a checkout of the last carries all three, and merging them in that
order keeps the versions as written. Each branch carries this roadmap as of 2026-09-15.

- [ ] **Jacobi exponent truncation.** `expand_jacobi` (`polynomials.pyx`) declared its
  exponents $\alpha, \beta$ as `int` while the example pipelines pass floats (e.g. `7.875`),
  so the published settings silently ran at truncated integer exponents. Issue #6, PR #7,
  branch `fix-alpha-beta-truncation`.
- [ ] **2B upper-triangle packing.** `calculate_3b_upper` packs an index triangle its
  flattening formula is not a bijection on. Issue #11, PR #12, branch `fix-2b-upper-packing`.
- [ ] **Absent configured species.** `_initialize_distances` (`fingerprints.py`) crashes in
  the neighbour search when a configured species has no atom in the structure being
  featurized — which is every structure of a multi-species impurity dataset. The species now
  yields zero blocks, the physically correct answer for a pair of elements that never
  co-occur. Alone among the three it changes no value a working input produces, only inputs
  that previously raised, so nothing needs regenerating. Issue #13, PR #14, branch
  `2026-09-10-absent-species`.

## Package standards — remaining

- [ ] LICENSE file + `license` field in `pyproject.toml` (pending the Sanvito group's choice —
  can land last).
- [ ] Zenodo release→DOI: enable the GitHub–Zenodo integration, then cut a release.
- [ ] `sample_charge` (`tools.py`) normalises with `prob_chg /= sum(prob_chg)`. If every
  Gaussian weight underflows to zero — many near-zero `chg` voxels — this divides by zero,
  giving NaN probabilities and a failing `rng.choice(p=...)`. (Found in the Copilot review of
  PR #2.) `develop` already answers this in commit `54b1f2f`, masking voxels below `1e-10` to
  zero probability, which is the weight's own limit there and a better fix than the uniform
  fallback first proposed; it still leaves the all-underflow case open. Changes sampling
  behaviour, so it needs the Sanvito group's sign-off.

## Branches inherited from the original authors

`develop` forked from `stable` in February 2023 and holds three content commits by Urvesh,
none of which ever reached `stable`. Raised with the group at the 2026-09-15 code review
meeting; reviewed here so the question is not reopened.

- [x] **Multispecies guard — superseded, nothing to carry over.** Commit `408388f` ("fix
  multispecies") guards the same call site as PR #14 with the same idea, skipping the
  neighbour-tree build for an absent species. Its diff is large only because the file was
  reformatted in the same commit. Four of its six placeholder arrays have the wrong length,
  dtype and dimensionality, harmless only because every consumer tests the neighbour count
  first; it has no warning and no tests. PR #14 is a superset.
- [ ] **One-body double-vanishing basis — not on `stable`, possibly wanted.** Commit
  `fc4f47f` adds a `double_shifted_1b` switch letting the 1B radial basis vanish at both ends
  of its mapped interval rather than at the cutoff alone, costing one radial order per
  species. It looks complete and self-consistent. Taken from `develop` as is it would inherit
  the boundary bug fixed in 0.1.6, so it needs re-applying on top of `stable`. Ask the group
  whether it was used for anything published.
- [ ] **Retire `develop` and `parallel_predict`?** `parallel_predict` (2024, "untested code
  for prediction in parallel") is superseded by the `fast_predictor` merged in 0.1.3.
  `develop` has nothing left once the one-body item above is settled. The group's call.

## Performance: parallelize `create()` over centers, in-process

Process parallelism over centers shipped in 0.1.5 as `fast_fingerprints.JLGridFingerprints`
(11.2× at 16 processes, break-even ~18,500 centers per call), and the misplaced OpenMP inside
the contraction kernels was removed in the same release. Two in-process routes remain, in
increasing order of effort:

- [ ] **Move the center loop into Cython.** Rewrite `create()` so `for io in range(self._n_centers)`
  runs in a `prange`, which requires passing neighbour data as flat arrays rather than Python
  lists. The natural next step: it removes the ~2 s process-spawn cost that sets the break-even
  above, avoids pickling descriptors back to the parent, and the inner-kernel `prange` that
  would have fought it is gone as of 0.1.5.
- [ ] **Vectorize over the batch dimension.** Restructure `create_2b_jl` / `create_3b_jl` to
  operate on all centers at once via NumPy broadcasting or `einsum`, eliminating the Python
  loop entirely.

## Performance: parallelize the example pipelines over frames

- [ ] The four `*/ml_model/data_ml/create_data.py` copies call `create()` once per frame with
  0.5% of the voxels — 13,720 centers for a 140³ grid, *below* the ~18,500-center break-even of
  `fast_fingerprints` (measured 0.67×, i.e. slower), because ~2 s of process spawn cannot be
  amortised over ~1.6 s of work. Frames are independent, so the frame loop is the right axis:
  the fixed cost is then paid once per dataset rather than once per frame. Deliberately not
  changed in 0.1.5 — these are example pipelines owned by the Sanvito group.

## Merge the CJL extension back into this repository

- [ ] Placeholder, no design yet. The covariant Jacobi-Legendre (CJL) extension exists only as
  the Zenodo record accompanying Focassio et al., *Phys. Rev. B* **110**, 184106 (2024)
  ([doi 10.5281/zenodo.13772980](https://doi.org/10.5281/zenodo.13772980)) — a fork of this
  package rather than a branch of it, so it has no living home here.

  **What must not be lost in the merge-back:** the CJL fork predates both open fixes and carries
  neither. The Jacobi exponent truncation defect (issue #6, PR #7) is present in its
  `polynomials.pyx`, and the 2B upper-triangle packing defect (issue #11, PR #12) is present
  *twice* in its `jlcontraction.pyx` — in `calculate_3b_upper` and in `calculate_3b_upper_l0`.
  Whoever does the merge must apply both, or consciously decide not to.
