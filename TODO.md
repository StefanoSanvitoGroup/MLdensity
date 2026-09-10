# TODO

Open items only. Completed work is recorded in [CHANGELOG.md](CHANGELOG.md) and in the
[GitHub releases](https://github.com/StefanoSanvitoGroup/MLdensity/releases), by version.

## Open pull requests — awaiting the Sanvito group's decision

Both change numerical behaviour, so neither merges without review. The measured evidence
lives in the issue and the pull request; the patch version each takes is decided by merge
order, per the rule in issue #8.

- [ ] **Jacobi exponent truncation.** `expand_jacobi` (`polynomials.pyx`) declared its
  exponents $\alpha, \beta$ as `int` while the example pipelines pass floats (e.g. `7.875`),
  so the published settings silently ran at truncated integer exponents. Issue #6, PR #7,
  branch `fix-alpha-beta-truncation`.
- [ ] **2B upper-triangle packing.** `calculate_3b_upper` packs an index triangle its
  flattening formula is not a bijection on. Issue #11, PR #12, branch `fix-2b-upper-packing`.

## Package standards — remaining

- [ ] LICENSE file + `license` field in `pyproject.toml` (pending the Sanvito group's choice —
  can land last).
- [ ] Zenodo release→DOI: enable the GitHub–Zenodo integration, then cut a release.
- [ ] `sample_charge` (`tools.py`) normalises with `prob_chg /= sum(prob_chg)`. If every
  Gaussian weight underflows to zero — many near-zero `chg` voxels — this divides by zero,
  giving NaN probabilities and a failing `rng.choice(p=...)`. Falling back to uniform
  probabilities would fix it but changes sampling behaviour, so it needs the Sanvito group's
  sign-off. (Found in the Copilot review of PR #2.)

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
