# TODO

## Performance: parallelize `create()` over centers, not inside the contraction kernel

- [ ] OpenMP `prange` in `calculate_3b_upper` / `calculate_3b` (`jlcontraction.pyx`) parallelizes the inner descriptor contraction (~147 elements for Al settings). This function is called once per grid point from a **Python loop** in `create()` (`fingerprints.py` lines 195–208), amounting to ~2.8M tiny parallel calls per structure at full-grid prediction scale. Thread overhead dominates; more threads = slower.

  The parallelism should be over the outer loop of centers (each grid point is independent). Options in order of increasing effort:

  - [ ] **Move center loop into Cython `prange`**: rewrite `create()` so the `for io in range(self._n_centers)` loop runs in a Cython `prange`, replacing per-element OpenMP inside the contraction kernels. Requires passing neighbor data as flat arrays rather than Python lists.
  - [ ] **Vectorize over the batch dimension**: restructure `create_2b_jl` / `create_3b_jl` to operate on all centers at once via NumPy broadcasting or `einsum`, eliminating the Python loop entirely.
  - [ ] **Quick partial fix**: at minimum, remove `prange` from the inner kernels to stop the thread-overhead regression when `OMP_NUM_THREADS > 1`.

  Context: observed on iffSLURM `th1-2020-64` partition (Intel x86, 128 cores / 128 GB). `OMP_NUM_THREADS=1` is currently the fastest setting. Mac M2 is ~1.5x faster than the cluster at serial execution due to higher single-core throughput and lower memory latency.
