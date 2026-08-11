Paper **"Linear Jacobi-Legendre expansion of the charge density for machine learning-accelerated electronic 
structure calculations"** at **[DOI:10.1038/s41524-023-01053-0](https://doi.org/10.1038/s41524-023-01053-0)**

Full dataset available (CHGCAR and POSCAR): https://doi.org/10.5281/zenodo.7922012

## Installation

The recommended environment manager is [uv](https://docs.astral.sh/uv/).

```bash
# one-time install
uv pip install .

# editable install for development
uv pip install -e .
```

This compiles the Cython extensions automatically as part of the build. No separate compilation step is needed.

## Package layout

```
jlgridfingerprints/
    fingerprints.py       # JLGridFingerprints descriptor class
    fast_fingerprints.py  # parallel subclass of it (see "Parallelization")
    predictor.py          # JLPredictor: descriptor -> model -> CHGCAR
    fast_predictor.py     # parallel subclass of it (see "Parallelization")
    tools.py
    lib/
    src/
        polynomials.pyx   # Jacobi-Legendre polynomials evaluation
        jlcontraction.pyx # 2-body and 3-body contraction kernels
        utils.pyx         # vector norm, dot, versors
        geometry.pyx      # neighbour list (KDTree + PBC)
```

The `.pyx` files are compiled by Cython into shared libraries in `jlgridfingerprints/lib/`. The build is driven by 
the root-level `setup.py` / `pyproject.toml`.

## Usage

### Quickstart

Evaluate Jacobi-Legendre fingerprints on a grid of points for a structure:

```python
from ase.build import bulk

from jlgridfingerprints.fingerprints import JLGridFingerprints
from jlgridfingerprints.tools import create_grid_coords

# Descriptor hyperparameters (see the paper for definitions).
# body "1" = 1B radial term, "2" = 2B radial-angular term.
settings = {
    "rcut": 4.08,
    "nmax": [15, 6],  # radial orders for the 1B and 2B terms
    "lmax": 6,  # angular order for the 2B term
    "alpha": [7.875, 5.875],
    "beta": [3.624, 1.751],
    "rmin": -0.74,
    "species": ["Al"],
    "body": "1+2",
    "periodic": True,
    "double_shifted": True,
}

atoms = bulk("Al", "fcc", a=4.05)
jl = JLGridFingerprints(**settings)

# Grid of points spanning the cell (small grid here for illustration).
centers = create_grid_coords(
    grid_size=(20, 20, 20),
    return_cartesian_coords=True,
    a_vectors=atoms.get_cell().array,
)

X = jl.create(atoms, centers)  # shape: (n_centers, jl._n_features)
```

To predict a full charge density and write a VASP CHGCAR, use
`jlgridfingerprints.predictor.JLPredictor` with a trained model. Both of these
have parallel counterparts — see [Parallelization](#parallelization) for which to
use.

### Examples

Full training and prediction pipelines are in the `aluminium/`, `benzene/`,
`molybdenium/`, and `2d_mos2/` directories.

Trained models `example/ml_model/train_ml/*.p` are part of the repo. 
The `CHGCAR` files required for training these models are in the published dataset.

## Parallelization

There are two parallel classes. **They parallelize the same thing** — the loop
over grid points — so you pick one, never both.

### Which one do I want?

**Start here:** are you producing a charge density, or descriptors?

| You want… | Use | Import |
|---|---|---|
| A predicted charge density / CHGCAR | **`fast_predictor.JLPredictor`** | `from jlgridfingerprints.fast_predictor import JLPredictor` |
| The descriptor matrix itself, ≳ 20,000 points per call | **`fast_fingerprints.JLGridFingerprints`** | `from jlgridfingerprints.fast_fingerprints import JLGridFingerprints` |
| The descriptor matrix, only a few thousand points per call | the plain serial class | `from jlgridfingerprints.fingerprints import JLGridFingerprints` |
| To process many small structures/frames | parallelize *your* loop over frames | — |

**If in doubt, prefer `fast_predictor`.** Predicting a density is the common
case, and `fast_predictor` is strictly better at it: it parallelizes the
descriptor work *and* the model evaluation, and it never holds the full
descriptor matrix in memory (for a 140³ grid that matrix is **2.6 GB**; the
predictor streams one batch at a time and keeps one float per point).

Reach for `fast_fingerprints` only when you genuinely need the descriptors —
training-data generation, analysis, feeding a model this package doesn't own.

### Both classes are drop-in subclasses

Each subclasses its serial counterpart and inherits everything else unchanged,
so the numerics are identical and you can swap the import back at any time. With
default arguments they behave exactly like the serial versions.

**Parallel prediction** — pass `num_proc` and a `batch_size`:

```python
from jlgridfingerprints.fast_predictor import JLPredictor

jl = JLPredictor(jl_settings=settings, model_path="model.p", grid_size=(140, 140, 140))
chg = jl.predict_chgcar(atoms, nelect=96, batch_size=20000, num_proc=8, normalize=True)
```

Two differences from the serial predictor to know about:

- `normalize` defaults to `False` here, whereas the serial predictor *always*
  renormalizes to `nelect`. Pass `normalize=True` to match it.
- It adds `predict_key_chgcar(...)`, which predicts on a point set you supply
  instead of the full FFT grid.

**Parallel descriptors** — same two arguments, on `create`:

```python
from jlgridfingerprints.fast_fingerprints import JLGridFingerprints

jl = JLGridFingerprints(**settings)  # keyword arguments required here
X = jl.create(atoms, centers, batch_size=20000, num_proc=8)
```

`num_proc > 1` requires a `batch_size` in both classes (you get a `ValueError`
otherwise). A good starting point is a batch size that gives each worker a few
batches: `batch_size = n_points // (4 * num_proc)`, clamped to at least ~10,000.

### How much does it help?

Measured on 2,744,000 points (a 140³ aluminium grid) on a 20-core node:

| `num_proc` | Wall time | Speedup | Efficiency |
|---|---|---|---|
| serial | 355 s | 1.00× | — |
| 2 | 185 s | 1.92× | 96% |
| 4 | 99 s | 3.60× | 90% |
| 8 | 56 s | 6.39× | 80% |
| 16 | 32 s | 11.20× | 70% |

Roughly linear up to the core count, tailing off gently. Both classes give
essentially this same curve, because both are parallelizing the same loop.

### When it *hurts*

Starting worker processes costs about **2 seconds** (the `spawn` start method
builds fresh Python interpreters that re-import numpy, ase and the compiled
extensions). That cost is fixed, so it only pays off if there is enough work
behind it. Break-even is around **18,000 points per call**; below that the
parallel path is slower, and at 1,000 points it is *16× slower*.

This is why the "many small frames" row above says to parallelize your own frame
loop: one 13,000-point frame is ~1.6 s of work, which cannot absorb a 2 s
startup, but a whole dataset of frames easily can. Measure your own case with
`scripts/benchmark_fingerprints/benchmark_fingerprints.py` or
`scripts/benchmark_predictors/benchmark_predictors.py`.

### `OMP_NUM_THREADS`

Since v0.1.5 the Cython extensions no longer use OpenMP, so **fingerprint work
needs no environment variable**. (Before v0.1.5, forgetting `OMP_NUM_THREADS=1`
with `num_proc > 1` caused a ~20–30× *slowdown*.)

Setting it is still worth doing for `fast_predictor`, because scikit-learn pulls
its own threaded BLAS into every worker process, and those threads oversubscribe
the cores the same way:

```bash
OMP_NUM_THREADS=1 python your_prediction_script.py
```

### Do not nest them

`fast_predictor` already calls the serial `create()` inside each of its workers,
so wrapping `fast_fingerprints` inside it is the same parallelism twice. It fails
loudly rather than silently oversubscribing — `multiprocessing.Pool` workers are
daemonic, so you get `AssertionError: daemonic processes are not allowed to have
children`.

## Containerized development

The Cython extensions require a Linux build environment. On macOS (Apple Silicon)
the recommended approach is to build and run the package inside a container using
[Podman](https://podman.io) or Docker.

### Setup

```bash
# (MacOS only) Initialize and start the Linux VM
# (--rootful=true is required in case of IDE integration)
podman machine init --rootful=true
podman machine start

# Build the image 'mldensity'
podman build -t mldensity .

# Run the container 'mldensity-dev' based on the image
podman run -d --name mldensity-dev -v ./:/MLdensity -it mldensity bash

# Open a shell in the running container
podman exec -it mldensity-dev bash   

# Install the package
cd /MLdensity && uv pip install -e .

# Use the package
python -c "import jlgridfingerprints" # verify

# Exit the shell when done
exit
```

### Container management

```bash
podman ps -a                        # list all containers
podman stop container-name          # stop container
podman start container-name         # restart
podman rm container-name            # remove container
podman images                       # list images
podman rmi image-name               # remove image
```
