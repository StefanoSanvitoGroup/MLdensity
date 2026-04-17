Paper **"Linear Jacobi-Legendre expansion of the charge density for machine learning-accelerated electronic 
structure calculations"** at **[DOI:10.1038/s41524-023-01053-0](https://doi.org/10.1038/s41524-023-01053-0)**

Full data available (CHGCAR and POSCAR): https://doi.org/10.5281/zenodo.7922012

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
    predictor.py
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

## Containerized development

A `Dockerfile` using uv Python environment is provided at the repo root for container managers (Podman, Docker). 
Alternatives for other Python env managers (conda, plain venv) are under `containers/`.

```bash
podman build -t mldensity .
podman run --rm -v .:/MLdensity -it mldensity bash
# inside container:
cd MLdensity/ && uv pip install -e .
```

> **Note (macOS / rootless Podman):** If `ls` inside the container returns `Permission denied` on the mounted directory, the `:z` volume label above should fix it. If not, try `--userns=keep-id` instead of `:z`.

## Usage

Play with the examples in the `aluminium/`, `benzene/`, `molybdenium/`, and `2d_mos2/` directories.
