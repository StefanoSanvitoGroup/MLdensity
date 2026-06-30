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
    "nmax": [15, 6],          # radial orders for the 1B and 2B terms
    "lmax": 6,                # angular order for the 2B term
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

X = jl.create(atoms, centers)   # shape: (n_centers, jl._n_features)
```

To predict a full charge density and write a VASP CHGCAR, use
`jlgridfingerprints.predictor.JLPredictor` with a trained model.

### Examples

Full training and prediction pipelines are in the `aluminium/`, `benzene/`,
`molybdenium/`, and `2d_mos2/` directories.

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
