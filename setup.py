import os

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# Ensure the output directory for compiled extensions exists.
# Without this, the build fails after a fresh git clone because the standard
# Python .gitignore template contains a bare `lib/` pattern that prevents
# jlgridfingerprints/lib/ from being tracked in git.
os.makedirs("jlgridfingerprints/lib", exist_ok=True)

fnames = ["polynomials", "jlcontraction", "utils", "geometry"]

extensions = []
for fname in fnames:
    extensions += [
        Extension(
            "jlgridfingerprints.lib." + fname,
            ["jlgridfingerprints/src/" + fname + ".pyx"],
            include_dirs=[".", np.get_include()],
            libraries=["m"],
            # No -fopenmp: the kernels no longer use prange. The OpenMP loops
            # used to sit inside a single grid point's ~150-element
            # contraction, called once per center (~2.8M times for a 140^3
            # grid), so thread-team setup dominated and more threads made it
            # slower. Parallelism now lives one level up, over centers, in
            # jlgridfingerprints.fast_fingerprints / fast_predictor.
            extra_compile_args=["-O3", "-march=native"],
            extra_link_args=["-O3"],
        )
    ]

setup(
    packages=["jlgridfingerprints", "jlgridfingerprints.lib"],
    ext_modules=cythonize(extensions, annotate=True),
)
