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
            extra_compile_args=["-O3", "-fopenmp", "-march=native"],
            extra_link_args=["-O3", "-fopenmp"],
        )
    ]

setup(
    packages=["jlgridfingerprints", "jlgridfingerprints.lib"],
    ext_modules=cythonize(extensions, annotate=True),
)
