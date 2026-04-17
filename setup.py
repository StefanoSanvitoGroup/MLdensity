from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

fnames = ['polynomials', 'jlcontraction', 'utils', 'geometry']

extensions = []
for fname in fnames:
    extensions += [Extension(
        "jlgridfingerprints.lib." + fname,
        ["jlgridfingerprints/src/" + fname + ".pyx"],
        include_dirs=['.', np.get_include()],
        libraries=["m"],
        extra_compile_args=['-O3', '-fopenmp', '-march=native'],
        extra_link_args=['-O3', '-fopenmp']
    )]

setup(
    name='jlgridfingerprints',
    packages=['jlgridfingerprints', 'jlgridfingerprints.lib'],
    ext_modules=cythonize(extensions, annotate=True)
)