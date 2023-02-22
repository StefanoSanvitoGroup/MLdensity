Paper **"Linear Jacobi-Legendre expansion of the charge density for machine learning-accelerated electronic structure calculations"** on [arXiv:2301.13550](https://arxiv.org/abs/2301.13550)

Full data available (CHGCAR and POSCAR): https://zenodo.org/record/7599897#.Y_ZqoNJKiV4

First, install required packages:

```
conda env create --name jlchg_tutorial --file=jlgridfingerprints_environment.yml
conda activate jlchg_tutorial
```

Second, to compile cython files within the code use the following command from inside the `jlgridfingerprints` directory:

```
LDSHARED="gcc -shared" CC=gcc python setup.py build_ext --inplace
```

Third, play with the examples
