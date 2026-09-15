# Contributing

Thanks for your interest in improving `jlgridfingerprints`. This is a small 
scientific research software package; the notes below keep it consistent.

## Development setup

The Cython extensions need a Linux build toolchain. On macOS build and run
inside a container (see the "Containerized development" section of the
[README](README.md)). Then, for an editable install with test tools:

```bash
pip install -e ".[test]"
```

## Code style

- **Formatting & linting**: [ruff](https://docs.astral.sh/ruff/). Run
  `ruff check .` and `ruff format .`, or install the git hook with
  `pre-commit install` (config in `.pre-commit-config.yaml`).
- **Docstrings**: [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html),
  for both `.py` and `.pyx` files. Every parameter in a signature must have a
  docstring entry, with the default behaviour noted for optional parameters.

## Citing measurements

`reports/` is gitignored: it holds local development write-ups that exist on one
machine and are not part of the repository. **Tracked files must therefore not cite a
`reports/` path** — for anyone else, and for any future checkout, such a pointer
resolves to nothing.

Cite the most durable thing that actually contains the measurement, in this order:

1. a **release tag** for work that has shipped (`v0.1.5`) — immutable and aligned with
   the version a CHANGELOG section already names;
2. a **pull request or issue** for work still in review (`PR #7`) — the write-up lives
   in its description;
3. a **commit hash** when neither exists.

If nothing public holds it, say so in those words rather than linking a local path, and
inline the one number the reader needs. Before citing, check the target genuinely
contains the content — a pointer to a release whose notes omit the measurement is the
same dead end in a nicer costume.

## Tests

`pytest`. The `tools` tests run anywhere; the fingerprint/predictor tests need
the compiled extensions and otherwise skip. CI (GitHub Actions) builds the
extensions and runs the full suite on Python 3.10 and 3.14.

## Scope of changes

This package implements published research code (see the paper linked in the
[README](README.md)). Packaging, docs, tests, CI, and formatting changes are
welcome. **Changes that alter numerical behaviour** (the fingerprint math,
contraction kernels, Jacobi/Legendre evaluation, or predictions) should be
discussed with the maintainers first — please open an issue describing the
change rather than altering published results silently.
