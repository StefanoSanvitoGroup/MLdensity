# Changelog

## [0.1.1] - 2026-04-21 — branch `improve-package-setup`

### Packaging / install

- [x] Modernize packaging: replace conda `environment.yml` + manual `jlgridfingerprints/setup.py` with `pyproject.toml` + root `setup.py`. Cython extensions now build automatically on `pip install -e .` (5572df1)
- [x] Fix Cython import: bare `from lib.utils import ...` → relative import in `polynomials.pyx`, broken by packaging refactor (5572df1)
- [x] Add `requirements.txt` mirroring `pyproject.toml` deps, for IDE environments where the package itself cannot be installed (8175433)
- [x] Remove conda lock file `environment.yml` (8f92d87)
- [x] Add `Dockerfile` (uv-based) and `containers/` with conda/venv alternatives (5572df1)

### Bug fixes

- [x] Fix `jlgridfingerprints/lib/` not present after `git clone`: bare `lib/` pattern in `.gitignore` (from Python template) silently ignored the directory. Added negation patterns and `.gitkeep`; `setup.py` also now creates the directory explicitly before build
- [x] Update sklearn API across all example scripts: replace deprecated `metrics.mean_squared_error(..., squared=False)` (removed in sklearn 1.6) with `metrics.root_mean_squared_error(...)`. Tested on sklearn 1.8 (3f68610)

### Misc

- [x] Update `.gitignore` with Cython build artifacts and Python standard entries (61a8edd)
- [x] Update `README.md` with new install instructions and package layout (5572df1, 10e1ca2, 39a379b)
