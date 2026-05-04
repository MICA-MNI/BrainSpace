# Contributing to BrainSpace

Thank you for your interest in contributing! BrainSpace is a Python and MATLAB
toolbox for cortical gradients. This guide describes the workflow for
contributing code, documentation, or bug reports.

## Reporting issues

- **Bugs** — please include a minimal reproducer, the version of BrainSpace
  (`brainspace.__version__`), Python version, OS, and the full traceback.
- **Feature requests** — describe the use case before proposing an API.
- **Security vulnerabilities** — please do not file a public issue. See
  [SECURITY.md](SECURITY.md) for the private reporting channel.

Search the existing [issue tracker](https://github.com/MICA-MNI/BrainSpace/issues)
before opening a new one.

## Development setup

```bash
git clone https://github.com/MICA-MNI/BrainSpace.git
cd BrainSpace
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

To run the test suite:

```bash
pytest brainspace/
```

Plotting tests need a display. On a headless machine install one of
`vtk-osmesa` / `vtk-egl` (see [docs/pages/install.rst](docs/pages/install.rst))
or run under `xvfb-run`.

## Pull request workflow

1. Fork the repository and create a topic branch from `master`. Name it after
   the issue it addresses, e.g. `fix-123-short-description`.
2. Keep the change focused — one logical change per PR. Unrelated cleanup
   belongs in its own PR.
3. Add or update tests for any code change. New code should not regress
   coverage.
4. Run `pytest` locally before pushing.
5. Open a PR against `master`. Reference the issue it closes
   (`Closes #NNN`) in the description and include a short test plan.
6. The CI matrix runs on Linux, macOS, and Windows across the supported
   Python versions; PRs must be green before review.

## Code style

- Python: follow PEP 8; match the surrounding style in the file you're editing.
- Public APIs need NumPy-style docstrings.
- MATLAB: match the existing function-header style in `matlab/analysis_code/`.

## Releases

Maintainers tag releases via `Release v0.X.Y` commits that bump
`brainspace/_version.py`. The `tagged_release` workflow then publishes to
PyPI. Contributors should not bump the version in PRs.
