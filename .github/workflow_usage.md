# GitHub Actions Workflows for RepoMetrics

This directory contains GitHub Actions workflows that automate testing, building, and deployment of RepoMetric.

## Workflows

### 1. Check Version and Release (`check-version.yml`)

This workflow automatically checks for version changes in `Cargo.toml` and `pyproject.toml` files. When a version change is detected, it:

1. Creates a new Git tag matching the version (e.g., `v0.1.2`)
2. Creates a GitHub Release
3. Triggers the PyPI publishing workflow

**Trigger:** Runs on pushes to `main` branch that modify `Cargo.toml` or `pyproject.toml` files.

### 2. Build and Publish to PyPI (`publish.yml`)

This workflow builds and publishes the package to PyPI.

1. Builds wheel files for multiple platforms (Linux, macOS, Windows)
2. Builds source distribution (sdist)
3. Uploads all packages to PyPI

**Trigger:** Runs when a new tag matching `v*` is created (automatically by the check-version workflow or manually).

## Setting Up Secrets

To use the PyPI publishing workflow, you need to set up the following secrets in your GitHub repository:

1. `PYPI_USERNAME`: Your PyPI username (or `__token__` if using API tokens)
2. `PYPI_API_TOKEN`: Your PyPI password or API token


