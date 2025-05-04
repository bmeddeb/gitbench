# Contributing to GitFleet

Thank you for your interest in contributing to GitFleet! This document provides guidelines and instructions for contributing to the project. We welcome contributions of all kinds, including bug reports, feature requests, documentation improvements, and code changes.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Building and Testing](#building-and-testing)
- [Contribution Workflow](#contribution-workflow)
- [Coding Standards](#coding-standards)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Pull Requests](#pull-requests)
- [Release Process](#release-process)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. We expect all contributors to be respectful, inclusive, and considerate of others.

## Getting Started

### Prerequisites

Before you begin contributing, ensure you have the following installed:

- Rust (1.60+)
- Python (3.8+)
- Cargo and pip
- Git

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/bmeddeb/GitFleet.git
cd GitFleet
```

3. Add the upstream repository as a remote:

```bash
git remote add upstream https://github.com/bmeddeb/GitFleet.git
```

## Development Environment

### Setup

1. Create a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
```

2. Install development dependencies:

```bash
pip install -e ".[dev]"
```

3. Install Rust dependencies:

```bash
cargo install maturin
```

### Building

To build the project locally:

```bash
maturin develop
```

This will compile the Rust code and install the Python package in development mode.

## Building and Testing

### Running Tests

To run the Rust tests:

```bash
cargo test
```

To run the Python tests:

```bash
pytest
```

To run linting:

```bash
# Python
black .
isort .
flake8

# Rust
cargo clippy
cargo fmt --check
```

## Contribution Workflow

1. **Choose an Issue**: Start by finding an issue you'd like to work on, or create a new one if you've identified a bug or enhancement.

2. **Create a Branch**: Create a branch for your work based on the `main` branch:

```bash
git checkout -b feature/your-feature-name
```

Use a prefix like `feature/`, `bugfix/`, `docs/`, etc., to indicate the type of change.

3. **Make Changes**: Implement your changes, following the coding standards below.

4. **Write Tests**: Add tests that cover your changes.

5. **Update Documentation**: Update any relevant documentation.

6. **Commit Changes**: Commit your changes with a descriptive commit message:

```bash
git commit -m "Add feature X to solve problem Y"
```

7. **Push to GitHub**: Push your branch to your fork:

```bash
git push origin feature/your-feature-name
```

8. **Create a Pull Request**: Create a pull request from your branch to the main repository's `main` branch.

## Coding Standards

### Rust Code Style

- Follow the [Rust Style Guide](https://github.com/rust-dev-tools/fmt-rfcs/blob/master/guide/guide.md)
- Use `cargo fmt` to format your code
- Use `cargo clippy` to check for common issues
- Use meaningful variable and function names
- Document public functions with doc comments
- Prefer returning `Result` types for functions that can fail

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use type hints for function arguments and return values
- Document functions and classes with docstrings
- Prefer async/await for asynchronous code

### Commit Messages

- Use clear, descriptive commit messages
- Start with a short summary line (max 72 characters)
- Include a more detailed description if necessary
- Reference issue numbers when relevant (e.g., "Fixes #123")

## Documentation

Good documentation is essential for any project. When contributing:

- Update the README.md if your changes affect usage, installation, or features
- Document new functions, classes, and modules with docstrings
- Update or add Markdown files in the docs/ directory as needed
- Add example code demonstrating new features

### Documentation Standards

- Use Markdown for all documentation
- Use clear, concise language
- Include code examples when relevant
- Break up long documentation with headings and lists
- Link to other relevant documentation when appropriate

## Issue Reporting

When reporting issues:

1. Use the issue template if provided
2. Include a clear description of the problem
3. Provide steps to reproduce the issue
4. Include relevant information like OS, Python/Rust version, etc.
5. Attach logs or screenshots if applicable

## Pull Requests

When submitting a pull request:

1. Fill out the pull request template completely
2. Link to any related issues
3. Describe the changes and the problem they solve
4. List any new dependencies introduced
5. Include screenshots or output examples for UI/UX changes
6. Ensure all tests pass
7. Be responsive to code review feedback

## Release Process

The GitFleet release process is as follows:

1. Version bumps follow [Semantic Versioning](https://semver.org/)
2. Releases are tagged in Git with the version number (e.g., `v1.0.0`)
3. Release notes are created for each release
4. Releases are published to PyPI

### Release Checklist

Before a release:

- All tests must pass
- Documentation must be up-to-date
- CHANGELOG.md must be updated
- Version number must be updated in appropriate files

## Architecture Overview

For a deeper understanding of GitFleet's architecture, see the [Architecture Documentation](architecture.md) and [Python/Rust Bridge](python-rust-bridge.md) pages.

## Getting Help

If you need help with contributing:

- Ask questions in the issue where you're contributing
- Join our community chat/forum (if available)
- Reach out to maintainers via email

Thank you for contributing to GitFleet! Your efforts help make this project better for everyone.