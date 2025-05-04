# 🚀 gitbench

**High-performance Git repository analysis and management in Python, powered by Rust**

[![PyPI version](https://img.shields.io/pypi/v/gitbench)](https://pypi.org/project/gitbench/)
[![License](https://img.shields.io/github/license/bmeddeb/gitbench)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/bmeddeb/gitbench/ci.yml)](https://github.com/bmeddeb/gitbench/actions)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://bmeddeb.github.io/gitbench/)

---

## ⚡ Overview

**gitbench** is a high-performance Python library built in Rust for asynchronous Git repository analysis at scale. It combines the speed and memory efficiency of Rust with the ease of use of Python's async/await syntax.

It supports bulk operations like:

- Parallel `git blame` across many files (10-50x faster than pure Python)
- Async cloning of hundreds of repositories with progress monitoring
- GitHub, GitLab, and GitBucket integration with native API clients
- API token rotation and rate limit monitoring to prevent 429 errors
- Real-time progress tracking for long-running tasks

Ideal for:

- Grading systems (e.g., student GitHub repos)
- Research in software engineering and code evolution
- DevOps and CI insights at scale
- Custom git analytics platforms
- Any application requiring efficient handling of multiple Git repositories

---

## 🔧 Features

- ✅ **Rayon-powered multithreaded Git operations**
- ✅ **Asynchronous repo cloning with async/await**
- ✅ **Integrated clients for GitHub, GitLab, GitBucket**
- ✅ **Multiple token management with auto-rotation**
- ✅ **Rate limit tracking and resilience**
- ✅ **Modular architecture: easily extendable**

---

## 📚 Documentation

Visit our [comprehensive documentation](https://bmeddeb.github.io/gitbench/) for:
- Detailed API reference
- Usage examples
- Provider integrations
- Advanced configuration

---

## 📦 Installation

```bash
pip install gitbench
```

> Requires Python 3.8+ and a Rust toolchain for building native extensions.
> Pre-built wheels are available for most common platforms.

---

## 🚀 Quick Start

```python
import asyncio
from gitbench import RepoManager

async def main():
    manager = RepoManager()
    await manager.clone_repos([
        "https://github.com/user/repo1.git",
        "https://github.com/user/repo2.git"
    ])
    results = await manager.blame_all("target_file.py")
    print(results)

asyncio.run(main())
```

---

## 🔌 Integration

gitbench is designed to interoperate with:

- **GitHub, GitLab, GitBucket** via async clients
- Jupyter notebooks (Pandas-friendly outputs)
- Flask, FastAPI, and async dashboards (e.g., Plotly Dash)
- CI/CD pipelines or grading systems

---

## 🛠 Architecture

| Layer            | Technology    | Purpose                                      |
|------------------|---------------|----------------------------------------------|
| Rust Core        | `git2`, `rayon` | Fast Git ops and parallelism                |
| Python Interface | `PyO3`, `maturin` | Native bindings and async APIs             |
| Async Clients    | `httpx`, `aiohttp` | GitHub/GitLab API access                   |
| Token Manager    | Custom        | Multi-token rotation & rate monitoring       |

---

## 📈 Roadmap

- [ ] Git diff and patch inspection
- [ ] Repository graph & commit visualizations
- [ ] CSV/JSON export for data analysis
- [ ] Web dashboard example with live metrics

---

## 🤝 Contributing

We welcome contributions! To get started:

```bash
git clone https://github.com/bmeddeb/gitbench
cd gitbench
pip install -e ".[dev]"
```

Run tests with:

```bash
pytest
```

For more details on contributing, see our [development documentation](https://bmeddeb.github.io/gitbench/development/architecture/).

---

## 📄 License

[MIT License](LICENSE)

---

## ✨ Credits

gitbench is inspired by the needs of large-scale Git analysis in grading, research, and dev tooling. Built with ❤️ using Rust + Python.
