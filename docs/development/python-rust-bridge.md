# Python-Rust Bridge

gitbench uses a hybrid architecture with a Rust core for performance-critical Git operations and Python for high-level APIs and provider clients. This page explains how the Python-Rust bridge works and how to use it effectively.

## Architecture Overview

gitbench's architecture consists of three main layers:

1. **Rust Core Library**: High-performance, memory-safe implementation of key Git operations (cloning, blame, commit analysis)
2. **PyO3 Bridge Layer**: Exposes Rust functionality to Python with asyncio integration
3. **Python Interface**: User-friendly API with provider clients, data models, and utility functions

## How PyO3 is Used in gitbench

The PyO3 library allows gitbench to expose Rust functionality to Python in a way that feels natural to Python developers. Here's how it works:

### Rust Struct Definitions with PyO3 Annotations

```rust
use pyo3::prelude::*;

#[pyclass(name = "RepoManager", module = "gitbench")]
struct RepoManager {
    inner: Arc<InternalRepoManagerLogic>,
}

#[pymethods]
impl RepoManager {
    #[new]
    fn new(urls: Vec<String>, github_username: String, github_token: String) -> Self {
        let string_urls: Vec<&str> = urls.iter().map(|s| s.as_str()).collect();
        Self {
            inner: Arc::new(InternalRepoManagerLogic::new(
                &string_urls,
                &github_username,
                &github_token,
            )),
        }
    }
    
    /// Clones all repositories configured in this manager instance asynchronously.
    #[pyo3(name = "clone_all")]
    fn clone_all<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        tokio::future_into_py(py, async move {
            inner.clone_all().await;
            Python::with_gil(|py| Ok(py.None()))
        })
    }
    
    // ...other methods
}
```

### Async Bridge with pyo3-async-runtimes

gitbench uses [pyo3-async-runtimes](https://github.com/PyO3/pyo3-async-runtimes) to bridge between Rust's async model (Tokio) and Python's asyncio:

```rust
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio;

#[pymethods]
impl RepoManager {
    /// Performs 'git blame' on multiple files within a cloned repository asynchronously.
    #[pyo3(name = "bulk_blame")]
    fn bulk_blame<'py>(
        &self,
        py: Python<'py>,
        repo_path: String,
        file_paths: Vec<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        tokio::future_into_py(py, async move {
            let result_map = inner
                .bulk_blame(&PathBuf::from(repo_path), file_paths)
                .await;
            Python::with_gil(|py| -> PyResult<Py<PyAny>> {
                // Convert result to Python object
                // ...
            })
        })
    }
}
```

## What's Implemented in Rust vs Python

gitbench uses a strategic split between Rust and Python implementations:

### Rust Implementation (Performance-Critical Operations)

The following operations are implemented in Rust for maximum performance:

| Rust Module | Functionality |
|-------------|--------------|
| `clone.rs` | Repository cloning with progress monitoring |
| `blame.rs` | Git blame analysis for file content |
| `commits.rs` | Commit history extraction and analysis |
| `repo.rs` | Core repository management logic |

These operations benefit significantly from Rust's performance characteristics, including:
- Memory efficiency through Rust's ownership model
- Parallel processing with Rayon for commit analysis
- No Python GIL limitations during intensive operations
- Direct use of git2-rs for native Git operations

### Python Implementation (API & Provider Layer)

The following components are implemented in Python:

| Python Module | Functionality |
|---------------|--------------|
| `providers/github.py` | GitHub API client |
| `providers/token_manager.py` | Token management with rotation |
| `providers/base.py` | Provider abstractions and interfaces |
| `models/common.py` | Data models for API requests/responses |
| `models/repo.py` | Wrappers for Rust objects with Pydantic |
| `utils/auth.py` | Authentication utilities |
| `utils/converters.py` | Data conversion utilities |
| `utils/rate_limit.py` | Rate limiting logic |

These components leverage Python's strengths:
- Rich ecosystem for HTTP requests and API clients
- Pydantic for data validation and serialization
- Intuitive asyncio interface for Python users
- Easier integration with Python frameworks

## Data Conversion Between Python and Rust

gitbench handles data conversion between Python and Rust through PyO3:

### Python to Rust
- Python strings → Rust `String`
- Python lists → Rust `Vec<T>`
- Python dicts → Rust `HashMap<K, V>` or custom structs
- Python None → Rust `Option<T>` as `None`

### Rust to Python
- Rust `String` → Python strings
- Rust `Vec<T>` → Python lists
- Rust `HashMap<K, V>` → Python dicts
- Rust structs → Python objects via PyO3
- Rust `Result<T, E>` → Python return value or exception
- Rust `Option<T>` → Python value or None

## Exposed Rust Objects as Python Classes

The Rust library exposes the following classes to Python:

| Rust Class | Python Class | Description |
|------------|--------------|-------------|
| `RepoManager` | `gitbench.RepoManager` | Main entry point for repository operations |
| `ExposedCloneStatus` | `gitbench.CloneStatus` | Repository clone status information |
| `ExposedCloneTask` | `gitbench.CloneTask` | Repository clone task with status |

## Performance Considerations

The hybrid Rust/Python architecture offers significant performance benefits:

- **Clone Operations**: Parallel, efficient repository cloning
- **Blame Analysis**: Fast line-by-line blame extraction (5-10x faster than pure Python)
- **Commit Processing**: Efficient commit history extraction with parallel processing
- **Memory Usage**: Lower memory footprint for large repositories
- **GIL Avoidance**: Compute-intensive operations run outside Python's GIL

## Contributing to the Bridge Layer

When contributing to the Python-Rust bridge:

1. Determine whether the new functionality is performance-critical (use Rust) or API-related (use Python)
2. For Rust changes:
   - Implement the core functionality in the appropriate Rust module
   - Add PyO3 bindings to expose the functionality to Python
   - Update tests for both Rust and Python interfaces
3. For Python changes:
   - Implement in the appropriate Python module
   - Ensure compatibility with the Rust-exposed objects
   - Add Python-specific tests

## Related Documentation

- [Architecture Overview](architecture.md)
- [Contributing Guide](contributing.md)
- [Performance Tips](../advanced/performance.md)