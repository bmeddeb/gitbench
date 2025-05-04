# gitbench

gitbench is a high-performance Git operations library with Python bindings, powered by Rust. It provides efficient repository management, blame analysis, commit extraction, and integration with Git hosting providers.

## 🚀 Key Features

- **High-Performance Core**: Core operations implemented in Rust for maximum speed
- **Asynchronous API**: All repository operations are non-blocking using asyncio
- **Git Provider Integration**: Support for GitHub APIs with GitLab and BitBucket coming soon
- **Pydantic Models**: Strong validation and serialization for all data
- **Token Management**: Automatic token rotation and rate limit handling
- **Pandas Integration**: Convert results to DataFrames for analysis

## 🏗️ Architecture

gitbench uses a hybrid architecture:

```
┌────────────────────────┐
│   Python Interface     │   User-friendly API, asyncio integration
├────────────────────────┤
│   PyO3 Bridge Layer    │   Seamless Rust-Python interoperability
├────────────────────────┤
│   Rust Core Library    │   High-performance Git operations
└────────────────────────┘
```

## 📋 Main Components

### Core Repository Operations

- [**Repository Manager**](RepoManager.md): Main interface for managing repositories
- [**Clone Operations**](api/clone-monitoring.md): Non-blocking repository cloning with progress tracking
- [**Blame Analysis**](api/blame-commit.md): Extract and analyze blame information
- [**Commit Extraction**](api/blame-commit.md): Analyze commit history and statistics

### Provider API Clients

- [**GitHub Client**](providers/github.md): Complete API client for GitHub
- [**Token Manager**](token-management.md): Token rotation and rate limit handling
- [**Provider Models**](providers/models.md): Type-safe models for all provider data

## 🔧 Installation

```bash
pip install gitbench
```

See the [Installation Guide](installation.md) for detailed instructions.

## 🚦 Quick Start

```python
import asyncio
from gitbench import RepoManager

async def main():
    # Initialize repository manager
    repo_manager = RepoManager(
        urls=["https://github.com/user/repo"],
        github_username="username",
        github_token="token"
    )
    
    # Clone repositories
    await repo_manager.clone_all()
    
    # Get clone tasks
    tasks = await repo_manager.fetch_clone_tasks()
    
    # Find a cloned repository
    repo_path = next(
        (task.temp_dir for task in tasks.values() 
         if task.status.status_type == "completed"),
        None
    )
    
    if repo_path:
        # Analyze blame
        blame = await repo_manager.bulk_blame(
            repo_path, ["README.md"]
        )
        
        # Extract commits
        commits = await repo_manager.extract_commits(repo_path)
    
    # Clean up
    await repo_manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

See the [Basic Usage Example](examples/basic-usage.md) for a more complete example.

## 📚 Documentation

- [**Installation**](installation.md): Detailed installation instructions
- [**API Reference**](api/index.md): Core API documentation
- [**Provider APIs**](providers/index.md): Git provider API clients
- [**Examples**](examples/basic-usage.md): Code examples for various use cases
- [**Advanced**](advanced/performance.md): Performance tips and advanced usage
  - [**Datetime Handling**](advanced/datetime-handling.md): Working with dates and times from APIs

## 🤝 Contributing

We welcome contributions! See the [Contributing Guide](development/contributing.md) for details.

## 📄 License

gitbench is released under the MIT License. See the LICENSE file for details.