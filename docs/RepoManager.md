# RepoManager

The `RepoManager` class is the main interface for managing multiple Git repositories, performing high-performance clone operations, running blame analysis, and extracting commit history. It is designed for asynchronous use, allowing you to efficiently manage and analyze many repositories in parallel.

The `RepoManager` is implemented in Rust for maximum performance, with Python bindings via PyO3.

> **Note:** All major methods of `RepoManager` are asynchronous and should be awaited. You can use Python's `asyncio` or any compatible event loop to run these methods.

## Overview

- Manage a set of repositories (clone, track status, cleanup)
- Perform bulk blame operations on files
- Extract commit history from repositories
- All operations are designed to be non-blocking and scalable

## Initialization

```python
from gitbench import RepoManager

# List of repository URLs to manage
urls = [
    "https://github.com/owner/repo1.git",
    "https://github.com/owner/repo2.git",
]

# Create a RepoManager instance
manager = RepoManager(urls, github_username="your-username", github_token="your-token")
```

## Methods

### clone_all()
Clones all repositories configured in this manager instance asynchronously.

```python
await manager.clone_all()
```

### fetch_clone_tasks()
Fetches the current status of all cloning tasks asynchronously. Returns a dictionary mapping repository URLs to `RustCloneTask` objects, which include a `RustCloneStatus`.

**Return Type**: `Dict[str, RustCloneTask]`

```python
# Get raw Rust objects
rust_tasks = await manager.fetch_clone_tasks()
for url, task in rust_tasks.items():
    print(f"{url}: {task.status.status_type}")
    
# Or convert to Pydantic models for additional features
from gitbench import convert_clone_tasks
pydantic_tasks = convert_clone_tasks(rust_tasks)
for url, task in pydantic_tasks.items():
    print(f"Task: {url}")
    print(f"Status: {task.status.status_type}")
    print(f"JSON: {task.model_dump_json()}")
```

### clone(url)
Clones a single repository specified by URL asynchronously.

```python
await manager.clone("https://github.com/owner/repo3.git")
```

### bulk_blame(repo_path, file_paths)
Performs 'git blame' on multiple files within a cloned repository asynchronously. Returns a dictionary mapping file paths to blame results.

```python
blame_results = await manager.bulk_blame("/path/to/repo", ["file1.py", "file2.py"])
for file, lines in blame_results.items():
    print(f"Blame for {file}:", lines)
```

### extract_commits(repo_path)
Extracts commit data from a cloned repository asynchronously. Returns a list of commit dictionaries.

```python
commits = await manager.extract_commits("/path/to/repo")
print(f"Found {len(commits)} commits.")
```

### cleanup()
Cleans up all temporary directories created for cloned repositories. Returns a dictionary with repository URLs as keys and cleanup results as values (True for success, or an error message).

```python
cleanup_results = manager.cleanup()
for url, result in cleanup_results.items():
    print(f"Cleanup for {url}: {result}")
```

---

## Simple Example

```python
import asyncio
from gitbench import RepoManager

async def main():
    urls = ["https://github.com/owner/repo1.git"]
    manager = RepoManager(urls, github_username="username", github_token="token")
    await manager.clone_all()
    tasks = await manager.fetch_clone_tasks()
    print(tasks)
    await manager.bulk_blame("/path/to/repo", ["file1.py"])
    await manager.extract_commits("/path/to/repo")
    manager.cleanup()

asyncio.run(main())
```

## Advanced Example: Managing Multiple Repositories

```python
import asyncio
from gitbench import RepoManager

async def main():
    urls = [
        "https://github.com/owner/repo1.git",
        "https://github.com/owner/repo2.git",
        "https://github.com/owner/repo3.git",
    ]
    manager = RepoManager(urls, github_username="username", github_token="token")
    await manager.clone_all()
    clone_tasks = await manager.fetch_clone_tasks()
    for url, task in clone_tasks.items():
        print(f"{url}: {task.status.status_type}")
    # Run blame on all files in all repos (example)
    for url, task in clone_tasks.items():
        if task.temp_dir:
            blame = await manager.bulk_blame(task.temp_dir, ["main.py", "utils.py"])
            print(f"Blame for {url}: {blame}")
    # Extract commits
    for url, task in clone_tasks.items():
        if task.temp_dir:
            commits = await manager.extract_commits(task.temp_dir)
            print(f"Commits for {url}: {len(commits)} found.")
    # Cleanup
    results = manager.cleanup()
    print("Cleanup results:", results)

asyncio.run(main())
```

---

## Working with Pydantic Models

gitbench provides Pydantic models that mirror the Rust objects returned by `RepoManager`. These models add serialization, validation, and other Pydantic features:

```python
import asyncio
from gitbench import RepoManager
from gitbench import to_pydantic_task, to_pydantic_status, convert_clone_tasks

async def main():
    # Create a repo manager
    manager = RepoManager(urls=["https://github.com/owner/repo.git"], 
                         github_username="username", github_token="token")
    
    # Get tasks from the manager (returns Rust objects)
    rust_tasks = await manager.fetch_clone_tasks()
    
    # Convert individual tasks to Pydantic models
    for url, task in rust_tasks.items():
        pydantic_task = to_pydantic_task(task)
        
        # Now you can use Pydantic features
        task_json = pydantic_task.model_dump_json(indent=2)
        print(task_json)
        
        # Convert a status to Pydantic
        pydantic_status = to_pydantic_status(task.status)
        print(pydantic_status.model_dump())
    
    # Or convert all tasks at once
    pydantic_tasks = convert_clone_tasks(rust_tasks)
    print(f"Converted {len(pydantic_tasks)} tasks to Pydantic models")

asyncio.run(main())
```

For details on the Rust objects `RustCloneTask` and `RustCloneStatus`, and their Pydantic counterparts `PydanticCloneTask` and `PydanticCloneStatus`, see the [CloneTask](./CloneTask.md) and [CloneStatus](./CloneStatus.md) documentation pages.
