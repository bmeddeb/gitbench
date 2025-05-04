# CloneTask

In GitFleet, there are two types of `CloneTask` objects that represent repository cloning tasks and their statuses:

1. **RustCloneTask**: The native Rust implementation returned by `RepoManager.fetch_clone_tasks()`
2. **PydanticCloneTask**: A Pydantic model version with additional validation and serialization features

Both types provide the same core functionality but serve different use cases.

## RustCloneTask Fields

The `RustCloneTask` class represents a repository cloning task directly from the Rust implementation:

- `url` (`str`): The URL of the repository being cloned.
- `status` (`RustCloneStatus`): The current status of the cloning operation.
- `temp_dir` (`Optional[str]`): The path to the temporary directory where the repository was cloned, or `None` if cloning has not completed or failed.

## PydanticCloneTask Fields

The `PydanticCloneTask` class provides a Pydantic model with the same fields as `RustCloneTask` but with added validation and serialization:

- `url` (`str`): The URL of the repository being cloned.
- `status` (`PydanticCloneStatus`): The current status of the cloning operation as a Pydantic model.
- `temp_dir` (`Optional[str]`): The path to the temporary directory where the repository was cloned, or `None` if cloning has not completed or failed.

## Typical Usage

### Using RustCloneTask Directly

```python
import asyncio
from GitFleet import RepoManager

async def main():
    # Create a repo manager
    urls = ["https://github.com/owner/repo1.git"]
    manager = RepoManager(urls, "username", "token")
    
    # Clone repositories
    await manager.clone_all()
    
    # Get RustCloneTask objects directly from the manager
    rust_tasks = await manager.fetch_clone_tasks()
    
    # Access task information
    for url, task in rust_tasks.items():
        print(f"Repo: {url}")
        print(f"  Status: {task.status.status_type}")
        print(f"  Temp dir: {task.temp_dir}")
        
        # Check status type with string comparison
        if task.status.status_type == "completed":
            print("  Cloning completed successfully")

asyncio.run(main())
```

### Converting to Pydantic Models

```python
import asyncio
from GitFleet import RepoManager, convert_clone_tasks

async def main():
    # Create a repo manager
    urls = ["https://github.com/owner/repo1.git"]
    manager = RepoManager(urls, "username", "token")
    
    # Clone repositories
    await manager.clone_all()
    
    # Get tasks and convert to Pydantic models
    rust_tasks = await manager.fetch_clone_tasks()
    pydantic_tasks = convert_clone_tasks(rust_tasks)
    
    # Use Pydantic features
    for url, task in pydantic_tasks.items():
        # Access the same fields
        print(f"Repo: {url}")
        print(f"  Status: {task.status.status_type}")
        
        # Use Pydantic features
        task_json = task.model_dump_json(indent=2)
        print(f"  JSON: {task_json}")

asyncio.run(main())
```

### Using temp_dir for Further Operations

After cloning, you can use the `temp_dir` field to perform further operations:

```python
# ... after fetching clone_tasks ...
for url, task in rust_tasks.items():
    if task.temp_dir:
        blame = await manager.bulk_blame(task.temp_dir, ["main.py"])
        print(f"Blame for {url}: {blame}")
```

---

For details on the `status` field, see the [CloneStatus](./CloneStatus.md) documentation.