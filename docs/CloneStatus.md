# CloneStatus

In GitFleet, there are two types of `CloneStatus` objects that represent the status of repository cloning operations:

1. **RustCloneStatus**: The native Rust implementation returned by `RepoManager.fetch_clone_tasks()`
2. **PydanticCloneStatus**: A Pydantic model version with additional validation and serialization features

Both types provide the same core functionality but serve different use cases.

## Fields

Both `RustCloneStatus` and `PydanticCloneStatus` share the same fields:

- `status_type` (`str`): The type of status. Possible values are:
  - `'queued'`: The cloning task is waiting to start.
  - `'cloning'`: The repository is currently being cloned. See `progress` for completion percentage.
  - `'completed'`: The repository has been successfully cloned.
  - `'failed'`: The cloning operation failed. See `error` for details.
- `progress` (`Optional[int]`): The percentage of completion (0-100) if the task is currently cloning, or `None` otherwise.
- `error` (`Optional[str]`): An error message if the cloning operation failed, or `None` otherwise.

## Typical Usage

You will most often encounter `CloneStatus` as part of a `CloneTask` when checking the status of repository cloning operations.

### Example: Using RustCloneStatus Directly

```python
import asyncio
from GitFleet import RepoManager

async def main():
    urls = ["https://github.com/owner/repo1.git"]
    manager = RepoManager(urls, "username", "token")
    await manager.clone_all()
    clone_tasks = await manager.fetch_clone_tasks()
    for url, task in clone_tasks.items():
        status = task.status  # This is a RustCloneStatus object
        print(f"Repo: {url}")
        print(f"  Status: {status.status_type}")
        if status.status_type == "cloning":
            print(f"  Progress: {status.progress}%")
        if status.status_type == "failed":
            print(f"  Error: {status.error}")

asyncio.run(main())
```

### Example: Converting to PydanticCloneStatus

```python
import asyncio
from GitFleet import RepoManager, to_pydantic_status

async def main():
    urls = ["https://github.com/owner/repo1.git"]
    manager = RepoManager(urls, "username", "token")
    await manager.clone_all()
    clone_tasks = await manager.fetch_clone_tasks()
    for url, task in clone_tasks.items():
        # Convert RustCloneStatus to PydanticCloneStatus
        status = to_pydantic_status(task.status)
        print(f"Repo: {url}")
        print(f"  Status: {status.status_type}")
        
        # Use Pydantic features
        status_json = status.model_dump_json(indent=2)
        print(f"  Status JSON: {status_json}")

asyncio.run(main())
```

---

The `CloneStatus` objects help you monitor and react to the state of repository cloning operations in your workflow.