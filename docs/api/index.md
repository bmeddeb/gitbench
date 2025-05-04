# gitbench Core API Reference

The gitbench Core API provides a set of high-performance operations for Git repositories, implemented in Rust with Python bindings. This page serves as a central reference for the main components of the Core API.

## Main Components

### Repository Management

The [RepoManager](../RepoManager.md) class is the central interface for working with Git repositories. It provides methods for:

- Cloning repositories
- Monitoring clone status
- Extracting blame information
- Analyzing commit history
- Managing temporary repositories

### Clone Operations

gitbench provides a robust system for asynchronously cloning repositories and monitoring their status:

- [CloneStatus](../CloneStatus.md): Represents the status of a repository cloning operation
- [CloneTask](../CloneTask.md): Represents a repository cloning task with metadata
- [Clone Monitoring](clone-monitoring.md): Advanced techniques for monitoring clone operations

### Blame and Commit Analysis

gitbench excels at high-performance blame and commit analysis:

- [Blame & Commit Analysis](blame-commit.md): Extract detailed blame information and commit history

## Performance Features

The Core API is designed for high performance:

- **Rust Implementation**: Core operations implemented in Rust for maximum speed
- **Asynchronous Operations**: All repository operations are non-blocking using asyncio
- **Parallel Processing**: Multiple repositories can be processed concurrently
- **Efficient Memory Usage**: Optimized data structures for large repositories

## Common Usage Patterns

```python
import asyncio
from gitbench import RepoManager

async def main():
    # Initialize repository manager
    repo_manager = RepoManager(
        urls=["https://github.com/user/repo"],
        github_username="your-username",
        github_token="your-token"
    )
    
    # Clone repositories
    await repo_manager.clone_all()
    
    # Get clone tasks
    clone_tasks = await repo_manager.fetch_clone_tasks()
    
    # Get a repository path from a successful clone
    repo_path = next(
        (task.temp_dir for task in clone_tasks.values() 
         if task.status.status_type == "completed"),
        None
    )
    
    if repo_path:
        # Analyze blame for files
        file_paths = ["README.md", "src/main.py"]
        blame_results = await repo_manager.bulk_blame(repo_path, file_paths)
        
        # Extract commit history
        commits = await repo_manager.extract_commits(repo_path)
    
    # Clean up temporary directories
    await repo_manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

## Error Handling

gitbench provides comprehensive error handling through:

- Clear error messages
- Exception hierarchies
- Status objects with detailed error information

See the [Error Handling](error-handling.md) guide for more details.

## Next Steps

- [Repository Manager](../RepoManager.md): Detailed documentation for the main RepoManager class
- [CloneStatus](../CloneStatus.md) and [CloneTask](../CloneTask.md): Working with clone operations
- [Complete API Reference](../gitbenchAPI.md): Comprehensive reference for all classes and methods
- [Examples](../examples/basic-usage.md): Practical examples of using the Core API