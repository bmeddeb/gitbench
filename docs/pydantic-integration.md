# Pydantic Integration

GitFleet provides comprehensive integration with [Pydantic](https://docs.pydantic.dev/), offering enhanced validation, serialization, and better type safety for all data models.

## Overview

Pydantic integration in GitFleet provides the following benefits:

- **Strong Validation**: All API responses are validated against defined schemas
- **Serialization**: Easy conversion to JSON, dictionaries, and other formats
- **Type Safety**: Improved IDE completions and static type checking
- **Consistent Interface**: Same API across Rust and Python components

## Data Models

All GitFleet data models are built using Pydantic's `BaseModel` class, including:

- Provider models (`RepoInfo`, `UserInfo`, etc.)
- Repository management models (`CloneStatus`, `CloneTask`)
- Token and credential management models

## Example Usage

### Basic Model Usage

```python
from GitFleet import RepoInfo

# Create a model instance with validation
repo = RepoInfo(
    name="example-repo",
    full_name="user/example-repo",
    clone_url="https://github.com/user/example-repo.git",
    provider_type="github"
)

# Access properties with proper types
print(repo.name)  # example-repo
print(repo.fork)  # False (default value)

# Convert to dictionary
repo_dict = repo.model_dump()
print(repo_dict)

# Convert to JSON with custom formatting
repo_json = repo.model_dump_json(indent=2)
print(repo_json)

# Parse created_at date if available
if repo.created_at:
    dt = repo.created_datetime()
    if dt:
        print(f"Created on: {dt.strftime('%Y-%m-%d')}")
```

### Working with API Responses

When using the GitHubClient, all API responses are automatically converted to Pydantic models:

```python
from GitFleet import GitHubClient

async def main():
    client = GitHubClient(token="your-token")
    
    # Returns a list of RepoInfo objects
    repos = await client.fetch_repositories("octocat")
    
    # Use model methods
    for repo in repos:
        print(f"Repository: {repo.full_name}")
        print(f"Created: {repo.created_datetime()}")
        
        # Convert to JSON
        print(repo.model_dump_json())
```

### Rust Type Integration

GitFleet provides Pydantic wrappers for the Rust-generated classes:

```python
from GitFleet import RepoManager, CloneTask, CloneStatus, CloneStatusType

async def main():
    # Create a RepoManager
    manager = RepoManager(
        urls=["https://github.com/user/repo.git"],
        github_username="username",
        github_token="token"
    )
    
    # Clone repositories
    await manager.clone_all()
    
    # Get clone tasks as Pydantic models
    tasks = await manager.fetch_clone_tasks()
    
    for url, task in tasks.items():
        # Access properties with proper type hints
        print(f"Repository: {url}")
        print(f"Status: {task.status.status_type}")
        
        # Use Pydantic features
        print(task.model_dump_json(indent=2))
        
        # Enums for type safety
        if task.status.status_type == CloneStatusType.COMPLETED:
            print(f"Cloned to: {task.temp_dir}")
```

## Installation

To use Pydantic features, install GitFleet with the Pydantic extra:

```bash
pip install "gitfleet[pydantic]"
```

Or install with multiple extras:

```bash
pip install "gitfleet[pydantic,pandas]"
```

## Conversion Utilities

GitFleet provides utility functions for working with Pydantic models:

- `to_dict(obj)`: Convert any object to a dictionary
- `to_json(obj, indent=None)`: Convert any object to a JSON string
- `to_dataframe(data)`: Convert models to a pandas DataFrame
- `flatten_dataframe(df)`: Flatten nested structures in a DataFrame

## Type Stubs

GitFleet includes comprehensive type stubs (`.pyi` files) for all Pydantic models, ensuring proper IDE completions and static type checking support.