# Quick Start Guide

This guide will help you get started with gitbench quickly. We'll cover installation, basic setup, and common operations.

## Installation

Install gitbench using pip:

```bash
pip install gitbench
```

For development installations:

```bash
git clone https://github.com/bmeddeb/gitbench.git
cd gitbench
pip install -e .
```

## Basic Usage

### Initializing the Repository Manager

```python
import asyncio
from gitbench import RepoManager

# Replace with your credentials
github_username = "your-username"
github_token = "your-github-token"

# List of repositories to work with
repos = [
    "https://github.com/user/repo1",
    "https://github.com/user/repo2"
]

# Initialize the repository manager
repo_manager = RepoManager(
    urls=repos,
    github_username=github_username,
    github_token=github_token
)
```

### Cloning Repositories

```python
async def main():
    # Clone all repositories asynchronously
    clone_future = repo_manager.clone_all()
    
    # Wait for cloning to complete
    await clone_future
    
    # Get the status of all clone tasks
    clone_tasks = await repo_manager.fetch_clone_tasks()
    
    # Find successfully cloned repositories
    for url, task in clone_tasks.items():
        if task.status.status_type == "completed":
            print(f"Successfully cloned {url} to {task.temp_dir}")
        elif task.status.status_type == "failed":
            print(f"Failed to clone {url}: {task.status.error}")

# Run the async function
asyncio.run(main())
```

### Analyzing Repository Data

#### Blame Analysis

```python
async def analyze_blame(repo_path, file_paths):
    # Get blame information for specified files
    blame_results = await repo_manager.bulk_blame(repo_path, file_paths)
    
    # Process the results
    for file_path, blame_info in blame_results.items():
        if isinstance(blame_info, list):  # Success case
            print(f"Blame for {file_path}:")
            # Count lines by author
            authors = {}
            for line in blame_info:
                author = line["author_name"]
                authors[author] = authors.get(author, 0) + 1
            
            # Show top contributors
            for author, count in sorted(authors.items(), key=lambda x: x[1], reverse=True):
                print(f"  {author}: {count} lines")
        else:  # Error case
            print(f"Error analyzing {file_path}: {blame_info}")
```

#### Commit Analysis

```python
async def analyze_commits(repo_path):
    # Extract commit history
    commits = await repo_manager.extract_commits(repo_path)
    
    if isinstance(commits, list):  # Success case
        print(f"Found {len(commits)} commits")
        
        # Show recent commits
        for commit in commits[:5]:
            print(f"Commit: {commit['sha'][:7]}")
            print(f"Author: {commit['author_name']}")
            print(f"Message: {commit['message'].split('\n')[0]}")
            print(f"Changes: +{commit['additions']} -{commit['deletions']}")
    else:  # Error case
        print(f"Error analyzing commits: {commits}")
```

### Cleaning Up

```python
async def cleanup():
    # Clean up all temporary directories
    cleanup_results = await repo_manager.cleanup()
    
    # Check cleanup results
    for url, result in cleanup_results.items():
        if result is True:
            print(f"Successfully cleaned up {url}")
        else:
            print(f"Failed to clean up {url}: {result}")
```

## Complete Example

Here's a complete example putting everything together:

```python
import asyncio
import os
from gitbench import RepoManager

async def main():
    # Initialize repository manager with credentials from environment
    repo_manager = RepoManager(
        urls=["https://github.com/bmeddeb/gradelib"],
        github_username=os.environ.get("GITHUB_USERNAME"),
        github_token=os.environ.get("GITHUB_TOKEN")
    )
    
    # Clone repositories
    await repo_manager.clone_all()
    
    # Get clone statuses
    clone_tasks = await repo_manager.fetch_clone_tasks()
    
    # Find a successfully cloned repository
    repo_path = None
    for url, task in clone_tasks.items():
        if task.status.status_type == "completed":
            repo_path = task.temp_dir
            print(f"Analyzing repository: {url}")
            break
    
    if repo_path:
        # Find Python files in the repository
        file_paths = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".py"):
                    rel_path = os.path.join(root, file).replace(repo_path + os.sep, "")
                    file_paths.append(rel_path)
                    if len(file_paths) >= 3:  # Limit to 3 files
                        break
            if len(file_paths) >= 3:
                break
        
        # Analyze blame
        if file_paths:
            blame_results = await repo_manager.bulk_blame(repo_path, file_paths)
            for file_path, blame_info in blame_results.items():
                if isinstance(blame_info, list):
                    authors = {}
                    for line in blame_info:
                        author = line["author_name"]
                        authors[author] = authors.get(author, 0) + 1
                    
                    print(f"\nBlame summary for {file_path}:")
                    for author, count in sorted(authors.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {author}: {count} lines")
        
        # Analyze commits
        commits = await repo_manager.extract_commits(repo_path)
        if isinstance(commits, list):
            print(f"\nFound {len(commits)} commits")
            print("Recent commits:")
            for commit in commits[:3]:
                print(f"  {commit['sha'][:7]} - {commit['message'].split('\\n')[0]}")
    
    # Clean up
    await repo_manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

## Next Steps

- [Repository Manager](RepoManager.md): More details on repository management
- [Clone Operations](api/clone-monitoring.md): Advanced clone monitoring
- [Provider APIs](providers/index.md): Working with Git hosting providers
- [Examples](examples/basic-usage.md): More complete examples