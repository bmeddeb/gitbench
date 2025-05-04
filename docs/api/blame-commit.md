# Blame and Commit Analysis

GitFleet provides powerful tools for analyzing repository blame information and commit history. This page explains how to use these features and interpret the results.

## Blame Analysis

Blame analysis identifies the author and commit information for each line of code in a file. GitFleet's blame analysis is implemented in Rust for maximum performance, making it significantly faster than pure Python implementations.

### Using the `bulk_blame` Method

The `bulk_blame` method allows you to analyze blame information for multiple files at once:

```python
import asyncio
from GitFleet import RepoManager

async def analyze_blame():
    # Initialize repository manager
    repo_manager = RepoManager(urls=["https://github.com/user/repo"])
    
    # Clone the repository
    await repo_manager.clone_all()
    
    # Get clone tasks
    clone_tasks = await repo_manager.fetch_clone_tasks()
    
    # Find a successfully cloned repository
    repo_path = None
    for task in clone_tasks.values():
        if task.status.status_type == "completed":
            repo_path = task.temp_dir
            break
    
    if repo_path:
        # Specify files to analyze
        file_paths = [
            "README.md",
            "src/main.py",
            "tests/test_main.py"
        ]
        
        # Get blame information
        blame_results = await repo_manager.bulk_blame(repo_path, file_paths)
        
        # Process the results
        for file_path, blame_info in blame_results.items():
            if isinstance(blame_info, list):  # Success case
                print(f"Blame for {file_path}:")
                
                # Create a summary by author
                authors = {}
                for line in blame_info:
                    author = line["author_name"]
                    authors[author] = authors.get(author, 0) + 1
                
                # Print the summary
                for author, count in sorted(authors.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {author}: {count} lines")
            else:  # Error case
                print(f"Error analyzing {file_path}: {blame_info}")
```

### Blame Result Structure

The `bulk_blame` method returns a dictionary where:
- Keys are file paths
- Values are either:
  - Lists of line blame information (success case)
  - Error messages (error case)

For each line in a file, the blame information includes:

| Field | Type | Description |
|-------|------|-------------|
| `commit_id` | string | SHA-1 hash of the commit |
| `author_name` | string | Name of the author |
| `author_email` | string | Email of the author |
| `orig_line_no` | int | Original line number in the commit |
| `final_line_no` | int | Current line number in the file |
| `line_content` | string | Content of the line |

### Advanced Blame Analysis

You can perform more advanced blame analysis by processing the detailed information:

```python
# Group by commit
commits = {}
for line in blame_info:
    commit_id = line["commit_id"]
    if commit_id not in commits:
        commits[commit_id] = 0
    commits[commit_id] += 1

# Find the most significant commits
significant_commits = sorted(commits.items(), key=lambda x: x[1], reverse=True)[:5]
print("Most significant commits:")
for commit_id, count in significant_commits:
    print(f"  {commit_id[:7]}: {count} lines")
```

## Commit Analysis

Commit analysis extracts detailed information about the commit history of a repository. This includes commit metadata, authorship information, and change statistics.

### Using the `extract_commits` Method

```python
import asyncio
from GitFleet import RepoManager
from datetime import datetime

async def analyze_commits():
    # Initialize repository manager
    repo_manager = RepoManager(urls=["https://github.com/user/repo"])
    
    # Clone the repository
    await repo_manager.clone_all()
    
    # Get clone tasks
    clone_tasks = await repo_manager.fetch_clone_tasks()
    
    # Find a successfully cloned repository
    repo_path = None
    for task in clone_tasks.values():
        if task.status.status_type == "completed":
            repo_path = task.temp_dir
            break
    
    if repo_path:
        # Extract commit history
        commits = await repo_manager.extract_commits(repo_path)
        
        if isinstance(commits, list):  # Success case
            print(f"Found {len(commits)} commits")
            
            # Show recent commits
            print("\nRecent commits:")
            for commit in commits[:5]:
                # Convert timestamp to datetime
                timestamp = commit["author_timestamp"]
                date = datetime.fromtimestamp(timestamp)
                
                print(f"Commit: {commit['sha'][:7]}")
                print(f"Author: {commit['author_name']} <{commit['author_email']}>")
                print(f"Date: {date.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Message: {commit['message'].split('\\n')[0]}")
                print(f"Changes: +{commit['additions']} -{commit['deletions']}")
                print()
        else:  # Error case
            print(f"Error extracting commits: {commits}")
```

### Commit Result Structure

The `extract_commits` method returns either:
- A list of commit information objects (success case)
- An error message (error case)

Each commit information object includes:

| Field | Type | Description |
|-------|------|-------------|
| `sha` | string | Full SHA-1 hash of the commit |
| `repo_name` | string | Name of the repository |
| `message` | string | Full commit message |
| `author_name` | string | Name of the author |
| `author_email` | string | Email of the author |
| `author_timestamp` | int | Author timestamp (Unix epoch) |
| `author_offset` | int | Author timezone offset in minutes |
| `committer_name` | string | Name of the committer |
| `committer_email` | string | Email of the committer |
| `committer_timestamp` | int | Committer timestamp (Unix epoch) |
| `committer_offset` | int | Committer timezone offset in minutes |
| `additions` | int | Number of lines added |
| `deletions` | int | Number of lines deleted |
| `is_merge` | bool | Whether this is a merge commit |

### Advanced Commit Analysis

You can perform more advanced commit analysis to extract insights:

```python
# Analyze commit activity over time
from collections import defaultdict
import time

# Group commits by month
months = defaultdict(int)
for commit in commits:
    # Get the month from the timestamp
    date = datetime.fromtimestamp(commit["author_timestamp"])
    month_key = f"{date.year}-{date.month:02d}"
    months[month_key] += 1

# Print activity by month
print("Commit activity by month:")
for month, count in sorted(months.items()):
    print(f"  {month}: {count} commits")

# Analyze authors
authors = defaultdict(int)
for commit in commits:
    author = commit["author_name"]
    authors[author] += 1

# Print top contributors
print("\nTop contributors:")
for author, count in sorted(authors.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {author}: {count} commits")

# Analyze code changes
total_additions = sum(commit["additions"] for commit in commits)
total_deletions = sum(commit["deletions"] for commit in commits)
print(f"\nTotal changes: +{total_additions} -{total_deletions}")
```

## Integrating with Pandas

GitFleet's results can be easily converted to pandas DataFrames for advanced analysis:

```python
import pandas as pd

# Convert blame results to DataFrame
def blame_to_dataframe(blame_info):
    return pd.DataFrame(blame_info)

# Convert commit results to DataFrame
def commits_to_dataframe(commits):
    df = pd.DataFrame(commits)
    
    # Convert timestamps to datetime
    df["author_date"] = pd.to_datetime(df["author_timestamp"], unit="s")
    df["committer_date"] = pd.to_datetime(df["committer_timestamp"], unit="s")
    
    return df

# Usage
blame_df = blame_to_dataframe(blame_info)
commits_df = commits_to_dataframe(commits)

# Example analyses
# Author contribution by month
author_monthly = commits_df.groupby([
    commits_df["author_date"].dt.year, 
    commits_df["author_date"].dt.month, 
    "author_name"
])["sha"].count()

# Lines added/removed ratio
commits_df["change_ratio"] = commits_df["additions"] / (commits_df["deletions"] + 1)
```

## Performance Considerations

- **Blame Analysis**: For large repositories or files, blame analysis can be resource-intensive. Consider limiting the number of files analyzed at once.
- **Commit Extraction**: The `extract_commits` method retrieves all commits by default. For repositories with very large histories, this can be slow and memory-intensive.
- **Parallelism**: The Rust implementation automatically parallelizes operations where possible, but system resources still impact performance.

## Related Documentation

- [Repository Manager](../RepoManager.md): Main interface for repository operations
- [Basic Usage Example](../examples/basic-usage.md): Complete example of blame and commit analysis
- [Performance Tips](../advanced/performance.md): Tips for maximizing performance