# Performance Tips

GitFleet is designed with performance in mind, leveraging Rust's speed and memory safety for core Git operations while providing a convenient Python interface. This guide provides tips and best practices for optimizing GitFleet's performance in your applications.

## Performance Architecture

GitFleet's performance architecture is built on several key components:

1. **Rust Implementation for Critical Operations**:
   - Repository cloning
   - Git blame analysis
   - Commit history extraction
   - Core repository management

2. **Asynchronous Processing**:
   - Non-blocking I/O via asyncio and Tokio
   - Parallel operations where appropriate

3. **Efficient Memory Management**:
   - Streaming results for large repositories
   - Temporary directory cleanup

## General Performance Tips

### Use Asynchronous APIs

Always prefer asynchronous APIs when available. GitFleet's async functions leverage Rust's Tokio runtime for optimal performance:

```python
# Good: Using async/await
async def process_repos():
    repos = await github_client.fetch_repositories("octocat")
    return repos

# Bad: Blocking the main thread
def process_repos_blocking():
    # Hypothetical synchronous version would block
    repos = github_client.fetch_repositories_sync("octocat")
    return repos
```

### Batch Operations

Group related operations into batches rather than making multiple individual calls:

```python
# Good: Batch operation
results = await repo_manager.bulk_blame(repo_path, ["file1.py", "file2.py", "file3.py"])

# Less efficient: Individual operations
result1 = await repo_manager.blame(repo_path, "file1.py")
result2 = await repo_manager.blame(repo_path, "file2.py")
result3 = await repo_manager.blame(repo_path, "file3.py")
```

### Process Data Incrementally

When working with large repositories, process data incrementally rather than loading everything into memory:

```python
# Process commits in batches
async for commit_batch in repo_manager.yield_commits(repo_path, batch_size=100):
    # Process each batch without loading everything into memory
    process_batch(commit_batch)
```

## Repository Cloning Performance

### Optimize Clone Options

When cloning repositories, consider using these options for better performance:

```python
# Clone with minimal history (faster)
repo_manager = RepoManager(
    urls=["https://github.com/example/repo.git"],
    clone_options={"depth": 1}  # Only fetch the latest commit
)
```

### Efficient Clone Monitoring

When monitoring clone progress, be mindful of polling frequency:

```python
# Good: Reasonable refresh rate
while not clone_future.done():
    clone_tasks = await repo_manager.fetch_clone_tasks()
    # Process tasks...
    await asyncio.sleep(1)  # 1 second interval

# Bad: Excessive polling
while not clone_future.done():
    clone_tasks = await repo_manager.fetch_clone_tasks()
    # Process tasks...
    await asyncio.sleep(0.01)  # 10ms interval is too frequent
```

### Clean Up Temporary Directories

Always clean up temporary directories when you're done with them:

```python
# Clean up after operations complete
cleanup_results = repo_manager.cleanup()
```

## API Interaction Performance

### Rate Limit Management

Efficiently manage rate limits with TokenManager:

```python
# Create a token manager with multiple tokens
token_manager = TokenManager()
for token in github_tokens:
    token_manager.add_token(token, ProviderType.GITHUB)

# Use token manager for auto-rotation
github = GitHubClient(
    token=github_tokens[0],  # Default token
    token_manager=token_manager  # Will rotate tokens as needed
)
```

### Local Caching

Implement caching for frequently accessed data:

```python
# Simple in-memory cache
repo_cache = {}

async def get_repo_info(repo_name):
    if repo_name in repo_cache:
        return repo_cache[repo_name]
    
    # Fetch from API
    repo_info = await github_client.fetch_repository_details("owner", repo_name)
    repo_cache[repo_name] = repo_info
    return repo_info
```

## Memory Optimization

### Stream Large Repository Data

For large repositories, use streaming operations where available:

```python
# Process large files incrementally
async def process_large_file(repo_path, filename):
    results = []
    async for blame_batch in repo_manager.stream_blame(repo_path, filename, batch_size=1000):
        # Process each batch
        results.extend(analyze_blame(blame_batch))
    return results
```

## Parallel Processing

### Concurrent API Operations

Use concurrent operations for independent API tasks:

```python
import asyncio

async def analyze_multiple_repos():
    # Start multiple operations concurrently
    tasks = [
        github_client.fetch_repositories("user1"),
        github_client.fetch_repositories("user2"),
        github_client.fetch_repositories("user3"),
    ]
    
    # Wait for all to complete
    results = await asyncio.gather(*tasks)
    return results
```

### Parallel Git Operations

For operations on multiple repositories, use concurrent processing:

```python
async def analyze_multiple_repos(repo_paths):
    # Extract commits from multiple repositories concurrently
    tasks = [
        repo_manager.extract_commits(repo_path)
        for repo_path in repo_paths
    ]
    
    # Wait for all to complete
    results = await asyncio.gather(*tasks)
    return results
```

## Git Operation Performance

### Blame Analysis Optimization

When using blame operations, consider these optimizations:

```python
# Limit to specific files of interest rather than entire repo
file_patterns = ["*.py", "*.rs", "*.js"]
relevant_files = []

# Find relevant files (better than recursive search)
for pattern in file_patterns:
    pattern_files = await repo_manager.find_files(repo_path, pattern)
    relevant_files.extend(pattern_files)

# Then perform blame on the filtered set
blame_results = await repo_manager.bulk_blame(repo_path, relevant_files)
```

### Commit Extraction Optimization

For commit extraction, filter to reduce processing time:

```python
# Extract only recent commits (faster)
recent_commits = await repo_manager.extract_commits(
    repo_path, 
    max_count=100,  # Only most recent 100 commits
    since="2023-01-01"  # Only commits after this date
)
```

## Monitoring and Profiling

### Memory Usage Monitoring

Monitor memory usage for large operations:

```python
import psutil
import os

def log_memory_usage(label):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # Convert to MB
    print(f"Memory usage ({label}): {mem:.2f} MB")

async def memory_intensive_operation():
    log_memory_usage("Before operation")
    result = await repo_manager.extract_commits(repo_path)
    log_memory_usage("After operation")
    return result
```

### Performance Timing

Time critical operations to identify bottlenecks:

```python
import time

async def measure_execution_time(coroutine, *args, **kwargs):
    start_time = time.time()
    result = await coroutine(*args, **kwargs)
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"{coroutine.__name__} took {execution_time:.2f} seconds")
    
    return result

# Example usage
results = await measure_execution_time(
    repo_manager.bulk_blame, repo_path, file_paths
)
```

## Implementation-Specific Optimizations

### Rust vs Python Implementation

GitFleet implements performance-critical operations in Rust, while providing Python implementations for higher-level functionality:

**Rust-powered operations** (leverage these for best performance):
- Repository cloning
- Git blame analysis
- Commit history extraction

**Python-implemented features**:
- API clients
- Token management
- Credential handling
- Data conversion utilities

### When to Process Locally vs. Via API

For best performance, consider whether to use local Git operations (faster, implemented in Rust) or API operations (more convenient, implemented in Python):

```python
# Fast local operation using Rust implementation
blame_info = await repo_manager.bulk_blame(local_repo_path, ["file.py"])

# Convenient but potentially slower API operation
file_content = await github_client.fetch_file_content("owner", "repo", "file.py")
```

## Related Topics

- [Architecture Overview](../development/architecture.md) - Understanding GitFleet's performance architecture
- [Python-Rust Bridge](../development/python-rust-bridge.md) - How the Rust and Python components interact