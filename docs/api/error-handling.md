# Error Handling

GitFleet provides a comprehensive error handling system that bridges the gap between Rust's Result-based error handling and Python's exception system. This document explains how errors are handled in GitFleet and how to properly handle them in your code.

## Error Propagation Model

GitFleet follows a consistent pattern for error propagation:

1. **Rust Core**: Uses the Rust `Result<T, E>` pattern for all operations that can fail
2. **Python Bridge**: Automatically converts Rust errors into appropriate Python exceptions
3. **Python API**: Provides error details and contextual information for debugging

## Common Error Types

GitFleet exposes several error types that you might encounter:

### `GitFleetError`

The base exception class for all GitFleet-specific errors.

```python
try:
    result = await repo_manager.clone_all()
except GitFleetError as e:
    print(f"GitFleet operation failed: {e}")
```

### `CloneError`

Raised when a repository clone operation fails.

```python
try:
    result = await repo_manager.clone_all()
except CloneError as e:
    print(f"Clone failed: {e.message}")
    print(f"Repository URL: {e.url}")
```

### `AuthenticationError`

Raised when authentication fails (e.g., invalid tokens or credentials).

```python
try:
    client = GitHubClient(token="invalid_token")
    is_valid = await client.validate_credentials()
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
```

### `RateLimitError`

Raised when API rate limits are exceeded.

```python
try:
    repos = await github_client.fetch_repositories("octocat")
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
    print(f"Reset time: {e.reset_time}")
```

### `NetworkError`

Raised when network operations fail.

```python
try:
    result = await repo_manager.clone_all()
except NetworkError as e:
    print(f"Network error: {e}")
    print(f"URL: {e.url}")
```

## Best Practices for Error Handling

### Use Specific Exception Types

Catch specific exceptions rather than using a broad catch-all:

```python
try:
    # Operation that might fail
    result = await repo_manager.clone_all()
except CloneError as e:
    # Handle clone errors
    print(f"Clone failed: {e}")
except AuthenticationError as e:
    # Handle authentication errors
    print(f"Authentication failed: {e}")
except RateLimitError as e:
    # Handle rate limit errors
    print(f"Rate limit exceeded, reset at: {e.reset_time}")
except GitFleetError as e:
    # Handle other GitFleet errors
    print(f"GitFleet operation failed: {e}")
```

### Async Error Handling

Remember that GitFleet is primarily async-based, so use the appropriate async error handling patterns:

```python
async def safe_operation():
    try:
        # Async operation that might fail
        result = await repo_manager.clone_all()
        return result
    except GitFleetError as e:
        # Log or handle the error
        print(f"Error in async operation: {e}")
        # Re-raise, handle, or return a fallback
        raise
```

### Error Recovery

GitFleet provides utility methods for recovery after errors:

```python
try:
    result = await repo_manager.clone_all()
except CloneError as e:
    # Attempt recovery
    print(f"Clone failed, retrying with different settings...")
    result = await repo_manager.retry_failed_clones(timeout=300)
```

## Working with the Rust/Python Boundary

GitFleet bridges the Rust and Python error handling systems. Here's what you need to know:

1. **Rust Errors**: Rust's `Result<T, E>` errors are automatically converted to Python exceptions
2. **Error Context**: Additional context is added during the conversion
3. **Custom Errors**: You can define custom error types that work on both sides of the boundary

When defining your own extensions or plugins, follow the established error handling patterns:

```python
from GitFleet import GitFleetError

class MyCustomError(GitFleetError):
    """Custom error type for my extension."""
    def __init__(self, message, context=None):
        super().__init__(message)
        self.context = context
```

## Performance Considerations

Error handling in GitFleet is designed to be efficient. However, keep these tips in mind:

1. **Avoid Deep Try Blocks**: Don't place large, long-running tasks in a single try block
2. **Use Error Callbacks**: For long-running operations, consider using callbacks instead of exceptions
3. **Granular Error Handling**: Handle errors at the appropriate level of abstraction

## Related Topics

- [Performance Tips](../advanced/performance.md) - Optimizing GitFleet performance, including error handling
- [Clone Monitoring](clone-monitoring.md) - Monitoring clone operations, including error states