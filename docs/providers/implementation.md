# Dual Implementation Architecture

GitFleet supports both Rust-based and pure Python implementations for its Git provider APIs. This architecture provides performance benefits when the Rust components are available, with a fallback to a pure Python implementation for maximum compatibility.

## Architecture Overview

```
┌───────────────────────────────────────┐
│            Python Interface           │
│                                       │
│         GitProviderClient (ABC)       │
│                  │                    │
│                  ▼                    │
│          Specific Providers           │
│     (GitHubClient, GitLabClient)      │
└─────────────────┬─────────────────────┘
                  │
      ┌───────────┴───────────┐
      │                       │
┌─────▼─────┐         ┌───────▼───────┐
│  Rust     │         │    Python     │
│ Backend   │         │   Backend     │
└───────────┘         └───────────────┘
```

## Implementation Detection

When you create a provider client (like `GitHubClient`), GitFleet automatically detects if the Rust implementation is available:

```python
from GitFleet import GitHubClient

# The client will use Rust implementation if available,
# otherwise it will fall back to Python
client = GitHubClient(token="your-token")
```

## Forcing Python Implementation

You can force GitFleet to use the Python implementation even when Rust is available:

```python
# Force Python implementation
client = GitHubClient(
    token="your-token",
    use_python_impl=True
)
```

This can be useful for:
- Debugging
- Ensuring consistent behavior across environments
- When you need features only available in the Python implementation

## Checking Implementation Status

You can check which implementation is being used:

```python
from GitFleet import GitHubClient

client = GitHubClient(token="your-token")

# Check if Rust implementation is available
print(f"Rust available: {client._use_rust}")
```

## Implementation Differences

Both implementations provide the same interface and functionality, but there are some differences:

### Performance

The Rust implementation typically offers:
- Faster request processing
- More efficient memory usage
- Better handling of concurrent requests

### Features

The Python implementation may offer:
- More granular control over HTTP requests
- Easier debugging and customization
- Additional helper methods in some cases

### Error Handling

Both implementations use the same error types, but error messages may differ slightly.
The Python implementation provides more detailed error messages in some cases, while
the Rust implementation typically provides more precise error codes.

## Switching at Runtime

You cannot switch implementations on an existing client instance. If you need to switch,
create a new client:

```python
# Start with Rust implementation
rust_client = GitHubClient(token="your-token")

# Later, create a Python implementation if needed
python_client = GitHubClient(token="your-token", use_python_impl=True)
```

## Token Management

Both implementations support the TokenManager for handling multiple tokens and rate limits:

```python
from GitFleet import GitHubClient
from GitFleet.providers import TokenManager

# Create a token manager
token_manager = TokenManager()
token_manager.add_token("token1", "github")
token_manager.add_token("token2", "github")

# Use with Rust or Python implementation
client = GitHubClient(
    token="token1",
    token_manager=token_manager
)
```

## Implementation Details

### Rust Implementation

The Rust implementation:
- Uses the `reqwest` crate for HTTP requests
- Implements efficient serialization with `serde`
- Uses Tokio for async runtime
- Leverages Rust's memory safety and performance optimizations

### Python Implementation

The Python implementation:
- Uses `httpx` for async HTTP requests
- Provides detailed error messages and debugging
- Has no native dependencies besides Python's standard library
- Provides fully typed interfaces with dataclasses

## Future Compatibility

As GitFleet evolves:
- Both implementations will maintain API compatibility
- New features will be added to both implementations when possible
- The Rust implementation will continue to focus on performance
- The Python implementation will focus on flexibility and integration