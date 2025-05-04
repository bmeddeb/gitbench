# Git Provider API Clients

gitbench includes API clients for various Git hosting providers that allow you to interact with repositories, users, and other provider-specific information.

## Available Providers

The following providers are currently implemented:

- [GitHub](github.md): Complete API client for GitHub with support for repositories, users, branches, and more.

Coming soon:
- GitLab: API client for GitLab (planned for v0.4.0)
- BitBucket: API client for BitBucket (planned for v0.5.0)

## Key Features

- **[Data Models](models.md)**: Type-safe dataclasses for all provider data
- **[Dual Implementation](implementation.md)**: Both Rust and Python implementations
- **[Token Management](../token-management.md)**: Built-in token rotation and rate limit handling 
- **Error Handling**: Comprehensive error hierarchy for all providers
- **Async Support**: Full async/await support for concurrent operations
- **Pandas Integration**: Easy conversion to pandas DataFrames

## Common Features

All provider clients share a common interface through the `GitProviderClient` base class, making it easy to work with different providers using the same code patterns.

Common functionality includes:

- Repository information retrieval
- User data access
- Branch and contributor details
- Rate limit handling

## Basic Usage

```python
import asyncio
from gitbench import GitHubClient

async def main():
    # Initialize the client with your token
    # Will automatically use Rust implementation if available
    github = GitHubClient(token="your-github-token")
    
    # Fetch repositories for a user
    repos = await github.fetch_repositories("octocat")
    
    # Get user information (returns UserInfo dataclass)
    user = await github.fetch_user_info()
    
    # Check rate limits (returns RateLimitInfo dataclass)
    rate_limit = await github.get_rate_limit()
    
    # Work with typed dataclass objects
    print(f"Found {len(repos)} repositories")
    print(f"Authenticated as: {user.login} ({user.name})")
    print(f"API calls remaining: {rate_limit.remaining}/{rate_limit.limit}")
    
    # Explore repository details
    for repo in repos[:3]:  # First 3 repos
        print(f"\nRepository: {repo.full_name}")
        print(f"  Description: {repo.description}")
        print(f"  Language: {repo.language}")
        print(f"  Stars: {repo.stargazers_count}")
        print(f"  Forks: {repo.forks_count}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Token Management

gitbench includes a built-in `TokenManager` for handling rate limits and authentication across multiple tokens:

```python
from gitbench import GitHubClient
from gitbench.providers import TokenManager, ProviderType

# Create a token manager
token_manager = TokenManager()
token_manager.add_token("token1", ProviderType.GITHUB)
token_manager.add_token("token2", ProviderType.GITHUB)

# Create a client with the token manager
github = GitHubClient(
    token="token1",
    token_manager=token_manager  # Auto-rotation of tokens
)

# The client will automatically use the next available token
# when rate limits are hit
repos = await github.fetch_repositories("octocat")
```

See the [Token Management](../token-management.md) guide for more details.

## Data Analysis with Pandas

All provider clients support converting their responses to pandas DataFrames for data analysis:

```python
# Option 1: Use the utility function
from gitbench import to_dataframe
repos = await github.fetch_repositories("octocat")
df = to_dataframe(repos)

# Option 2: Use the client method
repos = await github.fetch_repositories("octocat")
df = await github.to_pandas(repos)

# Analyze the data
popular_repos = df.sort_values("stargazers_count", ascending=False)
languages = df["language"].value_counts()

# Print the top 5 most popular repositories
print(popular_repos[["name", "stargazers_count"]].head(5))
```

## Force Python Implementation

You can force gitbench to use the pure Python implementation if needed:

```python
# Force Python implementation
github = GitHubClient(
    token="your-github-token",
    use_python_impl=True  # Force Python implementation
)
```

See the [Implementation](implementation.md) guide for more details.