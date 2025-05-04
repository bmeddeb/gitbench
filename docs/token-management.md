# Token Management

GitFleet includes a built-in token management system that helps you handle API rate limits and authentication across multiple tokens. This is especially useful for high-volume applications that need to maximize their API usage.

## Why Token Management?

Git hosting providers like GitHub, GitLab, and BitBucket implement rate limiting on their APIs to ensure fair usage. When you exceed these limits, your requests will be rejected until the rate limit resets.

Using multiple tokens allows you to:

1. Increase the total number of requests you can make
2. Continue operations when one token hits its rate limit
3. Distribute API load across multiple accounts
4. Maintain availability even when some tokens become invalid

## Built-in TokenManager

GitFleet provides a `TokenManager` class that automatically handles token rotation, rate limit tracking, and fallback logic:

```python
import asyncio
from GitFleet import GitHubClient
from GitFleet.providers import TokenManager, ProviderType

# Create a token manager
token_manager = TokenManager()

# Add tokens for GitHub
token_manager.add_token("your-token-1", ProviderType.GITHUB)
token_manager.add_token("your-token-2", ProviderType.GITHUB)
token_manager.add_token("your-token-3", ProviderType.GITHUB)

# Create a client with the token manager
github = GitHubClient(
    token="your-token-1",  # Default token
    token_manager=token_manager  # Token manager for rotation
)

# Now use the client as normal - it will automatically rotate tokens as needed
async def main():
    # This will automatically use the next available token if rate limits are hit
    repos = await github.fetch_repositories("octocat")
    user = await github.fetch_user_info()
    
    # Check how many tokens are still available
    available = token_manager.count_available_tokens(ProviderType.GITHUB)
    print(f"{available} GitHub tokens available")

asyncio.run(main())
```

## Token Status

Each token in the manager has a `TokenStatus` object that tracks its state:

- **Valid and Available**: The token is valid and has API calls remaining
- **Rate Limited**: The token has exceeded its rate limit and will become available again after the reset time
- **Invalid**: The token is invalid (e.g., revoked, expired)

The token manager automatically:
- Tracks rate limits for all tokens
- Skips rate-limited tokens until they reset
- Marks invalid tokens
- Uses the least recently used available token

## Using Multiple Providers

You can register tokens for different providers:

```python
from GitFleet.providers import TokenManager, ProviderType

# Create a token manager
token_manager = TokenManager()

# Add tokens for different providers
token_manager.add_token("github-token-1", ProviderType.GITHUB)
token_manager.add_token("github-token-2", ProviderType.GITHUB)
token_manager.add_token("gitlab-token-1", ProviderType.GITLAB)  # (Future)
token_manager.add_token("gitlab-token-2", ProviderType.GITLAB)  # (Future)

# Create provider clients with the same token manager
github = GitHubClient(token="github-token-1", token_manager=token_manager)
# gitlab = GitLabClient(token="gitlab-token-1", token_manager=token_manager)  # (Future)
```

## Manual Token Management

For simple cases, you can also manually handle tokens:

```python
import asyncio
from GitFleet import GitHubClient
from GitFleet.providers.base import RateLimitError

# Initialize clients with different tokens
github1 = GitHubClient(token="token1")
github2 = GitHubClient(token="token2")

async def fetch_with_fallback(owner):
    try:
        # Try with the first token
        repos = await github1.fetch_repositories(owner)
        return repos
    except RateLimitError:
        # Fallback to the second token
        repos = await github2.fetch_repositories(owner)
        return repos
```

## Rate Limit Awareness

With the built-in `TokenManager`, rate limits are tracked automatically. However, you can also manually check rate limits:

```python
# Check rate limits for a specific client
rate_limit = await github.get_rate_limit()
print(f"Remaining: {rate_limit.remaining}/{rate_limit.limit}")
print(f"Reset time: {rate_limit.reset_time}")

# Check all tokens in a token manager
from GitFleet.providers import TokenManager, ProviderType

token_manager = TokenManager()
token_manager.add_token("token1", ProviderType.GITHUB)
token_manager.add_token("token2", ProviderType.GITHUB)

# Get all GitHub tokens
github_tokens = token_manager.get_all_tokens(ProviderType.GITHUB)
for i, token_info in enumerate(github_tokens):
    if token_info.rate_limit:
        print(f"Token {i+1}: {token_info.rate_limit.remaining}/{token_info.rate_limit.limit}")
        print(f"Status: {token_info.status}")
    else:
        print(f"Token {i+1}: Rate limit not yet fetched")
```

## Token Classes

### TokenInfo

The `TokenInfo` class stores information about each token:

```python
@dataclass
class TokenInfo:
    token: str                  # The actual token string
    provider: ProviderType      # Which provider this token is for
    username: Optional[str]     # Optional username associated with token
    status: TokenStatus         # Current status information
    
    # Get a secure version of the token
    @property
    def secret_token(self) -> SecretStr:
        return SecretStr(self.token)
```

### TokenStatus

The `TokenStatus` class tracks the status of a token:

```python
@dataclass
class TokenStatus:
    is_valid: bool              # Whether the token is valid
    remaining_calls: int        # Number of API calls remaining
    reset_time: Optional[int]   # When rate limit resets (UNIX timestamp)
    last_used: Optional[float]  # When the token was last used
    
    # Useful properties
    @property
    def is_rate_limited(self) -> bool:
        """Check if the token is currently rate limited."""
        # Returns True if no calls remaining and reset time hasn't passed
        
    @property
    def is_available(self) -> bool:
        """Check if the token is available for use."""
        # Returns True if valid and not rate limited
```

Example of checking token status:

```python
# Check token status
if token_info.status.is_available:
    print("Token is available")
elif token_info.status.is_rate_limited:
    print(f"Token is rate limited until {time.ctime(token_info.status.reset_time)}")
elif not token_info.status.is_valid:
    print("Token is invalid or revoked")
print(f"Remaining calls: {token_info.status.remaining_calls}")
```

## Multiple Tokens Example

Here's a complete example using the token manager with multiple tokens from environment variables:

```python
import os
import asyncio
from GitFleet import GitHubClient
from GitFleet.providers import TokenManager, ProviderType

async def main():
    # Get tokens from environment (comma-separated)
    tokens = os.environ.get("GITHUB_TOKENS", "").split(",")
    if not tokens or not tokens[0]:
        print("No tokens found. Set GITHUB_TOKENS environment variable.")
        return
        
    # Create token manager and add tokens
    token_manager = TokenManager()
    for token in tokens:
        token_manager.add_token(token, ProviderType.GITHUB)
        
    # Create client with token manager
    github = GitHubClient(
        token=tokens[0],  # First token as default
        token_manager=token_manager
    )
    
    # Fetch repositories for multiple users
    users = ["octocat", "torvalds", "gvanrossum", "kennethreitz"]
    
    for user in users:
        try:
            repos = await github.fetch_repositories(user)
            print(f"Found {len(repos)} repositories for {user}")
            
            # The token manager automatically rotated tokens if needed
        except Exception as e:
            print(f"Error fetching repos for {user}: {e}")
            
    # Check token statuses after operations
    print("\nToken statuses:")
    for i, token_info in enumerate(token_manager.get_all_tokens(ProviderType.GITHUB)):
        if token_info.status:
            if token_info.status.is_rate_limited:
                status = f"Rate limited (resets at {time.ctime(token_info.status.reset_time)})"
            elif not token_info.status.is_valid:
                status = "Invalid"
            else:
                status = f"Available ({token_info.status.remaining_calls} calls remaining)"
        else:
            status = "Unknown"
            
        print(f"Token {i+1}: {status}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Best Practices

1. **Store tokens securely**: Never hard-code tokens in your source code. Use environment variables or secure secret management.

2. **Use the built-in TokenManager**: Let GitFleet handle token rotation and rate limiting automatically.

3. **Handle rate limit errors**: Always catch `RateLimitError` exceptions and implement appropriate fallback logic.

4. **Respect API limits**: Even with multiple tokens, be respectful of API limits and avoid making unnecessary requests.

5. **Implement exponential backoff**: When all tokens are rate-limited, implement exponential backoff before retrying.

```python
import asyncio
import time
import random
from GitFleet.providers.base import RateLimitError

async def fetch_with_backoff(client, owner, max_retries=5):
    for attempt in range(max_retries):
        try:
            return await client.fetch_repositories(owner)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise  # Give up after max_retries
            
            # Calculate wait time with exponential backoff and jitter
            wait_time = min(2 ** attempt + random.random(), 60)
            print(f"Rate limited. Waiting {wait_time:.2f} seconds before retry...")
            await asyncio.sleep(wait_time)
```

## Secure Token Storage

For production applications, consider using environment variables or a secure credential store:

```python
import os
from GitFleet import GitHubClient
from GitFleet.providers import TokenManager, ProviderType

# Get tokens from environment variables
token = os.environ.get("GITHUB_TOKEN")
if not token:
    raise ValueError("GITHUB_TOKEN environment variable not set")

# Create the client
github = GitHubClient(token=token)
```

For more secure storage in Python applications, consider using packages like:
- `python-dotenv` for loading from .env files
- `keyring` for system keychain integration
- `vault` for HashiCorp Vault integration