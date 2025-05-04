# Token Manager Example

This example demonstrates how to use GitFleet's `TokenManager` to handle multiple authentication tokens for GitHub API operations. The token manager automatically rotates tokens when rate limits are reached, ensuring your application can continue making API requests without interruption.

## Code Example

```python
#!/usr/bin/env python3
"""
Token Manager Example

This example demonstrates how to use multiple tokens with the GitFleet library
to handle rate limiting across different API providers.

Optional dependencies:
- cryptography: For secure token encryption (pip install cryptography)
  Install with: pip install "gitfleet[crypto]"
"""

import os
import asyncio
import time
from pprint import pprint

from GitFleet import GitHubClient
from GitFleet.providers import TokenManager, TokenStatus, ProviderType


async def main():
    # Get GitHub tokens from environment variables
    # Format: TOKEN1,TOKEN2,TOKEN3
    github_tokens = os.environ.get("GITHUB_TOKENS", "").split(",")
    
    if not github_tokens or not github_tokens[0]:
        # Fall back to single token
        github_tokens = [os.environ.get("GITHUB_TOKEN", "")]
        if not github_tokens[0]:
            print("⚠️ No GitHub tokens found in environment variables.")
            print("Set the GITHUB_TOKENS or GITHUB_TOKEN environment variable to use this example.")
            return
    
    print(f"Using {len(github_tokens)} GitHub token(s)")
    
    # Create a token manager and add tokens
    token_manager = TokenManager()
    for token in github_tokens:
        token_manager.add_token(token, ProviderType.GITHUB)
    
    # Create a single client with token manager for auto-rotation
    github = GitHubClient(
        token=github_tokens[0],  # Use first token as default
        token_manager=token_manager  # Token manager for rotation
    )
    
    # Check rate limits for all tokens
    print("\n📊 API Rate Limits for all tokens:")
    all_tokens = token_manager.get_all_tokens(ProviderType.GITHUB)
    
    # First fetch the rate limits to populate token information
    try:
        await github.get_rate_limit()
    except Exception as e:
        print(f"Error fetching initial rate limit: {e}")
    
    # Display token information
    for i, token_info in enumerate(all_tokens, 1):
        if hasattr(token_info, "rate_limit") and token_info.rate_limit:
            rl = token_info.rate_limit
            print(f"Token {i}: {rl.remaining}/{rl.limit} requests remaining")
            print(f"       Resets at: {time.ctime(rl.reset_time)}")
            print(f"       Status: {token_info.status}")
        else:
            print(f"Token {i}: Rate limit info not yet fetched")
    
    # Function to demonstrate using token manager with automatic rotation
    async def fetch_repositories_with_token_manager(owners):
        print(f"\n📚 Fetching repositories for {len(owners)} users:")
        
        results = {}
        
        for owner in owners:
            try:
                print(f"Fetching repos for {owner} (token manager will auto-select token)")
                repos = await github.fetch_repositories(owner)
                results[owner] = repos
                print(f"  ✅ Found {len(repos)} repositories for {owner}")
                
                # Check how many tokens are still available
                available = token_manager.count_available_tokens(ProviderType.GITHUB)
                print(f"  📊 {available}/{len(all_tokens)} tokens available")
                
            except Exception as e:
                print(f"  ❌ Error fetching repos for {owner}: {e}")
                print(f"  ℹ️ All tokens may be rate limited or invalid")
        
        return results
    
    # Test with multiple users
    sample_users = ["octocat", "torvalds", "gvanrossum", "kennethreitz", "yyx990803"]
    
    print("\n🔄 Using TokenManager for automatic token rotation:")
    repo_results = await fetch_repositories_with_token_manager(sample_users)
    
    # Print summary of all repos found
    print("\n📋 Summary:")
    total_repos = sum(len(repos) for repos in repo_results.values())
    print(f"Total repositories found: {total_repos}")
    for owner, repos in repo_results.items():
        print(f"  {owner}: {len(repos)} repositories")
    
    # Final token status
    print("\n📊 Final token status:")
    for i, token_info in enumerate(token_manager.get_all_tokens(ProviderType.GITHUB), 1):
        status = "Available"
        if hasattr(token_info, "status"):
            if token_info.status == TokenStatus.RATE_LIMITED and hasattr(token_info, "rate_limit"):
                reset_time = token_info.rate_limit.reset_time if token_info.rate_limit else 0
                status = f"Rate limited (resets at {time.ctime(reset_time)})"
            elif token_info.status == TokenStatus.INVALID:
                status = "Invalid"
        print(f"Token {i}: {status}")


if __name__ == "__main__":
    # Set up the event loop and run the main async function
    asyncio.run(main())
```

## Key Features Demonstrated

This example demonstrates several key features of GitFleet's token management capabilities:

1. **Multiple Token Management**: Managing and tracking multiple API tokens
2. **Automatic Token Rotation**: Switching to a different token when rate limits are reached
3. **Rate Limit Monitoring**: Checking and displaying rate limit information for each token
4. **Token Status Tracking**: Monitoring the status of each token (available, rate limited, invalid)
5. **Exception Handling**: Properly handling API rate limit and authentication exceptions

## Token Manager Architecture

GitFleet's `TokenManager` is designed to solve the common problem of API rate limiting when working with GitHub and other Git hosting providers. The architecture includes:

1. **Token Collection**: Store and manage multiple tokens per provider
2. **Rate Limit Tracking**: Track remaining API calls for each token
3. **Reset Time Monitoring**: Record when rate limits will reset
4. **Status Classification**: Categorize tokens as available, rate limited, or invalid
5. **Automatic Selection**: Choose the best available token for each request

## Token Status Types

The `TokenManager` tracks the following status types for each token:

- **AVAILABLE**: The token is valid and has remaining API calls
- **RATE_LIMITED**: The token has reached its API rate limit
- **INVALID**: The token is invalid or has been revoked

## Running the Example

To run this example:

1. Install GitFleet:
   ```bash
   pip install gitfleet
   ```

2. Set up multiple GitHub tokens as environment variables:
   ```bash
   # Option 1: Multiple tokens in a comma-separated list
   export GITHUB_TOKENS=token1,token2,token3
   
   # Option 2: Single token
   export GITHUB_TOKEN=your_github_token
   ```

3. Run the example:
   ```bash
   python examples/token_manager.py
   ```

## Integration with API Clients

The token manager integrates seamlessly with GitFleet's API clients:

```python
# Create a token manager
token_manager = TokenManager()
for token in github_tokens:
    token_manager.add_token(token, ProviderType.GITHUB)

# Create a client with token manager
github = GitHubClient(
    token=github_tokens[0],  # Initial token
    token_manager=token_manager  # For auto-rotation
)

# Now API calls will automatically use a different token when rate limits are reached
repos = await github.fetch_repositories("octocat")
```

## Best Practices for Token Management

### Token Security

Always store tokens securely:

```python
from GitFleet.utils.auth import CredentialManager

# Create a secure credential manager
credential_manager = CredentialManager.from_password(password="secure_password")

# Save tokens securely
for token in tokens:
    credential_manager.save_credential(
        provider=ProviderType.GITHUB,
        token=token,
        username="your_username"
    )
```

### Token Rotation Strategies

Different strategies can be implemented for token rotation:

1. **Round-Robin**: Cycle through tokens evenly
2. **Priority-Based**: Use certain tokens before others
3. **Rate-Aware**: Select tokens with the most remaining rate limit

GitFleet's `TokenManager` uses a rate-aware strategy by default.

### Handling Rate Limit Exhaustion

When all tokens are rate-limited:

```python
try:
    repos = await github.fetch_repositories("octocat")
except RateLimitError as e:
    # All tokens are rate limited
    earliest_reset = token_manager.get_earliest_reset_time(ProviderType.GITHUB)
    wait_seconds = earliest_reset - time.time()
    
    if wait_seconds > 0:
        print(f"All tokens rate limited. Waiting {wait_seconds} seconds...")
        await asyncio.sleep(wait_seconds)
        
        # Try again after waiting
        repos = await github.fetch_repositories("octocat")
```

## Related Examples

- [Secure Credentials](secure-credentials.md): Securely storing authentication tokens
- [GitHub Client](github-client.md): Working with the GitHub API client