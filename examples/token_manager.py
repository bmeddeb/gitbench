#!/usr/bin/env python3
"""
Token Manager Example

This example demonstrates how to use multiple tokens with the gitbench library
to handle rate limiting across different API providers.

Optional dependencies:
- cryptography: For secure token encryption (pip install cryptography)
  Install with: pip install "gitbench[crypto]"
"""

import os
import asyncio
import time
from pprint import pprint

from gitbench.providers import TokenManager, TokenStatus, ProviderType, GitHubClient


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
        if token_info.status:
            if token_info.status.is_rate_limited:
                reset_time = token_info.status.reset_time or 0
                status = f"Rate limited (resets at {time.ctime(reset_time)})"
            elif not token_info.status.is_valid:
                status = "Invalid"
            else:
                status = f"Available ({token_info.status.remaining_calls} calls remaining)"
        else:
            status = "Unknown"
        print(f"Token {i}: {status}")


if __name__ == "__main__":
    # Set up the event loop and run the main async function
    asyncio.run(main())