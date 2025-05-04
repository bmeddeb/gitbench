"""
gitbench Provider API Clients

This package contains API clients for various Git hosting providers:
- GitHub
- GitLab
- BitBucket

These clients allow you to interact with repository information, 
user data, and other provider-specific features.
"""

from .base import GitProviderClient, ProviderType
from .github import (AuthError, GitHubClient, GitHubError, RateLimitError)
# These imports will be uncommented as we implement them
# from .gitlab import GitLabClient
# from .bitbucket import BitBucketClient
from .token_manager import (TokenInfo, TokenManager, TokenStatus)

__all__ = [
    # Base classes
    "GitProviderClient",
    "ProviderType",
    # Token management
    "TokenManager",
    "TokenInfo",
    "TokenStatus",
    # GitHub client
    "GitHubClient",
    "GitHubError",
    "AuthError",
    "RateLimitError",
    # These will be implemented later
    # "GitLabClient",
    # "BitBucketClient",
]
