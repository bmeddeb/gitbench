"""
Type stubs for Git provider API clients.
"""

from gitbench.providers.base import GitProviderClient, ProviderType
from gitbench.providers.github import (
    GitHubClient,
    GitHubError,
    AuthError,
    RateLimitError,
)
from gitbench.providers.token_manager import TokenManager, TokenInfo, TokenStatus

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
]
