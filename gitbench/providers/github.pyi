"""
Type stubs for GitHub API client.
"""

import time
import re
import httpx
from typing import Dict, List, Any, Optional, Union, TypeVar, Type, cast
from datetime import datetime

from .base import GitProviderClient, ProviderType, ProviderError, AuthError as BaseAuthError, RateLimitError as BaseRateLimitError
from .token_manager import TokenManager, TokenInfo
from ..models.common import (
    UserInfo,
    RepoInfo,
    RepoDetails,
    RateLimitInfo,
    BranchInfo,
    ContributorInfo,
)

T = TypeVar("T")

# Constants for Rust implementation
RUST_AVAILABLE: bool

class GitHubError(ProviderError):
    """Base exception for GitHub API errors."""
    
    def __init__(self, message: str) -> None: ...

class RateLimitError(BaseRateLimitError):
    """Exception raised when GitHub rate limits are exceeded."""
    pass

class AuthError(BaseAuthError):
    """Exception raised for GitHub authentication failures."""
    pass

class GitHubClient(GitProviderClient):
    token: str
    base_url: str
    token_manager: Optional[TokenManager]
    use_python_impl: bool
    _use_rust: bool
    _client: Any  # RustGitHubClient when available

    def __init__(
        self,
        token: str,
        base_url: Optional[str] = None,
        token_manager: Optional[TokenManager] = None,
        use_python_impl: bool = False,
    ) -> None: ...
    
    async def _request(self, method: str, endpoint: str, **kwargs: Any) -> Any: ...
    
    def _convert_to_model(self, data: Dict[str, Any], model_class: Type[T]) -> T: ...
    
    def _handle_error(self, error: Exception) -> None: ...
    
    async def fetch_repositories(self, owner: str) -> List[RepoInfo]: ...
    
    async def fetch_user_info(self) -> UserInfo: ...
    
    async def get_rate_limit(self) -> RateLimitInfo: ...
    
    async def fetch_repository_details(self, owner: str, repo: str) -> RepoDetails: ...
    
    async def fetch_contributors(
        self, owner: str, repo: str
    ) -> List[ContributorInfo]: ...
    
    async def fetch_branches(self, owner: str, repo: str) -> List[BranchInfo]: ...
    
    async def validate_credentials(self) -> bool: ...
