"""
GitHub API Client

This module provides a client for interacting with the GitHub API.
Both pure Python and Rust-based implementations are available.
"""

import time
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

import httpx

# Import provider base and error classes
from .base import GitProviderClient, ProviderError, RateLimitError, AuthError

# Import models from common module
from ..models.common import (
    ProviderType, RepoInfo, UserInfo, RateLimitInfo, 
    RepoDetails, ContributorInfo, BranchInfo
)

# Import from token manager
from .token_manager import TokenInfo, TokenManager

# Try to import the Rust implementation
try:
    from gitbench import GitHubClient as RustGitHubClient
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

T = TypeVar("T")


class GitHubError(ProviderError):
    """Base exception for GitHub API errors."""
    
    def __init__(self, message: str):
        super().__init__(message, ProviderType.GITHUB)


class GitHubClient(GitProviderClient):
    """GitHub API client for interacting with GitHub repositories and users.
    
    This client provides methods for fetching repositories, user information,
    and other GitHub-specific data. It can use either a pure Python implementation
    or the Rust-based implementation for better performance.
    
    Args:
        token: GitHub personal access token
        base_url: Optional custom base URL for GitHub Enterprise
        token_manager: Optional token manager for rate limit handling
        use_python_impl: Force using the Python implementation even if Rust is available
    """
    
    def __init__(
        self,
        token: str,
        base_url: Optional[str] = None,
        token_manager: Optional[TokenManager] = None,
        use_python_impl: bool = False
    ):
        """Initialize the GitHub client.
        
        Args:
            token: GitHub personal access token
            base_url: Optional custom base URL for GitHub Enterprise
            token_manager: Optional token manager for rate limit handling
            use_python_impl: Force using the Python implementation even if Rust is available
        """
        super().__init__(ProviderType.GITHUB)
        self.token = token
        self.base_url = base_url or "https://api.github.com"
        self.use_python_impl = use_python_impl
        
        # Setup token management
        self.token_manager = token_manager
        if token_manager:
            token_manager.add_token(token, ProviderType.GITHUB)
            
        # Use Rust implementation if available and not forced to use Python
        if RUST_AVAILABLE and not use_python_impl:
            self._client = RustGitHubClient(token, base_url)
            self._use_rust = True
        else:
            self._use_rust = False
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        """Make an authenticated request to the GitHub API using the Python implementation."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        # Get a token from the manager if available, otherwise use the default
        token_to_use = self.token
        token_info = None
        if self.token_manager:
            token_info = await self.token_manager.get_next_available_token(
                ProviderType.GITHUB
            )
            if token_info:
                token_to_use = token_info.token

        headers = {
            "Authorization": f"token {token_to_use}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "gitbench-Client",
        }

        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method, url=url, headers=headers, **kwargs
            )

            # Update rate limit info if token manager is available
            if self.token_manager and token_info:
                remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
                reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                await self.token_manager.update_rate_limit(
                    token_to_use, ProviderType.GITHUB, remaining, reset_time
                )

            # Handle rate limiting
            if (
                response.status_code == 403
                and "X-RateLimit-Remaining" in response.headers
            ):
                if int(response.headers["X-RateLimit-Remaining"]) == 0:
                    reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                    raise RateLimitError(
                        f"Rate limit exceeded. Resets at {reset_time}", 
                        ProviderType.GITHUB,
                        reset_time
                    )

            # Handle authentication errors
            if response.status_code == 401:
                if self.token_manager and token_info:
                    await self.token_manager.mark_token_invalid(
                        token_to_use, ProviderType.GITHUB
                    )
                raise AuthError("Invalid GitHub token", ProviderType.GITHUB)

            # Raise for other error status codes
            response.raise_for_status()

            return response.json()
    
    def _convert_to_model(self, data: Dict[str, Any], model_class: Type[T]) -> T:
        """Convert API response data to a model instance for the Python implementation."""
        # Special processing for nested structures before validation
        # No need to convert owner ID to string as model now expects integers
        
        # Handle special case for RateLimitInfo where reset is named differently
        if model_class == RateLimitInfo and "reset" in data:
            data["reset_time"] = data.pop("reset")
            
        # Handle special case for BranchInfo where commit data is nested
        if model_class == BranchInfo and "commit" in data and isinstance(data["commit"], dict):
            # For backward compatibility, extract SHA
            data["commit_sha"] = data["commit"]["sha"]
            
            # For full commit data, create a CommitRef object
            from ..models.common import CommitRef
            data["commit"] = CommitRef.model_validate(data["commit"])
            
        # Add provider type to data
        data["provider_type"] = ProviderType.GITHUB
            
        # Use Pydantic's model_validate for validation and conversion
        try:
            return cast(T, model_class.model_validate(data))
        except Exception as e:
            raise ValueError(f"Error validating {model_class.__name__}: {str(e)}")
    
    def _handle_error(self, error: Exception) -> None:
        """Handle errors from the Rust client.
        
        Args:
            error: Exception from the Rust client
            
        Raises:
            Appropriate Python exception based on the error type
        """
        error_message = str(error)
        
        if "Authentication error" in error_message:
            raise AuthError(error_message, self.provider_type)
        elif "Rate limit exceeded" in error_message:
            # Extract reset time from error message if available
            reset_match = re.search(r"resets at timestamp: (\d+)", error_message)
            reset_time = int(reset_match.group(1)) if reset_match else 0
            raise RateLimitError(error_message, self.provider_type, reset_time)
        elif "Resource not found" in error_message:
            raise ProviderError(error_message, self.provider_type)
        else:
            raise ProviderError(error_message, self.provider_type)
    
    async def fetch_repositories(self, owner: str) -> List[RepoInfo]:
        """Fetch repositories for an owner or organization."""
        if self._use_rust:
            try:
                repos = await self._client.fetch_repositories(owner)
                return repos
            except Exception as e:
                self._handle_error(e)
        else:
            data = await self._request("GET", f"/users/{owner}/repos?per_page=100")
            return [self._convert_to_model(repo, RepoInfo) for repo in data]

    async def fetch_user_info(self) -> UserInfo:
        """Fetch information about the authenticated user."""
        if self._use_rust:
            try:
                user_info = await self._client.fetch_user_info()
                return user_info
            except Exception as e:
                self._handle_error(e)
        else:
            data = await self._request("GET", "/user")
            return self._convert_to_model(data, UserInfo)

    async def get_rate_limit(self) -> RateLimitInfo:
        """Get current rate limit information."""
        if self._use_rust:
            try:
                rate_limit = await self._client.get_rate_limit()
                return rate_limit
            except Exception as e:
                self._handle_error(e)
        else:
            response = await self._request("GET", "/rate_limit")
            return self._convert_to_model(response["resources"]["core"], RateLimitInfo)

    async def fetch_repository_details(self, owner: str, repo: str) -> RepoDetails:
        """Fetch detailed information about a specific repository."""
        if self._use_rust:
            try:
                repo_details = await self._client.fetch_repository_details(owner, repo)
                return repo_details
            except Exception as e:
                self._handle_error(e)
        else:
            data = await self._request("GET", f"/repos/{owner}/{repo}")
            # Convert to RepoDetails instead of RepoInfo to include all detailed fields
            return RepoDetails.model_validate(data)

    async def fetch_contributors(self, owner: str, repo: str) -> List[ContributorInfo]:
        """Fetch contributors for a repository."""
        if self._use_rust:
            try:
                contributors = await self._client.fetch_contributors(owner, repo)
                return contributors
            except Exception as e:
                self._handle_error(e)
        else:
            data = await self._request("GET", f"/repos/{owner}/{repo}/contributors")
            return [self._convert_to_model(contributor, ContributorInfo) for contributor in data]

    async def fetch_branches(self, owner: str, repo: str) -> List[BranchInfo]:
        """Fetch branches for a repository."""
        if self._use_rust:
            try:
                branches = await self._client.fetch_branches(owner, repo)
                return branches
            except Exception as e:
                self._handle_error(e)
        else:
            data = await self._request("GET", f"/repos/{owner}/{repo}/branches")
            return [self._convert_to_model(branch, BranchInfo) for branch in data]

    async def validate_credentials(self) -> bool:
        """Check if the current token is valid."""
        if self._use_rust:
            try:
                is_valid = await self._client.validate_credentials()
                return is_valid
            except Exception as e:
                self._handle_error(e)
        else:
            try:
                await self.fetch_user_info()
                return True
            except AuthError:
                return False
            except Exception:
                raise  # Re-raise other exceptions
