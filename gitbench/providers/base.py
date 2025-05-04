"""
Base Git Provider Client

This module provides the base class for all Git provider API clients.
"""

from abc import ABC, abstractmethod
import asyncio
from typing import Dict, List, Optional, Any, Union

# Import models from the models package
from ..models.common import (
    ProviderType, RepoInfo, UserInfo, RateLimitInfo, 
    RepoDetails, ContributorInfo, BranchInfo
)


class ProviderError(Exception):
    """Base exception for provider-related errors."""
    
    def __init__(self, message: str, provider_type: ProviderType):
        self.message = message
        self.provider_type = provider_type
        super().__init__(f"{provider_type.value}: {message}")


class RateLimitError(ProviderError):
    """Exception raised when rate limits are exceeded."""
    
    def __init__(self, message: str, provider_type: ProviderType, reset_time: int):
        self.reset_time = reset_time
        super().__init__(
            f"{message} (resets at {reset_time})", provider_type
        )


class AuthError(ProviderError):
    """Exception raised for authentication failures."""
    pass


class GitProviderClient(ABC):
    """Base class for Git provider API clients.
    
    This abstract class defines the common interface that all provider-specific
    clients should implement.
    """
    
    def __init__(self, provider_type: ProviderType):
        """Initialize the base provider client.
        
        Args:
            provider_type: The type of Git provider this client handles
        """
        self.provider_type = provider_type
    
    @abstractmethod
    async def fetch_repositories(self, owner: str) -> List[RepoInfo]:
        """Fetch repositories for an owner/organization.
        
        Args:
            owner: Username or organization name
            
        Returns:
            List of repository information objects
            
        Raises:
            ProviderError: If the API request fails
            AuthError: If authentication fails
            RateLimitError: If rate limits are exceeded
        """
        pass
    
    @abstractmethod
    async def fetch_user_info(self) -> UserInfo:
        """Fetch information about the authenticated user.
        
        Returns:
            User information for the authenticated user
            
        Raises:
            ProviderError: If the API request fails
            AuthError: If authentication fails
            RateLimitError: If rate limits are exceeded
        """
        pass
    
    @abstractmethod
    async def get_rate_limit(self) -> RateLimitInfo:
        """Get current rate limit information.
        
        Returns:
            Current rate limit status
            
        Raises:
            ProviderError: If the API request fails
            AuthError: If authentication fails
        """
        pass
    
    @abstractmethod
    async def fetch_repository_details(self, owner: str, repo: str) -> RepoDetails:
        """Fetch detailed information about a specific repository.
        
        Args:
            owner: Username or organization name
            repo: Repository name
            
        Returns:
            Detailed repository information
            
        Raises:
            ProviderError: If the API request fails
            AuthError: If authentication fails
            RateLimitError: If rate limits are exceeded
        """
        pass
    
    @abstractmethod
    async def fetch_contributors(self, owner: str, repo: str) -> List[ContributorInfo]:
        """Fetch contributors for a repository.
        
        Args:
            owner: Username or organization name
            repo: Repository name
            
        Returns:
            List of contributor information
            
        Raises:
            ProviderError: If the API request fails
            AuthError: If authentication fails
            RateLimitError: If rate limits are exceeded
        """
        pass
    
    @abstractmethod
    async def fetch_branches(self, owner: str, repo: str) -> List[BranchInfo]:
        """Fetch branches for a repository.
        
        Args:
            owner: Username or organization name
            repo: Repository name
            
        Returns:
            List of branch information
            
        Raises:
            ProviderError: If the API request fails
            AuthError: If authentication fails
            RateLimitError: If rate limits are exceeded
        """
        pass
    
    @abstractmethod
    async def validate_credentials(self) -> bool:
        """Check if the current token/credentials are valid.
        
        Returns:
            True if credentials are valid, False otherwise
            
        Raises:
            ProviderError: If the validation check fails for reasons other than auth
        """
        pass
    
    async def to_pandas(self, data: Union[List[Any], Any]) -> "pandas.DataFrame":
        """Convert provider data to pandas DataFrame.
        
        This is a helper method for data analysis that converts API response data
        to a pandas DataFrame. If pandas is not installed, raises an ImportError.
        
        Args:
            data: API response data (list of objects or single object)
            
        Returns:
            pandas DataFrame with the data
            
        Raises:
            ImportError: If pandas is not installed
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for this functionality. "
                "Install it with 'pip install pandas'."
            )
        
        # Use our utility function from converters module to handle Pydantic models properly
        from ..utils.converters import to_dataframe
        return to_dataframe(data)
