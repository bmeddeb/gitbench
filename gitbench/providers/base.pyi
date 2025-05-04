"""
Type stubs for base provider interface.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Union

from ..models.common import (
    ProviderType, RepoInfo, UserInfo, RateLimitInfo, 
    RepoDetails, ContributorInfo, BranchInfo
)

class ProviderError(Exception):
    """Base exception for provider-related errors."""
    message: str
    provider_type: ProviderType
    
    def __init__(self, message: str, provider_type: ProviderType) -> None: ...

class RateLimitError(ProviderError):
    """Exception raised when rate limits are exceeded."""
    reset_time: int
    
    def __init__(self, message: str, provider_type: ProviderType, reset_time: int) -> None: ...

class AuthError(ProviderError):
    """Exception raised for authentication failures."""
    pass

class GitProviderClient(ABC):
    provider_type: ProviderType

    def __init__(self, provider_type: ProviderType) -> None: ...
    
    @abstractmethod
    async def fetch_repositories(self, owner: str) -> List[RepoInfo]: ...
    
    @abstractmethod
    async def fetch_user_info(self) -> UserInfo: ...
    
    @abstractmethod
    async def get_rate_limit(self) -> RateLimitInfo: ...
    
    @abstractmethod
    async def fetch_repository_details(
        self, owner: str, repo: str
    ) -> RepoDetails: ...
    
    @abstractmethod
    async def fetch_contributors(
        self, owner: str, repo: str
    ) -> List[ContributorInfo]: ...
    
    @abstractmethod
    async def fetch_branches(self, owner: str, repo: str) -> List[BranchInfo]: ...
    
    @abstractmethod
    async def validate_credentials(self) -> bool: ...
    
    async def to_pandas(self, data: Union[List[Any], Any]) -> "pandas.DataFrame": ...
