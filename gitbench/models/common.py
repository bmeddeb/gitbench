"""
Common data models shared across different Git providers.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, cast

from pydantic import BaseModel, ConfigDict, Field, AnyHttpUrl


class ProviderType(str, Enum):
    """Enumeration of supported Git provider types."""

    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"


class UserInfo(BaseModel):
    """User information from a Git provider."""

    id: int  # GitHub API returns integers for IDs
    login: str
    name: Optional[str] = None
    email: Optional[str] = None
    avatar_url: Optional[str] = None
    provider_type: ProviderType = ProviderType.GITHUB
    raw_data: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(
        frozen=True,  # Make instances immutable
        extra='ignore',  # Ignore extra fields from API responses
    )


class RepoInfo(BaseModel):
    """Repository information from a Git provider.
    
    This model represents the common fields across different Git providers,
    with GitHub being the primary reference implementation.
    """

    # Basic repository information
    id: int  # GitHub API returns integer IDs
    name: str
    full_name: str  # Format: "owner/repo"
    clone_url: str  # Git URL for cloning
    
    # Repository metadata
    description: Optional[str] = None
    default_branch: str = "main"
    created_at: Optional[str] = None  # ISO8601 format
    updated_at: Optional[str] = None  # ISO8601 format
    language: Optional[str] = None
    
    # Repository stats
    fork: bool = False
    forks_count: int = 0
    stargazers_count: Optional[int] = None
    watchers_count: Optional[int] = None
    
    # GitHub URLs - optional for other providers
    html_url: Optional[str] = None  # Web URL for the repo
    
    # Repository status
    private: bool = False  # Whether the repo is private
    visibility: str = "public"  # "public", "private", or "internal"
    
    # Ownership
    owner: Optional[UserInfo] = None
    
    # Provider information
    provider_type: ProviderType = ProviderType.GITHUB
    raw_data: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(
        frozen=True,
        extra='ignore',
    )
    
    def created_datetime(self) -> Optional[datetime]:
        """Convert created_at string to datetime object.
        
        Returns:
            A timezone-aware datetime object (with UTC timezone),
            or None if conversion fails.
            
        Note:
            When performing datetime arithmetic with the result,
            be aware that this returns a timezone-aware datetime. 
            To subtract from a naive datetime like datetime.now(),
            you should either:
            1. Remove the timezone: dt.replace(tzinfo=None)
            2. Make both aware: use datetime.now(tz=timezone.utc)
        """
        if not self.created_at:
            return None
        try:
            return datetime.fromisoformat(self.created_at.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return None
    
    def updated_datetime(self) -> Optional[datetime]:
        """Convert updated_at string to datetime object.
        
        Returns:
            A timezone-aware datetime object (with UTC timezone),
            or None if conversion fails.
            
        Note:
            When performing datetime arithmetic with the result,
            be aware that this returns a timezone-aware datetime. 
            To subtract from a naive datetime like datetime.now(),
            you should either:
            1. Remove the timezone: dt.replace(tzinfo=None)
            2. Make both aware: use datetime.now(tz=timezone.utc)
        """
        if not self.updated_at:
            return None
        try:
            return datetime.fromisoformat(self.updated_at.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return None


class RateLimitInfo(BaseModel):
    """Rate limit information from a Git provider.
    
    Note: GitHub API returns this data with a 'reset' field, but
    we convert it to 'reset_time' in the GitHub client's _convert_to_model
    method to maintain a consistent naming convention across providers.
    """

    limit: int
    remaining: int
    reset_time: int  # UNIX timestamp when the rate limit resets
    used: int
    provider_type: ProviderType = ProviderType.GITHUB
    
    model_config = ConfigDict(frozen=True)
    
    def seconds_until_reset(self) -> int:
        """Get seconds until rate limit resets."""
        now = int(datetime.now().timestamp())
        return max(0, self.reset_time - now)


class RepoDetails(RepoInfo):
    """Detailed repository information structure with additional GitHub fields."""
    
    # Repository features and settings
    topics: List[str] = Field(default_factory=list)
    homepage: Optional[str] = None
    has_wiki: bool = False
    has_issues: bool = False
    has_projects: bool = False
    has_pages: Optional[bool] = None
    has_downloads: Optional[bool] = None
    allow_forking: Optional[bool] = None
    
    # Repository status
    archived: bool = False
    disabled: Optional[bool] = None
    
    # Repository metrics
    pushed_at: Optional[str] = None  # ISO8601 format
    size: int = 0  # Size in KB
    open_issues_count: Optional[int] = None
    network_count: Optional[int] = None
    subscribers_count: Optional[int] = None
    
    # License information
    license: Optional[Dict[str, Any]] = None  # Full license info from GitHub
    
    def pushed_datetime(self) -> Optional[datetime]:
        """Convert pushed_at string to datetime object."""
        if not self.pushed_at:
            return None
        try:
            return datetime.fromisoformat(self.pushed_at.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return None


class CommitRef(BaseModel):
    """Reference to a commit in a repository."""
    
    sha: str
    url: Optional[str] = None
    
    model_config = ConfigDict(frozen=True, extra='ignore')


class BranchInfo(BaseModel):
    """Branch information from a Git provider.
    
    Note: GitHub API returns commit data as a nested object with 'sha' and 'url' fields.
    We automatically extract the SHA in the GitHub client's _convert_to_model method
    to populate the 'commit_sha' field for backward compatibility.
    """

    name: str
    commit: Optional[CommitRef] = None  # Full commit object from GitHub API
    commit_sha: str  # Direct SHA for backward compatibility
    protected: bool = False
    provider_type: ProviderType = ProviderType.GITHUB
    raw_data: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(frozen=True, extra='ignore')
    
    def model_post_init(self, __context: Any) -> None:
        """Ensure commit_sha is populated from commit object if needed."""
        if self.commit and not self.commit_sha:
            object.__setattr__(self, 'commit_sha', self.commit.sha)


class ContributorInfo(BaseModel):
    """Contributor information from a Git provider."""

    id: int  # GitHub API returns integers for IDs
    login: str
    contributions: int
    avatar_url: Optional[str] = None
    provider_type: ProviderType = ProviderType.GITHUB
    raw_data: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(frozen=True, extra='ignore')
