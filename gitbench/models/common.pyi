"""
Type stubs for common data models.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, ClassVar, Type

from pydantic import BaseModel, ConfigDict, Field, AnyHttpUrl

class ProviderType(str, Enum):
    GITHUB: str = "github"
    GITLAB: str = "gitlab"
    BITBUCKET: str = "bitbucket"

class UserInfo(BaseModel):
    id: int  # Changed from str to int to match implementation
    login: str
    name: Optional[str] = None
    email: Optional[str] = None
    avatar_url: Optional[str] = None
    provider_type: ProviderType = ProviderType.GITHUB
    raw_data: Optional[Dict[str, Any]] = None
    
    model_config: ClassVar[ConfigDict]
    
    def model_dump(self) -> Dict[str, Any]: ...
    def model_dump_json(self, indent: Optional[int] = None) -> str: ...
    
    @classmethod
    def model_validate(cls, obj: Any, from_attributes: bool = False) -> "UserInfo": ...

class RepoInfo(BaseModel):
    id: int  # Added missing id field
    name: str
    full_name: str
    clone_url: str
    description: Optional[str] = None
    default_branch: str = "main"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    language: Optional[str] = None
    fork: bool = False
    forks_count: int = 0
    stargazers_count: Optional[int] = None
    watchers_count: Optional[int] = None  # Added missing field
    html_url: Optional[str] = None  # Added missing field
    private: bool = False  # Added missing field
    provider_type: ProviderType = ProviderType.GITHUB
    visibility: str = "public"
    owner: Optional[UserInfo] = None
    raw_data: Optional[Dict[str, Any]] = None
    
    model_config: ClassVar[ConfigDict]
    
    def model_dump(self) -> Dict[str, Any]: ...
    def model_dump_json(self, indent: Optional[int] = None) -> str: ...
    def created_datetime(self) -> Optional[datetime]: ...
    def updated_datetime(self) -> Optional[datetime]: ...
    
    @classmethod
    def model_validate(cls, obj: Any, from_attributes: bool = False) -> "RepoInfo": ...

class RateLimitInfo(BaseModel):
    limit: int
    remaining: int
    reset_time: int
    used: int
    provider_type: ProviderType = ProviderType.GITHUB
    
    model_config: ClassVar[ConfigDict]
    
    def model_dump(self) -> Dict[str, Any]: ...
    def model_dump_json(self, indent: Optional[int] = None) -> str: ...
    def seconds_until_reset(self) -> int: ...
    
    @classmethod
    def model_validate(cls, obj: Any, from_attributes: bool = False) -> "RateLimitInfo": ...

class CommitRef(BaseModel):
    """Reference to a commit in a repository."""
    sha: str
    url: Optional[str] = None
    
    model_config: ClassVar[ConfigDict]
    
    def model_dump(self) -> Dict[str, Any]: ...
    def model_dump_json(self, indent: Optional[int] = None) -> str: ...
    
    @classmethod
    def model_validate(cls, obj: Any, from_attributes: bool = False) -> "CommitRef": ...

class BranchInfo(BaseModel):
    name: str
    commit: Optional[CommitRef] = None  # Added missing commit field
    commit_sha: str
    protected: bool = False
    provider_type: ProviderType = ProviderType.GITHUB
    raw_data: Optional[Dict[str, Any]] = None
    
    model_config: ClassVar[ConfigDict]
    
    def model_dump(self) -> Dict[str, Any]: ...
    def model_dump_json(self, indent: Optional[int] = None) -> str: ...
    def model_post_init(self, __context: Any) -> None: ...  # Added missing method
    
    @classmethod
    def model_validate(cls, obj: Any, from_attributes: bool = False) -> "BranchInfo": ...

class ContributorInfo(BaseModel):
    id: int  # Changed from str to int to match implementation
    login: str
    contributions: int
    avatar_url: Optional[str] = None
    provider_type: ProviderType = ProviderType.GITHUB
    raw_data: Optional[Dict[str, Any]] = None
    
    model_config: ClassVar[ConfigDict]
    
    def model_dump(self) -> Dict[str, Any]: ...
    def model_dump_json(self, indent: Optional[int] = None) -> str: ...
    
    @classmethod
    def model_validate(cls, obj: Any, from_attributes: bool = False) -> "ContributorInfo": ...

class RepoDetails(RepoInfo):
    topics: List[str] = Field(default_factory=list)  # Updated to match implementation
    homepage: Optional[str] = None
    has_wiki: bool = False
    has_issues: bool = False
    has_projects: bool = False
    has_pages: Optional[bool] = None  # Added missing field
    has_downloads: Optional[bool] = None  # Added missing field
    allow_forking: Optional[bool] = None  # Added missing field
    archived: bool = False
    disabled: Optional[bool] = None  # Added missing field
    pushed_at: Optional[str] = None
    size: int = 0
    open_issues_count: Optional[int] = None
    network_count: Optional[int] = None  # Added missing field
    subscribers_count: Optional[int] = None  # Added missing field
    license: Optional[Dict[str, Any]] = None  # Changed type to match implementation
    
    def pushed_datetime(self) -> Optional[datetime]: ...
    
    @classmethod
    def model_validate(cls, obj: Any, from_attributes: bool = False) -> "RepoDetails": ...
