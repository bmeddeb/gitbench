"""
Type stubs for gitbench package.
"""

from typing import Any, Awaitable, Dict, List, Optional, Union, TypeVar, Type

# Import from core Rust bindings
class RustCloneStatus:
    status_type: str
    progress: Optional[int]
    error: Optional[str]

class RustCloneTask:
    url: str
    status: RustCloneStatus
    temp_dir: Optional[str]

class RepoManager:
    def __init__(
        self, urls: List[str], github_username: str, github_token: str
    ) -> None: ...
    def clone_all(self) -> Awaitable[None]: ...
    def fetch_clone_tasks(self) -> Awaitable[Dict[str, RustCloneTask]]: ...
    def clone(self, url: str) -> Awaitable[None]: ...
    def bulk_blame(
        self, repo_path: str, file_paths: List[str]
    ) -> Awaitable[Dict[str, Union[List[Dict[str, Any]], str]]]: 
        """Execute blame on multiple files at once.
        
        Returns a dictionary where:
        - Keys are file paths
        - Values are either:
          - Lists of line blame information with fields:
            - commit_id: str
            - author_name: str
            - author_email: str
            - orig_line_no: int
            - final_line_no: int
            - line_content: str
          - Error message strings
        
        Note: Blame line information does NOT include timestamps.
        """
        ...
    def extract_commits(
        self, repo_path: str
    ) -> Awaitable[Union[List[Dict[str, Any]], str]]: 
        """Extract commit history from a repository.
        
        Returns either:
        - A list of commit dictionaries with fields:
          - sha: str
          - repo_name: str
          - message: str
          - author_name: str
          - author_email: str
          - author_timestamp: int (Unix epoch in seconds)
          - author_offset: int
          - committer_name: str
          - committer_email: str
          - committer_timestamp: int (Unix epoch in seconds)
          - committer_offset: int
          - additions: int
          - deletions: int
          - is_merge: bool
        - Error message string
        
        Note: Timestamps are in Unix epoch format (seconds since 1970-01-01) 
        and need to be converted to datetime objects for human-readable format.
        """
        ...
    def cleanup(self) -> Dict[str, Union[bool, str]]: ...

# Pydantic models
class CloneStatusType:
    QUEUED: str
    CLONING: str
    COMPLETED: str
    FAILED: str

class PydanticCloneStatus:
    status_type: str
    progress: Optional[int]
    error: Optional[str]
    
    def model_dump(self) -> Dict[str, Any]: ...
    def model_dump_json(self, **kwargs: Any) -> str: ...

class PydanticCloneTask:
    url: str
    status: PydanticCloneStatus
    temp_dir: Optional[str]
    
    def model_dump(self) -> Dict[str, Any]: ...
    def model_dump_json(self, **kwargs: Any) -> str: ...

# Conversion functions
def to_pydantic_status(rust_status: RustCloneStatus) -> PydanticCloneStatus: ...
def to_pydantic_task(rust_task: RustCloneTask) -> PydanticCloneTask: ...
def convert_clone_tasks(rust_tasks: Dict[str, RustCloneTask]) -> Dict[str, PydanticCloneTask]: ...

# Import provider clients
from gitbench.providers import (
    GitHubClient,
    GitProviderClient,
    ProviderType,
    TokenManager,
    TokenInfo,
    TokenStatus,
    GitHubError,
    AuthError,
    RateLimitError,
)

# Import utility functions
from gitbench.utils import (
    CredentialManager,
    CredentialEntry,
    RateLimiter,
    rate_limited,
    to_dataframe,
    to_dict,
    to_json,
    flatten_dataframe,
)

# Import models
from gitbench.models import (
    UserInfo,
    RepoInfo,
    RepoDetails,
    RateLimitInfo,
    BranchInfo,
    ContributorInfo,
    CommitRef,
)

# Availability flag
RUST_AVAILABLE: bool

# Export all names
__all__: List[str]
__version__: str
