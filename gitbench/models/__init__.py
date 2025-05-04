"""
Data models for Git provider API responses and repository management.
"""

from gitbench.models.common import (BranchInfo, ContributorInfo, ProviderType,
                                   RateLimitInfo, RepoDetails, RepoInfo, UserInfo)

# Try to import the repository management models if the Rust extension is available
try:
    from gitbench.models.repo import (
        CloneStatus, CloneTask, CloneStatusType, RepoManager,
        clone_status_to_pydantic, clone_task_to_pydantic
    )
    __all__ = [
        # Provider models
        "ProviderType",
        "UserInfo", 
        "RepoInfo",
        "RepoDetails",
        "RateLimitInfo",
        "BranchInfo",
        "ContributorInfo",
        
        # Repository management models
        "CloneStatus",
        "CloneTask",
        "CloneStatusType",
        "RepoManager",
        "clone_status_to_pydantic",
        "clone_task_to_pydantic",
    ]
except ImportError:
    # If the Rust extension is not available, only export the provider models
    __all__ = [
        "ProviderType",
        "UserInfo",
        "RepoInfo",
        "RepoDetails",
        "RateLimitInfo",
        "BranchInfo",
        "ContributorInfo",
    ]
