"""
Tests for Repository Models (CloneStatus, CloneTask, RepoManager)

This module contains tests for the Pydantic models associated with 
repository management including CloneStatus and CloneTask,
and the RepoManager wrapper.
"""

import pytest
import asyncio
import os
from typing import Dict, List, Any, Optional, Union

from gitbench.models.repo import (
    CloneStatusType, 
    PydanticCloneStatus, 
    PydanticCloneTask,
    to_pydantic_status,
    to_pydantic_task,
    convert_clone_tasks,
)
from gitbench import RepoManager

# Import test-specific implementations
from gitbench.testing.mocks import (
    MockRepoManager,
    MockCloneStatus,
    MockCloneTask,
    create_test_blame_data,
    create_test_commit_data,
)
from gitbench.testing.fixtures import mock_git_repo, mock_git_repo_with_branches

# Legacy mock classes - kept for backward compatibility
class MockRustCloneStatus:
    """Mock Rust CloneStatus for testing."""
    def __init__(
        self,
        status_type: str,
        progress: Optional[int] = None,
        error: Optional[str] = None
    ):
        self.status_type = status_type
        self.progress = progress
        self.error = error

class MockRustCloneTask:
    """Mock Rust CloneTask for testing."""
    def __init__(
        self,
        url: str,
        status: MockRustCloneStatus,
        temp_dir: Optional[str] = None
    ):
        self.url = url
        self.status = status
        self.temp_dir = temp_dir

class MockRustRepoManager:
    """Mock Rust RepoManager for testing."""
    def __init__(self, urls: List[str], github_username: str, github_token: str):
        self.urls = urls
        self.github_username = github_username
        self.github_token = github_token
        self._tasks = {}
        
        # Initialize with queued status for all repos
        for url in urls:
            status = MockRustCloneStatus(status_type="queued")
            self._tasks[url] = MockRustCloneTask(url=url, status=status)
            
    async def clone_all(self) -> None:
        """Simulate cloning all repositories."""
        # In a real implementation, this would be async
        for url in self.urls:
            await self.clone(url)
            
    async def clone(self, url: str) -> None:
        """Simulate cloning a single repository."""
        # Mark as cloning
        self._tasks[url].status = MockRustCloneStatus(status_type="cloning", progress=0)
        
        # Simulate progress
        for i in range(0, 101, 20):
            self._tasks[url].status = MockRustCloneStatus(status_type="cloning", progress=i)
            await asyncio.sleep(0.01)  # Small delay to simulate actual work
            
        # Mark as completed with a temp directory
        self._tasks[url].status = MockRustCloneStatus(status_type="completed")
        self._tasks[url].temp_dir = f"/tmp/mock_repo_{url.split('/')[-1]}"
        
    async def fetch_clone_tasks(self) -> Dict[str, MockRustCloneTask]:
        """Return the current clone tasks."""
        return self._tasks
    
    async def bulk_blame(
        self, repo_path: str, file_paths: List[str]
    ) -> Dict[str, Union[List[Dict[str, Any]], str]]:
        """Mock blame operation.
        
        Returns a dictionary where:
        - Keys are file paths
        - Values are either:
          - Lists of line blame information with fields matching the Rust implementation
          - Error message strings
        """
        mock_blame_info = {
            file: [
                {
                    "commit_id": "abcdef1234567890",
                    "author_name": "Test User",
                    "author_email": "test@example.com",
                    "orig_line_no": 1,
                    "final_line_no": 1,
                    "line_content": "Test content line"
                }
            ] 
            for file in file_paths
        }
        
        # Add an error for one file to test error handling
        if len(file_paths) > 1:
            mock_blame_info[file_paths[1]] = "Error: File not found"
            
        return mock_blame_info
    
    async def extract_commits(
        self, repo_path: str
    ) -> Union[List[Dict[str, Any]], str]:
        """Mock commit extraction.
        
        Returns a list of commit dictionaries with fields matching the Rust implementation.
        """
        mock_commits = [
            {
                "sha": "abcdef1234567890",
                "repo_name": repo_path.split('/')[-1],
                "message": "Test commit message",
                "author_name": "Test Author",
                "author_email": "author@example.com",
                "author_timestamp": 1622548800,  # Unix timestamp (2021-06-01)
                "author_offset": 0,
                "committer_name": "Test Committer",
                "committer_email": "committer@example.com",
                "committer_timestamp": 1622548800,  # Unix timestamp (2021-06-01)
                "committer_offset": 0,
                "additions": 10,
                "deletions": 5,
                "is_merge": False
            }
        ]
        return mock_commits
    
    def cleanup(self) -> Dict[str, Union[bool, str]]:
        """Mock cleanup operation.
        
        Returns a dictionary mapping URLs to either:
        - True for successful cleanup
        - Error message string for failed cleanup
        """
        return {url: True for url in self.urls}

# Create modern mock module for gitbench.gitbench using test-specific implementations
class TestGitbenchModule:
    """Test-specific mock module for gitbench.gitbench."""
    RepoManager = MockRepoManager
    CloneStatus = MockCloneStatus
    CloneTask = MockCloneTask

# Patch the Rust modules for testing with the new test-specific implementations
@pytest.fixture
def patch_rust_modules(monkeypatch):
    """Replace Rust modules with test-specific implementations.
    
    This fixture creates mocks for the Rust modules using our test-specific
    implementations that don't make real network calls or require real repositories.
    """
    import sys
    import types
    import gitbench
    from gitbench.models import repo
    
    # Save original modules
    original_modules = {}
    for name in ['gitbench.gitbench']:
        if name in sys.modules:
            original_modules[name] = sys.modules[name]
    
    # Create and install test module
    test_module = types.ModuleType('gitbench.gitbench')
    test_module.RepoManager = MockRepoManager
    test_module.CloneStatus = MockCloneStatus
    test_module.CloneTask = MockCloneTask
    sys.modules['gitbench.gitbench'] = test_module
    
    # Patch gitbench package attributes
    monkeypatch.setattr(gitbench, "RepoManager", MockRepoManager)
    monkeypatch.setattr(gitbench, "RustCloneStatus", MockCloneStatus)
    monkeypatch.setattr(gitbench, "RustCloneTask", MockCloneTask)
    
    # Ensure RUST_AVAILABLE is True for tests
    monkeypatch.setattr(repo, "RUST_AVAILABLE", True)
    
    # Define conversion functions using test-specific implementations
    def test_to_pydantic_status(test_status):
        """Convert a test CloneStatus to a Pydantic model."""
        return PydanticCloneStatus(
            status_type=test_status.status_type,
            progress=test_status.progress,
            error=test_status.error,
        )
    
    def test_to_pydantic_task(test_task):
        """Convert a test CloneTask to a Pydantic model."""
        return PydanticCloneTask(
            url=test_task.url,
            status=test_to_pydantic_status(test_task.status),
            temp_dir=test_task.temp_dir,
        )
    
    def test_convert_clone_tasks(test_tasks):
        """Convert test clone tasks to Pydantic models."""
        return {url: test_to_pydantic_task(task) for url, task in test_tasks.items()}
    
    # Patch conversion functions
    monkeypatch.setattr(repo, "to_pydantic_status", test_to_pydantic_status)
    monkeypatch.setattr(repo, "to_pydantic_task", test_to_pydantic_task)
    monkeypatch.setattr(repo, "convert_clone_tasks", test_convert_clone_tasks)
    
    yield
    
    # Restore original modules
    for name, module in original_modules.items():
        sys.modules[name] = module
    
    # Remove test modules
    for name in ['gitbench.gitbench']:
        if name in sys.modules and name not in original_modules:
            del sys.modules[name]

# Tests for CloneStatusType
def test_clone_status_type():
    """Test CloneStatusType enum values."""
    assert CloneStatusType.QUEUED.value == "queued"
    assert CloneStatusType.CLONING.value == "cloning"
    assert CloneStatusType.COMPLETED.value == "completed"
    assert CloneStatusType.FAILED.value == "failed"
    
    # Test string conversion using value property
    assert CloneStatusType.QUEUED.value == "queued"
    
    # Test equality
    assert CloneStatusType.QUEUED == "queued"
    assert CloneStatusType.CLONING == "cloning"

# Tests for PydanticCloneStatus
def test_clone_status_creation():
    """Test PydanticCloneStatus model creation and validation."""
    # Test basic initialization
    status = PydanticCloneStatus(status_type=CloneStatusType.QUEUED)
    assert status.status_type == CloneStatusType.QUEUED
    assert status.progress is None
    assert status.error is None
    
    # Test with progress
    status = PydanticCloneStatus(status_type=CloneStatusType.CLONING, progress=50)
    assert status.status_type == CloneStatusType.CLONING
    assert status.progress == 50
    
    # Test with error
    status = PydanticCloneStatus(status_type=CloneStatusType.FAILED, error="Connection failed")
    assert status.status_type == CloneStatusType.FAILED
    assert status.error == "Connection failed"

def test_clone_status_validation():
    """Test PydanticCloneStatus validation rules."""
    # Test valid status types
    for status_type in CloneStatusType:
        status = PydanticCloneStatus(status_type=status_type)
        assert status.status_type == status_type
    
    # Test string status type (should convert to enum)
    status = PydanticCloneStatus(status_type="queued")
    assert status.status_type == CloneStatusType.QUEUED
    
    # Test progress validation (should be 0-100)
    status = PydanticCloneStatus(status_type=CloneStatusType.CLONING, progress=0)
    assert status.progress == 0
    
    status = PydanticCloneStatus(status_type=CloneStatusType.CLONING, progress=100)
    assert status.progress == 100
    
    # Progress should be None for non-cloning status
    status = PydanticCloneStatus(status_type=CloneStatusType.COMPLETED)
    assert status.progress is None
    
    # Error should only be set for failed status
    status = PydanticCloneStatus(status_type=CloneStatusType.FAILED, error="Test error")
    assert status.error == "Test error"

@pytest.mark.asyncio
async def test_to_pydantic_status(patch_rust_modules):
    """Test to_pydantic_status conversion function."""
    # Create a mock Rust CloneStatus
    rust_status = MockRustCloneStatus(
        status_type="cloning",
        progress=75,
        error=None
    )
    
    # Convert to Pydantic model
    status = to_pydantic_status(rust_status)
    
    # Verify fields
    assert status.status_type == CloneStatusType.CLONING
    assert status.progress == 75
    assert status.error is None
    
    # Test with error
    rust_status = MockRustCloneStatus(
        status_type="failed",
        progress=None,
        error="Git error: repository not found"
    )
    
    status = to_pydantic_status(rust_status)
    assert status.status_type == CloneStatusType.FAILED
    assert status.progress is None
    assert status.error == "Git error: repository not found"

# Tests for PydanticCloneTask
def test_clone_task_creation():
    """Test PydanticCloneTask model creation and validation."""
    # Test basic initialization
    status = PydanticCloneStatus(status_type=CloneStatusType.QUEUED)
    task = PydanticCloneTask(
        url="https://github.com/user/repo.git",
        status=status
    )
    
    assert task.url == "https://github.com/user/repo.git"
    assert task.status == status
    assert task.temp_dir is None
    
    # Test with temp_dir
    task = PydanticCloneTask(
        url="https://github.com/user/repo.git",
        status=status,
        temp_dir="/tmp/repo"
    )
    
    assert task.temp_dir == "/tmp/repo"

@pytest.mark.asyncio
async def test_to_pydantic_task(patch_rust_modules):
    """Test to_pydantic_task conversion function."""
    # Create a mock Rust CloneStatus
    rust_status = MockRustCloneStatus(
        status_type="completed",
        progress=None
    )
    
    # Create a mock Rust CloneTask
    rust_task = MockRustCloneTask(
        url="https://github.com/user/repo.git",
        status=rust_status,
        temp_dir="/tmp/test_repo"
    )
    
    # Convert to Pydantic model
    task = to_pydantic_task(rust_task)
    
    # Verify fields
    assert task.url == "https://github.com/user/repo.git"
    assert task.status.status_type == CloneStatusType.COMPLETED
    assert task.temp_dir == "/tmp/test_repo"
    
    # Test without temp_dir
    rust_task = MockRustCloneTask(
        url="https://github.com/user/repo.git",
        status=MockRustCloneStatus(status_type="queued"),
        temp_dir=None
    )
    
    task = to_pydantic_task(rust_task)
    assert task.temp_dir is None

# Tests for RepoManager
@pytest.mark.asyncio
async def test_repo_manager_initialization(patch_rust_modules):
    """Test RepoManager initialization."""
    urls = ["https://github.com/user/repo1.git", "https://github.com/user/repo2.git"]
    manager = RepoManager(urls, github_username="test_user", github_token="test_token")
    
    # Should initialize without errors
    assert isinstance(manager, RepoManager)

@pytest.mark.skip("Use test_repo_direct.py instead for direct testing without patching")
@pytest.mark.asyncio
async def test_repo_manager_clone_all(patch_rust_modules):
    """Test RepoManager.clone_all using test-specific implementation."""
    urls = ["https://github.com/user/repo1.git", "https://github.com/user/repo2.git"]
    manager = MockRepoManager(urls, github_username="test_user", github_token="test_token")
    
    # Clone all repositories
    await manager.clone_all()
    
    # Fetch and check tasks
    test_tasks = await manager.fetch_clone_tasks()
    
    # Create Pydantic models directly
    clone_tasks = {}
    for url, task in test_tasks.items():
        status = PydanticCloneStatus(
            status_type=task.status.status_type,
            progress=task.status.progress,
            error=task.status.error
        )
        clone_tasks[url] = PydanticCloneTask(
            url=task.url,
            status=status,
            temp_dir=task.temp_dir
        )
    
    assert len(clone_tasks) == 2
    assert "https://github.com/user/repo1.git" in clone_tasks
    assert "https://github.com/user/repo2.git" in clone_tasks
    
    # All tasks should have completed status
    for url, task in clone_tasks.items():
        assert task.status.status_type == CloneStatusType.COMPLETED
        assert task.temp_dir is not None

@pytest.mark.skip("Use test_repo_direct.py instead for direct testing without patching")
@pytest.mark.asyncio
async def test_repo_manager_clone(patch_rust_modules):
    """Test RepoManager.clone (single repository) using test-specific implementation."""
    urls = ["https://github.com/user/repo1.git"]
    manager = MockRepoManager(urls, github_username="test_user", github_token="test_token")
    
    # Clone a single repository
    await manager.clone("https://github.com/user/repo1.git")
    
    # Check task status directly
    test_tasks = await manager.fetch_clone_tasks()
    test_task = test_tasks["https://github.com/user/repo1.git"]
    
    # Create Pydantic model for assertion
    status = PydanticCloneStatus(
        status_type=test_task.status.status_type,
        progress=test_task.status.progress,
        error=test_task.status.error
    )
    task = PydanticCloneTask(
        url=test_task.url,
        status=status,
        temp_dir=test_task.temp_dir
    )
    
    assert task.status.status_type == CloneStatusType.COMPLETED
    assert task.temp_dir is not None

@pytest.mark.skip("Use test_repo_direct.py instead for direct testing without patching")
@pytest.mark.asyncio
async def test_repo_manager_fetch_clone_tasks(patch_rust_modules):
    """Test RepoManager.fetch_clone_tasks using test-specific implementation."""
    urls = ["https://github.com/user/repo1.git", "https://github.com/user/repo2.git"]
    manager = MockRepoManager(urls, github_username="test_user", github_token="test_token")
    
    # Fetch tasks before cloning
    test_tasks = await manager.fetch_clone_tasks()
    
    # Create Pydantic models for pre-clone tasks
    pre_clone_tasks = {}
    for url, task in test_tasks.items():
        status = PydanticCloneStatus(
            status_type=task.status.status_type,
            progress=task.status.progress,
            error=task.status.error
        )
        pre_clone_tasks[url] = PydanticCloneTask(
            url=task.url,
            status=status,
            temp_dir=task.temp_dir
        )
    
    assert len(pre_clone_tasks) == 2
    for url, task in pre_clone_tasks.items():
        assert task.status.status_type == CloneStatusType.QUEUED
        assert task.temp_dir is None
    
    # Clone and check again
    await manager.clone_all()
    test_tasks = await manager.fetch_clone_tasks()
    
    # Create Pydantic models for post-clone tasks
    post_clone_tasks = {}
    for url, task in test_tasks.items():
        status = PydanticCloneStatus(
            status_type=task.status.status_type,
            progress=task.status.progress,
            error=task.status.error
        )
        post_clone_tasks[url] = PydanticCloneTask(
            url=task.url,
            status=status,
            temp_dir=task.temp_dir
        )
    
    for url, task in post_clone_tasks.items():
        assert task.status.status_type == CloneStatusType.COMPLETED
        assert task.temp_dir is not None

@pytest.mark.skip("Use test_repo_direct.py instead for direct testing without patching")
def test_repo_manager_cleanup(patch_rust_modules):
    """Test RepoManager.cleanup using test-specific implementation."""
    urls = ["https://github.com/user/repo1.git", "https://github.com/user/repo2.git"]
    manager = MockRepoManager(urls, github_username="test_user", github_token="test_token")
    
    # Cleanup should return results for all URLs
    results = manager.cleanup()
    
    assert len(results) == 2
    assert all(results.values())  # All should be True (success)

@pytest.mark.skip("Use test_repo_direct.py instead for direct testing without patching")
@pytest.mark.asyncio
async def test_repo_manager_bulk_blame(patch_rust_modules, mock_git_repo):
    """Test RepoManager.bulk_blame using test-specific implementation and mock repo."""
    # Create a direct instance of our test repo manager
    urls = ["https://github.com/user/repo1.git"]
    manager = MockRepoManager(urls, github_username="test_user", github_token="test_token")
    
    # Set up mock repository path
    temp_dir = mock_git_repo
    
    # Create test files to analyze in the mock repo
    with open(os.path.join(temp_dir, "sample.py"), "w") as f:
        f.write("def test_function():\n    return 'Hello, World!'\n")
    
    # Run bulk blame directly with our test implementation
    file_paths = ["sample.py", "nonexistent.py"]
    blame_results = await manager.bulk_blame(temp_dir, file_paths)
    
    # Check results based on our test implementation's behavior
    assert len(blame_results) == 2
    assert "sample.py" in blame_results
    assert "nonexistent.py" in blame_results
    
    # Our test implementation should return blame info for all files at even indices
    sample_blame = blame_results["sample.py"]
    assert isinstance(sample_blame, list)
    assert len(sample_blame) > 0
    
    # Check blame entry has expected fields
    blame_entry = sample_blame[0]
    assert "commit_id" in blame_entry
    assert "author_name" in blame_entry
    assert "author_email" in blame_entry
    assert "orig_line_no" in blame_entry
    assert "final_line_no" in blame_entry
    assert "line_content" in blame_entry
    
    # Our test implementation returns error strings for odd indexed files
    nonexistent_blame = blame_results["nonexistent.py"]
    assert isinstance(nonexistent_blame, str)

@pytest.mark.skip("Use test_repo_direct.py instead for direct testing without patching")
@pytest.mark.asyncio
async def test_repo_manager_extract_commits(patch_rust_modules, mock_git_repo):
    """Test RepoManager.extract_commits using test-specific implementation and mock repo."""
    # Create a direct instance of TestRepoManager
    urls = ["https://github.com/user/repo1.git"]
    manager = MockRepoManager(urls, github_username="test_user", github_token="test_token")
    
    # Use the mock git repo directly
    temp_dir = mock_git_repo
    
    # Extract commits using our test implementation
    commits = await manager.extract_commits(temp_dir)
    
    # Check results from our test implementation
    # Our test implementation creates 3 test commits
    assert len(commits) == 3
    
    # Verify commit fields match expected structure
    commit = commits[0]
    assert "sha" in commit
    assert "repo_name" in commit
    assert "message" in commit
    assert "author_name" in commit
    assert "author_email" in commit
    assert "author_timestamp" in commit
    assert "author_offset" in commit
    assert "committer_name" in commit
    assert "committer_email" in commit
    assert "committer_timestamp" in commit
    assert "committer_offset" in commit
    assert "additions" in commit
    assert "deletions" in commit
    assert "is_merge" in commit
    
    # Verify timestamp is stored as expected (Unix epoch in seconds)
    assert isinstance(commit["author_timestamp"], int)
    assert isinstance(commit["committer_timestamp"], int)