"""
Direct tests for repository operations using test-specific implementations.

This module contains tests that directly use the test-specific implementations
without relying on monkeypatching. This avoids issues with the patching approach
and provides a cleaner way to test the functionality.
"""

import pytest
import os
import asyncio
from typing import Dict, List, Any, Optional, Union

from gitbench.models.repo import (
    CloneStatusType, 
    PydanticCloneStatus, 
    PydanticCloneTask,
)

# Import test-specific implementations directly
from gitbench.testing.mocks import (
    MockRepoManager,
    MockCloneStatus,
    MockCloneTask,
    create_test_blame_data,
    create_test_commit_data,
)
from gitbench.testing.fixtures import mock_git_repo, mock_git_repo_with_branches


# Direct tests that don't rely on patching

@pytest.mark.asyncio
async def test_direct_repo_manager_clone_all():
    """Test MockRepoManager.clone_all directly."""
    urls = ["https://github.com/user/repo1.git", "https://github.com/user/repo2.git"]
    manager = MockRepoManager(urls, github_username="test_user", github_token="test_token")
    
    # Verify initial state
    tasks = await manager.fetch_clone_tasks()
    assert len(tasks) == 2
    for url, task in tasks.items():
        assert task.status.status_type == "queued"
        assert task.temp_dir is None
    
    # Clone all repositories
    await manager.clone_all()
    
    # Check tasks after cloning
    tasks = await manager.fetch_clone_tasks()
    assert len(tasks) == 2
    for url, task in tasks.items():
        assert task.status.status_type == "completed"
        assert task.temp_dir is not None
        assert task.temp_dir.startswith("/tmp/mock_repo_")


@pytest.mark.asyncio
async def test_direct_repo_manager_clone():
    """Test MockRepoManager.clone directly."""
    urls = ["https://github.com/user/repo1.git"]
    manager = MockRepoManager(urls, github_username="test_user", github_token="test_token")
    
    # Clone a single repository
    await manager.clone("https://github.com/user/repo1.git")
    
    # Check task status
    tasks = await manager.fetch_clone_tasks()
    task = tasks["https://github.com/user/repo1.git"]
    
    assert task.status.status_type == "completed"
    assert task.temp_dir is not None
    assert task.temp_dir.startswith("/tmp/mock_repo_")


@pytest.mark.asyncio
async def test_direct_repo_manager_bulk_blame(mock_git_repo):
    """Test MockRepoManager.bulk_blame directly with a mock repo."""
    urls = ["https://github.com/user/repo1.git"]
    manager = MockRepoManager(urls, github_username="test_user", github_token="test_token")
    
    # Create a test file in the mock repo
    test_file = os.path.join(mock_git_repo, "sample.py")
    with open(test_file, "w") as f:
        f.write("def test_function():\n    return 'Hello, World!'\n")
    
    # Run bulk blame
    file_paths = ["sample.py", "missing.py"]
    blame_results = await manager.bulk_blame(mock_git_repo, file_paths)
    
    # Verify results based on our test implementation
    assert len(blame_results) == 2
    assert "sample.py" in blame_results
    assert "missing.py" in blame_results
    
    # Our test implementation returns a list for even-indexed files (sample.py)
    sample_blame = blame_results["sample.py"]
    assert isinstance(sample_blame, list)
    assert len(sample_blame) > 0
    
    blame_entry = sample_blame[0]
    assert "commit_id" in blame_entry
    assert "author_name" in blame_entry
    assert "author_email" in blame_entry
    assert "orig_line_no" in blame_entry
    assert "final_line_no" in blame_entry
    assert "line_content" in blame_entry
    
    # Our test implementation returns an error string for odd-indexed files (missing.py)
    missing_blame = blame_results["missing.py"]
    assert isinstance(missing_blame, str)


@pytest.mark.asyncio
async def test_direct_repo_manager_extract_commits(mock_git_repo):
    """Test MockRepoManager.extract_commits directly with a mock repo."""
    urls = ["https://github.com/user/repo1.git"]
    manager = MockRepoManager(urls, github_username="test_user", github_token="test_token")
    
    # Extract commits
    commits = await manager.extract_commits(mock_git_repo)
    
    # Our test implementation creates 3 test commits
    assert len(commits) == 3
    
    # Verify commit structure
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
    
    # Verify timestamp format
    assert isinstance(commit["author_timestamp"], int)
    assert isinstance(commit["committer_timestamp"], int)


def test_direct_repo_manager_cleanup():
    """Test MockRepoManager.cleanup directly."""
    urls = ["https://github.com/user/repo1.git", "https://github.com/user/repo2.git"]
    manager = MockRepoManager(urls, github_username="test_user", github_token="test_token")
    
    # Run cleanup
    results = manager.cleanup()
    
    # Verify results
    assert len(results) == 2
    assert all(results.values())
    for url in urls:
        assert url in results
        assert results[url] is True


# Test helper functions

def test_create_test_blame_data():
    """Test the create_test_blame_data helper function."""
    blame_data = create_test_blame_data(num_lines=5)
    
    assert len(blame_data) == 5
    
    for entry in blame_data:
        assert "commit_id" in entry
        assert "author_name" in entry
        assert "author_email" in entry
        assert "orig_line_no" in entry
        assert "final_line_no" in entry
        assert "line_content" in entry


def test_create_test_commit_data():
    """Test the create_test_commit_data helper function."""
    repo_name = "test-repo"
    commits = create_test_commit_data(repo_name=repo_name, num_commits=3)
    
    assert len(commits) == 3
    
    for commit in commits:
        assert "sha" in commit
        assert commit["repo_name"] == repo_name
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