"""
Mock implementations for testing gitbench.

This module provides test-specific implementations of gitbench classes
that can be used for testing without making real network calls.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
import time


class MockCloneStatus:
    """Test-specific implementation of CloneStatus.
    
    This class mirrors the Rust CloneStatus class but doesn't require the Rust backend.
    It's intended for use in tests only.
    """
    def __init__(
        self,
        status_type: str,
        progress: Optional[int] = None,
        error: Optional[str] = None
    ):
        self.status_type = status_type
        self.progress = progress
        self.error = error


class MockCloneTask:
    """Test-specific implementation of CloneTask.
    
    This class mirrors the Rust CloneTask class but doesn't require the Rust backend.
    It's intended for use in tests only.
    """
    def __init__(
        self,
        url: str,
        status: MockCloneStatus,
        temp_dir: Optional[str] = None
    ):
        self.url = url
        self.status = status
        self.temp_dir = temp_dir


class MockRepoManager:
    """Test-specific implementation of RepoManager.
    
    This class provides a mock implementation of the Rust-based RepoManager
    that doesn't make real network calls or require the Rust backend.
    It simulates all operations for testing purposes.
    """
    def __init__(self, urls: List[str], github_username: str, github_token: str):
        self.urls = urls
        self.github_username = github_username
        self.github_token = github_token
        self._tasks = {}
        
        # Initialize with queued status for all repos
        for url in urls:
            status = MockCloneStatus(status_type="queued")
            self._tasks[url] = MockCloneTask(url=url, status=status)
            
    async def clone_all(self) -> None:
        """Simulate cloning all repositories without real network calls."""
        for url in self.urls:
            await self.clone(url)
            
    async def clone(self, url: str) -> None:
        """Simulate cloning a single repository without real network calls."""
        # Mark as cloning
        self._tasks[url].status = MockCloneStatus(status_type="cloning", progress=0)
        
        # Simulate progress
        for i in range(0, 101, 20):
            self._tasks[url].status = MockCloneStatus(status_type="cloning", progress=i)
            await asyncio.sleep(0.01)  # Small delay to simulate work
            
        # Mark as completed with a temp directory
        self._tasks[url].status = MockCloneStatus(status_type="completed")
        self._tasks[url].temp_dir = f"/tmp/mock_repo_{url.split('/')[-1]}"
        
    async def fetch_clone_tasks(self) -> Dict[str, MockCloneTask]:
        """Return the current clone tasks."""
        return self._tasks
    
    async def bulk_blame(
        self, repo_path: str, file_paths: List[str]
    ) -> Dict[str, Union[List[Dict[str, Any]], str]]:
        """Simulate blame operation without real repository access."""
        result = {}
        for i, file_path in enumerate(file_paths):
            # Simulate an error for every other file to test error handling
            if i % 2 == 1:
                result[file_path] = f"Error: Could not find file {file_path}"
            else:
                result[file_path] = create_test_blame_data(num_lines=5)
        return result
    
    async def extract_commits(
        self, repo_path: str
    ) -> Union[List[Dict[str, Any]], str]:
        """Simulate commit extraction without real repository access."""
        # Create test commits data
        return create_test_commit_data(repo_name=repo_path.split('/')[-1], num_commits=3)
    
    def cleanup(self) -> Dict[str, Union[bool, str]]:
        """Simulate cleanup operation."""
        # Simulate successful cleanup for all URLs
        return {url: True for url in self.urls}


def create_test_blame_data(num_lines: int = 10) -> List[Dict[str, Any]]:
    """Create fake blame data for testing.
    
    Args:
        num_lines: Number of lines of blame data to generate
        
    Returns:
        List of blame data dictionaries with realistic test data
    """
    return [
        {
            "commit_id": f"abcdef{i}0123456789",
            "author_name": f"Test Author {i % 3 + 1}",
            "author_email": f"author{i % 3 + 1}@example.com",
            "orig_line_no": i,
            "final_line_no": i,
            "line_content": f"Line {i}: Test content for blame"
        }
        for i in range(1, num_lines + 1)
    ]


def create_test_commit_data(repo_name: str, num_commits: int = 5) -> List[Dict[str, Any]]:
    """Create fake commit data for testing.
    
    Args:
        repo_name: Name of the repository
        num_commits: Number of commits to generate
        
    Returns:
        List of commit data dictionaries with realistic test data
    """
    # Current timestamp
    now = int(time.time())
    
    # Generate commits (most recent first)
    return [
        {
            "sha": f"abcdef{i}0123456789",
            "repo_name": repo_name,
            "message": f"Test commit message {i}",
            "author_name": f"Test Author {i % 3 + 1}",
            "author_email": f"author{i % 3 + 1}@example.com",
            "author_timestamp": now - (i * 86400),  # Each commit is 1 day apart
            "author_offset": 0,
            "committer_name": f"Test Committer {i % 2 + 1}",
            "committer_email": f"committer{i % 2 + 1}@example.com",
            "committer_timestamp": now - (i * 86400),
            "committer_offset": 0,
            "additions": 10 + i,
            "deletions": 5 + (i % 3),
            "is_merge": (i % 5 == 0)  # Every 5th commit is a merge
        }
        for i in range(num_commits)
    ]