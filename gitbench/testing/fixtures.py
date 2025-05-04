"""
Test fixtures for gitbench.

This module provides reusable fixtures for testing gitbench functionality.
It includes fixtures for creating temporary Git repositories with predefined content
that don't require real network connections.
"""

import os
import tempfile
import subprocess
from typing import Iterator, Optional, List, Tuple

import pytest


@pytest.fixture
def mock_git_repo() -> Iterator[str]:
    """Create a temporary Git repository with sample content for testing.
    
    This fixture creates a temporary Git repository with predefined content
    that can be used for testing Git operations without network calls.
    
    Yields:
        Path to the temporary Git repository
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize Git repo
        subprocess.run(["git", "init"], cwd=temp_dir, check=True, capture_output=True)
        
        # Configure Git (required for commits)
        subprocess.run(
            ["git", "config", "user.name", "Test User"], 
            cwd=temp_dir, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], 
            cwd=temp_dir, check=True, capture_output=True
        )
        
        # Create sample files
        _create_sample_files(temp_dir)
        
        # Add and commit files
        subprocess.run(["git", "add", "."], cwd=temp_dir, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=temp_dir, check=True, capture_output=True
        )
        
        # Create more files and make additional commits
        _make_additional_commits(temp_dir, num_commits=3)
        
        yield temp_dir


@pytest.fixture
def mock_git_repo_with_branches() -> Iterator[Tuple[str, List[str]]]:
    """Create a temporary Git repository with multiple branches.
    
    This fixture creates a temporary Git repository with multiple branches
    and sample content for testing branch-related operations.
    
    Yields:
        Tuple containing:
        - Path to the temporary Git repository
        - List of branch names created
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize and set up the repository similar to mock_git_repo
        subprocess.run(["git", "init"], cwd=temp_dir, check=True, capture_output=True)
        
        # Configure Git
        subprocess.run(
            ["git", "config", "user.name", "Test User"], 
            cwd=temp_dir, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], 
            cwd=temp_dir, check=True, capture_output=True
        )
        
        # Create initial files and commit
        _create_sample_files(temp_dir)
        subprocess.run(["git", "add", "."], cwd=temp_dir, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=temp_dir, check=True, capture_output=True
        )
        
        # Create branches
        branches = ["feature-1", "feature-2", "bugfix"]
        for branch in branches:
            # Create and switch to new branch
            subprocess.run(
                ["git", "checkout", "-b", branch],
                cwd=temp_dir, check=True, capture_output=True
            )
            
            # Create branch-specific files
            with open(os.path.join(temp_dir, f"{branch}.txt"), "w") as f:
                f.write(f"Content for {branch}\n")
            
            # Commit branch changes
            subprocess.run(["git", "add", "."], cwd=temp_dir, check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", f"Add {branch} content"],
                cwd=temp_dir, check=True, capture_output=True
            )
            
            # Switch back to main
            subprocess.run(
                ["git", "checkout", "master"],
                cwd=temp_dir, check=True, capture_output=True
            )
        
        yield temp_dir, branches


def _create_sample_files(repo_dir: str) -> None:
    """Create sample files in the repository directory.
    
    Args:
        repo_dir: Directory path where files should be created
    """
    # Create Python file
    with open(os.path.join(repo_dir, "sample.py"), "w") as f:
        f.write("""def hello_world():
    \"\"\"Print hello world message.\"\"\"
    print("Hello, World!")
    
if __name__ == "__main__":
    hello_world()
""")
    
    # Create text file
    with open(os.path.join(repo_dir, "README.md"), "w") as f:
        f.write("""# Sample Repository
        
This is a sample repository for testing gitbench.

## Features

- Sample Python code
- README with markdown
- Configuration files
""")
    
    # Create config file
    with open(os.path.join(repo_dir, "config.ini"), "w") as f:
        f.write("""[DEFAULT]
debug = false
log_level = INFO

[app]
name = SampleApp
version = 1.0.0
""")


def _make_additional_commits(repo_dir: str, num_commits: int = 3) -> None:
    """Make additional commits in the repository.
    
    Args:
        repo_dir: Path to the Git repository
        num_commits: Number of additional commits to make
    """
    for i in range(1, num_commits + 1):
        # Create or modify a file
        with open(os.path.join(repo_dir, f"file{i}.txt"), "w") as f:
            f.write(f"Content for file {i}\n")
            f.write(f"This line was added in commit {i}\n")
        
        # Add and commit
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", f"Commit {i}: Add file{i}.txt"],
            cwd=repo_dir, check=True, capture_output=True
        )