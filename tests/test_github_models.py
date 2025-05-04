"""
Tests for GitHub Pydantic model validation against real API responses.
"""

import json
import os
import pytest
import asyncio
from typing import Dict, Any

from gitbench.providers.github import GitHubClient
from gitbench.models.common import (
    UserInfo,
    RepoInfo,
    RepoDetails,
    RateLimitInfo,
    BranchInfo,
    ContributorInfo,
    CommitRef
)

# Load GitHub token from environment with fallback to dotenv file
def get_github_token():
    """Get GitHub token from environment or .env file."""
    token = os.environ.get("GITHUB_TOKEN")
    
    # If token not in environment, try loading from .env file
    if not token:
        try:
            import dotenv
            dotenv.load_dotenv()
            token = os.environ.get("GITHUB_TOKEN")
        except ImportError:
            # dotenv not installed, continue without it
            pass
            
    return token

# Skip tests if no token is available
GITHUB_TOKEN = get_github_token()
requires_token = pytest.mark.skipif(
    not GITHUB_TOKEN, reason="GitHub token not available. Set GITHUB_TOKEN environment variable."
)

# Sample data for offline testing
SAMPLE_REPO = "bmeddeb/gitbench"
SAMPLE_OWNER = "bmeddeb"
SAMPLE_REPO_NAME = "gitbench"


@pytest.fixture
def github_client():
    """Create a GitHub client for testing."""
    if not GITHUB_TOKEN:
        return None
    return GitHubClient(token=GITHUB_TOKEN)


@pytest.mark.asyncio
@requires_token
async def test_user_info(github_client):
    """Test UserInfo model with actual GitHub API response."""
    user_info = await github_client.fetch_user_info()
    
    # Verify model fields
    assert isinstance(user_info, UserInfo)
    assert isinstance(user_info.id, int)
    assert user_info.login
    assert user_info.provider_type.value == "github"
    
    # Print model for debugging
    print(f"UserInfo: {user_info}")
    

@pytest.mark.asyncio
@requires_token
async def test_rate_limit(github_client):
    """Test RateLimitInfo model with actual GitHub API response."""
    rate_limit = await github_client.get_rate_limit()
    
    # Verify model fields
    assert isinstance(rate_limit, RateLimitInfo)
    assert isinstance(rate_limit.limit, int)
    assert isinstance(rate_limit.remaining, int)
    assert isinstance(rate_limit.reset_time, int)
    assert isinstance(rate_limit.used, int)
    assert rate_limit.provider_type.value == "github"
    
    # Verify helper method
    assert isinstance(rate_limit.seconds_until_reset(), int)
    
    # Print model for debugging
    print(f"RateLimitInfo: {rate_limit}")


@pytest.mark.asyncio
@requires_token
async def test_repo_info(github_client):
    """Test RepoInfo model with actual GitHub API response."""
    repos = await github_client.fetch_repositories(SAMPLE_OWNER)
    
    # Verify we got at least one repository
    assert repos
    repo = repos[0]
    
    # Verify model fields
    assert isinstance(repo, RepoInfo)
    assert isinstance(repo.id, int)
    assert repo.name
    assert repo.full_name
    assert repo.clone_url
    assert isinstance(repo.fork, bool)
    assert isinstance(repo.forks_count, int)
    assert repo.provider_type.value == "github"
    
    # If owner is present, verify its structure
    if repo.owner:
        assert isinstance(repo.owner.id, int)
        assert repo.owner.login
    
    # Print model for debugging
    print(f"RepoInfo: {repo}")


@pytest.mark.asyncio
@requires_token
async def test_repo_details(github_client):
    """Test RepoDetails model with actual GitHub API response."""
    repo_details = await github_client.fetch_repository_details(SAMPLE_OWNER, SAMPLE_REPO_NAME)
    
    # Verify model fields
    assert isinstance(repo_details, RepoDetails)
    assert isinstance(repo_details.id, int)
    assert repo_details.name == SAMPLE_REPO_NAME
    assert repo_details.full_name == SAMPLE_REPO
    assert repo_details.clone_url
    assert isinstance(repo_details.topics, list)
    assert isinstance(repo_details.has_wiki, bool)
    assert isinstance(repo_details.has_issues, bool)
    assert isinstance(repo_details.archived, bool)
    assert repo_details.provider_type.value == "github"
    
    # Print model for debugging
    print(f"RepoDetails: {repo_details}")


@pytest.mark.asyncio
@requires_token
async def test_branches(github_client):
    """Test BranchInfo model with actual GitHub API response."""
    branches = await github_client.fetch_branches(SAMPLE_OWNER, SAMPLE_REPO_NAME)
    
    # Verify we got at least one branch
    assert branches
    branch = branches[0]
    
    # Verify model fields
    assert isinstance(branch, BranchInfo)
    assert branch.name
    assert branch.commit_sha
    assert isinstance(branch.protected, bool)
    assert branch.provider_type.value == "github"
    
    # Verify the commit object
    if branch.commit:
        assert isinstance(branch.commit, CommitRef)
        assert branch.commit.sha == branch.commit_sha
    
    # Print model for debugging
    print(f"BranchInfo: {branch}")


@pytest.mark.asyncio
@requires_token
async def test_contributors(github_client):
    """Test ContributorInfo model with actual GitHub API response."""
    contributors = await github_client.fetch_contributors(SAMPLE_OWNER, SAMPLE_REPO_NAME)
    
    # Some repos might not have contributors yet
    if not contributors:
        pytest.skip("No contributors found for the repository")
    
    contributor = contributors[0]
    
    # Verify model fields
    assert isinstance(contributor, ContributorInfo)
    assert isinstance(contributor.id, int)
    assert contributor.login
    assert isinstance(contributor.contributions, int)
    assert contributor.provider_type.value == "github"
    
    # Print model for debugging
    print(f"ContributorInfo: {contributor}")



if __name__ == "__main__":
    # Manual test runner
    async def run_tests():
        client = GitHubClient(token=GITHUB_TOKEN)
        await test_user_info(client)
        await test_rate_limit(client)
        await test_repo_info(client)
        await test_repo_details(client)
        await test_branches(client)
        await test_contributors(client)
    
    if GITHUB_TOKEN:
        asyncio.run(run_tests())
    else:
        print("GitHub token not available. Set GITHUB_TOKEN environment variable to run tests.")