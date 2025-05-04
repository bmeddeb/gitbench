"""
Test suite for the Git provider API clients.
"""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from gitbench.models.common import RepoInfo, UserInfo
from gitbench.providers import GitHubClient, ProviderType
from gitbench.providers.token_manager import (TokenInfo, TokenManager,
                                              TokenStatus)


class TestGitHubClient(unittest.TestCase):
    """Test cases for the GitHub API client."""

    def setUp(self):
        """Set up test environment."""
        self.token = "fake_token"
        self.client = GitHubClient(self.token)

    @patch("gitbench.providers.github.httpx.AsyncClient")
    def test_fetch_user_info(self, mock_client):
        """Test fetching user info."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "id": 123,
            "login": "testuser",
            "name": "Test User",
            "email": "test@example.com",
            "avatar_url": "https://example.com/avatar.png",
        }
        mock_response.headers = {}

        # Set up the mock client
        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.request = AsyncMock(return_value=mock_response)
        mock_client.return_value = mock_client_instance

        # Run the test
        result = asyncio.run(self.client.fetch_user_info())

        # Verify the result
        self.assertEqual(result.login, "testuser")
        self.assertEqual(result.name, "Test User")
        self.assertEqual(result.email, "test@example.com")
        self.assertEqual(result.avatar_url, "https://example.com/avatar.png")
        self.assertEqual(result.provider_type, ProviderType.GITHUB)

        # Verify the mock was called with the right arguments
        mock_client_instance.request.assert_called_once_with(
            method="GET",
            url="https://api.github.com/user",
            headers={
                "Authorization": "token fake_token",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "gitbench-Client",
            },
        )

    @patch("gitbench.providers.github.httpx.AsyncClient")
    def test_fetch_repositories(self, mock_client):
        """Test fetching repositories."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {
                "id": 123,
                "name": "repo1",
                "full_name": "testuser/repo1",
                "clone_url": "https://github.com/testuser/repo1.git",
                "description": "Test repository 1",
                "default_branch": "main",
                "language": "Python",
                "fork": False,
                "forks_count": 2,
                "stargazers_count": 10,
                "owner": {
                    "id": 456,
                    "login": "testuser",
                    "name": None,
                    "email": None,
                    "avatar_url": "https://example.com/avatar.png",
                },
            }
        ]
        mock_response.headers = {}

        # Set up the mock client
        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.request = AsyncMock(return_value=mock_response)
        mock_client.return_value = mock_client_instance

        # Run the test
        result = asyncio.run(self.client.fetch_repositories("testuser"))

        # Verify the result
        self.assertEqual(len(result), 1)
        repo = result[0]
        self.assertEqual(repo.name, "repo1")
        self.assertEqual(repo.full_name, "testuser/repo1")
        self.assertEqual(repo.description, "Test repository 1")
        self.assertEqual(repo.language, "Python")
        self.assertEqual(repo.stargazers_count, 10)
        self.assertEqual(repo.forks_count, 2)

        # Check owner information
        self.assertIsNotNone(repo.owner)
        self.assertEqual(repo.owner.login, "testuser")
        self.assertEqual(repo.owner.avatar_url, "https://example.com/avatar.png")

        # Verify the mock was called with the right arguments
        mock_client_instance.request.assert_called_once_with(
            method="GET",
            url="https://api.github.com/users/testuser/repos?per_page=100",
            headers={
                "Authorization": "token fake_token",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "gitbench-Client",
            },
        )


class TestTokenManager(unittest.TestCase):
    """Test cases for the token manager."""

    def setUp(self):
        """Set up test environment."""
        self.token_manager = TokenManager()
        self.token1 = "token1"
        self.token2 = "token2"
        self.token_manager.add_token(self.token1, ProviderType.GITHUB, "user1")
        self.token_manager.add_token(self.token2, ProviderType.GITHUB, "user2")

    def test_add_token(self):
        """Test adding tokens."""
        # Verify the tokens were added
        self.assertEqual(len(self.token_manager.tokens[ProviderType.GITHUB]), 2)
        self.assertEqual(
            self.token_manager.tokens[ProviderType.GITHUB][0].token, self.token1
        )
        self.assertEqual(
            self.token_manager.tokens[ProviderType.GITHUB][1].token, self.token2
        )

    def test_get_next_available_token(self):
        """Test getting the next available token."""
        # Get the first token
        token_info = asyncio.run(
            self.token_manager.get_next_available_token(ProviderType.GITHUB)
        )
        self.assertEqual(token_info.token, self.token1)

        # Get the second token
        token_info = asyncio.run(
            self.token_manager.get_next_available_token(ProviderType.GITHUB)
        )
        self.assertEqual(token_info.token, self.token2)

        # Get the first token again (round-robin)
        token_info = asyncio.run(
            self.token_manager.get_next_available_token(ProviderType.GITHUB)
        )
        self.assertEqual(token_info.token, self.token1)

    def test_mark_token_invalid(self):
        """Test marking a token as invalid."""
        # Mark the first token as invalid
        asyncio.run(
            self.token_manager.mark_token_invalid(self.token1, ProviderType.GITHUB)
        )

        # Verify the token is marked as invalid
        self.assertFalse(
            self.token_manager.tokens[ProviderType.GITHUB][0].status.is_valid
        )

        # Get the next available token (should be the second token)
        token_info = asyncio.run(
            self.token_manager.get_next_available_token(ProviderType.GITHUB)
        )
        self.assertEqual(token_info.token, self.token2)

    def test_update_rate_limit(self):
        """Test updating rate limit information."""
        # Update rate limit for the first token
        asyncio.run(
            self.token_manager.update_rate_limit(
                self.token1, ProviderType.GITHUB, 100, 1234567890
            )
        )

        # Verify the rate limit information was updated
        token_info = self.token_manager.tokens[ProviderType.GITHUB][0]
        self.assertEqual(token_info.status.remaining_calls, 100)
        self.assertEqual(token_info.status.reset_time, 1234567890)


if __name__ == "__main__":
    unittest.main()
