"""
Tests for the authentication utilities.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from gitbench.providers.base import ProviderType
from gitbench.utils.auth import CredentialManager, derive_key_from_password


class TestCredentialManager(unittest.TestCase):
    """Test cases for the CredentialManager."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary file for credentials
        self.temp_dir = tempfile.TemporaryDirectory()
        self.credentials_file = os.path.join(self.temp_dir.name, "credentials.json")

    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_base64_encoding(self):
        """Test basic functionality with base64 encoding."""
        # Suppress warning about base64 encoding
        with patch("warnings.warn"):
            manager = CredentialManager(credentials_file=self.credentials_file)

            # Test saving and retrieving credentials
            manager.save_credential(
                provider=ProviderType.GITHUB, token="test_token", username="test_user"
            )

            credentials = manager.get_credentials(ProviderType.GITHUB)
            self.assertEqual(len(credentials), 1)
            self.assertEqual(credentials[0].token, "test_token")
            self.assertEqual(credentials[0].username, "test_user")

            # Test removing credentials
            result = manager.remove_credential(ProviderType.GITHUB, "test_token")
            self.assertTrue(result)

            credentials = manager.get_credentials(ProviderType.GITHUB)
            self.assertEqual(len(credentials), 0)

    def test_aes_encryption(self):
        """Test AES-GCM encryption if cryptography is available."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            self.skipTest("cryptography package not available")

        # Generate a random key
        key = os.urandom(32)
        manager = CredentialManager(
            credentials_file=self.credentials_file, encryption_key=key
        )

        # Test saving and retrieving credentials
        manager.save_credential(
            provider=ProviderType.GITHUB, token="secret_token", username="aes_user"
        )

        credentials = manager.get_credentials(ProviderType.GITHUB)
        self.assertEqual(len(credentials), 1)
        self.assertEqual(credentials[0].token, "secret_token")
        self.assertEqual(credentials[0].username, "aes_user")

        # Create a new manager with the same key and check if it can decrypt
        manager2 = CredentialManager(
            credentials_file=self.credentials_file, encryption_key=key
        )

        credentials = manager2.get_credentials(ProviderType.GITHUB)
        self.assertEqual(len(credentials), 1)
        self.assertEqual(credentials[0].token, "secret_token")

        # Create a manager with a different key and verify it can't decrypt
        wrong_key = os.urandom(32)
        manager3 = CredentialManager(
            credentials_file=self.credentials_file, encryption_key=wrong_key
        )

        # Suppress warning about failed decryption
        with patch("warnings.warn"):
            credentials = manager3.get_credentials(ProviderType.GITHUB)
            self.assertEqual(len(credentials), 0)  # Should fail to decrypt

    def test_password_derived_key(self):
        """Test using a password-derived key if cryptography is available."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            self.skipTest("cryptography package not available")

        # Create a salt file
        salt_file = os.path.join(self.temp_dir.name, "salt.bin")

        # Create a manager with a password
        manager = CredentialManager.from_password(
            password="test_password",
            salt_file=salt_file,
            credentials_file=self.credentials_file,
        )

        # Test saving and retrieving credentials
        manager.save_credential(
            provider=ProviderType.GITHUB,
            token="password_protected_token",
            username="password_user",
        )

        # Create a new manager with the same password and salt file
        manager2 = CredentialManager.from_password(
            password="test_password",
            salt_file=salt_file,
            credentials_file=self.credentials_file,
        )

        credentials = manager2.get_credentials(ProviderType.GITHUB)
        self.assertEqual(len(credentials), 1)
        self.assertEqual(credentials[0].token, "password_protected_token")

        # Create a manager with a wrong password and verify it can't decrypt
        manager3 = CredentialManager.from_password(
            password="wrong_password",
            salt_file=salt_file,
            credentials_file=self.credentials_file,
        )

        # Suppress warning about failed decryption
        with patch("warnings.warn"):
            credentials = manager3.get_credentials(ProviderType.GITHUB)
            self.assertEqual(len(credentials), 0)  # Should fail to decrypt

    def test_custom_encryption(self):
        """Test using custom encryption functions."""

        # Simple XOR encryption for testing
        def test_encrypt(token: str) -> str:
            return "".join(chr(ord(c) ^ 42) for c in token)

        def test_decrypt(token: str) -> str:
            return "".join(chr(ord(c) ^ 42) for c in token)

        manager = CredentialManager(
            credentials_file=self.credentials_file,
            encrypt_func=test_encrypt,
            decrypt_func=test_decrypt,
        )

        # Test saving and retrieving credentials
        manager.save_credential(
            provider=ProviderType.GITHUB,
            token="custom_encrypted_token",
            username="custom_user",
        )

        credentials = manager.get_credentials(ProviderType.GITHUB)
        self.assertEqual(len(credentials), 1)
        self.assertEqual(credentials[0].token, "custom_encrypted_token")

        # Test removing credentials
        result = manager.remove_credential_by_username(
            ProviderType.GITHUB, "custom_user"
        )
        self.assertTrue(result)

        credentials = manager.get_credentials(ProviderType.GITHUB)
        self.assertEqual(len(credentials), 0)

    def test_invalid_encryption_setup(self):
        """Test invalid encryption configurations."""
        # Test providing both encryption_key and custom functions
        with self.assertRaises(ValueError):
            CredentialManager(
                credentials_file=self.credentials_file,
                encryption_key=os.urandom(32),
                encrypt_func=lambda x: x,
                decrypt_func=lambda x: x,
            )

        # Test providing only one custom function
        with self.assertRaises(ValueError):
            manager = CredentialManager(
                credentials_file=self.credentials_file, encrypt_func=lambda x: x
            )


if __name__ == "__main__":
    unittest.main()
