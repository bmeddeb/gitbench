"""
Example of using secure token encryption with CredentialManager.

This example demonstrates various ways to use the CredentialManager:
1. Basic usage with default settings (base64 encoding only)
2. Using AES-GCM encryption with a provided key
3. Using a password-derived key
4. Using custom encryption functions

Note: This is just an example. In a real application, never hardcode passwords
and properly manage encryption keys.
"""

import os
from getpass import getpass

from gitbench.providers.base import ProviderType
from gitbench.utils.auth import CredentialManager, derive_key_from_password


def basic_example():
    """Basic usage of CredentialManager (with warning about weak security)."""
    # This will show a warning about using base64 encoding only
    manager = CredentialManager()

    # Save a credential
    manager.save_credential(
        provider=ProviderType.GITHUB, token="github_token_123", username="example_user"
    )

    # Get credentials
    credentials = manager.get_credentials(ProviderType.GITHUB)
    print(f"Retrieved {len(credentials)} credentials")
    for cred in credentials:
        print(f"Username: {cred.username}, Token: {cred.token}")

    # Clean up
    manager.clear_credentials()


def aes_example():
    """Example using AES-GCM encryption with a generated key."""
    # Generate a random 32-byte key (AES-256)
    key = os.urandom(32)
    print(f"Generated key: {key.hex()}")

    # Create a manager with AES encryption
    manager = CredentialManager(encryption_key=key)

    # Save a credential
    manager.save_credential(
        provider=ProviderType.GITHUB, token="github_token_456", username="secure_user"
    )

    # Get credentials
    credentials = manager.get_credentials(ProviderType.GITHUB)
    print(f"Retrieved {len(credentials)} credentials")
    for cred in credentials:
        print(f"Username: {cred.username}, Token: {cred.token}")

    # Clean up
    manager.clear_credentials()

    print("\nIn a real application, you would need to securely store this key")
    print("and provide it each time you create the CredentialManager.")


def password_example():
    """Example using a password-derived key with salt."""
    # Path to store the salt
    salt_file = "~/.gitbench/salt.bin"

    # In a real application, you would prompt for the password securely
    # password = getpass("Enter your password: ")
    password = "secure_password_example"  # Don't do this in real code!

    # Create a manager from password (this will generate and save the salt)
    manager = CredentialManager.from_password(password=password, salt_file=salt_file)

    # Save a credential
    manager.save_credential(
        provider=ProviderType.GITHUB,
        token="github_token_789",
        username="password_protected_user",
    )

    # Get credentials
    credentials = manager.get_credentials(ProviderType.GITHUB)
    print(f"Retrieved {len(credentials)} credentials")
    for cred in credentials:
        print(f"Username: {cred.username}, Token: {cred.token}")

    # Clean up
    manager.clear_credentials()

    print("\nThe salt was saved to:", os.path.expanduser(salt_file))
    print("To access the same encrypted data, you would need both the password")
    print("and the salt file.")


def custom_encryption_example():
    """Example using custom encryption functions."""

    # Define custom encryption functions (very basic, don't use in real applications)
    def custom_encrypt(token: str) -> str:
        # Just a simple XOR with a fixed key (NOT secure!)
        key = "SECRETKEY"
        result = ""
        for i, char in enumerate(token):
            key_char = key[i % len(key)]
            result += chr(ord(char) ^ ord(key_char))
        return base64.b64encode(result.encode()).decode()

    def custom_decrypt(encrypted_token: str) -> str:
        # Reverse the XOR operation
        key = "SECRETKEY"
        token = base64.b64decode(encrypted_token.encode()).decode()
        result = ""
        for i, char in enumerate(token):
            key_char = key[i % len(key)]
            result += chr(ord(char) ^ ord(key_char))
        return result

    # Create manager with custom encryption
    manager = CredentialManager(
        encrypt_func=custom_encrypt, decrypt_func=custom_decrypt
    )

    # Save a credential
    manager.save_credential(
        provider=ProviderType.GITHUB,
        token="github_token_custom",
        username="custom_encryption_user",
    )

    # Get credentials
    credentials = manager.get_credentials(ProviderType.GITHUB)
    print(f"Retrieved {len(credentials)} credentials")
    for cred in credentials:
        print(f"Username: {cred.username}, Token: {cred.token}")

    # Clean up
    manager.clear_credentials()

    print("\nThis example shows how to provide your own encryption functions.")
    print("In a real application, you would use a strong encryption algorithm")
    print("and proper key management.")


if __name__ == "__main__":
    # Add import here to avoid issues with the custom encryption example
    import base64

    print("\n=== Basic Example (base64 encoding only) ===")
    basic_example()

    try:
        print("\n=== AES-GCM Encryption Example ===")
        aes_example()

        print("\n=== Password-Based Encryption Example ===")
        password_example()
    except ImportError:
        print("\nAES examples require the 'cryptography' package.")
        print("Install it with: pip install cryptography")

    print("\n=== Custom Encryption Example ===")
    custom_encryption_example()
