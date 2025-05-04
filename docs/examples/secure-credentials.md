# Secure Credentials Example

This example demonstrates how to use GitFleet's `CredentialManager` to securely store and retrieve authentication tokens and credentials. It covers different encryption methods, from simple base64 encoding to AES encryption and password-derived keys.

## Code Example

```python
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

from GitFleet.providers.base import ProviderType
from GitFleet.utils.auth import CredentialManager, derive_key_from_password


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
    salt_file = "~/.gitfleet/salt.bin"

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
```

## Key Features Demonstrated

This example demonstrates several secure credential management approaches:

1. **Basic Usage**: Simple base64 encoding for token storage (minimal security)
2. **AES Encryption**: Using AES-GCM encryption with a randomly generated key
3. **Password-Based Encryption**: Deriving an encryption key from a password using PBKDF2
4. **Custom Encryption**: Implementing your own encryption and decryption functions
5. **Multi-Provider Support**: Storing and retrieving credentials for different providers

## Security Approaches

### Base64 Encoding (Minimal Security)

The most basic approach uses simple base64 encoding:

```python
manager = CredentialManager()  # No encryption key provided
```

This provides minimal security (obfuscation only) and is not recommended for production use.

### AES-GCM Encryption (Recommended)

For strong security, use AES-GCM encryption with a randomly generated key:

```python
# Generate a secure random key
key = os.urandom(32)  # 256 bits

# Create manager with AES encryption
manager = CredentialManager(encryption_key=key)
```

The challenge with this approach is securely storing the encryption key.

### Password-Based Key Derivation (User-Friendly)

For user-friendly security, derive an encryption key from a password:

```python
# Create a manager using a password
manager = CredentialManager.from_password(
    password="user_password", 
    salt_file="~/.gitfleet/salt.bin"
)
```

This approach uses PBKDF2 with a salt to derive a secure encryption key.

### Custom Encryption (Advanced)

For specialized needs, you can provide custom encryption and decryption functions:

```python
manager = CredentialManager(
    encrypt_func=my_encrypt_function,
    decrypt_func=my_decrypt_function
)
```

This allows integration with hardware security modules or other encryption systems.

## Running the Example

To run this example:

1. Install GitFleet with crypto support:
   ```bash
   pip install "gitfleet[crypto]"
   ```

2. Run the example:
   ```bash
   python examples/secure_credentials.py
   ```

## Implementation Details

### CredentialManager Class

The `CredentialManager` class provides the following methods:

- **save_credential**: Store a credential for a specific provider
- **get_credentials**: Retrieve all credentials for a provider
- **clear_credentials**: Remove all stored credentials
- **from_password**: Class method to create a manager with password-based encryption

### Provider Types

GitFleet supports different credential provider types:

```python
from GitFleet.providers.base import ProviderType

# Available provider types
ProviderType.GITHUB    # GitHub credentials
ProviderType.GITLAB    # GitLab credentials
ProviderType.BITBUCKET # Bitbucket credentials
```

### Encryption Methods

The encryption methods used in GitFleet are:

1. **Base64**: Simple encoding with no cryptographic security
2. **AES-GCM**: Authenticated encryption with associated data (AEAD)
3. **PBKDF2**: Password-based key derivation for generating encryption keys

## Security Best Practices

When using `CredentialManager` in production:

1. **Never hardcode passwords or keys** in your application code
2. **Use environment variables** or secure credential stores for sensitive information
3. **Always use AES encryption** or stronger when handling authentication tokens
4. **Set appropriate file permissions** for salt and credential storage files
5. **Consider using a vault service** like HashiCorp Vault for enterprise applications

## Related Examples

- [Token Manager](token-manager.md): Multi-token management for API rate limiting
- [GitHub Client](github-client.md): Using authentication with the GitHub API client