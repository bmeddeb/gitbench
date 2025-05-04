"""
Authentication utilities for Git provider API clients.

This module provides secure token management for Git provider APIs, with flexible 
encryption options.
"""

import base64
import hashlib
import json
import os
import secrets
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Import cryptography only if used to avoid hard dependency
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
except ImportError:
    AESGCM = None
    PBKDF2HMAC = None
    hashes = None

from gitbench.providers.base import ProviderType


from pydantic import SecretStr
from pydantic.dataclasses import dataclass

@dataclass
class CredentialEntry:
    """Represents a stored credential for a Git provider."""

    provider: ProviderType
    token: str  # Still stored as a string for compatibility
    username: Optional[str] = None
    host: Optional[str] = None
    
    @property
    def secret_token(self) -> SecretStr:
        """Get the token as a SecretStr for secure handling."""
        return SecretStr(self.token)
        
    def get_token(self) -> str:
        """Get the raw token string (for compatibility)."""
        return self.token


def derive_key_from_password(password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    """Derive a secure encryption key from a password using PBKDF2.
    
    Args:
        password: The password to derive the key from
        salt: Optional salt bytes. If None, generates a random salt.
        
    Returns:
        Tuple of (key, salt)
        
    Raises:
        ImportError: If cryptography package is not installed
    """
    if PBKDF2HMAC is None or hashes is None:
        # Fallback to a basic derivation if cryptography is not available
        if salt is None:
            salt = os.urandom(16)
        
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000, dklen=32)
        return key, salt
    
    # Use cryptography's PBKDF2HMAC if available
    if salt is None:
        salt = os.urandom(16)
        
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,  # 32 bytes = 256 bits
        salt=salt,
        iterations=100000,
    )
    
    key = kdf.derive(password.encode())
    return key, salt


class CredentialManager:
    """Manages secure storage and retrieval of Git provider credentials.
    
    This class provides secure token storage with flexible encryption options.
    By default, it uses AES-GCM encryption if the cryptography package is installed.
    Users can provide their own encryption key or custom encryption/decryption functions.
    
    Security Recommendations:
    - Use a strong encryption key (32 bytes for AES-256-GCM)
    - Store the encryption key separately from the credentials
    - Consider using a key management service for production use
    - For maximum security, implement custom encryption using a hardware security module
    
    Examples:
        # Basic usage with base64 encoding (not secure for sensitive tokens)
        manager = CredentialManager()
        
        # Using AES-GCM encryption with a provided key
        import os
        key = os.urandom(32)  # Generate a random 32-byte key
        manager = CredentialManager(encryption_key=key)
        
        # Using a password-derived key
        key, salt = derive_key_from_password("my-secure-password")
        manager = CredentialManager(encryption_key=key)
        
        # Using custom encryption functions
        def custom_encrypt(token: str) -> str:
            # Your custom encryption logic here
            return encrypted_token
            
        def custom_decrypt(encrypted_token: str) -> str:
            # Your custom decryption logic here
            return token
            
        manager = CredentialManager(
            encrypt_func=custom_encrypt,
            decrypt_func=custom_decrypt
        )
    """

    _DEFAULT_CREDS_FILE = "~/.gitbench/credentials.json"

    def __init__(
        self,
        credentials_file: Optional[str] = None,
        encryption_key: Optional[bytes] = None,
        encrypt_func: Optional[Callable[[str], str]] = None,
        decrypt_func: Optional[Callable[[str], str]] = None,
    ):
        """Initialize the credential manager with flexible encryption options.

        Args:
            credentials_file: Path to the credentials file. If None, uses the default location.
            encryption_key: Optional key for AES-GCM encryption (should be 32 bytes for AES-256).
                If None and no custom functions are provided, a warning is issued and base64 
                encoding is used (not secure).
            encrypt_func: Optional custom function to encrypt tokens.
            decrypt_func: Optional custom function to decrypt tokens.
            
        Raises:
            ValueError: If both encryption_key and custom functions are provided,
                       or if only one custom function is provided but not both.
            ImportError: If encryption_key is provided but cryptography package is not installed.
        """
        self.credentials_file = credentials_file or os.path.expanduser(
            self._DEFAULT_CREDS_FILE
        )
        self._creds_cache: Dict[ProviderType, List[CredentialEntry]] = {}
        self._ensure_creds_dir()
        
        # Setup encryption
        if encryption_key is not None and (encrypt_func is not None or decrypt_func is not None):
            raise ValueError("Cannot provide both encryption_key and custom encryption functions")
        
        # Validate that both custom functions are provided if any
        if (encrypt_func is not None and decrypt_func is None) or \
           (encrypt_func is None and decrypt_func is not None):
            raise ValueError("Must provide both encrypt_func and decrypt_func if using custom encryption")
            
        if encrypt_func is not None and decrypt_func is not None:
            # Use custom encryption functions
            self._encrypt_token = encrypt_func
            self._decrypt_token = decrypt_func
        elif encryption_key is not None:
            # Use AES-GCM encryption
            if AESGCM is None:
                raise ImportError(
                    "The 'cryptography' package is required for AES-GCM encryption. "
                    "Install it with 'pip install cryptography'."
                )
            self._setup_aes_encryption(encryption_key)
        else:
            # Fallback to base64 encoding with warning
            warnings.warn(
                "No encryption key or custom functions provided. "
                "Tokens will only be encoded with base64, which is NOT secure. "
                "Consider providing an encryption_key or custom encryption functions.",
                UserWarning,
                stacklevel=2,
            )
            self._encrypt_token = self._base64_encode
            self._decrypt_token = self._base64_decode
    
    def _setup_aes_encryption(self, key: bytes) -> None:
        """Setup AES-GCM encryption with the provided key."""
        # Ensure key is the right length (32 bytes for AES-256)
        if len(key) not in (16, 24, 32):
            warnings.warn(
                f"AES key length is {len(key)} bytes. Recommended lengths are 16, 24, or 32 bytes.",
                UserWarning,
                stacklevel=3,
            )
        
        self._aesgcm = AESGCM(key)
        self._encrypt_token = self._aes_encrypt_token
        self._decrypt_token = self._aes_decrypt_token
    
    def _ensure_creds_dir(self) -> None:
        """Ensure the credentials directory exists."""
        creds_dir = os.path.dirname(self.credentials_file)
        os.makedirs(os.path.expanduser(creds_dir), exist_ok=True)
    
    def _base64_encode(self, token: str) -> str:
        """Simple encoding of token (not secure, just to prevent casual viewing)."""
        return base64.b64encode(token.encode()).decode()
    
    def _base64_decode(self, encoded_token: str) -> str:
        """Decode a base64-encoded token."""
        return base64.b64decode(encoded_token.encode()).decode()
    
    def _aes_encrypt_token(self, token: str) -> str:
        """Encrypt a token using AES-GCM."""
        # Generate a random 12-byte nonce
        nonce = secrets.token_bytes(12)
        # Encrypt the token
        ciphertext = self._aesgcm.encrypt(nonce, token.encode(), None)
        # Combine nonce and ciphertext and encode as base64
        return base64.b64encode(nonce + ciphertext).decode()
    
    def _aes_decrypt_token(self, encoded_token: str) -> str:
        """Decrypt a token using AES-GCM."""
        # Decode the base64
        data = base64.b64decode(encoded_token.encode())
        # Extract nonce (first 12 bytes) and ciphertext
        nonce, ciphertext = data[:12], data[12:]
        # Decrypt the token
        plaintext = self._aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode()

    def save_credential(
        self,
        provider: ProviderType,
        token: str,
        username: Optional[str] = None,
        host: Optional[str] = None,
    ) -> None:
        """Save a credential for a provider.

        Args:
            provider: The provider type
            token: The API token
            username: Optional username associated with the token
            host: Optional custom hostname (for enterprise instances)
            
        Note:
            Tokens are encrypted before being stored. The encryption method
            depends on the options provided when initializing the CredentialManager.
        """
        # Load existing credentials
        try:
            with open(os.path.expanduser(self.credentials_file), "r") as f:
                creds_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            creds_data = {}

        # Initialize provider list if needed
        if provider.value not in creds_data:
            creds_data[provider.value] = []

        # Add the new credential
        creds_data[provider.value].append(
            {"token": self._encrypt_token(token), "username": username, "host": host}
        )

        # Save updated credentials
        with open(os.path.expanduser(self.credentials_file), "w") as f:
            json.dump(creds_data, f, indent=2)

        # Clear cache
        self._creds_cache = {}

    def get_credentials(
        self, provider: ProviderType, host: Optional[str] = None
    ) -> List[CredentialEntry]:
        """Get all credentials for a provider.

        Args:
            provider: The provider type
            host: Optional host to filter by (for enterprise instances)

        Returns:
            List of credential entries with decrypted tokens
            
        Note:
            Tokens are decrypted when retrieved. The decryption method
            depends on the options provided when initializing the CredentialManager.
        """
        # Check cache first
        cache_key = (provider, host) if host else provider
        if provider in self._creds_cache:
            return self._creds_cache[provider]

        # Load credentials from file
        try:
            with open(os.path.expanduser(self.credentials_file), "r") as f:
                creds_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

        # Get credentials for the provider
        provider_creds = creds_data.get(provider.value, [])

        # Filter by host if specified
        if host:
            provider_creds = [
                cred
                for cred in provider_creds
                if cred.get("host") == host or cred.get("host") is None
            ]

        # Convert to CredentialEntry objects
        result = []
        for cred in provider_creds:
            try:
                decrypted_token = self._decrypt_token(cred["token"])
                result.append(
                    CredentialEntry(
                        provider=provider,
                        token=decrypted_token,
                        username=cred.get("username"),
                        host=cred.get("host"),
                    )
                )
            except Exception as e:
                warnings.warn(
                    f"Failed to decrypt token: {e}. This may happen if the encryption key "
                    f"or method has changed since the token was stored.",
                    UserWarning,
                    stacklevel=2,
                )

        # Update cache
        self._creds_cache[provider] = result
        return result

    def remove_credential(self, provider: ProviderType, token: str) -> bool:
        """Remove a credential.

        Args:
            provider: The provider type
            token: The token to remove (unencrypted form)

        Returns:
            True if the credential was removed, False if not found
            
        Note:
            This method requires comparing the encrypted form of the token with
            stored encrypted tokens, which may not work if the encryption is 
            non-deterministic (like AES-GCM with random nonce). In such cases,
            it's better to remove credentials by username or get all credentials
            and remove the specific one by its index.
        """
        # Load credentials
        try:
            with open(os.path.expanduser(self.credentials_file), "r") as f:
                creds_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return False

        # Check if provider exists
        if provider.value not in creds_data:
            return False

        # For non-deterministic encryption (like AES-GCM with random nonce),
        # we need to decrypt all tokens and compare with the provided token
        provider_creds = creds_data[provider.value]
        original_length = len(provider_creds)
        to_remove = []
        
        # First pass: identify credentials to remove by decrypting and comparing
        for i, cred in enumerate(provider_creds):
            try:
                decrypted = self._decrypt_token(cred["token"])
                if decrypted == token:
                    to_remove.append(i)
            except Exception:
                # Skip credentials that can't be decrypted
                pass
                
        # Second pass: remove identified credentials (in reverse order to avoid index issues)
        for i in sorted(to_remove, reverse=True):
            provider_creds.pop(i)
            
        # Update the data
        creds_data[provider.value] = provider_creds

        # Check if anything was removed
        if len(provider_creds) == original_length:
            return False

        # Save updated credentials
        with open(os.path.expanduser(self.credentials_file), "w") as f:
            json.dump(creds_data, f, indent=2)

        # Clear cache
        self._creds_cache = {}
        return True
        
    def remove_credential_by_username(self, provider: ProviderType, username: str) -> bool:
        """Remove all credentials for a specific username.

        Args:
            provider: The provider type
            username: The username to remove

        Returns:
            True if any credentials were removed, False if none found
        """
        # Load credentials
        try:
            with open(os.path.expanduser(self.credentials_file), "r") as f:
                creds_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return False

        # Check if provider exists
        if provider.value not in creds_data:
            return False

        # Find and remove credentials by username
        provider_creds = creds_data[provider.value]
        original_length = len(provider_creds)

        creds_data[provider.value] = [
            cred for cred in provider_creds if cred.get("username") != username
        ]

        # Check if anything was removed
        if len(creds_data[provider.value]) == original_length:
            return False

        # Save updated credentials
        with open(os.path.expanduser(self.credentials_file), "w") as f:
            json.dump(creds_data, f, indent=2)

        # Clear cache
        self._creds_cache = {}
        return True

    def clear_credentials(self, provider: Optional[ProviderType] = None) -> None:
        """Clear all credentials or just for a specific provider.

        Args:
            provider: Optional provider to clear credentials for. If None, clears all.
        """
        try:
            with open(os.path.expanduser(self.credentials_file), "r") as f:
                creds_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return

        if provider:
            # Clear just this provider
            if provider.value in creds_data:
                creds_data[provider.value] = []
        else:
            # Clear all providers
            creds_data = {}

        # Save updated credentials
        with open(os.path.expanduser(self.credentials_file), "w") as f:
            json.dump(creds_data, f, indent=2)

        # Clear cache
        self._creds_cache = {}
        
    @classmethod
    def from_password(
        cls, 
        password: str, 
        salt: Optional[bytes] = None,
        salt_file: Optional[str] = None,
        credentials_file: Optional[str] = None
    ) -> "CredentialManager":
        """Create a CredentialManager with a key derived from a password.
        
        This is a convenience method that derives an encryption key from a password
        using PBKDF2 and creates a CredentialManager with that key.
        
        Args:
            password: The password to derive the key from
            salt: Optional salt bytes. If None, reads from salt_file or generates a random salt.
            salt_file: Optional path to a file containing the salt. If the file doesn't exist
                       and salt is None, a new salt is generated and saved to this file.
            credentials_file: Optional path to the credentials file.
            
        Returns:
            CredentialManager instance with password-derived encryption
            
        Raises:
            ImportError: If cryptography package is not installed
        """
        # First, try to load salt from file if specified
        if salt is None and salt_file is not None:
            salt_file_path = os.path.expanduser(salt_file)
            try:
                with open(salt_file_path, 'rb') as f:
                    salt = f.read()
            except FileNotFoundError:
                # Salt file doesn't exist, we'll create it below
                pass
                
        # Derive key from password
        key, salt = derive_key_from_password(password, salt)
        
        # Save salt to file if specified
        if salt_file is not None:
            salt_file_path = os.path.expanduser(salt_file)
            os.makedirs(os.path.dirname(salt_file_path), exist_ok=True)
            with open(salt_file_path, 'wb') as f:
                f.write(salt)
                
        # Create CredentialManager with derived key
        return cls(credentials_file=credentials_file, encryption_key=key)
