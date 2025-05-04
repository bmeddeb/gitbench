"""
Type stubs for authentication utilities.
"""

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import SecretStr
from pydantic.dataclasses import dataclass

from ..providers.base import ProviderType

@dataclass
class CredentialEntry:
    provider: ProviderType
    token: str
    username: Optional[str] = None
    host: Optional[str] = None
    
    @property
    def secret_token(self) -> SecretStr: ...
    
    def get_token(self) -> str: ...

def derive_key_from_password(
    password: str, salt: Optional[bytes] = None
) -> Tuple[bytes, bytes]: ...

class CredentialManager:
    credentials_file: str
    _creds_cache: Dict[ProviderType, List[CredentialEntry]]
    _encrypt_token: Callable[[str], str]
    _decrypt_token: Callable[[str], str]
    _DEFAULT_CREDS_FILE: str = ...

    def __init__(
        self,
        credentials_file: Optional[str] = None,
        encryption_key: Optional[bytes] = None,
        encrypt_func: Optional[Callable[[str], str]] = None,
        decrypt_func: Optional[Callable[[str], str]] = None,
    ) -> None: ...
    def _setup_aes_encryption(self, key: bytes) -> None: ...
    def _ensure_creds_dir(self) -> None: ...
    def _base64_encode(self, token: str) -> str: ...
    def _base64_decode(self, encoded_token: str) -> str: ...
    def _aes_encrypt_token(self, token: str) -> str: ...
    def _aes_decrypt_token(self, encoded_token: str) -> str: ...
    def save_credential(
        self,
        provider: ProviderType,
        token: str,
        username: Optional[str] = None,
        host: Optional[str] = None,
    ) -> None: ...
    def get_credentials(
        self, provider: ProviderType, host: Optional[str] = None
    ) -> List[CredentialEntry]: ...
    def remove_credential(self, provider: ProviderType, token: str) -> bool: ...
    def remove_credential_by_username(
        self, provider: ProviderType, username: str
    ) -> bool: ...
    def clear_credentials(self, provider: Optional[ProviderType] = None) -> None: ...
    @classmethod
    def from_password(
        cls,
        password: str,
        salt: Optional[bytes] = None,
        salt_file: Optional[str] = None,
        credentials_file: Optional[str] = None,
    ) -> "CredentialManager": ...
