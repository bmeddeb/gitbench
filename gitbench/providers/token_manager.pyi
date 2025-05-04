"""
Type stubs for token management system.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple, Callable

from pydantic import Field, SecretStr
from pydantic.dataclasses import dataclass

from .base import ProviderType

@dataclass
class TokenStatus:
    is_valid: bool
    remaining_calls: int
    reset_time: Optional[int] = None
    last_used: Optional[float] = None

    @property
    def is_rate_limited(self) -> bool: ...
    @property
    def is_available(self) -> bool: ...

@dataclass
class TokenInfo:
    token: str
    provider: ProviderType
    username: Optional[str] = None
    status: Optional[TokenStatus] = None

    @property
    def secret_token(self) -> SecretStr: ...
    
    def __post_init__(self) -> None: ...

class TokenManager:
    tokens: Dict[ProviderType, List[TokenInfo]]
    current_indices: Dict[ProviderType, int]
    _lock: asyncio.Lock

    def __init__(self) -> None: ...
    
    def add_token(
        self, token: str, provider: ProviderType, username: Optional[str] = None
    ) -> None: ...
    
    async def get_next_available_token(
        self, provider: ProviderType
    ) -> Optional[TokenInfo]: ...
    
    async def update_rate_limit(
        self, token: str, provider: ProviderType, remaining: int, reset_time: int
    ) -> None: ...
    
    async def mark_token_invalid(self, token: str, provider: ProviderType) -> None: ...
    
    def get_all_tokens(self, provider: ProviderType) -> List[TokenInfo]: ...
    
    def count_available_tokens(self, provider: ProviderType) -> int: ...
