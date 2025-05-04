"""
Type stubs for rate limiting utilities.
"""

import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar, Union

T = TypeVar("T")

class RateLimitError(Exception):
    reset_time: Optional[int]
    message: str

    def __init__(self, message: str, reset_time: Optional[int] = None) -> None: ...

class RateLimiter:
    min_interval: float
    max_retries: int
    last_call_time: Dict[str, float]
    _lock: asyncio.Lock

    def __init__(self, calls_per_second: float = 1.0, max_retries: int = 3) -> None: ...
    async def acquire(self, key: str = "default") -> None: ...
    def reset(self, key: str = "default") -> None: ...

def rate_limited(
    limiter: Optional[RateLimiter] = None,
    key: Optional[str] = None,
    retry_on_limit: bool = True,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]: ...
