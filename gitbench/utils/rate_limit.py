"""
Rate limiting utilities for API clients.
"""

import asyncio
import time
from datetime import datetime
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar, Union

T = TypeVar("T")


class RateLimitError(Exception):
    """Exception raised when a rate limit is exceeded."""

    def __init__(self, message: str, reset_time: Optional[int] = None):
        self.reset_time = reset_time
        self.message = message
        super().__init__(message)


class RateLimiter:
    """Rate limiter for API calls with backoff strategy."""

    def __init__(self, calls_per_second: float = 1.0, max_retries: int = 3):
        """Initialize the rate limiter.

        Args:
            calls_per_second: Maximum number of calls per second
            max_retries: Maximum number of retries when rate limited
        """
        self.min_interval = 1.0 / calls_per_second
        self.max_retries = max_retries
        self.last_call_time: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, key: str = "default") -> None:
        """Acquire permission to make a request.

        Args:
            key: Identifier for the rate limit bucket
        """
        async with self._lock:
            now = time.time()
            if key in self.last_call_time:
                elapsed = now - self.last_call_time[key]
                if elapsed < self.min_interval:
                    # Need to wait
                    wait_time = self.min_interval - elapsed
                    await asyncio.sleep(wait_time)

            # Record the time of this call
            self.last_call_time[key] = time.time()

    def reset(self, key: str = "default") -> None:
        """Reset the rate limiter for a specific key."""
        if key in self.last_call_time:
            del self.last_call_time[key]


def rate_limited(
    limiter: Optional[RateLimiter] = None,
    key: Optional[str] = None,
    retry_on_limit: bool = True,
):
    """Decorator for rate limiting an async function.

    Args:
        limiter: Rate limiter instance to use. If None, creates a default one.
        key: Key to use for rate limiting. If None, uses the function name.
        retry_on_limit: Whether to retry if rate limited.

    Returns:
        Decorated function
    """
    # Create default rate limiter if none provided
    _limiter = limiter or RateLimiter()

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # Use function name as key if none provided
            limiter_key = key or func.__name__

            # Try to run the function with rate limiting
            retries = 0
            while True:
                try:
                    # Wait for rate limit slot
                    await _limiter.acquire(limiter_key)

                    # Call the function
                    return await func(*args, **kwargs)

                except RateLimitError as e:
                    # If we have a reset time and should retry
                    if (
                        retry_on_limit
                        and retries < _limiter.max_retries
                        and e.reset_time
                    ):
                        # Calculate wait time (add a small buffer)
                        wait_time = max(0, e.reset_time - time.time()) + 1.0

                        # Wait before retry
                        if wait_time > 0:
                            retries += 1
                            await asyncio.sleep(wait_time)
                            continue

                    # Otherwise, re-raise the exception
                    raise

        return wrapper

    return decorator
