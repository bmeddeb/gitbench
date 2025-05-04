"""
Utility functions for gitbench.
"""

from gitbench.utils.auth import CredentialEntry, CredentialManager
from gitbench.utils.converters import (flatten_dataframe, to_dataframe,
                                      to_dict, to_json)
from gitbench.utils.rate_limit import RateLimiter, RateLimitError, rate_limited

__all__ = [
    # Authentication utilities
    "CredentialManager",
    "CredentialEntry",
    # Rate limiting utilities
    "RateLimiter",
    "rate_limited",
    "RateLimitError",
    # Data conversion utilities
    "to_dict",
    "to_json",
    "to_dataframe",
    "flatten_dataframe",
]
