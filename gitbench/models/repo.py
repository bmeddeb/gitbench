"""
Repository management models and utilities for working with the Rust-based RepoManager.

This module provides:
1. Pydantic models that mirror the PyO3-generated classes from the Rust implementation
2. Utility functions to convert between Rust objects and Pydantic models
3. Type definitions for better IDE integration

The Pydantic models provide serialization, validation, and conversion capabilities.
"""

from typing import Dict, List, Optional, Any, Union, TypeVar, Type, cast
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

# Check if Rust implementation is available
try:
    # These imports are for type checking only
    # The actual imports are done in __init__.py
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from gitbench.gitbench import CloneStatus as RustCloneStatus
        from gitbench.gitbench import CloneTask as RustCloneTask
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


class CloneStatusType(str, Enum):
    """Enumeration of possible clone status types."""
    QUEUED = "queued"
    CLONING = "cloning"
    COMPLETED = "completed"
    FAILED = "failed"


class PydanticCloneStatus(BaseModel):
    """Pydantic model for clone status information.
    
    This model provides a Pydantic representation of the Rust-generated CloneStatus class,
    adding serialization, validation, and other Pydantic features.
    
    Attributes:
        status_type: The type of status (queued, cloning, completed, failed)
        progress: The percentage of completion (0-100) if cloning, or None
        error: An error message if failed, or None
    """
    status_type: CloneStatusType
    progress: Optional[int] = None
    error: Optional[str] = None
    
    model_config = ConfigDict(
        frozen=True,
        extra='ignore',
    )


class PydanticCloneTask(BaseModel):
    """Pydantic model for clone task information.
    
    This model provides a Pydantic representation of the Rust-generated CloneTask class,
    adding serialization, validation, and other Pydantic features.
    
    Attributes:
        url: The URL of the repository being cloned
        status: The current status of the cloning operation (PydanticCloneStatus)
        temp_dir: The path to the temporary directory where the repository
                 was cloned, or None if cloning has not completed or failed
    """
    url: str
    status: PydanticCloneStatus
    temp_dir: Optional[str] = None
    
    model_config = ConfigDict(
        frozen=True,
        extra='ignore',
    )


# Conversion functions for Rust to Pydantic models
def to_pydantic_status(rust_status: 'RustCloneStatus') -> PydanticCloneStatus:
    """Convert a Rust CloneStatus to a Pydantic model.
    
    Args:
        rust_status: The Rust CloneStatus object from the gitbench extension
        
    Returns:
        A PydanticCloneStatus object with the same data
        
    Raises:
        ImportError: If the Rust implementation is not available
    """
    if not RUST_AVAILABLE:
        raise ImportError("Rust implementation not available")
    
    return PydanticCloneStatus(
        status_type=rust_status.status_type,
        progress=rust_status.progress,
        error=rust_status.error,
    )


def to_pydantic_task(rust_task: 'RustCloneTask') -> PydanticCloneTask:
    """Convert a Rust CloneTask to a Pydantic model.
    
    Args:
        rust_task: The Rust CloneTask object from the gitbench extension
        
    Returns:
        A PydanticCloneTask object with the same data
        
    Raises:
        ImportError: If the Rust implementation is not available
    """
    if not RUST_AVAILABLE:
        raise ImportError("Rust implementation not available")
    
    return PydanticCloneTask(
        url=rust_task.url,
        status=to_pydantic_status(rust_task.status),
        temp_dir=rust_task.temp_dir,
    )


def convert_clone_tasks(rust_tasks: Dict[str, 'RustCloneTask']) -> Dict[str, PydanticCloneTask]:
    """Convert a dictionary of Rust CloneTasks to Pydantic models.
    
    This is a convenience function for converting the result of RepoManager.fetch_clone_tasks()
    
    Args:
        rust_tasks: Dictionary mapping URLs to Rust CloneTask objects
        
    Returns:
        Dictionary mapping URLs to PydanticCloneTask objects
    """
    return {url: to_pydantic_task(task) for url, task in rust_tasks.items()}