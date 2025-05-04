"""Type stubs for repository management models and utilities."""

from typing import Dict, List, Optional, Any, Union, ClassVar, Type, TypeVar, Protocol
from enum import Enum

from pydantic import BaseModel, ConfigDict

T = TypeVar('T')

# These are used for type checking only
# The actual classes are defined in the Rust extension
class RustCloneStatus(Protocol):
    """Protocol for Rust-generated CloneStatus class."""
    status_type: str
    progress: Optional[int]
    error: Optional[str]

class RustCloneTask(Protocol):
    """Protocol for Rust-generated CloneTask class."""
    url: str
    status: RustCloneStatus
    temp_dir: Optional[str]

class CloneStatusType(str, Enum):
    """Enumeration of possible clone status types."""
    QUEUED: str
    CLONING: str
    COMPLETED: str
    FAILED: str

class PydanticCloneStatus(BaseModel):
    """Pydantic model for clone status information."""
    status_type: CloneStatusType
    progress: Optional[int]
    error: Optional[str]
    
    model_config: ClassVar[ConfigDict]
    
    def model_dump(self) -> Dict[str, Any]: ...
    def model_dump_json(self, indent: Optional[int] = None) -> str: ...
    @classmethod
    def model_validate(cls: Type[T], obj: Any) -> T: ...

class PydanticCloneTask(BaseModel):
    """Pydantic model for clone task information."""
    url: str
    status: PydanticCloneStatus
    temp_dir: Optional[str]
    
    model_config: ClassVar[ConfigDict]
    
    def model_dump(self) -> Dict[str, Any]: ...
    def model_dump_json(self, indent: Optional[int] = None) -> str: ...
    @classmethod
    def model_validate(cls: Type[T], obj: Any) -> T: ...

# Conversion functions
def to_pydantic_status(rust_status: RustCloneStatus) -> PydanticCloneStatus: ...
def to_pydantic_task(rust_task: RustCloneTask) -> PydanticCloneTask: ...
def convert_clone_tasks(rust_tasks: Dict[str, RustCloneTask]) -> Dict[str, PydanticCloneTask]: ...