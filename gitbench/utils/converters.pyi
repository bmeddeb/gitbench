"""
Type stubs for data conversion utilities.
"""

import datetime
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, TypeVar, Type, Union, cast

try:
    from pydantic import BaseModel
except ImportError:
    class BaseModel: ...

T = TypeVar("T")

def to_dict(obj: Any) -> Dict[str, Any]: ...

def to_json(obj: Any, indent: Optional[int] = None) -> str: ...

def to_dataframe(
    data: Union[List[Dict[str, Any]], Dict[str, Any], List[Any], BaseModel, List[BaseModel]],
) -> "pandas.DataFrame": ...

def flatten_dataframe(
    df: "pandas.DataFrame", separator: str = "_"
) -> "pandas.DataFrame": ...
