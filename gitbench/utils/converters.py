"""
Data conversion utilities for working with provider API data.
"""

import datetime
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

try:
    from pydantic import BaseModel
except ImportError:
    # Define a dummy BaseModel class if pydantic is not installed
    class BaseModel:
        pass

T = TypeVar("T")


def to_dict(obj: Any) -> Dict[str, Any]:
    """Convert an object to a dictionary.

    Handles Pydantic models, dataclasses, objects with __dict__, and primitive types.

    Args:
        obj: The object to convert

    Returns:
        Dictionary representation of the object
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    elif is_dataclass(obj):
        return asdict(obj)
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        return obj


def to_json(obj: Any, indent: Optional[int] = None) -> str:
    """Convert an object to a JSON string.

    Args:
        obj: The object to convert
        indent: Optional indentation level

    Returns:
        JSON string representation of the object
    """
    # Use Pydantic's built-in JSON serialization for Pydantic models
    if isinstance(obj, BaseModel):
        return obj.model_dump_json(indent=indent)

    def default_serializer(o: Any) -> Any:
        if isinstance(o, datetime.datetime):
            return o.isoformat()
        elif isinstance(o, datetime.date):
            return o.isoformat()
        elif isinstance(o, BaseModel):
            return o.model_dump()
        elif is_dataclass(o):
            return asdict(o)
        elif hasattr(o, "__dict__"):
            return o.__dict__
        else:
            return str(o)

    return json.dumps(obj, default=default_serializer, indent=indent)


def to_dataframe(
    data: Union[List[Dict[str, Any]], Dict[str, Any], List[Any], BaseModel, List[BaseModel]],
) -> "pandas.DataFrame":
    """Convert data to a pandas DataFrame.

    Args:
        data: The data to convert. Can be Pydantic models, dataclasses, or other types.

    Returns:
        pandas DataFrame

    Raises:
        ImportError: If pandas is not installed
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for this functionality. "
            "Install it with 'pip install pandas'."
        )

    # Handle single object case
    if not isinstance(data, list):
        if isinstance(data, BaseModel):
            data = [data.model_dump()]
        else:
            data = [data]
    else:
        # Handle list of Pydantic models
        if data and isinstance(data[0], BaseModel):
            data = [item.model_dump() for item in data]

    # Convert dataclass instances to dicts
    converted_data = []
    for item in data:
        if isinstance(item, BaseModel):
            converted_data.append(item.model_dump())
        elif is_dataclass(item):
            converted_data.append(asdict(item))
        elif hasattr(item, "__dict__"):
            converted_data.append(item.__dict__)
        else:
            converted_data.append(item)

    return pd.DataFrame(converted_data)


def flatten_dataframe(
    df: "pandas.DataFrame", separator: str = "_"
) -> "pandas.DataFrame":
    """Flatten nested columns in a DataFrame.

    Args:
        df: The DataFrame to flatten
        separator: Separator to use between levels

    Returns:
        Flattened DataFrame
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for this functionality. "
            "Install it with 'pip install pandas'."
        )

    # Function to flatten a dictionary
    def flatten_dict(
        d: Dict[str, Any], parent_key: str = "", sep: str = separator
    ) -> Dict[str, Any]:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # Apply flattening to each row
    flattened_data = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        flattened_data.append(flatten_dict(row_dict))

    return pd.DataFrame(flattened_data)
