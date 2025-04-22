"""
Serialization utilities for the Prometheum framework.

This module provides tools for serializing and deserializing Prometheum components
such as DataFrameSchema, Pipeline, and transformers to/from JSON and YAML formats.
"""

import datetime
import json
import importlib
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, cast

import pandas as pd
import yaml
from pandas import DataFrame

from prometheum.core.base import (
    DataContainer, DataFrameContainer, DataProcessor, 
    DataTransformer, Pipeline, Serializable
)
from prometheum.core.exceptions import ConfigurationError
from prometheum.data.parser import (
    DataType, ColumnSchema, DataFrameSchema, 
    SchemaValidator, DataValidator
)
from prometheum.processing.pipeline import PipelineStep, PipelineBuilder
from prometheum.processing.transformers import (
    StandardScaler, MinMaxScaler, OneHotEncoder
)


class SerializationFormat(Enum):
    """Enumeration of supported serialization formats."""
    
    JSON = "json"
    YAML = "yaml"


class ComponentSerializer:
    """Class for serializing and deserializing Prometheum components."""
    
    @staticmethod
    def _serialize_enum(enum_obj: Enum) -> Dict[str, str]:
        """Serialize an Enum object.
        
        Args:
            enum_obj: The Enum object to serialize
            
        Returns:
            Dict[str, str]: Serialized representation
        """
        return {
            "__type__": f"{enum_obj.__class__.__module__}.{enum_obj.__class__.__name__}",
            "value": enum_obj.value
        }
    
    @staticmethod
    def _deserialize_enum(data: Dict[str, str]) -> Enum:
        """Deserialize data into an Enum object.
        
        Args:
            data: Serialized enum data
            
        Returns:
            Enum: Deserialized Enum object
            
        Raises:
            ConfigurationError: If deserialization fails
        """
        try:
            module_name, class_name = data["__type__"].rsplit(".", 1)
            module = importlib.import_module(module_name)
            enum_class = getattr(module, class_name)
            return enum_class(data["value"])
        except (ValueError, ImportError, AttributeError) as e:
            raise ConfigurationError(
                f"Failed to deserialize enum: {str(e)}",
                details={"serialized_data": data}
            ) from e
    
    @staticmethod
    def _serialize_datetime(dt: datetime.datetime) -> Dict[str, str]:
        """Serialize a datetime object.
        
        Args:
            dt: The datetime object to serialize
            
        Returns:
            Dict[str, str]: Serialized representation
        """
        return {
            "__type__": "datetime.datetime",
            "value": dt.isoformat()
        }
    
    @staticmethod
    def _deserialize_datetime(data: Dict[str, str]) -> datetime.datetime:
        """Deserialize data into a datetime object.
        
        Args:
            data: Serialized datetime data
            
        Returns:
            datetime.datetime: Deserialized datetime object
            
        Raises:
            ConfigurationError: If deserialization fails
        """
        try:
            return datetime.datetime.fromisoformat(data["value"])
        except ValueError as e:
            raise ConfigurationError(
                f"Failed to deserialize datetime: {str(e)}",
                details={"serialized_data": data}
            ) from e
    
    @staticmethod
    def _get_component_class(type_path: str) -> Type[Any]:
        """Get a component class from its module path.
        
        Args:
            type_path: The module.class path
            
        Returns:
            Type[Any]: The component class
            
        Raises:
            ConfigurationError: If class cannot be found
        """
        try:
            module_name, class_name = type_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ValueError, ImportError, AttributeError) as e:
            raise ConfigurationError(
                f"Failed to find component class {type_path}: {str(e)}"
            ) from e
    
    @classmethod
    def _serialize_object(cls, obj: Any) -> Dict[str, Any]:
        """Serialize a Prometheum component object.
        
        Args:
            obj: The object to serialize
            
        Returns:
            Dict[str, Any]: Serialized representation
            
        Raises:
            ConfigurationError: If serialization fails
        """
        # Handle basic types
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        
        # Handle Enum objects
        if isinstance(obj, Enum):
            return cls._serialize_enum(obj)
        
        # Handle datetime objects
        if isinstance(obj, datetime.datetime):
            return cls._serialize_datetime(obj)
        
        # Handle lists
        if isinstance(obj, list):
            return [cls._serialize_object(item) for item in obj]
        
        # Handle dictionaries
        if isinstance(obj, dict):
            return {key: cls._serialize_object(value) for key, value in obj.items()}
        
        # Handle DataFrame
        if isinstance(obj, pd.DataFrame):
            return {
                "__type__": "pandas.DataFrame",
                "value": obj.to_dict(orient="split")
            }
        
        # Handle objects with to_dict method (Serializable protocol)
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            data = obj.to_dict()
            data["__type__"] = f"{obj.__class__.__module__}.{obj.__class__.__name__}"
            return cls._serialize_object(data)
        
        # Handle other objects by serializing __dict__
        try:
            data = {
                "__type__": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                "attributes": {key: cls._serialize_object(value) for key, value in obj.__dict__.items() 
                              if not key.startswith("_")}
            }
            return data
        except Exception as e:
            raise ConfigurationError(
                f"Failed to serialize object {obj.__class__.__name__}: {str(e)}",
                details={"object_type": str(type(obj))}
            ) from e
    
    @classmethod
    def _deserialize_object(cls, data: Any) -> Any:
        """Deserialize data into a Prometheum component object.
        
        Args:
            data: Serialized data
            
        Returns:
            Any: Deserialized object
            
        Raises:
            ConfigurationError: If deserialization fails
        """
        # Handle basic types
        if data is None or isinstance(data, (str, int, float, bool)):
            return data
        
        # Handle lists
        if isinstance(data, list):
            return [cls._deserialize_object(item) for item in data]
        
        # Handle dictionaries with type information
        if isinstance(data, dict):
            if "__type__" in data:
                type_path = data["__type__"]
                
                # Handle special types
                if type_path == "datetime.datetime":
                    return cls._deserialize_datetime(data)
                
                if type_path == "pandas.DataFrame":
                    return pd.DataFrame(**data["value"])
                
                # Handle Enum types
                if "value" in data and len(data) == 2:
                    try:
                        component_class = cls._get_component_class(type_path)
                        if issubclass(component_class, Enum):
                            return cls._deserialize_enum(data)
                    except Exception:
                        pass
                
                # Handle component objects
                try:
                    component_class = cls._get_component_class(type_path)
                    
                    # Use from_dict if available
                    if hasattr(component_class, "from_dict") and callable(component_class.from_dict):
                        # Remove type info before passing to from_dict
                        obj_data = {k: v for k, v in data.items() if k != "__type__"}
                        return component_class.from_dict(obj_data)
                    
                    # Use attributes if available
                    if "attributes" in data:
                        attrs = {k: cls._deserialize_object(v) for k, v in data["attributes"].items()}
                        try:
                            # Try to create instance and set attributes
                            obj = component_class()
                            for key, value in attrs.items():
                                setattr(obj, key, value)
                            return obj
                        except Exception as e:
                            raise ConfigurationError(
                                f"Failed to instantiate {type_path}: {str(e)}",
                                details={"attributes": attrs}
                            ) from e
                except Exception as e:
                    raise ConfigurationError(
                        f"Failed to deserialize component {type_path}: {str(e)}",
                        details={"serialized_data": data}
                    ) from e
            
            # Regular dictionary without type info
            return {key: cls._deserialize_object(value) for key, value in data.items()}
        
        # Return as is for unhandled types
        return data

    @classmethod
    def serialize(cls, component: Any, format: SerializationFormat = SerializationFormat.JSON) -> str:
        """Serialize a Prometheum component to a string.
        
        Args:
            component: The component to serialize
            format: Output format (JSON or YAML)
            
        Returns:
            str: Serialized string
            
        Raises:
            ConfigurationError: If serialization fails
        """
        try:
            # Serialize component to a dictionary
            serialized = cls._serialize_object(component)
            
            # Convert to the requested format
            if format == SerializationFormat.JSON:
                return json.dumps(serialized, indent=2)
            elif format == SerializationFormat.YAML:
                return yaml.dump(serialized, sort_keys=False)
            else:
                raise ValueError(f"Unsupported serialization format: {format}")
        except Exception as e:
            raise ConfigurationError(
                f"Failed to serialize component: {str(e)}",
                details={"component_type": component.__class__.__name__}
            ) from e
    
    @classmethod
    def deserialize(cls, data: str, format: SerializationFormat = SerializationFormat.JSON) -> Any:
        """Deserialize a string into a Prometheum component.
        
        Args:
            data: Serialized string
            format: Input format (JSON or YAML)
            
        Returns:
            Any: Deserialized component
            
        Raises:
            ConfigurationError: If deserialization fails
        """
        try:
            # Parse the string
            if format == SerializationFormat.JSON:
                parsed = json.loads(data)
            elif format == SerializationFormat.YAML:
                parsed = yaml.safe_load(data)
            else:
                raise ValueError(f"Unsupported serialization format: {format}")
            
            # Deserialize to component
            return cls._deserialize_object(parsed)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to deserialize data: {str(e)}",
                details={"format": format.value}
            ) from e
    
    @classmethod
    def save_to_file(
        cls, 
        component: Any, 
        filepath: Union[str, Path], 
        format: Optional[SerializationFormat] = None
    ) -> None:
        """Save a component to a file.
        
        Args:
            component: The component to save
            filepath: Path to the output file
            format: Output format (inferred from file extension if None)
            
        Raises:
            ConfigurationError: If saving fails
        """
        filepath = Path(filepath)
        
        # Determine format from file extension if not specified
        if format is None:
            if filepath.suffix.lower() == ".json":
                format = SerializationFormat.JSON
            elif filepath.suffix.lower() in (".yml", ".yaml"):
                format = SerializationFormat.YAML
            else:
                format = SerializationFormat.JSON  # Default to JSON
        
        try:
            # Serialize the component
            serialized = cls.serialize(component, format)
            
            # Write to file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(serialized)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save component to {filepath}: {str(e)}",
                details={"filepath": str(filepath), "format": format.value}
            ) from e
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path], format: Optional[SerializationFormat] = None) -> Any:
        """Load a component from a file.
        
        Args:
            filepath: Path to the input file
            format: Input format (inferred from file extension if None)
            
        Returns:
            Any: The loaded component
            
        Raises:
            ConfigurationError: If loading fails
        """
        filepath = Path(filepath)
        
        # Determine format from file extension if not specified
        if format is None:
            if filepath.suffix.lower() == ".json":
                format = SerializationFormat.JSON
            elif filepath.suffix.lower() in (".yml", ".yaml"):
                format = SerializationFormat.YAML
            else:
                format = SerializationFormat.JSON  # Default to JSON
        
        try:
            # Read from file
            with open(filepath, "r", encoding="utf-8") as f:
                data = f.read()
            
            # Deserialize the data
            return cls.deserialize(data, format)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load component from {filepath}: {str(e)}",
                details={"filepath": str(filepath), "format": format.value}
            ) from e


# Implement serialization support for key components

def _implement_serializable(cls):
    """Decorator to add to_dict and from_dict methods to a class."""
    
    if not hasattr(cls, "to_dict") or not callable(cls.to_dict):
        def to_dict(self) -> Dict[str, Any]:
            """Convert object to a dictionary.
            
            Returns:
                Dict[str, Any]: Dictionary representation
            """
            result = {}
            for key, value in self.__dict__.items():
                if key.startswith("_"):
                    # Convert _name to name
                    clean_key = key[1:] if key.startswith("_") else key
                else:
                    clean_key = key
                    
                # Skip internal attributes and methods
                if callable(value) or key in ("__dict__", "__weakref__"):
                    continue
                
                # Add to result
                result[clean_key] = value
            
            return result
        
        cls.to_dict = to_dict
    
    if not hasattr(cls, "from_dict") or not callable(cls.from_dict):
        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> Any:
            """Create an instance from a dictionary.
            
            Args:
                data: Dictionary representation
                
            Returns:
                Any: Instantiated object
            """
            # Create a new instance
            instance = cls.__new__(cls)
            
            # Initialize basic attributes
            for key, value in data.items():
                setattr(instance, key, value)
            
            return instance
        
        cls.from_dict = from_dict
    
    return cls


# Apply serialization to key classes
_implement_serializable(ColumnSchema)
_implement_serializable(DataFrameSchema)
_implement_serializable(SchemaValidator)
_implement_serializable(PipelineStep)
_implement_serializable(PipelineBuilder)
