"""
Core base classes for the Prometheum data processing framework.

This module contains the abstract base classes that define the core interfaces
for the framework components such as data loaders, transformers, and processors.
These serve as the foundation for creating concrete implementations.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Generic, Iterator, List, Optional, Protocol, TypeVar, Union
import uuid

import pandas as pd
import numpy as np

from prometheum.core.exceptions import ValidationError

# Type variables for generic typing
T = TypeVar('T')  # Input data type
U = TypeVar('U')  # Output data type
DataT = TypeVar('DataT')  # Generic data container type


class Identifiable(abc.ABC):
    """Base class for objects that need a unique identifier."""
    
    def __init__(self) -> None:
        """Initialize with a unique identifier."""
        self._id = str(uuid.uuid4())
    
    @property
    def id(self) -> str:
        """Get the unique identifier for this object.
        
        Returns:
            str: The unique identifier
        """
        return self._id


class Configurable(abc.ABC):
    """Base class for components that are configurable."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize with configuration parameters.
        
        Args:
            config: Optional configuration dictionary
        """
        self._config = config or {}
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the current configuration.
        
        Returns:
            Dict[str, Any]: The current configuration dictionary
        """
        return self._config
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update the configuration with new parameters.
        
        Args:
            new_config: Dictionary containing new configuration parameters
        """
        self._config.update(new_config)
    
    def validate_config(self) -> bool:
        """Validate the current configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
            
        Raises:
            ValidationError: If configuration is invalid
        """
        return True


class DataContainer(Generic[DataT]):
    """Base class for data containers that hold processed or raw data."""
    
    def __init__(self, data: DataT, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the data container.
        
        Args:
            data: The data to be stored
            metadata: Optional metadata about the data
        """
        self._data = data
        self._metadata = metadata or {}
    
    @property
    def data(self) -> DataT:
        """Get the stored data.
        
        Returns:
            DataT: The stored data
        """
        return self._data
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the metadata.
        
        Returns:
            Dict[str, Any]: The metadata dictionary
        """
        return self._metadata
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add a new metadata entry.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self._metadata[key] = value


class DataFrameContainer(DataContainer[pd.DataFrame]):
    """Container specifically for pandas DataFrames."""
    
    def __init__(self, data: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Initialize with a pandas DataFrame.
        
        Args:
            data: The DataFrame to store
            metadata: Optional metadata about the DataFrame
        """
        super().__init__(data, metadata)
    
    def shape(self) -> tuple:
        """Get the shape of the DataFrame.
        
        Returns:
            tuple: The shape as (rows, columns)
        """
        return self._data.shape
    
    def head(self, n: int = 5) -> pd.DataFrame:
        """Get the first n rows of the DataFrame.
        
        Args:
            n: Number of rows to retrieve
            
        Returns:
            pd.DataFrame: The first n rows
        """
        return self._data.head(n)


class DataLoader(Identifiable, Configurable, Generic[T]):
    """Abstract base class for data loaders."""
    
    @abc.abstractmethod
    def load(self) -> T:
        """Load data from the source.
        
        Returns:
            T: The loaded data
            
        Raises:
            DataLoadError: If data cannot be loaded
        """
        pass
    
    @abc.abstractmethod
    def validate(self, data: T) -> bool:
        """Validate the loaded data.
        
        Args:
            data: The data to validate
            
        Returns:
            bool: True if valid, False otherwise
            
        Raises:
            ValidationError: If validation fails with specific errors
        """
        pass


class DataProcessor(Identifiable, Configurable, Generic[T, U]):
    """Abstract base class for data processors."""
    
    @abc.abstractmethod
    def process(self, data: T) -> U:
        """Process the input data.
        
        Args:
            data: Input data to process
            
        Returns:
            U: Processed output data
            
        Raises:
            ProcessingError: If processing fails
        """
        pass


class DataTransformer(DataProcessor[T, T]):
    """Abstract base class for data transformers that modify data of the same type."""
    
    @abc.abstractmethod
    def fit(self, data: T) -> None:
        """Fit the transformer to the data.
        
        Args:
            data: Data to fit the transformer to
            
        Raises:
            FittingError: If fitting fails
        """
        pass
    
    @abc.abstractmethod
    def transform(self, data: T) -> T:
        """Transform the data.
        
        Args:
            data: Data to transform
            
        Returns:
            T: Transformed data
            
        Raises:
            TransformationError: If transformation fails
        """
        pass
    
    def fit_transform(self, data: T) -> T:
        """Fit the transformer to the data and transform it.
        
        Args:
            data: Data to fit and transform
            
        Returns:
            T: Transformed data
            
        Raises:
            FittingError: If fitting fails
            TransformationError: If transformation fails
        """
        self.fit(data)
        return self.transform(data)
    
    def process(self, data: T) -> T:
        """Process implementation that calls transform.
        
        Args:
            data: Data to transform
            
        Returns:
            T: Transformed data
        """
        return self.transform(data)


class Pipeline(Identifiable, Configurable, Generic[T, U]):
    """A pipeline that chains multiple data processors together."""
    
    def __init__(
        self, 
        steps: List[DataProcessor],
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the pipeline with processing steps.
        
        Args:
            steps: Ordered list of processing steps
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self._steps = steps
    
    @property
    def steps(self) -> List[DataProcessor]:
        """Get the pipeline steps.
        
        Returns:
            List[DataProcessor]: The current pipeline steps
        """
        return self._steps
    
    def add_step(self, step: DataProcessor) -> None:
        """Add a step to the end of the pipeline.
        
        Args:
            step: The processor to add
        """
        self._steps.append(step)
    
    def process(self, data: Any) -> Any:
        """Process data through the pipeline.
        
        Args:
            data: Input data to process
            
        Returns:
            Any: The final processed output
            
        Raises:
            PipelineError: If any step fails
        """
        current_data = data
        for step in self._steps:
            current_data = step.process(current_data)
        return current_data


class DataSink(Identifiable, Configurable, Generic[T]):
    """Abstract base class for data output destinations."""
    
    @abc.abstractmethod
    def write(self, data: T) -> None:
        """Write data to the sink.
        
        Args:
            data: Data to write
            
        Raises:
            DataWriteError: If writing fails
        """
        pass


class Serializable(Protocol):
    """Protocol for objects that can be serialized."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert object to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Any:
        """Create object from a dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Any: The created object
        """
        ...

