"""
Custom exceptions for the Prometheum data processing framework.

This module defines a hierarchy of exceptions specific to the framework,
allowing for detailed error reporting and handling of various error conditions.
"""

from typing import Any, Dict, List, Optional, Union


class PrometheumError(Exception):
    """Base exception for all Prometheum-specific errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize with error message and optional details.
        
        Args:
            message: Error message
            details: Additional error details
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)
    
    def __str__(self) -> str:
        """Get string representation of the error.
        
        Returns:
            str: Error message including details if available
        """
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


# Configuration and Initialization Errors

class ConfigurationError(PrometheumError):
    """Error raised when there is an issue with configuration."""
    pass


class InitializationError(PrometheumError):
    """Error raised when component initialization fails."""
    pass


class ValidationError(PrometheumError):
    """Error raised when validation of data, configuration, or parameters fails."""
    
    def __init__(
        self, 
        message: str, 
        field: Optional[str] = None,
        invalid_value: Any = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize with validation error details.
        
        Args:
            message: Error message
            field: Optional name of the field that failed validation
            invalid_value: Optional value that caused validation to fail
            details: Additional error details
        """
        self.field = field
        self.invalid_value = invalid_value
        details_dict = details or {}
        if field:
            details_dict['field'] = field
        if invalid_value is not None:
            details_dict['invalid_value'] = str(invalid_value)
        super().__init__(message, details_dict)


# Data Processing Errors

class DataError(PrometheumError):
    """Base class for data-related errors."""
    pass


class DataLoadError(DataError):
    """Error raised when data loading fails."""
    
    def __init__(
        self, 
        message: str, 
        source: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize with data load error details.
        
        Args:
            message: Error message
            source: Optional data source identifier
            details: Additional error details
        """
        self.source = source
        details_dict = details or {}
        if source:
            details_dict['source'] = source
        super().__init__(message, details_dict)


class DataWriteError(DataError):
    """Error raised when writing data fails."""
    
    def __init__(
        self, 
        message: str, 
        destination: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize with data write error details.
        
        Args:
            message: Error message
            destination: Optional data destination identifier
            details: Additional error details
        """
        self.destination = destination
        details_dict = details or {}
        if destination:
            details_dict['destination'] = destination
        super().__init__(message, details_dict)


class ProcessingError(PrometheumError):
    """Base class for errors that occur during data processing."""
    
    def __init__(
        self, 
        message: str, 
        processor_id: Optional[str] = None,
        step_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize with processing error details.
        
        Args:
            message: Error message
            processor_id: Optional identifier of the processor
            step_name: Optional name of the processing step
            details: Additional error details
        """
        self.processor_id = processor_id
        self.step_name = step_name
        details_dict = details or {}
        if processor_id:
            details_dict['processor_id'] = processor_id
        if step_name:
            details_dict['step_name'] = step_name
        super().__init__(message, details_dict)


class TransformationError(ProcessingError):
    """Error raised when data transformation fails."""
    pass


class FittingError(ProcessingError):
    """Error raised when fitting a transformer fails."""
    pass


class PipelineError(ProcessingError):
    """Error raised when a pipeline operation fails."""
    
    def __init__(
        self, 
        message: str, 
        pipeline_id: Optional[str] = None,
        step_idx: Optional[int] = None,
        step_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize with pipeline error details.
        
        Args:
            message: Error message
            pipeline_id: Optional identifier of the pipeline
            step_idx: Optional index of the step that failed
            step_id: Optional identifier of the step that failed
            details: Additional error details
        """
        self.step_idx = step_idx
        self.step_id = step_id
        details_dict = details or {}
        if step_idx is not None:
            details_dict['step_idx'] = step_idx
        if step_id:
            details_dict['step_id'] = step_id
        super().__init__(message, pipeline_id, None, details_dict)


# Resource and Environment Errors

class ResourceError(PrometheumError):
    """Error raised when there is an issue with a resource."""
    pass


class FileSystemError(ResourceError):
    """Error raised when there is an issue with file system operations."""
    
    def __init__(
        self, 
        message: str, 
        path: Optional[str] = None, 
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize with file system error details.
        
        Args:
            message: Error message
            path: Optional file path
            operation: Optional operation being performed (read, write, etc.)
            details: Additional error details
        """
        self.path = path

