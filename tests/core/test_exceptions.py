"""
Tests for the custom exceptions in the Prometheum framework.

This module provides tests for the exception hierarchy, error details tracking,
error propagation, and proper string formatting of error messages.
"""

import pytest
import pandas as pd
from typing import Dict, Any, Optional

from prometheum.core.exceptions import (
    PrometheumError,
    ConfigurationError,
    ValidationError,
    DataError,
    DataLoadError,
    DataWriteError,
    ProcessingError,
    TransformationError,
    FittingError,
    PipelineError,
    ResourceError,
    FileSystemError
)
from prometheum.core.base import DataProcessor, Pipeline


class ErrorTestProcessor(DataProcessor):
    """Processor that raises different types of errors for testing."""
    
    def __init__(
        self, 
        error_type: str = "processing", 
        error_message: str = "Test error",
        additional_details: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize with error configuration.
        
        Args:
            error_type: Type of error to raise
            error_message: Custom error message
            additional_details: Additional error details
            config: Processor configuration
        """
        super().__init__(config)
        self.error_type = error_type
        self.error_message = error_message
        self.additional_details = additional_details or {}
    
    def process(self, data: Any) -> Any:
        """Raise the configured type of error."""
        if self.error_type == "processing":
            raise ProcessingError(
                self.error_message, 
                processor_id=self.id,
                details=self.additional_details
            )
        elif self.error_type == "transformation":
            raise TransformationError(
                self.error_message, 
                processor_id=self.id,
                details=self.additional_details
            )
        elif self.error_type == "fitting":
            raise FittingError(
                self.error_message, 
                processor_id=self.id,
                details=self.additional_details
            )
        elif self.error_type == "validation":
            raise ValidationError(
                self.error_message, 
                field=self.additional_details.get("field"),
                invalid_value=self.additional_details.get("invalid_value"),
                details=self.additional_details
            )
        elif self.error_type == "data_load":
            raise DataLoadError(
                self.error_message, 
                source=self.additional_details.get("source"),
                details=self.additional_details
            )
        elif self.error_type == "data_write":
            raise DataWriteError(
                self.error_message, 
                destination=self.additional_details.get("destination"),
                details=self.additional_details
            )
        elif self.error_type == "file_system":
            raise FileSystemError(
                self.error_message, 
                path=self.additional_details.get("path"),
                operation=self.additional_details.get("operation"),
                details=self.additional_details
            )
        elif self.error_type == "configuration":
            raise ConfigurationError(
                self.error_message, 
                details=self.additional_details
            )
        else:
            raise ValueError(f"Unknown error type: {self.error_type}")


class TestBaseException:
    """Test the base PrometheumError class."""
    
    def test_basic_error(self):
        """Test creating a basic error without details."""
        error = PrometheumError("Test error message")
        
        assert error.message == "Test error message"
        assert error.details == {}
        assert str(error) == "Test error message"
    
    def test_error_with_details(self):
        """Test creating an error with details."""
        details = {"source": "test", "value": 42}
        error = PrometheumError("Test error message", details)
        
        assert error.message == "Test error message"
        assert error.details == details
        assert "Test error message" in str(error)
        assert "Details" in str(error)
        assert "source" in str(error)
        assert "42" in str(error)
    
    def test_error_inheritance(self):
        """Test the inheritance hierarchy of errors."""
        # Configuration errors
        assert issubclass(ConfigurationError, PrometheumError)
        
        # Data errors
        assert issubclass(DataError, PrometheumError)
        assert issubclass(DataLoadError, DataError)
        assert issubclass(DataWriteError, DataError)
        
        # Processing errors
        assert issubclass(ProcessingError, PrometheumError)
        assert issubclass(TransformationError, ProcessingError)
        assert issubclass(FittingError, ProcessingError)
        assert issubclass(PipelineError, ProcessingError)
        
        # Resource errors
        assert issubclass(ResourceError, PrometheumError)
        assert issubclass(FileSystemError, ResourceError)


class TestValidationError:
    """Test the ValidationError class."""
    
    def test_simple_validation_error(self):
        """Test creating a simple validation error."""
        error = ValidationError("Invalid value")
        
        assert error.message == "Invalid value"
        assert error.field is None
        assert error.invalid_value is None
        assert error.details == {}
    
    def test_validation_error_with_field(self):
        """Test validation error with field information."""
        error = ValidationError("Invalid value", field="username")
        
        assert error.message == "Invalid value"
        assert error.field == "username"
        assert error.invalid_value is None
        assert error.details == {"field": "username"}
    
    def test_validation_error_with_invalid_value(self):
        """Test validation error with invalid value."""
        error = ValidationError("Invalid value", field="age", invalid_value=-5)
        
        assert error.message == "Invalid value"
        assert error.field == "age"
        assert error.invalid_value == -5
        assert error.details == {"field": "age", "invalid_value": "-5"}
        assert "age" in str(error)
        assert "-5" in str(error)
    
    def test_validation_error_with_additional_details(self):
        """Test validation error with additional details."""
        details = {"allowed_range": [0, 100], "severity": "high"}
        error = ValidationError(
            "Invalid age", 
            field="age", 
            invalid_value=-5, 
            details=details
        )
        
        assert error.message == "Invalid age"
        assert error.field == "age"
        assert error.invalid_value == -5
        assert error.details["field"] == "age"
        assert error.details["invalid_value"] == "-5"
        assert error.details["allowed_range"] == [0, 100]
        assert error.details["severity"] == "high"


class TestProcessingErrors:
    """Test the processing error classes."""
    
    def test_processing_error(self):
        """Test creating a basic processing error."""
        error = ProcessingError("Processing failed")
        
        assert error.message == "Processing failed"
        assert error.processor_id is None
        assert error.step_name is None
        assert error.details == {}
    
    def test_processing_error_with_context(self):
        """Test processing error with processor context."""
        error = ProcessingError(
            "Processing failed", 
            processor_id="proc-123", 
            step_name="normalize"
        )
        
        assert error.message == "Processing failed"
        assert error.processor_id == "proc-123"
        assert error.step_name == "normalize"
        assert error.details == {
            "processor_id": "proc-123",
            "step_name": "normalize"
        }
        assert "proc-123" in str(error)
        assert "normalize" in str(error)
    
    def test_transformation_error(self):
        """Test transformation error."""
        error = TransformationError(
            "Transformation failed", 
            processor_id="transformer-123"
        )
        
        assert isinstance(error, ProcessingError)
        assert error.message == "Transformation failed"
        assert error.processor_id == "transformer-123"
    
    def test_fitting_error(self):
        """Test fitting error."""
        error = FittingError(
            "Fitting failed", 
            processor_id="normalizer-123"
        )
        
        assert isinstance(error, ProcessingError)
        assert error.message == "Fitting failed"
        assert error.processor_id == "normalizer-123"


class TestPipelineError:
    """Test the PipelineError class."""
    
    def test_pipeline_error(self):
        """Test creating a basic pipeline error."""
        error = PipelineError("Pipeline execution failed")
        
        assert error.message == "Pipeline execution failed"
        assert error.processor_id is None
        assert error.step_idx is None
        assert error.step_id is None
    
    def test_pipeline_error_with_step_info(self):
        """Test pipeline error with step information."""
        error = PipelineError(
            "Step failed", 
            pipeline_id="pipeline-123",
            step_idx=2,
            step_id="transform"
        )
        
        assert error.message == "Step failed"
        assert error.processor_id == "pipeline-123"
        assert error.step_idx == 2
        assert error.step_id == "transform"
        assert error.details == {
            "step_idx": 2,
            "step_id": "transform"
        }
        assert "step_idx" in str(error)
        assert "transform" in str(error)
    
    def test_pipeline_error_with_details(self):
        """Test pipeline error with additional details."""
        sub_error = ProcessingError("Processing failed", processor_id="proc-456")
        details = {
            "original_error": str(sub_error),
            "data_source": "test.csv"
        }
        
        error = PipelineError(
            "Pipeline execution failed", 
            pipeline_id="pipeline-123",
            step_idx=1,
            step_id="process",
            details=details
        )
        
        assert error.details["step_idx"] == 1
        assert error.details["step_id"] == "process"
        assert error.details["original_error"] == str(sub_error)
        assert error.details["data_source"] == "test.csv"


class TestDataErrors:
    """Test the data error classes."""
    
    def test_data_load_error(self):
        """Test data load error."""
        error = DataLoadError("Failed to load data", source="file.csv")
        
        assert error.message == "Failed to load data"
        assert error.source == "file.csv"
        assert error.details["source"] == "file.csv"
        assert "file.csv" in str(error)
    
    def test_data_write_error(self):
        """Test data write error."""
        error = DataWriteError("Failed to write data", destination="output.json")
        
        assert error.message == "Failed to write data"
        assert error.destination == "output.json"
        assert error.details["destination"] == "output.json"
        assert "output.json" in str(error)
    
    def test_data_error_details(self):
        """Test data error with additional details."""
        details = {
            "format": "CSV",
            "rows": 1000,
            "encoding": "UTF-8"
        }
        error = DataLoadError(
            "Failed to load CSV data", 
            source="data.csv", 
            details=details
        )
        
        assert error.message == "Failed to load CSV data"
        assert error.source == "data.csv"
        assert error.details["source"] == "data.csv"
        assert error.details["format"] == "CSV"
        assert error.details["rows"] == 1000
        assert error.details["encoding"] == "UTF-8"
        
        # Verify string representation contains details
        error_str = str(error)
        assert "data.csv" in error_str
        assert "CSV" in error_str
        assert "1000" in error_str
        assert "UTF-8" in error_str


class TestResourceErrors:
    """Test the resource error classes."""
    
    def test_resource_error_basic(self):
        """Test creating a basic resource error."""
        error = ResourceError("Resource not available")
        
        assert error.message == "Resource not available"
        assert error.details == {}
    
    def test_filesystem_error(self):
        """Test file system error with path and operation."""
        error = FileSystemError(
            "File operation failed", 
            path="/path/to/file.txt", 
            operation="read"
        )
        
        assert error.message == "File operation failed"
        assert error.path == "/path/to/file.txt"
        assert hasattr(error, "operation")  # Attribute is defined in the class
        assert "path" in error.details
        assert "operation" in error.details
        assert error.details["path"] == "/path/to/file.txt"
        assert error.details["operation"] == "read"
        
        # Verify string representation
        error_str = str(error)
        assert "/path/to/file.txt" in error_str
        assert "read" in error_str
    
    def test_filesystem_error_with_details(self):
        """Test file system error with additional details."""
        details = {
            "file_size": 1024,
            "permissions": "rw-r--r--",
            "owner": "user"
        }
        error = FileSystemError(
            "Permission denied", 
            path="/path/to/protected.txt", 
            operation="write",
            details=details
        )
        
        assert error.message == "Permission denied"
        assert error.path == "/path/to/protected.txt"
        assert error.details["path"] == "/path/to/protected.txt"
        assert error.details["operation"] == "write"
        assert error.details["file_size"] == 1024
        assert error.details["permissions"] == "rw-r--r--"
        assert error.details["owner"] == "user"


class TestErrorPropagation:
    """Test error propagation through components."""
    
    def test_pipeline_error_propagation(self):
        """Test error propagation through a pipeline."""
        # Create processors with different error types
        first_processor = ErrorTestProcessor("processing", "First processor error")
        second_processor = ErrorTestProcessor("validation", "Validation error", 
                                          {"field": "test_field", "invalid_value": 42})
        
        # Create pipeline with error-raising processors
        pipeline = Pipeline([
            ("first", first_processor),
            ("second", second_processor)
        ])
        
        # The first processor should raise an error, which gets wrapped by the pipeline
        with pytest.raises(PipelineError) as exc_info:
            pipeline.process("test data")
        
        # Verify error details from pipeline
        pipeline_error = exc_info.value
        assert pipeline_error.details["step"] == "first"
        assert "First processor error" in pipeline_error.details["error"]
    
    def test_nested_error_context(self):
        """Test preservation of error context through nested components."""
        # Create a processor that will raise an error
        error_processor = ErrorTestProcessor("processing", "Inner error", 
                                          {"inner_context": "preserved"})
        
        # Create a pipeline with the error processor
        inner_pipeline = Pipeline([("error", error_processor)])
        
        # Create a pipeline that includes the inner pipeline
        outer_pipeline = Pipeline([("inner", inner_pipeline)])
        
        # Outer pipeline should propagate inner errors with context
        with pytest.raises(PipelineError) as exc_info:
            outer_pipeline.process("test data")
        
        # Verify error details are preserved through multiple levels
        outer_error = exc_info.value
        assert outer_error.details["step"] == "inner"
        assert "Inner error" in outer_error.details["error"]
        
        # The inner pipeline error should be in the details
        inner_error_str = outer_error.details["error"]
        assert "inner_context" in inner_error_str
        assert "preserved" in inner_error_str
    
    def test_error_with_cause(self):
        """Test error propagation with original cause."""
        try:
            # Raise a low-level error
            try:
                raise ValueError("Original low-level error")
            except ValueError as e:
                # Wrap with a more specific error
                raise DataLoadError(
                    "Failed to load data",
                    source="test.csv",
                    details={"original_error": str(e)}
                ) from e
        except DataLoadError as e:
            # Verify both error details and original cause are preserved
            assert e.message == "Failed to load data"
            assert e.source == "test.csv"
            assert e.details["original_error"] == "Original low-level error"
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)
            assert str(e.__cause__) == "Original low-level error"
    
    def test_error_handling_strategies(self):
        """Test different error handling strategies."""
        # Create processors with different error types
        processors = [
            ErrorTestProcessor("processing", "Processing error"),
            ErrorTestProcessor("validation", "Validation error"),
            ErrorTestProcessor("data_load", "Data load error")
        ]
        
        # Test catching specific error types
        for processor in processors:
            try:
                processor.process("test data")
                pytest.fail(f"Expected {processor.error_type} error but none was raised")
            except PrometheumError as e:
                # Verify we can catch all errors with the base class
                assert processor.error_message in str(e)
            
            # Test catching specific error types
            try:
                processor.process("test data")
            except ProcessingError as e:
                assert processor.error_type == "processing"
            except ValidationError as e:
                assert processor.error_type == "validation"
            except DataLoadError as e:
                assert processor.error_type == "data_load"
            except Exception:
                pytest.fail(f"Unexpected error type for {processor.error_type}")


# Documentation about the error handling test suite
"""
The error handling test suite provides comprehensive coverage of the exception framework:

1. Base Exceptions:
   - PrometheumError initialization and details handling
   - Error hierarchy and inheritance structure
   - String representation with details

2. Specialized Errors:
   - ValidationError with field and invalid value tracking
   - ProcessingError with processor context
   - PipelineError with step information
   - DataErrors with source/destination tracking
   - ResourceErrors with resource-specific details

3. Error Propagation:
   - Pipeline error wrapping
   - Error context preservation through nested components
   - Error causality chain with __cause__
   - Error handling strategies

The tests verify proper error construction, details preservation, context tracking,
and propagation to ensure robust error handling throughout the framework.
"""
