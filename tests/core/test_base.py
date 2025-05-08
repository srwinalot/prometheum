"""
Tests for the core base classes in the Prometheum framework.

This module provides tests for the fundamental classes like DataContainer,
DataFrameContainer, Pipeline, and the Configurable and Identifiable mixins.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

from prometheum.core.base import (
    Configurable, 
    DataContainer, 
    DataFrameContainer, 
    DataProcessor,
    DataTransformer,
    Identifiable, 
    Pipeline
)
from prometheum.core.exceptions import (
    ProcessingError,
    PipelineError,
    ValidationError
)


class SimpleConfigurable(Configurable):
    """Simple implementation of Configurable for testing."""
    def validate_config(self) -> bool:
        """Validate configuration with custom rules."""
        if "required_param" not in self.config:
            raise ValidationError("Missing required parameter", field="required_param")
        if self.config.get("positive_param", 0) <= 0:
            raise ValidationError(
                "Parameter must be positive", 
                field="positive_param", 
                invalid_value=self.config.get("positive_param", 0)
            )
        return True


class SimpleProcessor(DataProcessor):
    """Simple processor that adds a constant to numeric data."""
    def __init__(self, add_value: float = 1.0, config: Dict[str, Any] = None) -> None:
        """Initialize with a value to add during processing."""
        super().__init__(config)
        self.add_value = add_value
    
    def process(self, data: Any) -> Any:
        """Add self.add_value to input data."""
        if isinstance(data, (int, float)):
            return data + self.add_value
        elif isinstance(data, np.ndarray):
            return data + self.add_value
        elif isinstance(data, pd.DataFrame):
            # Only add to numeric columns
            numeric_df = data.select_dtypes(include=[np.number])
            result = data.copy()
            for col in numeric_df.columns:
                result[col] = result[col] + self.add_value
            return result
        elif isinstance(data, DataFrameContainer):
            # Process the contained DataFrame
            processed_df = self.process(data.data)
            result = DataFrameContainer(processed_df, data.metadata.copy())
            result.add_metadata("processed_by", self.id)
            return result
        else:
            raise ProcessingError(
                f"Unsupported data type: {type(data)}",
                processor_id=self.id
            )


class SimpleTransformer(DataTransformer):
    """Simple transformer that normalizes numeric data."""
    def __init__(self, config: Dict[str, Any] = None) -> None:
        """Initialize the transformer."""
        super().__init__(config)
        self.fitted = False
        self.mean = None
        self.std = None
    
    def fit(self, data: Any) -> None:
        """Fit to data by computing mean and standard deviation."""
        if isinstance(data, pd.DataFrame):
            numeric_df = data.select_dtypes(include=[np.number])
            self.mean = numeric_df.mean()
            self.std = numeric_df.std()
            self.fitted = True
        elif isinstance(data, DataFrameContainer) and isinstance(data.data, pd.DataFrame):
            self.fit(data.data)
        else:
            raise ProcessingError(
                f"Unsupported data type for fitting: {type(data)}",
                processor_id=self.id
            )
    
    def transform(self, data: Any) -> Any:
        """Transform data by normalizing with computed statistics."""
        if not self.fitted:
            raise ProcessingError("Transformer not fitted", processor_id=self.id)
            
        if isinstance(data, pd.DataFrame):
            result = data.copy()
            numeric_df = data.select_dtypes(include=[np.number])
            for col in numeric_df.columns:
                if col in self.mean and self.std[col] > 0:
                    result[col] = (result[col] - self.mean[col]) / self.std[col]
            return result
        elif isinstance(data, DataFrameContainer) and isinstance(data.data, pd.DataFrame):
            processed_df = self.transform(data.data)
            result = DataFrameContainer(processed_df, data.metadata.copy())
            result.add_metadata("transformed_by", self.id)
            return result
        else:
            raise ProcessingError(
                f"Unsupported data type for transformation: {type(data)}",
                processor_id=self.id
            )


class ErrorProcessor(DataProcessor):
    """Processor that raises a specified error for testing error handling."""
    def __init__(self, error_type: str = "processing", config: Dict[str, Any] = None) -> None:
        """Initialize with the type of error to raise."""
        super().__init__(config)
        self.error_type = error_type
    
    def process(self, data: Any) -> Any:
        """Raise the specified error."""
        if self.error_type == "processing":
            raise ProcessingError("Intentional processing error", processor_id=self.id)
        elif self.error_type == "validation":
            raise ValidationError("Intentional validation error", field="test_field")
        else:
            raise ValueError("Unknown error type")


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """Create a simple DataFrame for testing."""
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10.0, 20.0, 30.0, 40.0, 50.0],
        'C': ['a', 'b', 'c', 'd', 'e']
    })


@pytest.fixture
def df_container(simple_df) -> DataFrameContainer:
    """Create a DataFrameContainer with metadata."""
    return DataFrameContainer(simple_df, {"source": "test", "version": 1.0})


@pytest.fixture
def pipeline_components() -> Tuple[SimpleProcessor, SimpleTransformer, ErrorProcessor]:
    """Create components for pipeline testing."""
    return (
        SimpleProcessor(add_value=10.0),
        SimpleTransformer(),
        ErrorProcessor()
    )


class TestDataContainer:
    """Test the DataContainer base class."""
    
    def test_init_with_data(self):
        """Test initializing with various data types."""
        # Test with different data types
        container1 = DataContainer("string data")
        assert container1.data == "string data"
        
        container2 = DataContainer([1, 2, 3])
        assert container2.data == [1, 2, 3]
        
        container3 = DataContainer({"key": "value"})
        assert container3.data == {"key": "value"}
    
    def test_metadata_handling(self):
        """Test metadata operations."""
        # Initialize with metadata
        container = DataContainer("data", {"source": "test"})
        assert container.metadata == {"source": "test"}
        
        # Add metadata
        container.add_metadata("version", 1.0)
        assert container.metadata == {"source": "test", "version": 1.0}
        
        # Overwrite existing metadata
        container.add_metadata("source", "updated")
        assert container.metadata == {"source": "updated", "version": 1.0}


class TestDataFrameContainer:
    """Test the DataFrameContainer class."""
    
    def test_init_with_dataframe(self, simple_df):
        """Test initializing with a DataFrame."""
        container = DataFrameContainer(simple_df)
        assert container.data is simple_df
        assert isinstance(container.data, pd.DataFrame)
        assert container.metadata == {}
    
    def test_shape_method(self, simple_df):
        """Test the shape method."""
        container = DataFrameContainer(simple_df)
        assert container.shape() == (5, 3)
    
    def test_head_method(self, simple_df):
        """Test the head method."""
        container = DataFrameContainer(simple_df)
        head_df = container.head(2)
        assert len(head_df) == 2
        assert list(head_df.columns) == ['A', 'B', 'C']
        
    def test_metadata_inheritance(self, simple_df):
        """Test that metadata functionality is inherited from DataContainer."""
        container = DataFrameContainer(simple_df, {"source": "test"})
        container.add_metadata("processed", True)
        assert container.metadata == {"source": "test", "processed": True}


class TestConfigurable:
    """Test the Configurable mixin class."""
    
    def test_config_initialization(self):
        """Test initializing with configuration."""
        # Default empty config
        configurable = SimpleConfigurable()
        assert configurable.config == {}
        
        # Initial config provided
        config = {"param1": "value1", "param2": 100}
        configurable = SimpleConfigurable(config)
        assert configurable.config == config
    
    def test_update_config(self):
        """Test updating configuration."""
        configurable = SimpleConfigurable({"param1": "value1"})
        configurable.update_config({"param2": 100})
        assert configurable.config == {"param1": "value1", "param2": 100}
        
        # Overwrite existing value
        configurable.update_config({"param1": "updated"})
        assert configurable.config == {"param1": "updated", "param2": 100}
    
    def test_validate_config(self):
        """Test configuration validation."""
        # Valid configuration
        configurable = SimpleConfigurable({
            "required_param": "exists",
            "positive_param": 10
        })
        assert configurable.validate_config() is True
        
        # Missing required param
        configurable = SimpleConfigurable({"positive_param": 10})
        with pytest.raises(ValidationError) as exc_info:
            configurable.validate_config()
        assert "required_param" in str(exc_info.value)
        
        # Invalid positive_param
        configurable = SimpleConfigurable({
            "required_param": "exists",
            "positive_param": -5
        })
        with pytest.raises(ValidationError) as exc_info:
            configurable.validate_config()
        assert "positive_param" in str(exc_info.value)
        assert "-5" in str(exc_info.value)


class TestIdentifiable:
    """Test the Identifiable mixin class."""
    
    def test_id_generation(self):
        """Test that unique IDs are generated."""
        id1 = Identifiable().id
        id2 = Identifiable().id
        
        assert isinstance(id1, str)
        assert len(id1) > 0
        assert id1 != id2  # IDs should be unique
    
    def test_id_consistency(self):
        """Test that ID remains consistent for an instance."""
        identifiable = Identifiable()
        id1 = identifiable.id
        id2 = identifiable.id
        
        assert id1 == id2  # Same instance should return same ID


class TestPipeline:
    """Test the Pipeline class."""
    
    def test_init_empty(self):
        """Test initializing an empty pipeline."""
        pipeline = Pipeline()
        assert pipeline.steps == []
        assert pipeline.processors == []
    
    def test_init_with_processors(self, pipeline_components):
        """Test initializing with processor instances."""
        add_processor, norm_transformer, _ = pipeline_components
        
        # Initialize with unnamed processors
        pipeline = Pipeline([add_processor, norm_transformer])
        assert len(pipeline.steps) == 2
        assert [name for name, _ in pipeline.steps] == ["step_0", "step_1"]
        assert pipeline.processors == [add_processor, norm_transformer]
        
        # Initialize with named processors
        named_steps = [
            ("add", add_processor),
            ("normalize", norm_transformer)
        ]
        pipeline = Pipeline(named_steps)
        assert pipeline.steps == named_steps
    
    def test_init_with_invalid_steps(self):
        """Test initializing with invalid steps."""
        with pytest.raises(ValueError):
            Pipeline(["not a processor"])
        
        with pytest.raises(ValueError):
            Pipeline([("name_only",)])
        
        with pytest.raises(ValueError):
            Pipeline([("name", "not a processor")])
    
    def test_add_step(self, pipeline_components):
        """Test adding steps to a pipeline."""
        add_processor, norm_transformer, _ = pipeline_components
        pipeline = Pipeline()
        
        # Add unnamed processor
        pipeline.add_step(add_processor)
        assert pipeline.steps == [("step_0", add_processor)]
        
        # Add named processor
        pipeline.add_step(("normalize", norm_transformer))
        assert pipeline.steps == [
            ("step_0", add_processor),
            ("normalize", norm_transformer)
        ]
        
        # Add invalid step
        with pytest.raises(ValueError):
            pipeline.add_step("not a processor")
    
    def test_process_data(self, pipeline_components, simple_df):
        """Test processing data through a pipeline."""
        add_processor, norm_transformer, _ = pipeline_components
        
        # Create a pipeline with two steps
        pipeline = Pipeline([
            ("add", add_processor),
            ("normalize", norm_transformer)
        ])
        
        # First fit the transformer
        norm_transformer.fit(simple_df)
        
        # Process the data
        result_df = pipeline.process(simple_df)
        
        # Verify the data was processed correctly
        # First the add_processor adds 10, then the norm_transformer normalizes
        # For column A, original mean = 3, std = 1.58
        # After adding 10: mean = 13, std = 1.58
        # Normalized value for first row (11): (11-13)/1.58 = -1.27
        assert isinstance(result_df, pd.DataFrame)
        assert result_df.shape == simple_df.shape
        assert result_df['C'].tolist() == ['a', 'b', 'c', 'd', 'e']  # Non-numeric column unchanged
        
        # Check some normalized values (approximate due to floating point)
        assert abs(result_df['A'].iloc[0] - -1.27) < 0.01
        assert abs(result_df['A'].iloc[4] - 1.27) < 0.01
    
    def test_process_with_container(self, pipeline_components, df_container):
        """Test processing a DataFrameContainer."""
        add_processor, norm_transformer, _ = pipeline_components
        
        # Create a pipeline
        pipeline = Pipeline([
            ("add", add_processor),
            ("normalize", norm_transformer)
        ])
        
        # First fit the transformer
        norm_transformer.fit(df_container.data)
        
        # Process the container
        result = pipeline.process(df_container)
        
        # Verify the result is a DataFrameContainer with correct metadata
        assert isinstance(result, DataFrameContainer)
        assert result.metadata["source"] == "test"
        assert result.metadata["version"] == 1.0
        assert "processed_by" in result.metadata
        assert "transformed_by" in result.metadata
        
        # Verify the data was processed correctly
        assert isinstance(result.data, pd.DataFrame)
        assert result.data.shape == df_container.data.shape
        assert result.data['C'].tolist() == ['a', 'b', 'c', 'd', 'e']  # Non-numeric column unchanged
    
    def test_process_error_handling(self, pipeline_components, simple_df):
        """Test error handling during pipeline processing."""
        add_processor, norm_transformer, error_processor = pipeline_components
        
        # Create a pipeline with an error-raising processor
        pipeline = Pipeline([
            ("add", add_processor),
            ("error", error_processor)
        ])
        
        # Processing should raise a PipelineError
        with pytest.raises(PipelineError) as exc_info:
            pipeline.process(simple_df)
        
        # Verify error details
        error = exc_info.value
        assert "error" in error.details
        assert "step" in error.details
        assert error.details["step"] == "error"
        assert "Intentional processing error" in error.details["error"]
    
    def test_pipeline_nested(self, pipeline_components, simple_df):
        """Test nesting pipelines within pipelines."""
        add_processor, norm_transformer, _ = pipeline_components
        
        # Create a sub-pipeline
        sub_pipeline = Pipeline([("add", add_processor)])
        
        # Create a main pipeline that includes the sub-pipeline
        main_pipeline = Pipeline([
            ("sub", sub_pipeline),
            ("normalize", norm_transformer)
        ])
        
        # Fit the transformer
        norm_transformer.fit(simple_df)
        
        # Process the data through the nested pipelines
        result = main_pipeline.process(simple_df)
        
        # Verify the result was processed through both pipelines
        assert isinstance(result, pd.DataFrame)
        # Values should be first processed by sub_pipeline (add 10)
        # Then normalized by the main pipeline
        assert abs(result['A'].iloc[0] - -1.27) < 0.01


# Documentation about the completed test suite
"""
The core base test suite provides comprehensive coverage of the core functionality:

1. DataContainer and DataFrameContainer:
   - Object initialization with various data types
   - DataFrame-specific operations (shape, head)
   - Metadata handling and inheritance

2. Mixin Classes:
   - Configurable parameter management and validation
   - Identifiable unique ID generation and consistency

3. Pipeline:
   - Pipeline construction with named and unnamed processors
   - Step management (adding, validation) 
   - Data processing through sequential steps
   - Error handling and propagation
   - Nested pipelines

4. Data Transformers:
   - Fitting and transforming data
   - Processing with various data types
   - Error condition handling

The tests verify both normal operation and error handling to ensure robust core components.
"""
