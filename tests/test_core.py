"""
Tests for core components of the Prometheum framework.
"""

import pytest
import pandas as pd
import numpy as np

from prometheum.core.base import (
    DataContainer, 
    DataFrameContainer, 
    DataProcessor, 
    Pipeline
)
from prometheum.core.exceptions import (
    ProcessingError, 
    PipelineError
)


# Fixtures

@pytest.fixture
def sample_data():
    """Sample data for testing DataContainer."""
    return {"key1": "value1", "key2": [1, 2, 3], "key3": {"nested": "value"}}


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing containers."""
    return {"source": "test", "timestamp": "2025-04-10T12:00:00"}


@pytest.fixture
def sample_dataframe():
    """Sample pandas DataFrame for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "value": [10.5, 20.3, 30.1, 40.8, 50.2],
        "category": ["A", "B", "A", "C", "B"]
    })


class SimpleProcessor(DataProcessor):
    """Simple processor that adds a constant to numeric columns."""
    
    def __init__(self, add_value=1):
        super().__init__()
        self.add_value = add_value
    
    def process(self, data):
        if isinstance(data, DataFrameContainer):
            df = data.data.copy()
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col] + self.add_value
            return DataFrameContainer(df, data.metadata)
        return data


class FailingProcessor(DataProcessor):
    """Processor that always fails for testing error handling."""
    
    def process(self, data):
        raise ProcessingError("This processor always fails")


# Tests for DataContainer

def test_data_container_creation(sample_data, sample_metadata):
    """Test basic DataContainer creation and properties."""
    container = DataContainer(sample_data, sample_metadata)
    
    assert container.data == sample_data
    assert container.metadata == sample_metadata
    
    # Test adding metadata
    container.add_metadata("new_key", "new_value")
    assert container.metadata["new_key"] == "new_value"


def test_dataframe_container_creation(sample_dataframe, sample_metadata):
    """Test DataFrameContainer creation and properties."""
    container = DataFrameContainer(sample_dataframe, sample_metadata)
    
    assert container.data.equals(sample_dataframe)
    assert container.metadata == sample_metadata
    
    # Test shape method
    assert container.shape() == sample_dataframe.shape
    
    # Test head method
    assert container.head(2).equals(sample_dataframe.head(2))


# Tests for Pipeline

def test_pipeline_creation():
    """Test creating a pipeline with steps."""
    # Create processors
    proc1 = SimpleProcessor(add_value=1)
    proc2 = SimpleProcessor(add_value=2)
    
    # Create pipeline
    pipeline = Pipeline([proc1, proc2])
    
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0] == proc1
    assert pipeline.steps[1] == proc2


def test_pipeline_add_step():
    """Test adding steps to a pipeline."""
    pipeline = Pipeline()
    
    # Add steps one by one
    proc1 = SimpleProcessor(add_value=1)
    proc2 = SimpleProcessor(add_value=2)
    
    pipeline.add_step(proc1)
    pipeline.add_step(proc2)
    
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0] == proc1
    assert pipeline.steps[1] == proc2


def test_pipeline_process(sample_dataframe):
    """Test processing data through a pipeline."""
    # Create a simple pipeline
    pipeline = Pipeline([
        SimpleProcessor(add_value=1),
        SimpleProcessor(add_value=2)
    ])
    
    # Create input data
    data = DataFrameContainer(sample_dataframe)
    
    # Process data
    result = pipeline.process(data)
    
    # Verify result
    assert isinstance(result, DataFrameContainer)
    
    # Check that numeric values have been increased by 3 (1+2)
    assert result.data["id"].equals(sample_dataframe["id"] + 3)
    assert result.data["value"].equals(sample_dataframe["value"] + 3)
    
    # Check that non-numeric column is unchanged
    assert result.data["category"].equals(sample_dataframe["category"])


# Tests for Exception Handling

def test_exception_handling():
    """Test exception handling in pipeline."""
    # Create a pipeline with a failing processor
    pipeline = Pipeline([FailingProcessor()])
    
    # Create input data
    data = DataContainer({"test": "data"})
    
    # Process should raise PipelineError
    with pytest.raises(PipelineError):
        pipeline.process(data)


def test_pipeline_with_mixed_processors(sample_dataframe):
    """Test pipeline with mixed working and failing processors."""
    # Create a pipeline with working and failing processors
    pipeline = Pipeline([
        SimpleProcessor(add_value=1),
        FailingProcessor(),
        SimpleProcessor(add_value=2)
    ])
    
    # Create input data
    data = DataFrameContainer(sample_dataframe)
    
    # Process should raise PipelineError
    with pytest.raises(PipelineError):
        pipeline.process(data)


def test_error_propagation():
    """Test error details propagation."""
    # Create input data
    data = DataContainer({"test": "data"})
    
    # Create processor that raises a specific error
    class CustomProcessor(DataProcessor):
        def process(self, data):
            raise ProcessingError(
                "Custom error message",
                details={"custom_detail": "value"}
            )
    
    # Create pipeline with custom processor
    pipeline = Pipeline([CustomProcessor()])
    
    # Process and catch the error to inspect details
    try:
        pipeline.process(data)
        assert False, "Expected PipelineError was not raised"
    except PipelineError as e:
        # Check that details were propagated
        assert "Custom error message" in str(e)
        if hasattr(e, 'details'):
            assert "custom_detail" in e.details.get("original_error", "")

