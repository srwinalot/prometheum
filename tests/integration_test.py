"""
Integration test for Prometheum framework.

This script tests the integration of multiple components 
to ensure they work together correctly.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the parent directory to the path to import prometheum
sys.path.append(str(Path(__file__).parent.parent))

from prometheum import (
    # Core
    DataType, DataFrameContainer,
    # Data loading
    CSVDataLoader,
    # Schema validation
    ColumnSchema, DataFrameSchema, SchemaValidator,
    # Transformers
    MissingValueHandler, StandardScaler, ColumnSelector, MinMaxScaler,
    # Pipeline
    Pipeline, PipelineBuilder,
    # Serialization
    ComponentSerializer, SerializationFormat
)

def test_full_pipeline():
    """Test a complete data processing pipeline flow."""
    
    print("Starting integration test...")
    
    # 1. Create test data
    print("Creating test data...")
    test_data = pd.DataFrame({
        'id': range(1, 11),
        'value': [10, 20, np.nan, 40, 50, 60, 70, np.nan, 90, 100],
        'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
        'timestamp': pd.date_range(start='2023-01-01', periods=10)
    })
    
    # 2. Define schema
    print("Defining schema...")
    schema = DataFrameSchema(
        columns=[
            ColumnSchema(
                name="id", 
                data_type=DataType.INTEGER, 
                nullable=False, 
                unique=True
            ),
            ColumnSchema(
                name="value", 
                data_type=DataType.FLOAT, 
                nullable=True, 
                min_value=0, 
                max_value=200
            ),
            ColumnSchema(
                name="category", 
                data_type=DataType.STRING, 
                nullable=False,
                allowed_values=["A", "B", "C"]
            ),
            ColumnSchema(
                name="timestamp", 
                data_type=DataType.DATETIME, 
                nullable=False
            )
        ],
        allow_extra_columns=False,
        require_all_columns=True
    )
    
    # 3. Create data container
    container = DataFrameContainer(test_data, {"source": "test"})
    
    # 4. Validate against schema
    print("Validating schema...")
    validator = SchemaValidator(schema, raise_on_error=False)
    validation_result = validator.process(container)
    
    # Check validation result
    passed = validation_result.metadata.get("validation_passed", False)
    print(f"Validation {'passed' if passed else 'failed'}")
    
    # 5. Create processing pipeline
    print("Creating and executing pipeline...")
    pipeline = Pipeline([
        ("fill_missing", MissingValueHandler(strategy="mean")),
        ("scale", StandardScaler(columns=["value"])),
        ("select", ColumnSelector(columns=["id", "value", "category"]))
    ])
    
    # 6. Process data through pipeline
    result = pipeline.process(container)
    
    # Check execution status
    execution = result.metadata.get("execution", {})
    steps_executed = execution.get("steps_executed", 0)
    print(f"Pipeline executed {steps_executed} steps")
    
    # 7. Serialize the pipeline
    print("Testing serialization...")
    
    # Convert to JSON
    serialized = ComponentSerializer.serialize(pipeline, SerializationFormat.JSON)
    print(f"Serialized pipeline to JSON ({len(serialized)} characters)")
    
    # Deserialize
    deserialized = ComponentSerializer.deserialize(serialized, SerializationFormat.JSON)
    print("Successfully deserialized pipeline")
    
    # 8. Run the deserialized pipeline
    print("Running deserialized pipeline...")
    result2 = deserialized.process(container)
    
    # 9. Compare results
    original_columns = set(result.data.columns)
    deserialized_columns = set(result2.data.columns)
    
    print(f"Original columns: {original_columns}")
    print(f"Deserialized columns: {deserialized_columns}")
    print(f"Results match: {original_columns == deserialized_columns}")
    
    # 10. Test fluent builder API
    print("Testing fluent builder API...")
    builder_pipeline = (
        PipelineBuilder("Test Pipeline")
        .add(MissingValueHandler(strategy="mean"))
        .add(MinMaxScaler(columns=["value"]))
        .add(ColumnSelector(columns=["id", "value"]))
        .build()
    )
    
    builder_result = builder_pipeline.process(container)
    print(f"Builder pipeline executed successfully with columns: {list(builder_result.data.columns)}")
    
    print("Integration test completed successfully!")
    return True

if __name__ == "__main__":
    test_full_pipeline()

