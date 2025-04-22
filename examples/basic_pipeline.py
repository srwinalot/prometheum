"""
Basic Prometheum Pipeline Example

This example demonstrates how to:
1. Load data from a CSV file
2. Define a schema for validation
3. Create a processing pipeline with multiple transformers
4. Run data through the pipeline and examine results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the parent directory to sys.path to import prometheum
sys.path.append(str(Path(__file__).parent.parent))

from prometheum import (
    # Core components
    DataType, 
    # Schema validation
    create_schema, validate_schema,
    # Data loading
    read_csv,
    # Transformers
    MissingValueHandler, StandardScaler, ColumnSelector,
    # Pipeline
    create_pipeline
)

# Create sample data file if it doesn't exist
sample_file = Path(__file__).parent / "sample_data.csv"
if not sample_file.exists():
    # Generate sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'id': range(1, 101),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='D')
    })
    
    # Add some missing values
    df.loc[np.random.choice(df.index, 10), 'value'] = np.nan
    
    # Save to CSV
    df.to_csv(sample_file, index=False)
    print(f"Created sample data file: {sample_file}")

# 1. Load data
print("Loading data...")
data = read_csv(sample_file)
print(f"Loaded {len(data.data)} rows with columns: {list(data.data.columns)}")

# 2. Define and validate schema
print("\nDefining schema...")
schema = (
    create_schema()
    .column("id", DataType.INTEGER, nullable=False, unique=True)
    .with_min_value(1)
    .column("value", DataType.FLOAT, nullable=True)
    .with_min_value(0)
    .with_max_value(200)
    .column("category", DataType.STRING, nullable=False)
    .with_allowed_values(["A", "B", "C"])
    .column("timestamp", DataType.DATETIME, nullable=False)
    .build()
)

print("Validating data against schema...")
validation_result = validate_schema(data, schema)
print(f"Validation passed: {validation_result.metadata.get('validation_passed', False)}")

# 3. Create processing pipeline
print("\nCreating data processing pipeline...")
pipeline = (
    create_pipeline("Sample Data Processing")
    .add(MissingValueHandler(strategy="mean", columns=["value"]))
    .add(StandardScaler(columns=["value"]))
    .add(ColumnSelector(columns=["id", "value", "category"]))
    .build()
)

# 4. Process data through pipeline
print("Processing data through pipeline...")
result = pipeline.process(data)

# 5. Examine results
print("\nResults:")
print(f"Rows: {len(result.data)}")
print(f"Columns: {list(result.data.columns)}")
print("\nFirst 5 rows:")
print(result.data.head())

print("\nPipeline execution details:")
execution = result.metadata.get("execution", {})
print(f"Steps executed: {execution.get('steps_executed', 0)}")
print(f"Duration: {execution.get('duration_seconds', 0):.2f} seconds")

# Print step history
if "step_history" in execution:
    print("\nStep execution history:")
    for step in execution["step_history"]:
        print(f"- {step['name']}: {step['status']} ({step.get('execution_time', 0):.4f}s)")

print("\nExample completed successfully!")

