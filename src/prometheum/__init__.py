"""
Prometheum - A flexible and powerful framework for data processing and analysis.

This package provides tools for:
  - Data loading from various sources (CSV, JSON, SQL, URLs)
  - Data validation and parsing
  - Data transformation and processing
  - Pipeline orchestration

Quick Start:

```python
import pandas as pd
from prometheum import read_csv, create_pipeline, MissingValueHandler, StandardScaler

# Load data
data = read_csv("data.csv")

# Create processing pipeline
pipeline = (
    create_pipeline("Data Processing Pipeline")
    .add(MissingValueHandler(strategy="mean"))
    .add(StandardScaler())
    .build()
)

# Process data
result = pipeline.process(data)
```
"""

__version__ = "0.1.0"
__author__ = "Franklin Butahe"
__email__ = "franklinbutahe@example.com"

# Core imports
from prometheum.core.base import (
    DataContainer,
    DataFrameContainer,
    DataLoader,
    DataProcessor,
    DataSink,
    DataTransformer,
    Identifiable,
    Pipeline as CorePipeline,
    Serializable,
)

from prometheum.core.exceptions import (
    ConfigurationError,
    DataError,
    DataLoadError,
    DataWriteError,
    FittingError,
    PipelineError,
    ProcessingError,
    PrometheumError,
    ResourceError,
    TransformationError,
    ValidationError,
)

# Data module imports
from prometheum.data.loader import (
    CSVDataLoader,
    FileDataLoader,
    JSONDataLoader,
    SQLDataLoader,
    URLDataLoader,
)

from prometheum.data.parser import (
    ColumnSchema,
    DataFrameSchema,
    DataParser,
    DataProfiler,
    DataQualityReport,
    DataType,
    SchemaValidator,
    TypeConverter,
)

from prometheum.data.schema import (
    LengthValidator,
    ColumnDependencyValidator,
    RowValidator,
    SchemaBuilder,
)

# Processing module imports
from prometheum.processing.transformers import (
    ColumnSelector,
    MissingValueHandler,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

from prometheum.processing.pipeline import (
    ExecutionContext,
    Pipeline,
    PipelineBuilder,
    PipelineStep,
    StepStatus,
)

from prometheum.utils.serialization import (
    ComponentSerializer,
    SerializationFormat
)


# Convenience functions
def read_csv(filepath, **kwargs):
    """Load data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        **kwargs: Additional arguments for CSVDataLoader
        
    Returns:
        DataFrameContainer: Container with the loaded DataFrame
    """
    loader = CSVDataLoader(filepath, **kwargs)
    return loader.load()


def read_json(filepath, **kwargs):
    """Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        **kwargs: Additional arguments for JSONDataLoader
        
    Returns:
        DataFrameContainer: Container with the loaded DataFrame
    """
    loader = JSONDataLoader(filepath, **kwargs)
    return loader.load()


def read_sql(query, connection, **kwargs):
    """Load data from a SQL database.
    
    Args:
        query: SQL query to execute
        connection: Database connection string or SQLAlchemy engine
        **kwargs: Additional arguments for SQLDataLoader
        
    Returns:
        DataFrameContainer: Container with the loaded DataFrame
    """
    loader = SQLDataLoader(query, connection, **kwargs)
    return loader.load()


def create_schema():
    """Create a new DataFrame schema using the fluent builder API.
    
    Returns:
        SchemaBuilder: A builder for creating DataFrameSchema
    """
    return SchemaBuilder()


def validate_schema(data, schema, **kwargs):
    """Validate data against a schema.
    
    Args:
        data: DataFrame or DataFrameContainer to validate
        schema: DataFrameSchema to validate against
        **kwargs: Additional arguments for SchemaValidator
        
    Returns:
        DataFrameContainer: Container with validation results
    """
    # Convert DataFrame to DataFrameContainer if needed
    if not isinstance(data, DataFrameContainer):
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            data = DataFrameContainer(data, {})
    
    validator = SchemaValidator(schema, **kwargs)
    return validator.process(data)


def create_pipeline(name="Pipeline"):
    """Create a new data processing pipeline.
    
    Args:
        name: Name of the pipeline
        
    Returns:
        PipelineBuilder: A builder for creating pipelines
    """
    return PipelineBuilder(name)


__all__ = [
    # Core classes
    "DataContainer",
    "DataFrameContainer",
    "DataLoader",
    "DataProcessor",
    "DataSink",
    "DataTransformer",
    "Identifiable",
    "CorePipeline",
    "Serializable",
    
    # Exceptions
    "ConfigurationError",
    "DataError",
    "DataLoadError",
    "DataWriteError",
    "FittingError",
    "PipelineError",
    "ProcessingError",
    "PrometheumError",
    "ResourceError",
    "TransformationError",
    "ValidationError",
    
    # Data loaders
    "CSVDataLoader",
    "FileDataLoader",
    "JSONDataLoader",
    "SQLDataLoader",
    "URLDataLoader",
    
    # Data parsing
    "ColumnSchema",
    "DataFrameSchema",
    "DataParser",
    "DataProfiler",
    "DataQualityReport",
    "DataType",
    "SchemaValidator",
    "TypeConverter",
    
    # Schema utilities
    "LengthValidator",
    "ColumnDependencyValidator",
    "RowValidator",
    "SchemaBuilder",
    
    # Transformers
    "ColumnSelector",
    "MissingValueHandler",
    "MinMaxScaler",
    "OneHotEncoder",
    "StandardScaler",
    
    # Pipeline
    "ExecutionContext",
    "Pipeline",
    "PipelineBuilder",
    "PipelineStep",
    "StepStatus",
    
    # Serialization
    "ComponentSerializer",
    "SerializationFormat",
    
    # Convenience functions
    "read_csv",
    "read_json",
    "read_sql",
    "create_schema",
    "validate_schema",
    "create_pipeline",
]
