"""
Tests for data module components of the Prometheum framework.
"""

import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from prometheum.data.loader import (
    CSVDataLoader,
    JSONDataLoader,
    FileDataLoader
)
from prometheum.data.parser import (
    DataFrameSchema,
    ColumnSchema,
    DataType,
    DataParser,
    SchemaValidator,
    TypeConverter,
    NotNullValidator,
    RangeValidator,
    UniqueValidator,
    RegexValidator,
    AllowedValuesValidator
)
from prometheum.core.base import DataContainer, DataFrameContainer
from prometheum.core.exceptions import (
    DataLoadError,
    ValidationError,
    FileSystemError
)


# Fixtures

@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def csv_file_path(fixtures_dir):
    """Path to sample CSV file."""
    return os.path.join(fixtures_dir, "sample.csv")


@pytest.fixture
def json_file_path(fixtures_dir):
    """Path to sample JSON file."""
    return os.path.join(fixtures_dir, "sample.json")


@pytest.fixture
def nonexistent_file_path(fixtures_dir):
    """Path to a file that does not exist."""
    return os.path.join(fixtures_dir, "nonexistent.csv")


@pytest.fixture
def sample_dataframe():
    """Sample pandas DataFrame for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
        "value": [10.5, 20.3, 30.1, 40.8, 50.2],
        "active": [True, False, True, False, True],
        "created_at": [
            "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"
        ]
    })


@pytest.fixture
def sample_schema():
    """Sample schema for validation testing."""
    return DataFrameSchema(
        columns=[
            ColumnSchema(name="id", data_type=DataType.INTEGER, nullable=False, unique=True),
            ColumnSchema(name="name", data_type=DataType.STRING, nullable=False),
            ColumnSchema(name="value", data_type=DataType.FLOAT, min_value=0.0, max_value=100.0),
            ColumnSchema(name="active", data_type=DataType.BOOLEAN),
            ColumnSchema(
                name="created_at",
                data_type=DataType.STRING,
                regex_pattern=r"\d{4}-\d{2}-\d{2}"
            )
        ],
        allow_extra_columns=False,
        require_all_columns=True
    )


# Tests for CSVDataLoader

def test_csv_loader_init(csv_file_path):
    """Test initializing CSV loader."""
    loader = CSVDataLoader(csv_file_path)
    assert loader.filepath.name == "sample.csv"
    assert loader.delimiter == ","
    assert loader.encoding == "utf-8"


def test_csv_loader_load(csv_file_path):
    """Test loading CSV data."""
    loader = CSVDataLoader(csv_file_path)
    result = loader.load()
    
    # Check result type
    assert isinstance(result, DataFrameContainer)
    
    # Check data properties
    df = result.data
    assert len(df) == 5
    assert list(df.columns) == ["id", "name", "value", "active", "created_at"]
    
    # Check metadata
    assert "source" in result.metadata
    assert result.metadata["format"] == "csv"
    assert result.metadata["delimiter"] == ","
    assert result.metadata["row_count"] == 5
    assert result.metadata["column_count"] == 5


def test_csv_loader_nonexistent_file(nonexistent_file_path):
    """Test loader with nonexistent file."""
    loader = CSVDataLoader(nonexistent_file_path)
    
    with pytest.raises(FileSystemError):
        loader.load()


def test_csv_loader_validate(csv_file_path):
    """Test CSV data validation."""
    loader = CSVDataLoader(csv_file_path)
    data = loader.load()
    
    # Validation should pass
    assert loader.validate(data) is True
    
    # Create empty DataFrame to test validation failure
    empty_df = pd.DataFrame()
    empty_container = DataFrameContainer(empty_df)
    
    with pytest.raises(ValidationError):
        loader.validate(empty_container)


# Tests for JSONDataLoader

def test_json_loader_init(json_file_path):
    """Test initializing JSON loader."""
    loader = JSONDataLoader(json_file_path)
    assert loader.filepath.name == "sample.json"
    assert loader.encoding == "utf-8"


def test_json_loader_load(json_file_path):
    """Test loading JSON data."""
    loader = JSONDataLoader(json_file_path)
    result = loader.load()
    
    # Check result type (should be DataFrame since our JSON is an array of objects)
    assert isinstance(result, DataFrameContainer)
    
    # Check data properties
    df = result.data
    assert len(df) == 5
    assert "id" in df.columns
    assert "name" in df.columns
    assert "value" in df.columns
    assert "active" in df.columns
    assert "created_at" in df.columns
    assert "metadata.category" in df.columns  # Flattened nested objects
    
    # Check metadata
    assert "source" in result.metadata
    assert result.metadata["format"] == "json"
    assert result.metadata["row_count"] == 5


def test_json_loader_nonexistent_file(nonexistent_file_path):
    """Test JSON loader with nonexistent file."""
    loader = JSONDataLoader(nonexistent_file_path)
    
    with pytest.raises(FileSystemError):
        loader.load()


# Tests for Schema Validation

def test_schema_creation():
    """Test creating a schema."""
    schema = DataFrameSchema(
        columns=[
            ColumnSchema(name="id", data_type=DataType.INTEGER),
            ColumnSchema(name="name", data_type=DataType.STRING)
        ]
    )
    
    assert len(schema.columns) == 2
    assert schema.get_column_schema("id") is not None
    assert schema.get_column_schema("name") is not None
    assert schema.get_column_schema("nonexistent") is None


def test_schema_from_dataframe(sample_dataframe):
    """Test creating schema from DataFrame."""
    schema = DataFrameSchema.from_dataframe(sample_dataframe)
    
    assert len(schema.columns) == 5
    assert schema.get_column_schema("id").data_type == DataType.INTEGER
    assert schema.get_column_schema("value").data_type == DataType.FLOAT
    assert schema.get_column_schema("active").data_type == DataType.BOOLEAN
    assert schema.row_count_min == 5
    assert schema.row_count_max == 5


def test_schema_validator(sample_dataframe, sample_schema):
    """Test schema validation with valid data."""
    validator = SchemaValidator(sample_schema)
    container = DataFrameContainer(sample_dataframe)
    
    result = validator.process(container)
    
    # Validation should pass
    assert result.metadata["validation_passed"] is True
    assert result.metadata["error_count"] == 0


def test_schema_validator_with_errors(sample_dataframe, sample_schema):
    """Test schema validation with invalid data."""
    # Create invalid data (negative value should fail min_value constraint)
    invalid_df = sample_dataframe.copy()
    invalid_df.loc[0, "value"] = -10.0
    
    validator = SchemaValidator(sample_schema, raise_on_error=False)
    container = DataFrameContainer(invalid_df)
    
    result = validator.process(container)
    
    # Validation should fail
    assert result.metadata["validation_passed"] is False
    assert result.metadata["error_count"] > 0
    assert "errors" in result.metadata
    
    # Check that the error is about the range validation
    assert any("outside range" in error for error in result.metadata["errors"])


def test_schema_validator_raises_error(sample_dataframe, sample_schema):
    """Test schema validator raises error when configured to."""
    # Create invalid data
    invalid_df = sample_dataframe.copy()
    invalid_df.loc[0, "value"] = -10.0
    
    validator = SchemaValidator(sample_schema, raise_on_error=True)
    container = DataFrameContainer(invalid_df)
    
    with pytest.raises(ValidationError):
        validator.process(container)


# Tests for Type Converter

def test_type_converter(sample_dataframe):
    """Test type conversion."""
    # Create converter to convert id to float and created_at to datetime
    converter = TypeConverter({
        "id": DataType.FLOAT,
        "created_at": DataType.DATETIME
    })
    
    container = DataFrameContainer(sample_dataframe)
    result = converter.process(container)
    
    # Check conversions
    assert pd.api.types.is_float_dtype(result.data["id"].dtype)
    assert pd.api.types.is_datetime64_dtype(result.data["created_at"].dtype)
    
    # Metadata should track conversions
    assert "type_conversion" in result.metadata
    assert "id" in result.metadata["converted_columns"]
    assert "created_at" in result.metadata["converted_columns"]


def test_type_converter_error_handling():
    """Test type converter error handling."""
    # Create data with unconvertible value
    df = pd.DataFrame({
        "value": ["not_a_number", "10.5", "20.3"]
    })
    
    # Test with errors='raise'
    converter_raise = TypeConverter({"value": DataType.FLOAT}, errors="raise")
    with pytest.raises(Exception):
        converter_raise.process(DataFrameContainer(df))
    
    # Test with errors='ignore'
    converter_ignore = TypeConverter({"value": DataType.FLOAT}, errors="ignore")
    result_ignore = converter_ignore.process(DataFrameContainer(df))
    # Original dtype should be preserved (object for string)
    assert pd.api.types.is_object_dtype(result_ignore.data["value"].dtype)
    
    # Test with errors='coerce'
    converter_coerce = TypeConverter({"value": DataType.FLOAT}, errors="coerce")
    result_coerce = converter_coerce.process(DataFrameContainer(df))
    # dtype should be float, with NaN for the unconvertible value
    assert pd.api.types.is_float_dtype(result_coerce.data["value"].dtype)
    assert pd.isna(result_coerce.data["value"].iloc[0])


# Tests for Data Parser

def test_data_parser(sample_dataframe, sample_schema):
    """Test data parser with conversion and validation."""
    parser = DataParser(sample_schema, convert_types=True, validate=True)
    container = DataFrameContainer(sample_dataframe)
    
    result = parser.process(container)
    
    # Check that parsing was successful
    assert result.metadata["parsed"] is True
    assert result.metadata["type_conversion_applied"] is True
    assert result.metadata["schema_validation_applied"] is True
    assert result.metadata["validation_passed"] is True


def test_data_parser_with_errors(sample_dataframe, sample_schema):
    """Test data parser with validation errors."""
    # Create invalid data
    invalid_df = sample_dataframe.copy()
    invalid_df.loc[0, "value"] = -10.0  # Will fail min_value constraint
    
    # Parser with errors='ignore' should not raise
    parser = DataParser(sample_schema, errors="ignore")
    container = DataFrameContainer(invalid_df)
    
    result = parser.process(container)
    
    # Check that parsing captured error
    assert result.metadata["parsed"] is True
    assert result.metadata["validation_passed"] is False
    assert "validation_errors" in result.metadata
    
    # Parser with errors='raise' should raise
    parser_raise = DataParser(sample_schema, errors="raise")
    with pytest.raises(ValidationError):
        parser_raise.process(container)


# Tests for Individual Validators

def test_not_null_validator():
    """Test NotNullValidator."""
    validator = NotNullValidator("test_column")
    
    # Valid data (no nulls)
    valid_series = pd.Series([1, 2, 3, 4, 5])
    is_valid, errors = validator.validate(valid_series)
    assert is_valid is True
    assert len(errors) == 0
    
    # Invalid data (contains nulls)
    invalid_series = pd.Series([1, None, 3, None, 5])
    is_valid, errors = validator.validate(invalid_series)
    assert is_valid is False
    assert len(errors) == 1
    assert "contains null values" in errors[0]


def test_range_validator():
    """Test RangeValidator."""
    validator = RangeValidator("test_column", min_value=0, max_value=10)
    
    # Valid data (within range)
    valid_series = pd.Series([0, 1, 5, 9, 10])
    is_valid, errors = validator.validate(valid_series)
    assert is_valid is True
    assert len(errors) == 0
    
    # Invalid data (outside range)
    invalid_series = pd.Series([-1, 0, 5, 10, 11])
    is_valid, errors = validator.validate(invalid_series)
    assert is_valid is False
    assert len(errors) > 0


def test_unique_validator():
    """Test UniqueValidator."""
    validator = UniqueValidator("test_column")
    
    # Valid data (all unique)
    valid_series = pd.Series([1, 2, 3, 4, 5])
    is_valid, errors = validator.validate(valid_series)
    assert is_valid is True
    assert len(errors) == 0
    
    # Invalid data (contains duplicates)
    invalid_series = pd.Series([1, 2, 2, 3, 3])
    is_valid, errors = validator.validate(invalid_series)
    assert is_valid is False
    assert len(errors) > 0
    assert "duplicate values" in errors[0]


def test_regex_validator():
    """Test RegexValidator."""
    # Date format validator
    validator = RegexValidator("test_column", pattern=r"\d{4}-\d{2}-\d{2}")
    
    # Valid data (matches pattern)
    valid_series = pd.Series(["2025-01-01", "2025-02-15", "2025-12-31"])
    is_valid, errors = validator.validate(valid_series)
    assert is_valid is True
    assert len(errors) == 0
    
    #

