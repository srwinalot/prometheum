"""
Data parsing and validation components for the Prometheum framework.

This module provides tools for schema definition, data validation, and type conversion,
allowing users to ensure data quality and consistency in their processing pipelines.
"""

import datetime
import re
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, cast

import numpy as np
import pandas as pd
import pydantic
from pydantic import BaseModel, Field, field_validator

from prometheum.core.base import DataContainer, DataFrameContainer, DataProcessor, Identifiable, Configurable
from prometheum.core.exceptions import ProcessingError, ValidationError


class DataType(Enum):
    """Enumeration of supported data types for columns."""
    
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    CATEGORY = "category"
    JSON = "json"
    
    @classmethod
    def from_pandas_dtype(cls, pandas_dtype: Any) -> "DataType":
        """Convert pandas dtype to DataType.
        
        Args:
            pandas_dtype: Pandas data type
            
        Returns:
            DataType: Corresponding DataType enum value
            
        Raises:
            ValueError: If no matching DataType found
        """
        dtype_str = str(pandas_dtype)
        
        if "int" in dtype_str:
            return cls.INTEGER
        elif "float" in dtype_str:
            return cls.FLOAT
        elif "bool" in dtype_str:
            return cls.BOOLEAN
        elif "datetime" in dtype_str:
            return cls.DATETIME
        elif "date" in dtype_str:
            return cls.DATE
        elif "time" in dtype_str:
            return cls.TIME
        elif "category" in dtype_str:
            return cls.CATEGORY
        elif "object" in dtype_str:
            return cls.STRING  # Default for object types
        else:
            return cls.STRING  # Default fallback


class ColumnSchema(BaseModel):
    """Schema definition for a single data column."""
    
    name: str
    data_type: DataType
    nullable: bool = True
    unique: bool = False
    min_value: Optional[Union[int, float, datetime.datetime]] = None
    max_value: Optional[Union[int, float, datetime.datetime]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    regex_pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"
    
    @field_validator("min_value", "max_value")
    def check_value_type(cls, v, info):
        """Validate min_value and max_value types match data_type.
        
        Args:
            v: The value to validate
            info: Field validation info containing data and context 
            
        Returns:
            Any: The validated value
            
        Raises:
            ValueError: If value type doesn't match data_type
        """
        if v is None:
            return v
            
        # Access values from the model data
        data = info.data
        data_type = data.get("data_type") if data else None
        if data_type in (DataType.INTEGER, DataType.FLOAT):
            if not isinstance(v, (int, float)):
                raise ValueError(f"min_value/max_value must be numeric for {data_type}")
        elif data_type in (DataType.DATETIME, DataType.DATE, DataType.TIME):
            if not isinstance(v, (datetime.datetime, datetime.date, datetime.time)):
                raise ValueError(f"min_value/max_value must be datetime-like for {data_type}")
                
        return v


class DataFrameSchema(BaseModel):
    """Schema definition for a complete DataFrame."""
    
    columns: List[ColumnSchema]
    allow_extra_columns: bool = False
    require_all_columns: bool = True
    row_count_min: Optional[int] = None
    row_count_max: Optional[int] = None
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"
    
    def get_column_schema(self, column_name: str) -> Optional[ColumnSchema]:
        """Get schema for a specific column.
        
        Args:
            column_name: Name of the column
            
        Returns:
            Optional[ColumnSchema]: Column schema if found, None otherwise
        """
        for col in self.columns:
            if col.name == column_name:
                return col
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the schema
        """
        return self.dict()
    
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        nullable_detection: bool = True,
        infer_unique: bool = True
    ) -> "DataFrameSchema":
        """Infer schema from an existing DataFrame.
        
        Args:
            df: DataFrame to infer schema from
            nullable_detection: Whether to detect nullable columns
            infer_unique: Whether to detect unique columns
            
        Returns:
            DataFrameSchema: Inferred schema
        """
        columns = []
        
        for col_name, dtype in df.dtypes.items():
            # Infer data type
            data_type = DataType.from_pandas_dtype(dtype)
            
            # Check nullability
            nullable = True
            if nullable_detection:
                nullable = df[col_name].isna().any()
            
            # Check uniqueness
            unique = False
            if infer_unique:
                unique = df[col_name].nunique() == len(df)
            
            # Create column schema
            col_schema = ColumnSchema(
                name=col_name,
                data_type=data_type,
                nullable=nullable,
                unique=unique
            )
            columns.append(col_schema)
        
        return cls(
            columns=columns,
            row_count_min=len(df),
            row_count_max=len(df)
        )


class DataValidator:
    """Base class for data validators."""
    
    def __init__(
        self,
        error_message_template: str,
        column_name: Optional[str] = None
    ) -> None:
        """Initialize the validator.
        
        Args:
            error_message_template: Template for error messages
            column_name: Optional name of the column to validate
        """
        self.error_message_template = error_message_template
        self.column_name = column_name
    
    def validate(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate data.
        
        Args:
            data: Data to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def format_error(self, **kwargs) -> str:
        """Format error message.
        
        Args:
            **kwargs: Values to insert into template
            
        Returns:
            str: Formatted error message
        """
        message = self.error_message_template
        
        # Add column name if available
        if self.column_name:
            kwargs["column"] = self.column_name
            
        # Format the message
        return message.format(**kwargs)


class NotNullValidator(DataValidator):
    """Validator that checks for null values."""
    
    def __init__(
        self,
        column_name: str,
        error_message_template: str = "Column '{column}' contains null values"
    ) -> None:
        """Initialize the not-null validator.
        
        Args:
            column_name: Name of the column to validate
            error_message_template: Template for error messages
        """
        super().__init__(error_message_template, column_name)
    
    def validate(self, data: pd.Series) -> Tuple[bool, List[str]]:
        """Validate that the series contains no null values.
        
        Args:
            data: Series to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        if data.isna().any():
            null_count = data.isna().sum()
            percentage = (null_count / len(data)) * 100
            error = self.format_error(null_count=null_count, percentage=f"{percentage:.2f}%")
            return False, [error]
        return True, []


class RangeValidator(DataValidator):
    """Validator that checks value ranges."""
    
    def __init__(
        self,
        column_name: str,
        min_value: Optional[Union[int, float, datetime.datetime]] = None,
        max_value: Optional[Union[int, float, datetime.datetime]] = None,
        error_message_template: str = "Values in '{column}' outside range [{min_value}, {max_value}]"
    ) -> None:
        """Initialize the range validator.
        
        Args:
            column_name: Name of the column to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            error_message_template: Template for error messages
            
        Raises:
            ValueError: If both min_value and max_value are None
        """
        super().__init__(error_message_template, column_name)
        
        if min_value is None and max_value is None:
            raise ValueError("At least one of min_value or max_value must be specified")
            
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, data: pd.Series) -> Tuple[bool, List[str]]:
        """Validate that values are within range.
        
        Args:
            data: Series to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        
        # Skip NaN values
        non_null_data = data.dropna()
        
        # Check minimum value
        if self.min_value is not None:
            below_min = non_null_data[non_null_data < self.min_value]
            if not below_min.empty:
                count = len(below_min)
                error = self.format_error(
                    min_value=self.min_value,
                    max_value=self.max_value or "∞",
                    violation="below minimum",
                    count=count
                )
                errors.append(error)
        
        # Check maximum value
        if self.max_value is not None:
            above_max = non_null_data[non_null_data > self.max_value]
            if not above_max.empty:
                count = len(above_max)
                error = self.format_error(
                    min_value=self.min_value or "-∞",
                    max_value=self.max_value,
                    violation="above maximum",
                    count=count
                )
                errors.append(error)
        
        return len(errors) == 0, errors


class UniqueValidator(DataValidator):
    """Validator that checks for unique values."""
    
    def __init__(
        self,
        column_name: str,
        error_message_template: str = "Column '{column}' contains duplicate values"
    ) -> None:
        """Initialize the uniqueness validator.
        
        Args:
            column_name: Name of the column to validate
            error_message_template: Template for error messages
        """
        super().__init__(error_message_template, column_name)
    
    def validate(self, data: pd.Series) -> Tuple[bool, List[str]]:
        """Validate that the series contains only unique values.
        
        Args:
            data: Series to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        # Find duplicated values
        duplicates = data[data.duplicated()]
        
        if not duplicates.empty:
            dupes_count = len(duplicates)
            unique_dupes = duplicates.unique()
            sample = str(unique_dupes[:3].tolist())
            
            error = self.format_error(
                count=dupes_count,
                sample=sample,
                unique_count=len(unique_dupes)
            )
            return False, [error]
        
        return True, []


class RegexValidator(DataValidator):
    """Validator that checks string values against regex patterns."""
    
    def __init__(
        self,
        column_name: str,
        pattern: str,
        error_message_template: str = "Values in '{column}' do not match pattern: {pattern}"
    ) -> None:
        """Initialize the regex validator.
        
        Args:
            column_name: Name of the column to validate
            pattern: Regex pattern to match
            error_message_template: Template for error messages
            
        Raises:
            ValueError: If pattern is invalid
        """
        super().__init__(error_message_template, column_name)
        
        try:
            self.pattern = pattern
            self.compiled_regex = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {pattern}. Error: {str(e)}")
    
    def validate(self, data: pd.Series) -> Tuple[bool, List[str]]:
        """Validate that string values match the pattern.
        
        Args:
            data: Series to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        # Skip non-string values and nulls
        if not pd.api.types.is_string_dtype(data):
            return False, [f"Column '{self.column_name}' is not string type and cannot be regex validated"]
        
        # Get non-null values
        non_null = data.dropna()
        
        # Check pattern matches
        matches = non_null.str.match(self.pattern)
        non_matches = non_null[~matches]
        
        if not non_matches.empty:
            count = len(non_matches)
            sample = str(non_matches.iloc[:3].tolist())
            
            error = self.format_error(
                pattern=self.pattern,
                count=count,
                sample=sample
            )
            return False, [error]
        
        return True, []


class AllowedValuesValidator(DataValidator):
    """Validator that checks values against an allowed set."""
    
    def __init__(
        self,
        column_name: str,
        allowed_values: List[Any],
        error_message_template: str = "Column '{column}' contains disallowed values"
    ) -> None:
        """Initialize the allowed values validator.
        
        Args:
            column_name: Name of the column to validate
            allowed_values: List of allowed values
            error_message_template: Template for error messages
            
        Raises:
            ValueError: If allowed_values is empty
        """
        super().__init__(error_message_template, column_name)
        
        if not allowed_values:
            raise ValueError("allowed_values cannot be empty")
            
        self.allowed_values = set(allowed_values)
    
    def validate(self, data: pd.Series) -> Tuple[bool, List[str]]:
        """Validate that all values are in the allowed set.
        
        Args:
            data: Series to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        # Skip null values
        non_null = data.dropna()
        
        # Find disallowed values
        disallowed = non_null[~non_null.isin(self.allowed_values)]
        
        if not disallowed.empty:
            count = len(disallowed)
            unique_disallowed = disallowed.unique()
            sample = str(unique_disallowed[:3].tolist())
            
            error = self.format_error(
                count=count,
                sample=sample,
                unique_count=len(unique_disallowed),
                allowed=str(list(self.allowed_values)[:10])  # First 10 allowed values
            )
            return False, [error]
        
        return True, []


class SchemaValidator(DataProcessor):
    """Validator that applies schema validation to a DataFrame."""
    
    def __init__(
        self,
        schema: DataFrameSchema,
        raise_on_error: bool = True,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the schema validator.
        
        Args:
            schema: Schema to validate against
            raise_on_error: Whether to raise an exception on validation failure
            config: Optional configuration parameters
        """
        Identifiable.__init__(self)
        Configurable.__init__(self, config)
        self.schema = schema
        self.raise_on_error = raise_on_error
        self.validators = self._create_validators()
    
    def _create_validators(self) -> Dict[str, List[DataValidator]]:
        """Create validators for all columns in the schema.
        
        Returns:
            Dict[str, List[DataValidator]]: Dictionary mapping column names to validator lists
        """
        validators = {}
        for column_schema in self.schema.columns:
            column_validators = self._create_validators_for_column(column_schema)
            validators[column_schema.name] = column_validators
        return validators
    
    def _create_validators_for_column(self, column_schema: ColumnSchema) -> List[DataValidator]:
        """Create validators for a column based on its schema.
        
        Args:
            column_schema: Schema for the column
            
        Returns:
            List[DataValidator]: List of validators for the column
        """
        validators = []
        column_name = column_schema.name
        
        # Not null validator
        if not column_schema.nullable:
            validators.append(NotNullValidator(column_name))
        
        # Range validator
        if column_schema.min_value is not None or column_schema.max_value is not None:
            validators.append(RangeValidator(
                column_name,
                min_value=column_schema.min_value,
                max_value=column_schema.max_value
            ))
        
        # Unique validator
        if column_schema.unique:
            validators.append(UniqueValidator(column_name))
        
        # Regex validator
        if column_schema.regex_pattern is not None:
            validators.append(RegexValidator(
                column_name,
                pattern=column_schema.regex_pattern
            ))
        
        # Allowed values validator
        if column_schema.allowed_values is not None:
            validators.append(AllowedValuesValidator(
                column_name,
                allowed_values=column_schema.allowed_values
            ))
        
        return validators
    
    def process(self, data: DataFrameContainer) -> DataFrameContainer:
        """Validate the DataFrame against the schema.
        
        Args:
            data: Container with DataFrame to validate
            
        Returns:
            DataFrameContainer: The same container if validation passes
            
        Raises:
            ValidationError: If validation fails and raise_on_error is True
        """
        df = data.data.copy()  # Make a copy to avoid modifying the original
        errors = []
        
        # Validate column existence
        expected_columns = {col.name for col in self.schema.columns}
        actual_columns = set(df.columns)
        
        # Check for missing required columns
        if self.schema.require_all_columns:
            missing_columns = expected_columns - actual_columns
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for extra columns
        if not self.schema.allow_extra_columns:
            extra_columns = actual_columns - expected_columns
            if extra_columns:
                errors.append(f"Found unexpected columns: {extra_columns}")
        
        # Validate row count
        if self.schema.row_count_min is not None and len(df) < self.schema.row_count_min:
            errors.append(f"DataFrame has {len(df)} rows, less than minimum {self.schema.row_count_min}")
        
        if self.schema.row_count_max is not None and len(df) > self.schema.row_count_max:
            errors.append(f"DataFrame has {len(df)} rows, more than maximum {self.schema.row_count_max}")
        
        # Validate each column
        for column_name, validators in self.validators.items():
            # Skip if column doesn't exist
            if column_name not in df.columns:
                continue
                
            # Apply each validator
            for validator in validators:
                is_valid, validator_errors = validator.validate(df[column_name])
                if not is_valid:
                    errors.extend(validator_errors)
        
        # Add validation result to metadata
        validation_metadata = {
            "schema_validated": True,
            "validation_passed": len(errors) == 0,
            "error_count": len(errors)
        }
        
        if errors:
            validation_metadata["errors"] = errors[:10]  # Limit to first 10 errors
            if self.raise_on_error:
                error_details = {
                    "schema": self.schema.model_dump() if hasattr(self.schema, 'model_dump') else self.schema.dict(),
                    "error_count": len(errors),
                    "errors": errors[:10]
                }
                raise ValidationError(
                    f"Schema validation failed with {len(errors)} errors",
                    details=error_details
                )
            
        # Create a new container with updated metadata
        new_metadata = {**data.metadata, **validation_metadata}
        return DataFrameContainer(df, new_metadata)


class TypeConverter(DataProcessor):
    """Processor that converts column data types."""
    
    def __init__(
        self,
        column_types: Dict[str, Union[DataType, Type, str]],
        errors: str = "raise",
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the type converter.
        
        Args:
            column_types: Mapping of column names to target types
            errors: How to handle conversion errors ('raise', 'ignore', 'coerce')
            config: Optional configuration parameters
            
        Raises:
            ConfigurationError: If errors option is invalid
        """
        Identifiable.__init__(self)
        Configurable.__init__(self, config)
        self.column_types = column_types
        
        if errors not in ("raise", "ignore", "coerce"):
            raise ValueError("errors must be one of: 'raise', 'ignore', 'coerce'")
            
        self.errors = errors
    
    def _convert_type(self, series: pd.Series, target_type: Union[DataType, Type, str]) -> pd.Series:
        """Convert a series to the target type.
        
        Args:
            series: The series to convert
            target_type: The target type
            
        Returns:
            pd.Series: Converted series
            
        Raises:
            ProcessingError: If conversion fails and errors='raise'
        """
        try:
            # Handle DataType enum
            if isinstance(target_type, DataType):
                if target_type == DataType.INTEGER:
                    return pd.to_numeric(series, errors=self.errors, downcast="integer")
                elif target_type == DataType.FLOAT:
                    return pd.to_numeric(series, errors=self.errors, downcast="float")
                elif target_type == DataType.BOOLEAN:
                    return series.astype(bool)
                elif target_type == DataType.DATETIME:
                    return pd.to_datetime(series, errors=self.errors)
                elif target_type == DataType.DATE:
                    return pd.to_datetime(series, errors=self.errors).dt.date
                elif target_type == DataType.TIME:
                    return pd.to_datetime(series, errors=self.errors).dt.time
                elif target_type == DataType.CATEGORY:
                    return series.astype("category")
                elif target_type == DataType.STRING:
                    return series.astype(str)
                else:
                    return series  # Unknown type, return as is
            else:
                # Handle Python types or string dtype names
                return series.astype(target_type)
        except Exception as e:
            if self.errors == "raise":
                raise ProcessingError(
                    f"Failed to convert column to {target_type}: {str(e)}",
                    details={"original_error": str(e)}
                )
            elif self.errors == "ignore":
                return series  # Return original series
            else:  # coerce
                # Special handling for common types
                if target_type == DataType.INTEGER or target_type == int:
                    return pd.to_numeric(series, errors="coerce").astype("Int64")  # Nullable integer
                elif target_type == DataType.FLOAT or target_type == float:
                    return pd.to_numeric(series, errors="coerce")
                elif target_type == DataType.DATETIME:
                    return pd.to_datetime(series, errors="coerce")
                else:
                    return series
    
    def process(self, data: DataFrameContainer) -> DataFrameContainer:
        """Convert column types in the DataFrame.
        
        Args:
            data: Container with DataFrame to process
            
        Returns:
            DataFrameContainer: Container with processed DataFrame
            
        Raises:
            ProcessingError: If conversion fails and errors='raise'
        """
        df = data.data.copy()
        conversions = {}
        
        # Convert each specified column
        for column_name, target_type in self.column_types.items():
            if column_name in df.columns:
                original_type = df[column_name].dtype
                df[column_name] = self._convert_type(df[column_name], target_type)
                conversions[column_name] = {
                    "original_type": str(original_type),
                    "target_type": str(target_type),
                    "result_type": str(df[column_name].dtype)
                }
        
        # Add conversion info to metadata
        type_conversion_metadata = {
            "type_conversion": True,
            "converted_columns": list(conversions.keys()),
            "conversions": conversions
        }
        
        # Create a new container with updated metadata
        new_metadata = {**data.metadata, **type_conversion_metadata}
        return DataFrameContainer(df, new_metadata)


class DataParser(DataProcessor):
    """Processor that parses and validates data according to a schema."""
    
    def __init__(
        self,
        schema: DataFrameSchema,
        convert_types: bool = True,
        validate: bool = True,
        errors: str = "raise",
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the data parser.
        
        Args:
            schema: Schema defining the expected data structure
            convert_types: Whether to convert column types based on schema
            validate: Whether to validate against schema
            errors: How to handle conversion errors ('raise', 'ignore', 'coerce')
            config: Optional configuration parameters
        """
        Identifiable.__init__(self)
        Configurable.__init__(self, config)
        self.schema = schema
        self.convert_types = convert_types
        self.validate = validate
        self.errors = errors
        
        # Create sub-processors
        if convert_types:
            column_types = {
                col.name: col.data_type for col in schema.columns
            }
            self.type_converter = TypeConverter(column_types, errors=errors)
            
        if validate:
            self.schema_validator = SchemaValidator(schema, raise_on_error=(errors == "raise"))
    
    def process(self, data: DataContainer) -> DataContainer:
        """Parse and validate the data.
        
        Args:
            data: Container with data to process
            
        Returns:
            DataContainer: Container with processed data
            
        Raises:
            ValidationError: If validation fails and errors='raise'
            ProcessingError: If type conversion fails and errors='raise'
            TypeError: If data is not a DataFrameContainer
        """
        # Check if input is a DataFrame container
        if not isinstance(data, DataFrameContainer):
            raise TypeError(
                f"DataParser can only process DataFrameContainer, not {type(data).__name__}",
            )
        
        # Initialize processing metadata
        processing_metadata = {
            "parsed": True,
            "schema_name": self.schema.__class__.__name__,
            "parser_id": self.id
        }
        
        current_data = data
        
        try:
            # Step 1: Convert types if enabled
            if self.convert_types:
                current_data = self.type_converter.process(current_data)
                processing_metadata["type_conversion_applied"] = True
            
            # Step 2: Validate against schema if enabled
            if self.validate:
                current_data = self.schema_validator.process(current_data)
                processing_metadata["schema_validation_applied"] = True
                
                # If validation was successful, update metadata
                if current_data.metadata.get("validation_passed", False):
                    processing_metadata["validation_passed"] = True
                else:
                    processing_metadata["validation_passed"] = False
                    
                    # Include validation errors in metadata if available
                    if "errors" in current_data.metadata:
                        processing_metadata["validation_errors"] = current_data.metadata["errors"]
                
        except (ValidationError, ProcessingError) as e:
            # If errors='raise', re-raise the exception
            if self.errors == "raise":
                raise
                
            # Otherwise, capture error details in metadata
            processing_metadata["error_occurred"] = True
            processing_metadata["error_type"] = e.__class__.__name__
            processing_metadata["error_message"] = str(e)
            
            if hasattr(e, "details"):
                processing_metadata["error_details"] = e.details
        
        # Create final metadata by merging original and processing metadata
        final_metadata = {**current_data.metadata, **processing_metadata}
        
        # Return processed data with updated metadata
        df = current_data.data
        return DataFrameContainer(df, final_metadata)


class DataQualityReport:
    """Utility class for generating data quality reports."""
    
    @staticmethod
    def generate_report(data: DataFrameContainer) -> Dict[str, Any]:
        """Generate a comprehensive data quality report.
        
        Args:
            data: DataFrame container to analyze
            
        Returns:
            Dict[str, Any]: Report containing data quality metrics
        """
        df = data.data
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": {}
        }
        
        # Generate column-level statistics
        for column in df.columns:
            col_data = df[column]
            col_stats = {
                "dtype": str(col_data.dtype),
                "count": len(col_data),
                "null_count": col_data.isna().sum(),
                "null_percentage": round((col_data.isna().sum() / len(col_data)) * 100, 2),
                "unique_count": col_data.nunique()
            }
            
            # Add numeric statistics if applicable
            if pd.api.types.is_numeric_dtype(col_data):
                col_stats.update({
                    "min": col_data.min(),
                    "max": col_data.max(),
                    "mean": col_data.mean(),
                    "median": col_data.median(),
                    "std": col_data.std()
                })
            
            # Add string statistics if applicable
            elif pd.api.types.is_string_dtype(col_data):
                non_null = col_data.dropna()
                if len(non_null) > 0:
                    col_stats.update({
                        "min_length": non_null.str.len().min(),
                        "max_length": non_null.str.len().max(),
                        "avg_length": round(non_null.str.len().mean(), 2)
                    })
            
            report["columns"][column] = col_stats
        
        return report


class DataProfiler(DataProcessor):
    """Processor that generates data profiles and quality metrics."""
    
    def __init__(
        self,
        include_histograms: bool = False,
        include_correlations: bool = True,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the data profiler.
        
        Args:
            include_histograms: Whether to include histogram data
            include_correlations: Whether to include correlation matrix
            config: Optional configuration parameters
        """
        Identifiable.__init__(self)
        Configurable.__init__(self, config)
        self.include_histograms = include_histograms
        self.include_correlations = include_correlations
    
    def process(self, data: DataFrameContainer) -> DataFrameContainer:
        """Profile the data and add quality metrics to metadata.
        
        Args:
            data: Container with DataFrame to profile
            
        Returns:
            DataFrameContainer: Container with profiling metadata added
        """
        df = data.data
        
        # Generate basic quality report
        quality_report = DataQualityReport.generate_report(data)
        
        # Add correlation matrix if requested
        if self.include_correlations:
            try:
                # Get only numeric columns
                numeric_df = df.select_dtypes(include=['number'])
                if not numeric_df.empty and len(numeric_df.columns) > 1:
                    correlations = numeric_df.corr().to_dict()
                    quality_report["correlations"] = correlations
            except Exception as e:
                quality_report["correlation_error"] = str(e)
        
        # Add histograms if requested
        if self.include_histograms:
            histograms = {}
            for column in df.columns:
                try:
                    if pd.api.types.is_numeric_dtype(df[column]):
                        hist_data = df[column].value_counts(bins=10, sort=False)
                        histograms[column] = {
                            "bins": [str(b) for b in hist_data.index.astype(str)],
                            "counts": hist_data.values.tolist()
                        }
                except Exception as e:
                    histograms[column] = {"error": str(e)}
                    
            quality_report["histograms"] = histograms
        
        # Create a new container with the quality report added to metadata
        profiling_metadata = {
            "data_quality_report": quality_report,
            "profiled_at": datetime.datetime.now().isoformat(),
            "profiler_id": self.id
        }
        
        new_metadata = {**data.metadata, **profiling_metadata}
        return DataFrameContainer(df, new_metadata)
