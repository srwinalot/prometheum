"""
Schema definition and validation for the Prometheum framework.

This module provides classes for defining schemas for data validation,
including column constraints, data type validation, and schema enforcement.
It builds on the parser module to provide a simplified interface for schema
validation specifically.
"""

import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, cast

import numpy as np
import pandas as pd

from prometheum.core.base import DataContainer, DataFrameContainer, DataProcessor
from prometheum.core.exceptions import ValidationError, ProcessingError
from prometheum.data.parser import (
    DataType, ColumnSchema, DataFrameSchema, SchemaValidator,
    DataValidator, NotNullValidator, RangeValidator, UniqueValidator,
    RegexValidator, AllowedValuesValidator
)

# Re-export key classes for backward compatibility
__all__ = [
    "DataType", "ColumnSchema", "DataFrameSchema", "SchemaValidator",
    "DataValidator", "NotNullValidator", "RangeValidator", "UniqueValidator",
    "RegexValidator", "AllowedValuesValidator", "LengthValidator",
    "ColumnDependencyValidator", "RowValidator", "SchemaBuilder"
]


class LengthValidator(DataValidator):
    """Validator that checks string length."""
    
    def __init__(
        self,
        column_name: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        error_message_template: str = "Values in '{column}' have invalid length"
    ) -> None:
        """Initialize the length validator.
        
        Args:
            column_name: Name of the column to validate
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            error_message_template: Template for error messages
            
        Raises:
            ValueError: If both min_length and max_length are None
        """
        super().__init__(error_message_template, column_name)
        
        if min_length is None and max_length is None:
            raise ValueError("At least one of min_length or max_length must be specified")
            
        self.min_length = min_length
        self.max_length = max_length
    
    def validate(self, data: pd.Series) -> Tuple[bool, List[str]]:
        """Validate that string values have acceptable length.
        
        Args:
            data: Series to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        
        # Skip non-string values and nulls
        if not pd.api.types.is_string_dtype(data):
            return False, [f"Column '{self.column_name}' is not string type and cannot be length validated"]
        
        # Get non-null values
        non_null = data.dropna()
        
        # Apply string length function
        lengths = non_null.str.len()
        
        # Check minimum length
        if self.min_length is not None:
            too_short = non_null[lengths < self.min_length]
            if not too_short.empty:
                count = len(too_short)
                sample = str(too_short.iloc[:3].tolist())
                
                error = self.format_error(
                    min_length=self.min_length,
                    max_length=self.max_length or "∞",
                    violation="too short",
                    count=count,
                    sample=sample
                )
                errors.append(error)
        
        # Check maximum length
        if self.max_length is not None:
            too_long = non_null[lengths > self.max_length]
            if not too_long.empty:
                count = len(too_long)
                sample = str(too_long.iloc[:3].tolist())
                
                error = self.format_error(
                    min_length=self.min_length or "0",
                    max_length=self.max_length,
                    violation="too long",
                    count=count,
                    sample=sample
                )
                errors.append(error)
        
        return len(errors) == 0, errors


class ColumnDependencyValidator(DataValidator):
    """Validator that checks relationships between columns."""
    
    def __init__(
        self,
        column_name: str,
        dependent_column: str,
        dependency_checker: Callable[[Any, Any], bool],
        error_message_template: str = "Dependency failed between '{column}' and '{dependent_column}'"
    ) -> None:
        """Initialize the column dependency validator.
        
        Args:
            column_name: Name of the primary column
            dependent_column: Name of the dependent column
            dependency_checker: Function that checks the relationship
            error_message_template: Template for error messages
        """
        super().__init__(error_message_template, column_name)
        self.dependent_column = dependent_column
        self.dependency_checker = dependency_checker
    
    def validate(self, data: pd.Series, df: pd.DataFrame = None) -> Tuple[bool, List[str]]:
        """Validate the dependency between columns.
        
        Args:
            data: Primary column series to validate
            df: Full DataFrame containing both columns
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
            
        Raises:
            ValueError: If df is not provided
        """
        if df is None:
            return False, ["DataFrame must be provided for column dependency validation"]
        
        if self.dependent_column not in df.columns:
            return False, [f"Dependent column '{self.dependent_column}' not found in DataFrame"]
        
        # Get primary and dependent columns
        primary_col = data
        dependent_col = df[self.dependent_column]
        
        # Check dependency for non-null values
        mask = ~primary_col.isna() & ~dependent_col.isna()
        failures = []
        
        for i, (p, d) in enumerate(zip(primary_col[mask], dependent_col[mask])):
            if not self.dependency_checker(p, d):
                failures.append(i)
        
        if failures:
            # Get a sample of failing rows
            sample_idxs = failures[:3]
            sample_values = []
            for idx in sample_idxs:
                row_idx = mask.index[idx]
                sample_values.append({
                    self.column_name: primary_col.iloc[idx],
                    self.dependent_column: dependent_col.iloc[idx]
                })
            
            error = self.format_error(
                dependent_column=self.dependent_column,
                count=len(failures),
                sample=str(sample_values)
            )
            return False, [error]
        
        return True, []


class RowValidator(DataValidator):
    """Validator that checks entire rows using a custom validation function."""
    
    def __init__(
        self,
        validation_func: Callable[[pd.Series], bool],
        column_name: str = None,
        error_message_template: str = "Row validation failed"
    ) -> None:
        """Initialize the row validator.
        
        Args:
            validation_func: Function that validates a row (returns True if valid)
            column_name: Optional column to associate with validation
            error_message_template: Template for error messages
        """
        super().__init__(error_message_template, column_name)
        self.validation_func = validation_func
    
    def validate(self, data: pd.Series, df: pd.DataFrame = None) -> Tuple[bool, List[str]]:
        """Validate each row in the DataFrame.
        
        Args:
            data: Not used for row validation
            df: Full DataFrame to validate rows
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
            
        Raises:
            ValueError: If df is not provided
        """
        if df is None:
            return False, ["DataFrame must be provided for row validation"]
        
        failures = []
        
        # Apply validation function to each row
        for idx, row in df.iterrows():
            try:
                if not self.validation_func(row):
                    failures.append(idx)
            except Exception as e:
                # Count exceptions as validation failures
                failures.append(idx)
        
        if failures:
            # Get a sample of failing rows
            sample_idxs = failures[:3]
            sample_rows = [df.loc[idx].to_dict() for idx in sample_idxs]
            
            error = self.format_error(
                count=len(failures),
                sample=str(sample_rows)
            )
            return False, [error]
        
        return True, []


class SchemaBuilder:
    """Builder for creating DataFrameSchema objects with a fluent API."""
    
    def __init__(self, allow_extra_columns: bool = False, require_all_columns: bool = True) -> None:
        """Initialize the schema builder.
        
        Args:
            allow_extra_columns: Whether to allow extra columns in validation
            require_all_columns: Whether to require all schema columns to be present
        """
        self.columns = []
        self.allow_extra_columns = allow_extra_columns
        self.require_all_columns = require_all_columns
        self.row_count_min = None
        self.row_count_max = None
    
    def column(
        self,
        name: str,
        data_type: DataType,
        nullable: bool = True,
        unique: bool = False
    ) -> "SchemaBuilder":
        """Add a column to the schema.
        
        Args:
            name: Column name
            data_type: Data type
            nullable: Whether null values are allowed
            unique: Whether values must be unique
            
        Returns:
            SchemaBuilder: Self for method chaining
        """
        col_schema = ColumnSchema(
            name=name,
            data_type=data_type,
            nullable=nullable,
            unique=unique
        )
        self.columns.append(col_schema)
        return self
    
    def with_min_value(self, min_value: Union[int, float, datetime.datetime]) -> "SchemaBuilder":
        """Set minimum value for the most recently added column.
        
        Args:
            min_value: Minimum allowed value
            
        Returns:
            SchemaBuilder: Self for method chaining
            
        Raises:
            ValueError: If no columns have been added yet
        """
        if not self.columns:
            raise ValueError("Add a column before setting min_value")
        self.columns[-1].min_value = min_value
        return self
    
    def with_max_value(self, max_value: Union[int, float, datetime.datetime]) -> "SchemaBuilder":
        """Set maximum value for the most recently added column.
        
        Args:
            max_value: Maximum allowed value
            
        Returns:
            SchemaBuilder: Self for method chaining
            
        Raises:
            ValueError: If no columns have been added yet
        """
        if not self.columns:
            raise ValueError("Add a column before setting max_value")
        self.columns[-1].max_value = max_value
        return self
    
    def with_min_length(self, min_length: int) -> "SchemaBuilder":
        """Set minimum length for the most recently added column.
        
        Args:
            min_length: Minimum allowed length
            
        Returns:
            SchemaBuilder: Self for method chaining
            
        Raises:
            ValueError: If no columns have been added yet
        """
        if not self.columns:
            raise ValueError("Add a column before setting min_length")
        self.columns[-1].min_length = min_length
        return self
    
    def with_max_length(self, max_length: int) -> "SchemaBuilder":
        """Set maximum length for the most recently added column.
        
        Args:
            max_length: Maximum allowed length
            
        Returns:
            SchemaBuilder: Self for method chaining
            
        Raises:
            ValueError: If no columns have been added yet
        """
        if not self.columns:
            raise ValueError("Add a column before setting max_length")
        self.columns[-1].max_length = max_length
        return self
    
    def with_regex(self, pattern: str) -> "SchemaBuilder":
        """Set regex pattern for the most recently added column.
        
        Args:
            pattern: Regex pattern to match
            
        Returns:
            SchemaBuilder: Self for method chaining
            
        Raises:
            ValueError: If no columns have been added yet
        """
        if not self.columns:
            raise ValueError("Add a column before setting regex_pattern")
        self.columns[-1].regex_pattern = pattern
        return self
    
    def with_allowed_values(self, values: List[Any]) -> "SchemaBuilder":
        """Set allowed values for the most recently added column.
        
        Args:
            values: List of allowed values
            
        Returns:
            SchemaBuilder: Self for method chaining
            
        Raises:
            ValueError: If no columns have been added yet
        """
        if not self.columns:
            raise ValueError("Add a column before setting allowed_values")
        self.columns[-1].allowed_values = values
        return self
    
    def with_row_count(self, min_count: Optional[int] = None, max_count: Optional[int] = None) -> "SchemaBuilder":
        """Set row count constraints for the schema.
        
        Args:
            min_count: Minimum number of rows
            max_count: Maximum number of rows
            
        Returns:
            SchemaBuilder: Self for method chaining
        """
        self.row_count_min = min_count
        self.row_count_max = max_count
        return self
    
    def build(self) -> DataFrameSchema:
        """Build and return the DataFrameSchema.
        
        Returns:
            DataFrameSchema: The constructed schema
        """
        return DataFrameSchema(
            columns=self.columns,
            allow_extra_columns=self.allow_extra_columns,
            require_all_columns=self.require_all_columns,
            row_count_min=self.row_count_min,
            row_count_max=self.row_count_max
        )

