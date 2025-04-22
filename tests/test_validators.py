"""
Unit tests for Prometheum validators.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

# Add the parent directory to the path to import prometheum
sys.path.append(str(Path(__file__).parent.parent))

from prometheum import (
    DataType,
    ColumnSchema,
    DataFrameSchema,
    SchemaValidator,
    DataFrameContainer
)


class TestValidators(unittest.TestCase):
    """Test validator functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = pd.DataFrame({
            'id': range(1, 11),
            'value': [10, 20, np.nan, 40, 50, 60, 70, np.nan, 90, 100],
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
            'timestamp': pd.date_range(start='2023-01-01', periods=10)
        })
        self.container = DataFrameContainer(self.test_data, {"source": "test"})
    
    def test_not_null_validation(self):
        """Test not null validation."""
        # Schema requiring value to be not null
        schema = DataFrameSchema(
            columns=[
                ColumnSchema(
                    name="value", 
                    data_type=DataType.FLOAT, 
                    nullable=False
                )
            ],
            require_all_columns=False
        )
        
        validator = SchemaValidator(schema, raise_on_error=False)
        result = validator.process(self.container)
        
        # Should fail because value has nulls
        self.assertFalse(result.metadata.get("validation_passed", True))
    
    def test_range_validation(self):
        """Test range validation."""
        # Schema with min and max values
        schema = DataFrameSchema(
            columns=[
                ColumnSchema(
                    name="value", 
                    data_type=DataType.FLOAT, 
                    min_value=0, 
                    max_value=50
                )
            ],
            require_all_columns=False
        )
        
        validator = SchemaValidator(schema, raise_on_error=False)
        result = validator.process(self.container)
        
        # Should fail because value has entries > 50
        self.assertFalse(result.metadata.get("validation_passed", True))
    
    def test_allowed_values_validation(self):
        """Test allowed values validation."""
        # Schema with allowed values
        schema = DataFrameSchema(
            columns=[
                ColumnSchema(
                    name="category", 
                    data_type=DataType.STRING, 
                    allowed_values=["A", "B"]
                )
            ],
            require_all_columns=False
        )
        
        validator = SchemaValidator(schema, raise_on_error=False)
        result = validator.process(self.container)
        
        # Should fail because category has values not in allowed_values
        self.assertFalse(result.metadata.get("validation_passed", True))
    
    def test_unique_validation(self):
        """Test unique validation."""
        # Schema requiring id to be unique
        schema = DataFrameSchema(
            columns=[
                ColumnSchema(
                    name="id", 
                    data_type=DataType.INTEGER, 
                    unique=True
                ),
                ColumnSchema(
                    name="category", 
                    data_type=DataType.STRING, 
                    unique=False
                )
            ],
            require_all_columns=False
        )
        
        validator = SchemaValidator(schema, raise_on_error=False)
        result = validator.process(self.container)
        
        # Should pass because id is unique
        self.assertTrue(result.metadata.get("validation_passed", False))
    
    def test_data_type_validation(self):
        """Test data type validation."""
        # Schema with incorrect data type
        schema = DataFrameSchema(
            columns=[
                ColumnSchema(
                    name="category", 
                    data_type=DataType.INTEGER  # Wrong type
                )
            ],
            require_all_columns=False
        )
        
        validator = SchemaValidator(schema, raise_on_error=False)
        result = validator.process(self.container)
        
        # Should fail because category is STRING, not INTEGER
        self.assertFalse(result.metadata.get("validation_passed", True))
    
    def test_passing_validation(self):
        """Test a schema that should pass."""
        # Schema with correct constraints
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
                    nullable=True,  # Allows nulls
                    min_value=0
                ),
                ColumnSchema(
                    name="category", 
                    data_type=DataType.STRING, 
                    nullable=False,
                    allowed_values=["A", "B", "C"]  # All values present
                )
            ],
            require_all_columns=False
        )
        
        validator = SchemaValidator(schema, raise_on_error=False)
        result = validator.process(self.container)
        
        # Should pass all validations
        self.assertTrue(result.metadata.get("validation_passed", False))


if __name__ == "__main__":
    unittest.main()

