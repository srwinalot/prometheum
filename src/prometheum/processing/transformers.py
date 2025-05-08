"""
Data transformers for the Prometheum framework.

This module provides concrete implementations of DataTransformer classes
for various data transformation operations, including scaling, encoding,
and feature engineering.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from prometheum.core.base import DataFrameContainer, DataTransformer
from prometheum.core.exceptions import ProcessingError, TransformationError, FittingError, ValidationError


class ColumnSelector(DataTransformer):
    """Transformer that selects specific columns from a DataFrame."""
    
    def __init__(
        self,
        columns: List[str],
        drop: bool = False,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the column selector.
        
        Args:
            columns: List of column names to select or drop
            drop: If True, drop the specified columns; if False, keep only those columns
            config: Optional configuration parameters
        """
        super().__init__(config)
        self.columns = columns
        self.drop = drop
    
    def fit(self, data: DataFrameContainer) -> None:
        """Validate column names existence.
        
        Args:
            data: Container with DataFrame to validate
            
        Raises:
            FittingError: If any specified column doesn't exist
        """
        df = data.data
        missing_columns = [col for col in self.columns if col not in df.columns]
        
        if missing_columns:
            raise FittingError(
                f"Column(s) not found in DataFrame: {missing_columns}",
                details={"missing_columns": missing_columns}
            )
    
    def transform(self, data: DataFrameContainer) -> DataFrameContainer:
        """Select or drop columns from the DataFrame.
        
        Args:
            data: Container with DataFrame to transform
            
        Returns:
            DataFrameContainer: Container with transformed DataFrame
            
        Raises:
            TransformationError: If transformation fails
        """
        try:
            df = data.data
            
            if self.drop:
                # Drop specified columns
                result_df = df.drop(columns=[col for col in self.columns if col in df.columns])
            else:
                # Keep only specified columns
                available_columns = [col for col in self.columns if col in df.columns]
                result_df = df[available_columns]
            
            # Add transformation info to metadata
            selection_metadata = {
                "column_selection": True,
                "selection_mode": "drop" if self.drop else "keep",
                "selected_columns": self.columns,
                "resulting_columns": list(result_df.columns)
            }
            
            new_metadata = {**data.metadata, **selection_metadata}
            return DataFrameContainer(result_df, new_metadata)
            
        except Exception as e:
            if isinstance(e, TransformationError):
                raise
            raise TransformationError(
                f"Failed to select columns: {str(e)}",
                details={"original_error": str(e)}
            )


class MissingValueHandler(DataTransformer):
    """Transformer that handles missing values in a DataFrame."""
    
def __init__(
        self,
        strategy: str = "mean",
        fill_value: Optional[Any] = None,
        columns: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the missing value handler.
        
        Args:
            strategy: Strategy for filling missing values
                    ('mean', 'median', 'mode', 'constant', 'ffill', 'bfill')
            fill_value: Value to use when strategy is 'constant'
            columns: List of columns to process (if None, process all)
            config: Optional configuration parameters
            
        Raises:
            ValueError: If strategy is invalid
        """
        DataTransformer.__init__(self, config)  # This properly initializes the inheritance chain
        
        valid_strategies = ['mean', 'median', 'mode', 'constant', 'ffill', 'bfill']
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
            
        self.strategy = strategy
        self.fill_value = fill_value
        self.columns = columns
        self.fill_values = {}  # Will store fitted values
    
    def fit(self, data: DataFrameContainer) -> None:
        """Compute filling values based on the strategy.
        
        Args:
            data: Container with DataFrame to fit
            
        Raises:
            FittingError: If fitting fails
        """
        try:
            df = data.data
            
            # Determine columns to process
            columns_to_process = self.columns or df.columns
            
            for column in columns_to_process:
                if column not in df.columns:
                    continue
                
                series = df[column]
                
                # Skip if no missing values
                if not series.isna().any():
                    continue
                
                # Compute fill value based on strategy
                if self.strategy == 'constant':
                    self.fill_values[column] = self.fill_value
                elif is_numeric_dtype(series):
                    if self.strategy == 'mean':
                        self.fill_values[column] = series.mean()
                    elif self.strategy == 'median':
                        self.fill_values[column] = series.median()
                    elif self.strategy == 'mode':
                        mode_result = series.mode()
                        self.fill_values[column] = mode_result[0] if not mode_result.empty else np.nan
                elif self.strategy in ['ffill', 'bfill']:
                    # For forward/backward fill, we don't need pre-computed values
                    pass
                else:
                    # For non-numeric columns, use mode or constant
                    if self.strategy == 'mode':
                        mode_result = series.mode()
                        self.fill_values[column] = mode_result[0] if not mode_result.empty else np.nan
                    else:
                        self.fill_values[column] = self.fill_value
            
        except Exception as e:
            raise FittingError(
                f"Failed to fit missing value handler: {str(e)}",
                details={"original_error": str(e)}
            )
    
    def transform(self, data: DataFrameContainer) -> DataFrameContainer:
        """Fill missing values in the DataFrame.
        
        Args:
            data: Container with DataFrame to transform
            
        Returns:
            DataFrameContainer: Container with transformed DataFrame
            
        Raises:
            TransformationError: If transformation fails
        """
        try:
            df = data.data.copy()
            
            # Determine columns to process
            columns_to_process = self.columns or df.columns
            
            fill_stats = {}
            
            for column in columns_to_process:
                if column not in df.columns:
                    continue
                
                # Count missing values before filling
                missing_count = df[column].isna().sum()
                if missing_count == 0:
                    continue
                
                # Apply filling strategy
                if self.strategy == 'ffill':
                    df[column] = df[column].ffill()
                elif self.strategy == 'bfill':
                    df[column] = df[column].bfill()
                elif column in self.fill_values:
                    df[column] = df[column].fillna(self.fill_values[column])
                
                # Record statistics
                filled_count = missing_count - df[column].isna().sum()
                fill_stats[column] = {
                    "missing_before": int(missing_count),
                    "filled": int(filled_count),
                    "missing_after": int(df[column].isna().sum()),
                    "fill_method": self.strategy,
                    "fill_value": str(self.fill_values.get(column, "N/A"))
                }
            
            # Add transformation info to metadata
            fill_metadata = {
                "missing_value_handling": True,
                "strategy": self.strategy,
                "stats": fill_stats
            }
            
            new_metadata = {**data.metadata, **fill_metadata}
            return DataFrameContainer(df, new_metadata)
            
        except Exception as e:
            if isinstance(e, TransformationError):
                raise
            raise TransformationError(
                f"Failed to fill missing values: {str(e)}",
                details={"original_error": str(e)}
            )


class Scaler(DataTransformer):
    """Base class for scaling numeric columns."""
    
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the scaler.
        
        Args:
            columns: List of columns to scale (if None, scale all numeric columns)
            config: Optional configuration parameters
        """
        super().__init__(config)
        self.columns = columns
        self.scaling_params = {}  # Will store fitted scaling parameters
    
    def _check_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Check and return valid numeric columns.
        
        Args:
            df: DataFrame to check
            
        Returns:
            List[str]: List of valid numeric column names
            
        Raises:
            ValidationError: If no valid numeric columns found
        """
        # Determine columns to process
        if self.columns is not None:
            # Check that specified columns exist
            missing = [col for col in self.columns if col not in df.columns]
            if missing:
                raise ValidationError(f"Columns not found: {missing}")
                
            # Check that specified columns are numeric
            non_numeric = [col for col in self.columns if col in df.columns and not is_numeric_dtype(df[col])]
            if non_numeric:
                raise ValidationError(f"Non-numeric columns cannot be scaled: {non_numeric}")
                
            valid_columns = [col for col in self.columns if col in df.columns and is_numeric_dtype(df[col])]
        else:
            # Use all numeric columns
            valid_columns = [col for col in df.columns if is_numeric_dtype(df[col])]
        
        if not valid_columns:
            raise ValidationError("No valid numeric columns found for scaling")
            
        return valid_columns
    
    def fit(self, data: DataFrameContainer) -> None:
        """Compute scaling parameters.
        
        Args:
            data: Container with DataFrame to fit
            
        Raises:
            FittingError: If fitting fails
        """
        # Implementation depends on specific scaling method
        raise NotImplementedError("Subclasses must implement this method")
    
    def transform(self, data: DataFrameContainer) -> DataFrameContainer:
        """Scale numeric columns in the DataFrame.
        
        Args:
            data: Container with DataFrame to transform
            
        Returns:
            DataFrameContainer: Container with transformed DataFrame
            
        Raises:
            TransformationError: If transformation fails
        """
        # Implementation depends on specific scaling method
        raise NotImplementedError("Subclasses must implement this method")


class StandardScaler(Scaler):
    """Transformer that standardizes numeric columns to mean=0, std=1."""
    
    def fit(self, data: DataFrameContainer) -> None:
        """Compute means and standard deviations for scaling.
        
        Args:
            data: Container with DataFrame to fit
            
        Raises:
            FittingError: If fitting fails
        """
        try:
            df = data.data
            
            # Get valid numeric columns
            valid_columns = self._check_numeric_columns(df)
            
            # Compute means and standard deviations
            for column in valid_columns:
                mean = df[column].mean()
                std = df[column].std()
                
                # Ensure non-zero std to avoid division by zero
                if std == 0:
                    std = 1.0
                
                self.scaling_params[column] = {'mean': mean, 'std': std}
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise FittingError(str(e))
            raise FittingError(
                f"Failed to fit standard scaler: {str(e)}",
                details={"original_error": str(e)}
            )
    
    def transform(self, data: DataFrameContainer) -> DataFrameContainer:
        """Standardize numeric columns in the DataFrame.
        
        Args:
            data: Container with DataFrame to transform
            
        Returns:
            DataFrameContainer: Container with transformed DataFrame
            
        Raises:
            TransformationError: If transformation fails
        """
        try:
            df = data.data.copy()
            
            # Apply scaling to each column
            scaling_stats = {}
            
            for column, params in self.scaling_params.items():
                if column in df.columns:
                    mean = params['mean']
                    std = params['std']
                    
                    # Apply standardization
                    df[column] = (df[column] - mean) / std
                    
                    scaling_stats[column] = {
                        'method': 'standard',
                        'mean': float(mean),
                        'std': float(std),
                        'min_after': float(df[column].min()),
                        'max_after': float(df[column].max())
                    }
            
            # Add transformation info to metadata
            scaling_metadata = {
                'scaling_applied': True,
                'scaling_method': 'standard',
                'scaled_columns': list(scaling_stats.keys()),
                'scaling_stats': scaling_stats
            }
            
            new_metadata = {**data.metadata, **scaling_metadata}
            return DataFrameContainer(df, new_metadata)
            
        except Exception as e:
            if isinstance(e, TransformationError):
                raise
            raise TransformationError(
                f"Failed to apply standard scaling: {str(e)}",
                details={"original_error": str(e)}
            )


class MinMaxScaler(Scaler):
    """Transformer that scales numeric columns to a specified range."""
    
    def __init__(
        self,
        feature_range: Tuple[float, float] = (0, 1),
        columns: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the min-max scaler.
        
        Args:
            feature_range: Target range as (min, max)
            columns: List of columns to scale (if None, scale all numeric columns)
            config: Optional configuration parameters
        """
        super().__init__(columns, config)
        self.feature_range = feature_range
    
    def fit(self, data: DataFrameContainer) -> None:
        """Compute min and max values for scaling.
        
        Args:
            data: Container with DataFrame to fit
            
        Raises:
            FittingError: If fitting fails
        """
        try:
            df = data.data
            
            # Get valid numeric columns
            valid_columns = self._check_numeric_columns(df)
            
            # Compute min and max values
            for column in valid_columns:
                data_min = df[column].min()
                data_max = df[column].max()
                
                # Handle the case where min equals max
                if data_min == data_max:
                    # In this case, we'll center the constant value within the feature range
                    data_min = data_max - 1  # Create artificial range
                
                self.scaling_params[column] = {'min': data_min, 'max': data_max}
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise FittingError(str(e))
            raise FittingError(
                f"Failed to fit min-max scaler: {str(e)}",
                details={"original_error": str(e)}
            )
    
    def transform(self, data: DataFrameContainer) -> DataFrameContainer:
        """Scale numeric columns to the specified range.
        
        Args:
            data: Container with DataFrame to transform
            
        Returns:
            DataFrameContainer: Container with transformed DataFrame
            
        Raises:
            TransformationError: If transformation fails
        """
        try:
            df = data.data.copy()
            feature_min, feature_max = self.feature_range
            
            # Apply scaling to each column
            scaling_stats = {}
            
            for column, params in self.scaling_params.items():
                if column in df.columns:
                    data_min = params['min']
                    data_max = params['max']
                    
                    # Apply min-max scaling: X_scaled = (X - X_min) / (X_max - X_min) * (feature_max - feature_min) + feature_min
                    df[column] = (df[column] - data_min) / (data_max - data_min) * (feature_max - feature_min) + feature_min
                    
                    scaling_stats[column] = {
                        'method': 'min-max',
                        'original_min': float(data_min),
                        'original_max': float(data_max),
                        'target_min': float(feature_min),
                        'target_max': float(feature_max),
                        'min_after': float(df[column].min()),
                        'max_after': float(df[column].max())
                    }
            
            # Add transformation info to metadata
            scaling_metadata = {
                'scaling_applied': True,
                'scaling_method': 'min-max',
                'feature_range': self.feature_range,
                'scaled_columns': list(scaling_stats.keys()),
                'scaling_stats': scaling_stats
            }
            
            new_metadata = {**data.metadata, **scaling_metadata}
            return DataFrameContainer(df, new_metadata)
            
        except Exception as e:
            if isinstance(e, TransformationError):
                raise
            raise TransformationError(
                f"Failed to apply min-max scaling: {str(e)}",
                details={"original_error": str(e)}
            )


class OneHotEncoder(DataTransformer):
    """Transformer that one-hot encodes categorical columns."""
    
    def __init__(
        self,
        columns: List[str],
        drop_original: bool = True,
        prefix_separator: str = "_",
        sparse: bool = False,
        handle_unknown: str = 'error',
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the one-hot encoder.
        
        Args:
            columns: List of categorical columns to encode
            drop_original: Whether to drop the original columns
            prefix_separator: Separator between column name and category in the new column names
            sparse: Whether to return a sparse matrix (not implemented yet)
            handle_unknown: Strategy for handling unknown categories ('error' or 'ignore')
            config: Optional configuration parameters
        """
        super().__init__(config)
        self.columns = columns
        self.drop_original = drop_original
        self.prefix_separator = prefix_separator
        self.sparse = sparse
        self.handle_unknown = handle_unknown
        self.categories = {}  # Will store fitted categories
    
    def fit(self, data: DataFrameContainer) -> None:
        """Compute unique categories for each column.
        
        Args:
            data: Container with DataFrame to fit
            
        Raises:
            FittingError: If fitting fails
        """
        try:
            df = data.data
            
            # Check that specified columns exist
            missing_columns = [col for col in self.columns if col not in df.columns]
            if missing_columns:
                raise ValidationError(
                    f"Columns not found: {missing_columns}",
                    details={"missing_columns": missing_columns}
                )
            
            # Get unique categories for each column
            for column in self.columns:
                categories = df[column].dropna().unique().tolist()
                self.categories[column] = categories
                
                # Warn about columns with too many categories
                if len(categories) > 100:
                    import warnings
                    warnings.warn(f"Column '{column}' has {len(categories)} categories, which may result in a large number of new columns.")
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise FittingError(str(e))
            raise FittingError(
                f"Failed to fit one-hot encoder: {str(e)}",
                details={"original_error": str(e)}
            )
    
    def transform(self, data: DataFrameContainer) -> DataFrameContainer:
        """One-hot encode categorical columns in the DataFrame.
        
        Args:
            data: Container with DataFrame to transform
            
        Returns:
            DataFrameContainer: Container with transformed DataFrame
            
        Raises:
            TransformationError: If transformation fails
        """
        try:
            df = data.data.copy()
            encoding_stats = {}
            
            for column, categories in self.categories.items():
                if column not in df.columns:
                    continue
                    
                # Create dummy variables
                dummies = pd.get_dummies(
                    df[column],
                    prefix=column,
                    prefix_sep=self.prefix_separator
                )
                
                # Handle unknown categories
                if self.handle_unknown == 'ignore':
                    # Only keep known categories
                    known_dummy_cols = [f"{column}{self.prefix_separator}{cat}" for cat in categories]
                    dummies = dummies[[col for col in dummies.columns if col in known_dummy_cols]]
                
                # Create missing columns for known categories
                for category in categories:
                    dummy_col = f"{column}{self.prefix_separator}{category}"
                    if dummy_col not in dummies.columns:
                        dummies[dummy_col] = 0
                
                # Add dummy columns to the DataFrame
                df = pd.concat([df, dummies], axis=1)
                
                # Drop original column if requested
                if self.drop_original:
                    df = df.drop(columns=[column])
                
                # Record statistics
                encoding_stats[column] = {
                    'categories': categories,
                    'generated_columns': dummies.columns.tolist(),
                    'dropped_original': self.drop_original
                }
            
            # Add transformation info to metadata
            encoding_metadata = {
                'encoding_applied': True,
                'encoding_method': 'one-hot',
                'encoded_columns': list(encoding_stats.keys()),
                'encoding_stats': encoding_stats
            }
            
            new_metadata = {**data.metadata, **encoding_metadata}
            return DataFrameContainer(df, new_metadata)
            
        except Exception as e:
            if isinstance(e, TransformationError):
                raise
            raise TransformationError(
                f"Failed to apply one-hot encoding: {str(e)}",
                details={"original_error": str(e)}
            )
