"""
Data loader implementations for the Prometheum framework.

This module provides concrete implementations of DataLoader classes for
various data sources including files (CSV, JSON, etc.) and databases.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from prometheum.core.base import DataContainer, DataFrameContainer, DataLoader, Identifiable, Configurable
from prometheum.core.exceptions import (
    ConfigurationError, 
    DataLoadError, 
    FileSystemError,
    ValidationError
)


class FileDataLoader(DataLoader[Any]):
    """Base class for file-based data loaders."""
    
    def __init__(
        self, 
        filepath: Union[str, Path], 
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the file data loader.
        
        Args:
            filepath: Path to the file to load
            config: Optional configuration parameters
            
        Raises:
            ConfigurationError: If filepath is invalid
        """
        Identifiable.__init__(self)
        Configurable.__init__(self, config)
        self.filepath = Path(filepath)
        if not isinstance(filepath, (str, Path)):
            raise ConfigurationError(
                "filepath must be a string or Path object",
                {"provided_type": type(filepath).__name__}
            )
    
    def validate_file_exists(self) -> bool:
        """Check if the file exists.
        
        Returns:
            bool: True if file exists, False otherwise
            
        Raises:
            FileSystemError: If file does not exist
        """
        if not self.filepath.exists():
            raise FileSystemError(
                f"File not found: {self.filepath}",
                path=str(self.filepath),
                operation="read"
            )
        return True


class CSVDataLoader(FileDataLoader):
    """Data loader for CSV files."""
    
    def __init__(
        self, 
        filepath: Union[str, Path],
        delimiter: str = ",",
        encoding: str = "utf-8",
        header: Union[int, List[int], None] = 0,
        usecols: Optional[List[Union[str, int]]] = None,
        dtype: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the CSV data loader.
        
        Args:
            filepath: Path to the CSV file
            delimiter: Column delimiter character
            encoding: File encoding
            header: Row number(s) to use as column names
            usecols: List of columns to read
            dtype: Dict of column data types
            config: Additional configuration parameters
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        super().__init__(filepath, config)
        self.delimiter = delimiter
        self.encoding = encoding
        self.header = header
        self.usecols = usecols
        self.dtype = dtype
    
    def load(self) -> DataFrameContainer:
        """Load data from the CSV file into a DataFrame.
        
        Returns:
            DataFrameContainer: Container with the loaded DataFrame
            
        Raises:
            DataLoadError: If loading fails
            FileSystemError: If file does not exist
        """
        try:
            self.validate_file_exists()
            
            # Read the CSV file
            df = pd.read_csv(
                self.filepath,
                delimiter=self.delimiter,
                encoding=self.encoding,
                header=self.header,
                usecols=self.usecols,
                dtype=self.dtype,
                **self._config
            )
            
            # Create metadata
            metadata = {
                "source": str(self.filepath),
                "format": "csv",
                "delimiter": self.delimiter,
                "encoding": self.encoding,
                "row_count": len(df),
                "column_count": len(df.columns)
            }
            
            return DataFrameContainer(df, metadata)
            
        except FileSystemError as e:
            # Re-raise the original error
            raise e
        except Exception as e:
            # Convert other exceptions to DataLoadError
            raise DataLoadError(
                f"Failed to load CSV file: {str(e)}",
                source=str(self.filepath),
                details={"original_error": str(e)}
            )
    
    def validate(self, data: DataFrameContainer) -> bool:
        """Validate the loaded data.
        
        Args:
            data: The data container to validate
            
        Returns:
            bool: True if data is valid
            
        Raises:
            ValidationError: If validation fails
        """
        df = data.data
        
        # Check if DataFrame is empty
        if df.empty:
            raise ValidationError(
                "CSV data is empty",
                details={"source": str(self.filepath)}
            )
        
        # Additional validation logic can be added here
        return True


class JSONDataLoader(FileDataLoader):
    """Data loader for JSON files."""
    
    def __init__(
        self, 
        filepath: Union[str, Path],
        orient: str = "records",
        encoding: str = "utf-8",
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the JSON data loader.
        
        Args:
            filepath: Path to the JSON file
            orient: Expected JSON data orientation
            encoding: File encoding
            config: Additional configuration parameters
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        super().__init__(filepath, config)
        self.orient = orient
        self.encoding = encoding
    
    def load(self) -> Union[DataFrameContainer, DataContainer[Any]]:
        """Load data from the JSON file.
        
        Returns:
            Union[DataFrameContainer, DataContainer]: Container with the loaded data
            
        Raises:
            DataLoadError: If loading fails
            FileSystemError: If file does not exist
        """
        try:
            self.validate_file_exists()
            
            # Open and read the JSON file
            with open(self.filepath, "r", encoding=self.encoding) as f:
                json_data = json.load(f)
            
            # Create metadata
            metadata = {
                "source": str(self.filepath),
                "format": "json",
                "encoding": self.encoding
            }
            
            # Try to convert to DataFrame if possible (based on the structure)
            try:
                # For list-like structures
                if isinstance(json_data, list) or (isinstance(json_data, dict) and self.orient != "columns"):
                    df = pd.json_normalize(json_data)
                    metadata["row_count"] = len(df)
                    metadata["column_count"] = len(df.columns)
                    return DataFrameContainer(df, metadata)
                # For column-oriented data
                elif isinstance(json_data, dict) and self.orient == "columns":
                    df = pd.DataFrame.from_dict(json_data)
                    metadata["row_count"] = len(df)
                    metadata["column_count"] = len(df.columns)
                    return DataFrameContainer(df, metadata)
            except (ValueError, TypeError) as e:
                # If conversion to DataFrame fails, return as raw data
                metadata["conversion_error"] = str(e)
            
            # Return as raw data if not converted to DataFrame
            return DataContainer(json_data, metadata)
            
        except FileSystemError as e:
            # Re-raise the original error
            raise e
        except Exception as e:
            # Convert other exceptions to DataLoadError
            raise DataLoadError(
                f"Failed to load JSON file: {str(e)}",
                source=str(self.filepath),
                details={"original_error": str(e)}
            )
    
    def validate(self, data: Union[DataFrameContainer, DataContainer[Any]]) -> bool:
        """Validate the loaded data.
        
        Args:
            data: The data container to validate
            
        Returns:
            bool: True if data is valid
            
        Raises:
            ValidationError: If validation fails
        """
        # Check if data is empty
        if isinstance(data, DataFrameContainer):
            if data.data.empty:
                raise ValidationError(
                    "JSON data is empty (DataFrame)",
                    details={"source": str(self.filepath)}
                )
        elif isinstance(data.data, (list, dict)):
            if not data.data:
                raise ValidationError(
                    "JSON data is empty",
                    details={"source": str(self.filepath)}
                )
        
        # Additional validation logic can be added here
        return True


class SQLDataLoader(DataLoader[DataFrameContainer]):
    """Data loader for SQL database sources."""
    
    def __init__(
        self, 
        query: str,
        connection: Union[str, Engine],
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the SQL data loader.
        
        Args:
            query: SQL query to execute
            connection: SQLAlchemy connection string or engine
            params: Optional parameters for the SQL query
            config: Additional configuration parameters
            
        Raises:
            ConfigurationError: If query or connection is invalid
        """
        Identifiable.__init__(self)
        Configurable.__init__(self, config)
        self.query = query
        self.params = params or {}
        
        # Handle connection string or engine
        if isinstance(connection, str):
            try:
                self.engine = sa.create_engine(connection)
            except Exception as e:
                raise ConfigurationError(
                    f"Invalid database connection string: {str(e)}",
                    details={"connection_string": connection}
                )
        elif isinstance(connection, Engine):
            self.engine = connection
        else:
            raise ConfigurationError(
                "connection must be a SQLAlchemy Engine or connection string",
                details={"provided_type": type(connection).__name__}
            )
    
    def load(self) -> DataFrameContainer:
        """Load data from the SQL database.
        
        Returns:
            DataFrameContainer: Container with the loaded DataFrame
            
        Raises:
            DataLoadError: If loading fails
        """
        try:
            # Execute the query and load into DataFrame
            df = pd.read_sql(self.query, self.engine, params=self.params)
            
            # Create metadata
            metadata = {
                "source": "sql",
                "row_count": len(df),
                "column_count": len(df.columns),
                "query": self.query,
                "database": str(self.engine.url).split("@")[-1]  # Safe version of connection info
            }
            
            return DataFrameContainer(df, metadata)
            
        except SQLAlchemyError as e:
            raise DataLoadError(
                f"Database query failed: {str(e)}",
                source="sql",
                details={
                    "query": self.query,
                    "database": str(self.engine.url).split("@")[-1],
                    "original_error": str(e)
                }
            )
        except Exception as e:
            raise DataLoadError(
                f"Failed to load SQL data: {str(e)}",
                source="sql",
                details={"original_error": str(e)}
            )
    
    def validate(self, data: DataFrameContainer) -> bool:
        """Validate the loaded data.
        
        Args:
            data: The data container to validate
            
        Returns:
            bool: True if data is valid
            
        Raises:
            ValidationError: If validation fails
        """
        df = data.data
        
        # Check if DataFrame is empty
        if df.empty:
            raise ValidationError(
                "SQL query returned no data",
                details={"query": self.query}
            )
        
        # Additional validation logic can be added here
        return True


class URLDataLoader(DataLoader[Union[DataFrameContainer, DataContainer[Any]]]):
    """Data loader for URL-based sources."""
    
    def __init__(
        self, 
        url: str,
        format: str = "auto",
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the URL data loader.
        
        Args:
            url: URL to load data from
            format: Data format ('auto', 'csv', 'json', etc.)
            headers: Optional HTTP headers for the request
            timeout: Request timeout in seconds
            config: Additional configuration parameters
            
        Raises:
            ConfigurationError: If URL is invalid
        """
        Identifiable.__init__(self)
        Configurable.__init__(self, config)
        self.url = url
        self.format = format
        self.headers = headers or {}
        self.timeout = timeout
        
        # Validate URL
        try:
            self.parsed_url = urllib.parse.urlparse(url)
            if not self.parsed_url.scheme or not self.parsed_url.netloc:
                raise ValueError("Invalid URL format")
        except Exception as e:
            raise ConfigurationError(
                f"Invalid URL: {str(e)}",
                details={"url": url}
            )
    
    def load(self) -> Union[DataFrameContainer, DataContainer[Any]]:
        """Load data from the URL.
        
        Returns:
            Union[DataFrameContainer, DataContainer]: Container with the loaded data
            
        Raises:
            DataLoadError: If loading fails
        """
        try:
            # Create request
            request = urllib.request.Request(self.url, headers=self.headers)
            
            # Fetch data from URL
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                content_type = response.headers.get('Content-Type', '').lower()
                data = response.read()
                
                # Determine format if auto
                format_type = self.format
                if format_type == 'auto':
                    if 'json' in content_type:
                        format_type = 'json'
                    elif 'csv' in content_type:
                        format_type = 'csv'
                    elif 'text/plain' in content_type:
                        # Try to guess from URL
                        if self.url.endswith('.csv'):
                            format_type = 'csv'
                        elif self.url.endswith('.json'):
                            format_type = 'json'
                        else:
                            format_type = 'text'
                    else:
                        format_type = 'raw'
            
            # Create metadata
            metadata = {
                "source": self.url,
                "format": format_type,
                "content_type": content_type,
                "size": len(data)
            }
            
            # Process based on format
            if format_type == 'json':
                json_data = json.loads(data)
                
                # Try to convert to DataFrame if possible
                try:
                    if isinstance(json_data, list):
                        df = pd.json_normalize(json_data)
                        metadata["row_count"] = len(df)
                        metadata["column_count"] = len(df.columns)
                        return DataFrameContainer(df, metadata)
                except Exception:
                    # Return as raw JSON data if conversion to DataFrame fails
                    return DataContainer(json_data, metadata)
                
            elif format_type == 'csv':
                # Try to parse CSV data
                import io
                try:
                    df = pd.read_csv(io.StringIO(data.decode('utf-8')))
                    metadata["row_count"] = len(df)
                    metadata["column_count"] = len(df.columns)
                    return DataFrameContainer(df, metadata)
                except Exception as e:
                    # If CSV parsing fails, return raw data
                    metadata["conversion_error"] = str(e)
                    return DataContainer(data.decode('utf-8'), metadata)
            
            # For other formats, return raw data
            if format_type == 'text':
                return DataContainer(data.decode('utf-8'), metadata)
            else:
                return DataContainer(data, metadata)
            
        except Exception as e:
            # Convert exceptions to DataLoadError
            raise DataLoadError(
                f"Failed to load data from URL: {str(e)}",
                source=self.url,
                details={"original_error": str(e)}
            )
    
    def validate(self, data: Union[DataFrameContainer, DataContainer[Any]]) -> bool:
        """Validate the loaded data.
        
        Args:
            data: The data container to validate
            
        Returns:
            bool: True if data is valid
            
        Raises:
            ValidationError: If validation fails
        """
        # Check if data is empty
        if isinstance(data, DataFrameContainer):
            if data.data.empty:
                raise ValidationError(
                    "URL data is empty (DataFrame)",
                    details={"source": self.url}
                )
        elif isinstance(data.data, (list, dict)):
            if not data.data:
                raise ValidationError(
                    "URL data is empty (JSON/dict/list)",
                    details={"source": self.url}
                )
        elif isinstance(data.data, str):
            if not data.data.strip():
                raise ValidationError(
                    "URL data is empty (text)",
                    details={"source": self.url}
                )
        elif isinstance(data.data, bytes):
            if not data.data:
                raise ValidationError(
                    "URL data is empty (binary)",
                    details={"source": self.url}
                )
        
        # Additional validation logic can be added here
        return True
