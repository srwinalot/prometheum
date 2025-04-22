# Prometheum NAS Router OS

A comprehensive Network Attached Storage solution that can be deployed on routers or standalone hardware, providing local storage, backup capabilities, and AI-powered data management.

## Features

- **Custom Alpine Linux Base**: Lightweight, secure operating system optimized for NAS functionality
- **Cross-Platform Backup**: Seamlessly backup iOS, Android, Windows, and Mac devices to your local NAS
- **AI-Powered Data Management**: Local LLM integration for intelligent file organization and search
- **Self-Hosted Services**: Keep your data under your control and off the cloud
- **Containerized Applications**: Run additional services with Podman containerization
- **Scalable Architecture**: Scale based on your hardware resources

## System Requirements

- **Minimum**:
  - 2 CPU cores
  - 4GB RAM
  - 32GB storage
- **Recommended for AI Features**:
  - 4+ CPU cores
  - 16GB+ RAM
  - 512GB+ storage

## Project Structure

```
prometheum/
├── build/                      # Build artifacts and tools
│   ├── alpine/                 # Alpine Linux customization
│   ├── images/                 # System image creation
│   └── packages/               # Custom package builds
├── config/                     # System configurations
│   ├── network/                # Network service configurations
│   ├── storage/                # Storage system configurations
│   ├── containers/             # Container configurations
│   └── security/               # Security settings
├── scripts/                    # System management scripts
│   ├── install/                # Installation scripts
│   ├── backup/                 # Backup service scripts
│   └── update/                 # System update scripts
├── services/                   # Core services
│   ├── api/                    # System API
│   ├── web/                    # Web interface
│   ├── ai/                     # AI data management
│   ├── sync/                   # Device sync services
│   └── discovery/              # Network service discovery
├── clients/                    # Backup clients for different platforms
│   ├── ios/                    # iOS backup client
│   ├── android/                # Android backup client
│   ├── windows/                # Windows backup client
│   └── macos/                  # macOS backup client
└── docs/                       # Documentation
    ├── installation/           # Installation guides
    ├── configuration/          # Configuration guides
    ├── development/            # Development documentation
    └── api/                    # API documentation
```

## Getting Started

The project is currently under development. Check back soon for installation instructions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
from prometheum import DataFrameSchema, ColumnSchema, DataType, SchemaValidator

# Define a schema
schema = DataFrameSchema(
    columns=[
        ColumnSchema(name="id", data_type=DataType.INTEGER, nullable=False, unique=True),
        ColumnSchema(name="value", data_type=DataType.FLOAT, min_value=0, max_value=100),
        ColumnSchema(name="category", data_type=DataType.STRING, 
                    allowed_values=["A", "B", "C"])
    ],
    allow_extra_columns=False
)

# Validate data against schema
validator = SchemaValidator(schema)
validated_data = validator.process(data)
```

### Data Transformation

```python
from prometheum import StandardScaler, MissingValueHandler, OneHotEncoder

# Handle missing values
missing_handler = MissingValueHandler(strategy="mean")
clean_data = missing_handler.fit_transform(data)

# Scale numeric features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clean_data)

# Encode categorical features
encoder = OneHotEncoder(columns=["category"])
encoded_data = encoder.fit_transform(scaled_data)
```

### Pipeline Creation

```python
from prometheum import PipelineBuilder

# Create a pipeline
pipeline = (
    PipelineBuilder("Data Processing Pipeline")
    .add(MissingValueHandler(strategy="mean"))
    .add(StandardScaler())
    .add(OneHotEncoder(columns=["category"]))
    .pipeline
)

# Process data through the pipeline
result = pipeline.fit_transform(data)

# Access results and execution metadata
processed_df = result.data
execution_metadata = result.metadata["execution"]
```

## Documentation

Comprehensive documentation with examples and API reference is available in the `docs` directory.
## Core Components

Prometheum is organized into several key modules:

### Data Module

The data module provides tools for data loading and validation:

- **Loaders**: `CSVDataLoader`, `JSONDataLoader`, `SQLDataLoader`, `URLDataLoader`
- **Parsers**: `DataParser`, `SchemaValidator`, `TypeConverter`
- **Schema**: `ColumnSchema`, `DataFrameSchema`, `DataType`

### Processing Module

The processing module includes data transformation and pipeline tools:

- **Transformers**: `StandardScaler`, `MinMaxScaler`, `MissingValueHandler`, `OneHotEncoder`, `ColumnSelector`
- **Pipeline**: `Pipeline`, `PipelineBuilder`, `PipelineStep`

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone git@github.com:srwinalot/prometheum.git
cd prometheum

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

The project structure follows standard Python package conventions:

```
prometheum/
├── src/prometheum/       # Main package source code
│   ├── core/             # Core components and base classes
│   ├── data/             # Data loading and validation
│   └── processing/       # Data transformation and pipelines
├── tests/                # Test suite
├── docs/                 # Documentation
└── resources/            # Additional resources
```

### Creating Custom Components

Extend the framework with your own components:

```python
from prometheum import DataTransformer, DataFrameContainer

class MyCustomTransformer(DataTransformer):
    def __init__(self, param=1):
        super().__init__()
        self.param = param
        
    def fit(self, data: DataFrameContainer) -> None:
        # Fit logic here
        pass
        
    def transform(self, data: DataFrameContainer) -> DataFrameContainer:
        # Transformation logic here
        df = data.data.copy()
        # ... your transformations ...
        return DataFrameContainer(df, data.metadata)
```

### Running Tests

```bash
pytest
        }

## Extending the Framework

Prometheum is designed to be easily extended with custom components. Here are some examples:

### Creating a Custom Transformer

```python
from prometheum import DataTransformer, DataFrameContainer

class LogTransformer(DataTransformer):
    """Transformer that applies a logarithmic transformation to numeric columns."""
    
    def __init__(self, columns=None, base=10, offset=1.0):
        super().__init__()
        self.columns = columns
        self.base = base
        self.offset = offset
    
    def fit(self, data: DataFrameContainer) -> None:
        # No fitting needed for this transformer
        pass
        
    def transform(self, data: DataFrameContainer) -> DataFrameContainer:
        df = data.data.copy()
        
        # Determine columns to transform
        columns_to_transform = self.columns or df.select_dtypes(include=['number']).columns
        
        # Apply log transformation
        for col in columns_to_transform:
            if col in df.columns:
                df[col] = np.log(df[col] + self.offset) / np.log(self.base)
        
        # Update metadata
        new_metadata = {
            **data.metadata,
            "log_transform_applied": True,
            "log_base": self.base,
            "log_offset": self.offset
        }
        
        return DataFrameContainer(df, new_metadata)
```

### Creating a Custom Data Loader

```python
from prometheum import DataLoader, DataFrameContainer
import pandas as pd

class ExcelDataLoader(DataLoader):
    """Data loader for Excel files."""
    
    def __init__(self, filepath, sheet_name=0, **kwargs):
        super().__init__()
        self.filepath = filepath
        self.sheet_name = sheet_name
        self.kwargs = kwargs
    
    def load(self) -> DataFrameContainer:
        # Load Excel file
        df = pd.read_excel(self.filepath, sheet_name=self.sheet_name, **self.kwargs)
        
        # Create metadata
        metadata = {
            "source": str(self.filepath),
            "format": "excel",
            "sheet_name": self.sheet_name,
            "row_count": len(df),
            "column_count": len(df.columns)
        }
        
        return DataFrameContainer(df, metadata)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue to discuss potential changes or enhancements.
