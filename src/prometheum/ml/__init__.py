"""
Machine Learning Framework Integration for Prometheum.

This module provides adapters for integrating popular machine learning frameworks
with Prometheum data processing pipelines. It enables seamless transition from
data preparation to model training, evaluation, and inference.

Available adapters:
- SKLearnAdapter: Scikit-learn integration

Example usage:
```python
from prometheum.ml import get_adapter
from prometheum.processing import Pipeline

# Create and run a data processing pipeline
pipeline = Pipeline(...)
processed_data = pipeline.process(data)

# Get scikit-learn adapter and train a model
adapter = get_adapter('sklearn')
model = adapter.create_model('RandomForestClassifier', n_estimators=100)
X, y = adapter.prepare_data(processed_data, target_column='target')
model = adapter.train(model, X, y)

# Make predictions
predictions = adapter.predict(model, new_data)
```
"""

from .base import (
    MLAdapter,
    ModelInfo,
    FrameworkType
)

# Import framework-specific adapters conditionally
try:
    from .sklearn import SKLearnAdapter
except ImportError:
    SKLearnAdapter = None

def get_adapter(framework: str):
    """
    Get an adapter for the specified ML framework.
    
    Args:
        framework: Name of the ML framework ('sklearn', 'tensorflow', 'pytorch')
        
    Returns:
        An appropriate adapter instance
        
    Raises:
        ImportError: If the requested framework is not available
        ValueError: If the framework is not supported
    """
    framework = framework.lower()
    
    if framework in ('sklearn', 'scikit-learn'):
        if SKLearnAdapter is None:
            raise ImportError("scikit-learn is not installed or could not be imported")
        return SKLearnAdapter()
    
    # Add support for other frameworks as they're implemented
    raise ValueError(f"Unsupported ML framework: {framework}")

__version__ = '0.1.0'

