"""
Base interfaces and types for ML framework adapters.

This module defines the core abstractions for integrating machine learning frameworks
with Prometheum data processing pipelines. It provides a common interface that
specific framework adapters must implement.
"""

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Type, TypeVar, cast

import pandas as pd
import numpy as np

from prometheum.core.base import DataFrameContainer
from prometheum.core.exceptions import ProcessingError


class FrameworkType(enum.Enum):
    """Supported ML frameworks."""
    SKLEARN = "sklearn"
    TENSORFLOW = "tensorflow" 
    PYTORCH = "pytorch"
    XGBOOST = "xgboost"


class ModelType(enum.Enum):
    """Common types of machine learning models."""
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    CLUSTERER = "clusterer"
    DIMENSIONALITY_REDUCER = "dimensionality_reducer"
    TRANSFORMER = "transformer"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


@dataclass
class ModelInfo:
    """Information about a machine learning model."""
    
    name: str
    model_type: ModelType
    framework: FrameworkType
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Model metadata
    description: Optional[str] = None
    version: Optional[str] = None
    created_at: Optional[str] = None
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Feature information
    feature_names: List[str] = field(default_factory=list)
    target_name: Optional[str] = None
    
    # Serialization info
    serializable: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model info to a dictionary."""
        return {
            "name": self.name,
            "model_type": self.model_type.value,
            "framework": self.framework.value,
            "params": self.params,
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at,
            "metrics": self.metrics,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "serializable": self.serializable
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        """Create a ModelInfo instance from a dictionary."""
        model_type = ModelType(data.get("model_type", "custom"))
        framework = FrameworkType(data.get("framework", "sklearn"))
        
        return cls(
            name=data["name"],
            model_type=model_type,
            framework=framework,
            params=data.get("params", {}),
            description=data.get("description"),
            version=data.get("version"),
            created_at=data.get("created_at"),
            metrics=data.get("metrics", {}),
            feature_names=data.get("feature_names", []),
            target_name=data.get("target_name"),
            serializable=data.get("serializable", True)
        )


class MLError(ProcessingError):
    """Base exception for ML-related errors."""
    pass


class ModelCreationError(MLError):
    """Exception raised when model creation fails."""
    pass


class TrainingError(MLError):
    """Exception raised when model training fails."""
    pass


class PredictionError(MLError):
    """Exception raised when making predictions fails."""
    pass


class EvaluationError(MLError):
    """Exception raised when model evaluation fails."""
    pass


class SerializationError(MLError):
    """Exception raised when model serialization/deserialization fails."""
    pass


# Generic type for ML models
Model = TypeVar('Model')


class MLAdapter(ABC):
    """
    Base interface for ML framework adapters.
    
    This abstract class defines the common interface that all ML framework adapters
    must implement. It provides methods for creating, training, evaluating, and
    using machine learning models within the Prometheum ecosystem.
    """
    
    @property
    @abstractmethod
    def framework(self) -> FrameworkType:
        """Get the ML framework type this adapter supports."""
        pass
    
    @abstractmethod
    def create_model(self, model_name: str, **kwargs) -> Any:
        """
        Create a new model instance.
        
        Args:
            model_name: Name of the model class (e.g., 'RandomForestClassifier')
            **kwargs: Parameters to pass to the model constructor
            
        Returns:
            A new model instance
            
        Raises:
            ModelCreationError: If model creation fails
        """
        pass
    
    @abstractmethod
    def get_model_info(self, model: Any) -> ModelInfo:
        """
        Get information about a model.
        
        Args:
            model: The model instance
            
        Returns:
            ModelInfo: Information about the model
        """
        pass
    
    @abstractmethod
    def prepare_data(
        self, 
        data: Union[DataFrameContainer, pd.DataFrame],
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[Any, Optional[Any]]:
        """
        Prepare data for model training or prediction.
        
        Args:
            data: Input data (DataFrame or DataFrameContainer)
            target_column: Name of target column (for supervised learning)
            feature_columns: List of feature column names to use
            **kwargs: Additional preparation parameters
            
        Returns:
            Tuple of (X, y) where X is features and y is target (y may be None for unsupervised)
            
        Raises:
            ProcessingError: If data preparation fails
        """
        pass
    
    @abstractmethod
    def train(
        self,
        model: Any,
        X: Any,
        y: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        Train a model.
        
        Args:
            model: The model to train
            X: Feature data
            y: Target data (may be None for unsupervised models)
            **kwargs: Additional training parameters
            
        Returns:
            The trained model
            
        Raises:
            TrainingError: If training fails
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        model: Any,
        X: Any,
        **kwargs
    ) -> Any:
        """
        Make predictions with a model.
        
        Args:
            model: The trained model
            X: Feature data
            **kwargs: Additional prediction parameters
            
        Returns:
            Model predictions
            
        Raises:
            PredictionError: If prediction fails
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        model: Any,
        X: Any,
        y: Any,
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate a model's performance.
        
        Args:
            model: The trained model
            X: Feature data
            y: True target values
            metrics: List of metric names to compute
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary of metric names to values
            
        Raises:
            EvaluationError: If evaluation fails
        """
        pass
    
    @abstractmethod
    def save_model(self, model: Any, path: str) -> None:
        """
        Save a model to disk.
        
        Args:
            model: The model to save
            path: Path where the model should be saved
            
        Raises:
            SerializationError: If saving fails
        """
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> Any:
        """
        Load a model from disk.
        
        Args:
            path: Path from which to load the model
            
        Returns:
            The loaded model
            
        Raises:
            SerializationError: If loading fails
        """
        pass
    
    def create_pipeline_transformer(self, model: Any, **kwargs) -> Any:
        """
        Create a Prometheum transformer from this model.
        
        Args:
            model: The trained model
            **kwargs: Additional parameters for the transformer
            
        Returns:
            A transformer that can be used in a Prometheum pipeline
            
        Raises:
            ProcessingError: If transformer creation fails
        """
        raise NotImplementedError(
            f"Pipeline transformer creation not implemented for {self.framework.value}"
        )

