"""
Scikit-learn Adapter for Prometheum ML Framework.

This module provides an adapter for integrating scikit-learn models with
Prometheum data processing pipelines. It handles model creation, training,
evaluation, and serialization.
"""

import importlib
import inspect
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Type, cast

import numpy as np
import pandas as pd

try:
    import sklearn
    from sklearn.base import BaseEstimator
    from sklearn import metrics as sklearn_metrics
except ImportError:
    raise ImportError(
        "scikit-learn is required for this module. "
        "Install it with: pip install scikit-learn"
    )

from prometheum.core.base import DataFrameContainer, DataTransformer
from prometheum.core.exceptions import ProcessingError
from prometheum.ml.base import (
    FrameworkType,
    MLAdapter,
    ModelCreationError,
    ModelInfo,
    ModelType,
    TrainingError,
    PredictionError,
    EvaluationError,
    SerializationError,
)


def _get_sklearn_model_type(model: Any) -> ModelType:
    """
    Determine the type of a scikit-learn model.
    
    Args:
        model: A scikit-learn model instance
        
    Returns:
        ModelType: The type of the model
    """
    from sklearn.base import (
        ClassifierMixin,
        RegressorMixin,
        ClusterMixin,
        TransformerMixin,
    )
    
    if hasattr(model, "__module__") and "ensemble" in model.__module__:
        return ModelType.ENSEMBLE
    elif isinstance(model, ClassifierMixin):
        return ModelType.CLASSIFIER
    elif isinstance(model, RegressorMixin):
        return ModelType.REGRESSOR
    elif isinstance(model, ClusterMixin):
        return ModelType.CLUSTERER
    elif isinstance(model, TransformerMixin):
        return ModelType.TRANSFORMER
    
    return ModelType.CUSTOM


def _extract_model_params(model: Any) -> Dict[str, Any]:
    """
    Extract parameters from a scikit-learn model.
    
    Args:
        model: A scikit-learn model instance
        
    Returns:
        Dict[str, Any]: The model's parameters
    """
    if hasattr(model, "get_params"):
        params = model.get_params()
        # Convert non-serializable params to strings
        for key, value in params.items():
            if hasattr(value, "__name__"):
                params[key] = value.__name__
            elif not isinstance(value, (str, int, float, bool, list, dict, tuple, type(None))):
                params[key] = str(value)
        return params
    return {}


class SKLearnAdapter(MLAdapter):
    """
    Adapter for scikit-learn models.
    
    This adapter provides methods to create, train, evaluate, and save scikit-learn
    models within the Prometheum ecosystem.
    """
    
    @property
    def framework(self) -> FrameworkType:
        """Get the ML framework type."""
        return FrameworkType.SKLEARN
    
    def create_model(self, model_name: str, **kwargs) -> Any:
        """
        Create a scikit-learn model instance.
        
        Args:
            model_name: Name of the model class (e.g., 'RandomForestClassifier')
            **kwargs: Parameters to pass to the model constructor
            
        Returns:
            A new model instance
            
        Raises:
            ModelCreationError: If model creation fails
        """
        try:
            # Import the necessary scikit-learn module
            module_parts = model_name.split('.')
            if len(module_parts) == 1:
                # If no module specified, try common modules
                model_cls = None
                common_modules = [
                    "sklearn.ensemble",
                    "sklearn.linear_model",
                    "sklearn.tree",
                    "sklearn.svm",
                    "sklearn.neighbors",
                    "sklearn.cluster",
                    "sklearn.decomposition",
                    "sklearn.preprocessing",
                ]
                
                for module_name in common_modules:
                    try:
                        module = importlib.import_module(module_name)
                        if hasattr(module, model_name):
                            model_cls = getattr(module, model_name)
                            break
                    except (ImportError, AttributeError):
                        continue
                
                if model_cls is None:
                    raise ModelCreationError(
                        f"Could not find model class '{model_name}' in common scikit-learn modules",
                        details={"model_name": model_name}
                    )
            else:
                # If module is specified, import it directly
                module_name = '.'.join(module_parts[:-1])
                class_name = module_parts[-1]
                
                module = importlib.import_module(module_name)
                model_cls = getattr(module, class_name)
            
            # Create the model instance
            model = model_cls(**kwargs)
            
            # Verify it's a scikit-learn estimator
            if not isinstance(model, BaseEstimator):
                raise ModelCreationError(
                    f"Created object is not a scikit-learn estimator: {type(model)}",
                    details={"model_type": str(type(model))}
                )
            
            return model
            
        except (ImportError, AttributeError) as e:
            raise ModelCreationError(
                f"Failed to import model class '{model_name}': {str(e)}",
                details={"original_error": str(e)}
            )
        except Exception as e:
            if isinstance(e, ModelCreationError):
                raise
            raise ModelCreationError(
                f"Failed to create model '{model_name}': {str(e)}",
                details={"original_error": str(e)}
            )
    
    def get_model_info(self, model: Any) -> ModelInfo:
        """
        Get information about a scikit-learn model.
        
        Args:
            model: The scikit-learn model instance
            
        Returns:
            ModelInfo: Information about the model
        """
        if not isinstance(model, BaseEstimator):
            raise ValueError("Model is not a scikit-learn estimator")
        
        model_type = _get_sklearn_model_type(model)
        model_name = type(model).__name__
        params = _extract_model_params(model)
        
        # Extract feature names if available
        feature_names = []
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        
        # Extract created timestamp
        created_at = datetime.now().isoformat()
        
        # Check if model has been trained
        is_fitted = False
        try:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(model)
            is_fitted = True
        except Exception:
            pass
        
        return ModelInfo(
            name=model_name,
            model_type=model_type,
            framework=FrameworkType.SKLEARN,
            params=params,
            description=model.__doc__.split("\n")[0] if model.__doc__ else None,
            version=sklearn.__version__,
            created_at=created_at,
            metrics={},
            feature_names=feature_names,
            target_name=None,
            serializable=True
        )
    
    def prepare_data(
        self, 
        data: Union[DataFrameContainer, pd.DataFrame],
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare data for scikit-learn model training or prediction.
        
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
        try:
            # Extract DataFrame if wrapped in a container
            if isinstance(data, DataFrameContainer):
                df = data.data
            else:
                df = data
            
            # Validate input
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Expected pandas DataFrame, got {type(df)}")
            
            # Select feature columns if specified
            if feature_columns:
                missing_cols = [col for col in feature_columns if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Feature columns not found in data: {missing_cols}")
                X_df = df[feature_columns]
            else:
                # Use all columns except target
                if target_column:
                    if target_column not in df.columns:
                        raise ValueError(f"Target column '{target_column}' not found in data")
                    X_df = df.drop(columns=[target_column])
                else:
                    X_df = df
            
            # Convert to numpy array
            X = X_df.to_numpy()
            
            # Extract target if specified
            y = None
            if target_column:
                if target_column not in df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in data")
                y = df[target_column].to_numpy()
            
            return X, y
            
        except Exception as e:
            if isinstance(e, ProcessingError):
                raise
            raise ProcessingError(
                f"Failed to prepare data: {str(e)}",
                details={"original_error": str(e)}
            )
    
    def train(
        self,
        model: Any,
        X: Any,
        y: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        Train a scikit-learn model.
        
        Args:
            model: The scikit-learn model to train
            X: Feature data
            y: Target data (may be None for unsupervised models)
            **kwargs: Additional training parameters
            
        Returns:
            The trained model
            
        Raises:
            TrainingError: If training fails
        """
        try:
            # Validate inputs
            if not isinstance(model, BaseEstimator):
                raise ValueError("Model is not a scikit-learn estimator")
            
            # Determine if model is supervised or unsupervised
            from sklearn.base import ClassifierMixin, RegressorMixin
            is_supervised = isinstance(model, (ClassifierMixin, RegressorMixin))
            
            # Check if we have target data for supervised models
            if is_supervised and y is None:
                raise ValueError("Target data (y) is required for supervised models")
            
            # Apply fit method with appropriate arguments
            if is_supervised:
                model.fit(X, y, **kwargs)
            else:
                model.fit(X, **kwargs)
            
            return model
            
        except Exception as e:
            if isinstance(e, TrainingError):
                raise
            raise TrainingError(
                f"Failed to train model: {str(e)}",
                details={"original_error": str(e)}
            )
    
    def predict(
        self,
        model: Any,
        X: Any,
        **kwargs
    ) -> np.ndarray:
        """
        Make predictions with a scikit-learn model.
        
        Args:
            model: The trained scikit-learn model
            X: Feature data
            **kwargs: Additional prediction parameters
            
        Returns:
            Model predictions
            
        Raises:
            PredictionError: If prediction fails
        """
        try:
            # Validate inputs
            if not isinstance(model, BaseEstimator):
                raise ValueError("Model is not a scikit-learn estimator")
            
            # Determine prediction method to use
            from sklearn.base import ClassifierMixin
            
            # For classifiers, check if we want probabilities
            if isinstance(model, ClassifierMixin) and kwargs.get("return_probabilities", False):
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(X)
                else:
                    raise ValueError("Model does not support probability predictions")
            
            # Standard prediction
            return model.predict(X)
            
        except Exception as e:
            if isinstance(e, PredictionError):
                raise
            raise PredictionError(
                f"Failed to make predictions: {str(e)}",
                details={"original_error": str(e)}
            )
    
    def evaluate(
        self,
        model: Any,
        X: Any,
        y: Any,
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate a scikit-learn model's performance.
        
        Args:
            model: The trained scikit-learn model
            X: Feature data
            y: True target values
            metrics: List of metric names to compute
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary of metric names to values
            
        Raises:
            EvaluationError: If evaluation fails
        """
        try:
            # Validate inputs
            if not isinstance(model, BaseEstimator):
                raise ValueError("Model is not a scikit-learn estimator")
            
            # Make predictions
            y_pred = self.predict(model, X)
            
            # Determine model type to use appropriate metrics
            model_type = _get_sklearn_model_type(model)
            
            # Use provided metrics or default ones based on model type
            if metrics is None:
                if model_type == ModelType.CLASSIFIER:
                    metrics = ["accuracy", "precision", "recall", "f1"]
                elif model_type == ModelType.REGRESSOR:
                    metrics = ["r2", "mae", "mse", "rmse"]
                else:
                    metrics = []
            
            # Calculate all requested metrics
            results = {}
            
            for metric_name in metrics:
                metric_name = metric_name.lower()
                
                # Classification metrics
                if metric_name == "accuracy":
                    results[metric_name] = float(sklearn_metrics.accuracy_score(y, y_pred))
                elif metric_name in ("precision", "precision_weighted"):
                    results[metric_name] = float(sklearn_metrics.precision_score(y, y_pred, average="weighted"))
                elif metric_name in ("recall", "recall_weighted"):
                    results[metric_name] = float(sklearn_metrics.recall_score(y, y_pred, average="weighted"))
                elif metric_name in ("f1", "f1_weighted"):
                    results[metric_name] = float(sklearn_metrics.f1_score(y, y_pred, average="weighted"))
                
                # Regression metrics
                elif metric_name == "r2":
                    results[metric_name] = float(sklearn_metrics.r2_score(y, y_pred))
                elif metric_name in ("mae", "mean_absolute_error"):
                    results[metric_name] = float(sklearn_metrics.mean_absolute_error(y, y_pred))
                elif metric_name in ("mse", "mean_squared_error"):
                    results[

"""
Scikit-learn Adapter for Prometheum ML Framework.

This module provides an adapter for integrating scikit-learn models with
Prometheum data processing pipelines. It handles model creation, training,
evaluation, and serialization.
"""

import importlib
import inspect
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Type, cast

import numpy as np
import pandas as pd

try:
    import sklearn
    from sklearn.base import BaseEstimator
    from sklearn import metrics as sklearn_metrics
except ImportError:
    raise ImportError(
        "scikit-learn is required for this module. "
        "Install it with: pip install scikit-learn"
    )

from prometheum.core.base import DataFrameContainer, DataTransformer
from prometheum.core.exceptions import ProcessingError
from prometheum.ml.base import (
    FrameworkType,
    MLAdapter,
    ModelCreationError,
    ModelInfo,
    ModelType,
    TrainingError,
    PredictionError,
    EvaluationError,
    SerializationError,
)


def _get_sklearn_model_type(model: Any) -> ModelType:
    """
    Determine the type of a scikit-learn model.
    
    Args:
        model: A scikit-learn model instance
        
    Returns:
        ModelType: The type of the model
    """
    from sklearn.base import (
        ClassifierMixin,
        RegressorMixin,
        ClusterMixin,
        TransformerMixin,
    )
    
    if hasattr(model, "__module__") and "ensemble" in model.__module__:
        return ModelType.ENSEMBLE
    elif isinstance(model, ClassifierMixin):
        return ModelType.CLASSIFIER
    elif isinstance(model, RegressorMixin):
        return ModelType.REGRESSOR
    elif isinstance(model, ClusterMixin):
        return ModelType.CLUSTERER
    elif isinstance(model, TransformerMixin):
        return ModelType.TRANSFORMER
    
    return ModelType.CUSTOM


def _extract_model_params(model: Any) -> Dict[str, Any]:
    """
    Extract parameters from a scikit-learn model.
    
    Args:
        model: A scikit-learn model instance
        
    Returns:
        Dict[str, Any]: The model's parameters
    """
    if hasattr(model, "get_params"):
        params = model.get_params()
        # Convert non-serializable params to strings
        for key, value in params.items():
            if hasattr(value, "__name__"):
                params[key] = value.__name__
            elif not isinstance(value, (str, int, float, bool, list, dict, tuple, type(None))):
                params[key] = str(value)
        return params
    return {}


class SKLearnAdapter(MLAdapter):
    """
    Adapter for scikit-learn models.
    
    This adapter provides methods to create, train, evaluate, and save scikit-learn
    models within the Prometheum ecosystem.
    """
    
    @property
    def framework(self) -> FrameworkType:
        """Get the ML framework type."""
        return FrameworkType.SKLEARN
    
    def create_model(self, model_name: str, **kwargs) -> Any:
        """
        Create a scikit-learn model instance.
        
        Args:
            model_name: Name of the model class (e.g., 'RandomForestClassifier')
            **kwargs: Parameters to pass to the model constructor
            
        Returns:
            A new model instance
            
        Raises:
            ModelCreationError: If model creation fails
        """
        try:
            # Import the necessary scikit-learn module
            module_parts = model_name.split('.')
            if len(module_parts) == 1:
                # If no module specified, try common modules
                model_cls = None
                common_modules = [
                    "sklearn.ensemble",
                    "sklearn.linear_model",
                    "sklearn.tree",
                    "sklearn.svm",
                    "sklearn.neighbors",
                    "sklearn.cluster",
                    "sklearn.decomposition",
                    "sklearn.preprocessing",
                ]
                
                for module_name in common_modules:
                    try:
                        module = importlib.import_module(module_name)
                        if hasattr(module, model_name):
                            model_cls = getattr(module, model_name)
                            break
                    except (ImportError, AttributeError):
                        continue
                
                if model_cls is None:
                    raise ModelCreationError(
                        f"Could not find model class '{model_name}' in common scikit-learn modules",
                        details={"model_name": model_name}
                    )
            else:
                # If module is specified, import it directly
                module_name = '.'.join(module_parts[:-1])
                class_name = module_parts[-1]
                
                module = importlib.import_module(module_name)
                model_cls = getattr(module, class_name)
            
            # Create the model instance
            model = model_cls(**kwargs)
            
            # Verify it's a scikit-learn estimator
            if not isinstance(model, BaseEstimator):
                raise ModelCreationError(
                    f"Created object is not a scikit-learn estimator: {type(model)}",
                    details={"model_type": str(type(model))}
                )
            
            return model
            
        except (ImportError, AttributeError) as e:
            raise ModelCreationError(
                f"Failed to import model class '{model_name}': {str(e)}",
                details={"original_error": str(e)}
            )
        except Exception as e:
            if isinstance(e, ModelCreationError):
                raise
            raise ModelCreationError(
                f"Failed to create model '{model_name}': {str(e)}",
                details={"original_error": str(e)}
            )
    
    def get_model_info(self, model: Any) -> ModelInfo:
        """
        Get information about a scikit-learn model.
        
        Args:
            model: The scikit-learn model instance
            
        Returns:
            ModelInfo: Information about the model
        """
        if not isinstance(model, BaseEstimator):
            raise ValueError("Model is not a scikit-learn estimator")
        
        model_type = _get_sklearn_model_type(model)
        model_name = type(model).__name__
        params = _extract_model_params(model)
        
        # Extract feature names if available
        feature_names = []
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        
        # Extract created timestamp
        created_at = datetime.now().isoformat()
        
        # Check if model has been trained
        is_fitted = False
        try:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(model)
            is_fitted = True
        except Exception:
            pass
        
        return ModelInfo(
            name=model_name,
            model_type=model_type,
            framework=FrameworkType.SKLEARN,
            params=params,
            description=model.__doc__.split("\n")[0] if model.__doc__ else None,
            version=sklearn.__version__,
            created_at=created_at,
            metrics={},
            feature_names=feature_names,
            target_name=None,
            serializable=True
        )
    
    def prepare_data(
        self, 
        data: Union[DataFrameContainer, pd.DataFrame],
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare data for scikit-learn model training or prediction.
        
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
        try:
            # Extract DataFrame if wrapped in a container
            if isinstance(data, DataFrameContainer):
                df = data.data
            else:
                df = data
            
            # Validate input
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Expected pandas DataFrame, got {type(df)}")
            
            # Select feature columns if specified
            if feature_columns:
                missing_cols = [col for col in feature_columns if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Feature columns not found in data: {missing_cols}")
                X_df = df[feature_columns]
            else:
                # Use all columns except target
                if target_column:
                    if target_column not in df.columns:
                        raise ValueError(f"Target column '{target_column}' not found in data")
                    X_df = df.drop(columns=[target_column])
                else:
                    X_df = df
            
            # Convert to numpy array
            X = X_df.to_numpy()
            
            # Extract target if specified
            y = None
            if target_column:
                if target_column not in df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in data")
                y = df[target_column].to_numpy()
            
            return X, y
            
        except Exception as e:
            if isinstance(e, ProcessingError):
                raise
            raise ProcessingError(
                f"Failed to prepare data: {str(e)}",
                details={"original_error": str(e)}
            )
    
    def train(
        self,
        model: Any,
        X: Any,
        y: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        Train a scikit-learn model.
        
        Args:
            model: The scikit-learn model to train
            X: Feature data
            y: Target data (may be None for unsupervised models)
            **kwargs: Additional training parameters
            
        Returns:
            The trained model
            
        Raises:
            TrainingError: If training fails
        """
        try:
            # Validate inputs
            if not isinstance(model, BaseEstimator):
                raise ValueError("Model is not a scikit-learn estimator")
            
            # Determine if model is supervised or unsupervised
            from sklearn.base import ClassifierMixin, RegressorMixin
            is_supervised = isinstance(model, (ClassifierMixin, RegressorMixin))
            
            # Check if we have target data for supervised models
            if is_supervised and y is None:
                raise ValueError("Target data (y) is required for supervised models")
            
            # Apply fit method with appropriate arguments
            if is_supervised:
                model.fit(X, y, **kwargs)
            else:
                model.fit(X, **kwargs)
            
            return model
            
        except Exception as e:
            if isinstance(e, TrainingError):
                raise
            raise TrainingError(
                f"Failed to train model: {str(e)}",
                details={"original_error": str(e)}
            )
    
    def predict(
        self,
        model: Any,
        X: Any,
        **kwargs
    ) -> np.ndarray:
        """
        Make predictions with a scikit-learn model.
        
        Args:
            model: The trained scikit-learn model
            X: Feature data
            **kwargs: Additional prediction parameters
            
        Returns:
            Model predictions
            
        Raises:
            PredictionError: If prediction fails
        """
        try:
            # Validate inputs
            if not isinstance(model, BaseEstimator):
                raise ValueError("Model is not a scikit-learn estimator")
            
            # Determine prediction method to use
            from sklearn.base import ClassifierMixin
            
            # For classifiers, check if we want probabilities
            if isinstance(model, ClassifierMixin) and kwargs.get("return_probabilities", False):
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(X)
                else:
                    raise ValueError("Model does not support probability predictions")
            
            # Standard prediction
            return model.predict(X)
            
        except Exception as e:
            if isinstance(e, PredictionError):
                raise
            raise PredictionError(
                f"Failed to make predictions: {str(e)}",
                details={"original_error": str(e)}
            )
    
    def evaluate(
        self,
        model: Any,
        X: Any,
        y: Any,
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate a scikit-learn model's performance.
        
        Args:
            model: The trained scikit-learn model
            X: Feature data
            y: True target values
            metrics: List of metric names to compute
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary of metric names to values
            
        Raises:
            EvaluationError: If evaluation fails
        """
        try:
            # Validate inputs
            if not isinstance(model, BaseEstimator):
                raise ValueError("Model is not a scikit-learn estimator")
            
            # Make predictions
            y_pred = self.predict(model, X)
            
            # Determine model type to use appropriate metrics
            model_type = _get_sklearn_model_type(model)
            
            # Use provided metrics or default ones based on model type
            if metrics is None:
                if model_type == ModelType.CLASSIFIER:
                    metrics = ["accuracy", "precision", "recall", "f1"]
                elif model_type == ModelType.REGRESSOR:
                    metrics = ["r2", "mae", "mse", "rmse"]
                else:
                    metrics = []
            
            # Calculate all requested metrics
            results = {}
            
            for metric_name in metrics:
                metric_name = metric_name.lower()
                
                # Classification metrics
                if metric_name == "accuracy":
                    results[metric_name] = float(sklearn_metrics.accuracy_score(y, y_pred))
                elif metric_name in ("precision", "precision_weighted"):
                    results[metric_name] = float(sklearn_metrics.precision_score(y, y_pred, average="weighted"))
                elif metric_name in ("recall", "recall_weighted"):
                    results[metric_name] = float(sklearn_metrics.recall_score(y, y_pred, average="weighted"))
                elif metric_name in ("f1", "f1_weighted"):
                    results[metric_name] = float(sklearn_metrics.f1_score(y, y_pred, average="weighted"))
                
                # Regression metrics
                elif metric_name == "r2":
                    results[metric_name] = float(sklearn_metrics.r2_score(y, y_pred))
                elif metric_name in ("mae", "mean_absolute_error"):
                    results[metric_name] = float(sklearn_metrics.mean_absolute_error(y, y_pred))
                elif metric_name in ("mse", "mean_squared_error"):
                    results[metric_name] = float(sklearn_metrics.mean_squared_error(y, y_pred))
                elif metric

    def save_model(self, model: Any, path: str) -> None:
        """
        Save a scikit-learn model to disk.
        
        Args:
            model: The scikit-learn model to save
            path: Path where the model should be saved
            
        Raises:
            SerializationError: If saving fails
        """
        try:
            # Validate inputs
            if not isinstance(model, BaseEstimator):
                raise ValueError("Model is not a scikit-learn estimator")
            
            if not path.endswith('.pkl'):
                path = f"{path}.pkl"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            # Save the model info together with the model
            model_info = self.get_model_info(model)
            model._prometheum_info = model_info
            
            # Save the model using pickle
            with open(path, 'wb') as f:
                pickle.dump(model, f)
                
        except Exception as e:
            if isinstance(e, SerializationError):
                raise
            raise SerializationError(
                f"Failed to save model: {str(e)}",
                details={"original_error": str(e), "path": path}
            )
    
    def load_model(self, path: str) -> Any:
        """
        Load a scikit-learn model from disk.
        
        Args:
            path: Path from which to load the model
            
        Returns:
            The loaded scikit-learn model
            
        Raises:
            SerializationError: If loading fails
        """
        try:
            # Validate path
            if not os.path.exists(path):
                raise ValueError(f"Model file does not exist: {path}")
                
            if not path.endswith('.pkl'):
                path = f"{path}.pkl"
            
            # Load the model using pickle
            with open(path, 'rb') as f:
                model = pickle.load(f)
            
            # Validate the loaded model
            if not isinstance(model, BaseEstimator):
                raise ValueError(f"Loaded object is not a scikit-learn estimator: {type(model)}")
            
            return model
            
        except Exception as e:
            if isinstance(e, SerializationError):
                raise
            raise SerializationError(
                f"Failed to load model: {str(e)}",
                details={"original_error": str(e), "path": path}
            )
    
    def create_pipeline_transformer(self, model: Any, **kwargs) -> DataTransformer:
        """
        Create a Prometheum transformer from a scikit-learn model.
        
        This allows scikit-learn models to be used within Prometheum pipelines.
        The resulting transformer will apply the model's transform or predict 
        method to the input data.
        
        Args:
            model: The trained scikit-learn model
            **kwargs: Additional parameters for the transformer
            
        Returns:
            A transformer that can be used in a Prometheum pipeline
            
        Raises:
            ProcessingError: If transformer creation fails
        """
        try:
            # Validate the model
            if not isinstance(model, BaseEstimator):
                raise ValueError("Model is not a scikit-learn estimator")
            
            # Create a transformer wrapper for the scikit-learn model
            class SKLearnModelTransformer(DataTransformer):
                """Transformer that applies a scikit-learn model."""
                
                def __init__(self, model, target_column=None, output_column=None, **config):
                    """Initialize with a scikit-learn model."""
                    super().__init__(config)
                    self.model = model
                    self.model_type = _get_sklearn_model_type(model)
                    self.target_column = target_column
                    self.output_column = output_column or "prediction"
                    
                    # Determine if this is a transformer or predictor
                    self.is_transformer = hasattr(model, "transform")
                    self.is_predictor = hasattr(model, "predict")
                
                def fit(self, data: DataFrameContainer) -> None:
                    """
                    Fit is a no-op since we assume the model is already trained.
                    """
                    pass
                
                def transform(self, data: DataFrameContainer) -> DataFrameContainer:
                    """
                    Apply the scikit-learn model to the data.
                    
                    For transformers, uses model.transform()
                    For predictors, uses model.predict()
                    """
                    df = data.data.copy()
                    
                    # Prepare the input features
                    adapter = SKLearnAdapter()
                    X, _ = adapter.prepare_data(df, target_column=self.target_column)
                    
                    try:
                        if self.is_transformer:
                            # Apply transform method
                            result = self.model.transform(X)
                        elif self.is_predictor:
                            # Apply predict method
                            result = self.model.predict(X)
                        else:
                            raise ValueError("Model has no transform or predict method")
                            
                        # Handle different output formats
                        if isinstance(result, np.ndarray):
                            if len(result.shape) == 1:
                                # 1D array, ad

"""
Scikit-learn Adapter for Prometheum ML Framework.

This module provides an adapter for integrating scikit-learn models with
Prometheum data processing pipelines. It handles model creation, training,
evaluation, and serialization.
"""

import importlib
import inspect
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Type, cast

import numpy as np
import pandas as pd

try:
    import sklearn
    from sklearn.base import BaseEstimator
    from sklearn import metrics as sklearn_metrics
except ImportError:
    raise ImportError(
        "scikit-learn is required for this module. "
        "Install it with: pip install scikit-learn"
    )

from prometheum.core.base import DataFrameContainer, DataTransformer
from prometheum.core.exceptions import ProcessingError
from prometheum.ml.base import (
    FrameworkType,
    MLAdapter,
    ModelCreationError,
    ModelInfo,
    ModelType,
    TrainingError,
    PredictionError,
    EvaluationError,
    SerializationError,
)


def _get_sklearn_model_type(model: Any) -> ModelType:
    """
    Determine the type of a scikit-learn model.
    
    Args:
        model: A scikit-learn model instance
        
    Returns:
        ModelType: The type of the model
    """
    from sklearn.base import (
        ClassifierMixin,
        RegressorMixin,
        ClusterMixin,
        TransformerMixin,
    )
    
    if hasattr(model, "__module__") and "ensemble" in model.__module__:
        return ModelType.ENSEMBLE
    elif isinstance(model, ClassifierMixin):
        return ModelType.CLASSIFIER
    elif isinstance(model, RegressorMixin):
        return ModelType.REGRESSOR
    elif isinstance(model, ClusterMixin):
        return ModelType.CLUSTERER
    elif isinstance(model, TransformerMixin):
        return ModelType.TRANSFORMER
    
    return ModelType.CUSTOM


def _extract_model_params(model: Any) -> Dict[str, Any]:
    """
    Extract parameters from a scikit-learn model.
    
    Args:
        model: A scikit-learn model instance
        
    Returns:
        Dict[str, Any]: The model's parameters
    """
    if hasattr(model, "get_params"):
        params = model.get_params()
        # Convert non-serializable params to strings
        for key, value in params.items():
                elif metric_name in ("recall", "recall_weighted"):
                    results[metric_name] = float(sklearn_metrics.recall_score(y, y_pred, average="weighted"))
                elif metric_name in ("f1", "f1_weighted"):
                    results[metric_name] = float(sklearn_metrics.f1_score(y, y_pred, average="weighted"))
                
                # Regression metrics
                elif metric_name == "r2":
                    results[metric_name] = float(sklearn_metrics.r2_score(y, y_pred))
                elif metric_name in ("mae", "mean_absolute_error"):
                    results[metric_name] = float(sklearn_metrics.mean_absolute_error(y, y_pred))
                elif metric_name in ("mse", "mean_squared_error"):
                    results[metric_name] = float(sklearn_metrics.mean_squared_error(y, y_pred))
                elif metric_name in ("rmse", "root_mean_squared_error"):
                    results[metric_name] = float(np.sqrt(sklearn_metrics.mean_squared_error(y, y_pred)))
                else:
                    # Try to find metric in sklearn.metrics
                    if hasattr(sklearn_metrics, metric_name):
                        metric_func = getattr(sklearn_metrics, metric_name)
                        try:
                            results[metric_name] = float(metric_func(y, y_pred))
                        except Exception as e:
                            warn_msg = f"Could not compute metric '{metric_name}': {str(e)}"
                            results[f"{metric_name}_error"] = warn_msg
                    else:
                        warn_msg = f"Unknown metric: {metric_name}"
                        results[f"{metric_name}_error"] = warn_msg
            
            # Update model metrics if possible
            if hasattr(model, "_prometheum_info") and isinstance(model._prometheum_info, ModelInfo):
                model._prometheum_info.metrics.update(results)
            
            return results
            
        except Exception as e:
            if isinstance(e, EvaluationError):
                raise
            raise EvaluationError(
                f"Failed to evaluate model: {str(e)}",
                details={"original_error": str(e)}
            )
            elif not isinstance(value, (str, int, float, bool, list, dict, tuple, type(None))):
                params[key] = str(value)
        return params
    return {}
                elif metric_name in ("rmse", "root_mean_squared_error"):
                    results[metric_name] = float(np.sqrt(sklearn_metrics.mean_squared_error(y, y_pred)))
                else:
                    # Try to find metric in sklearn.metrics
                    if hasattr(sklearn_metrics, metric_name):
                        metric_func = getattr(sklearn_metrics, metric_name)
                        try:
                            results[metric_name] = float(metric_func(y, y_pred))
                        except Exception as e:
                            warn_msg = f"Could not compute metric '{metric_name}': {str(e)}"
                            results[f"{metric_name}_error"] = warn_msg
                    else:
                        warn_msg = f"Unknown metric: {metric_name}"
                        results[f"{metric_name}_error"] = warn_msg
            
            # Update model metrics if possible
            if hasattr(model, "_prometheum_info") and isinstance(model._prometheum_info, ModelInfo):
                model._prometheum_info.metrics.update(results)
            
            return results
            
        except Exception as e:
            if isinstance(e, EvaluationError):
                raise
            raise EvaluationError(
                f"Failed to evaluate model: {str(e)}",
                details={"original_error": str(e)}
            )
    
    def save_model(self, model: Any, path: str) -> None:
        """
        Save a scikit-learn model to disk.
        
        Args:
            model: The scikit-learn model to save
            path: Path where the model should be saved
            
        Raises:
            SerializationError: If saving fails
        """
        try:
            # Validate inputs
            if not isinstance(model, BaseEstimator):
                raise ValueError("Model is not a scikit-learn estimator")
            
            if not path.endswith('.pkl'):
                path = f"{path}.pkl"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            # Save the model info together with the model
            model_info = self.get_model_info(model)
            model._prometheum_info = model_info
            
            # Save the model using pickle
            with open(path, 'wb') as f:
                pickle.dump(model, f)
                
        except Exception as e:
            if isinstance(e, SerializationError):
                raise
            raise SerializationError(
                f"Failed to save model: {str(e)}",
                details={"original_error": str(e), "path": path}
            )
    
    def load_model(self, path: str) -> Any:
        """
        Load a scikit-learn model from disk.
        
        Args:
            path: Path from which to load the model
            
        Returns:
            The loaded scikit-learn model
            
        Raises:
            SerializationError: If loading fails
        """
        try:
            # Validate path
            if not os.path.exists(path):
                raise ValueError(f"Model file does not exist: {path}")
                
            if not path.endswith('.pkl'):
                path = f"{path}.pkl"
            
            # Load the model using pickle
            with open(path, 'rb') as f:
                model = pickle.load(f)
            
            # Validate the loaded model
            if not isinstance(model, BaseEstimator):
                raise ValueError(f"Loaded object is not a scikit-learn estimator: {type(model)}")
            
            return model
            
        except Exception as e:
            if isinstance(e, SerializationError):
                raise
            raise SerializationError(
                f"Failed to load model: {str(e)}",
                details={"original_error": str(e), "path": path}
            )
    
    def create_pipeline_transformer(self, model: Any, **kwargs) -> DataTransformer:
        """
        Create a Prometheum transformer from a scikit-learn model.
        
        This allows scikit-learn models to be used within Prometheum pipelines.
        The resulting transformer will apply the model's transform or predict 
        method to the input data.
        
        Args:
            model: The trained scikit-learn model
            **kwargs: Additional parameters for the transformer
            
        Returns:
            A transformer that can be used in a Prometheum pipeline
            
        Raises:
            ProcessingError: If transformer creation fails
        """
        try:
            # Import here to avoid circular imports
            from prometheum.core.base import DataTransformer
            
            # Validate the model
            if not isinstance(model, BaseEstimator):
                raise ValueError("Model is not a scikit-learn estimator")
            
            # Create a transformer wrapper for the scikit-learn model
            class SKLearnModelTransformer(DataTransformer):
                """Transformer that applies a scikit-learn model."""
                
                def __init__(self, model, target_column=None, output_column=None, **config):
                    """Initialize with a scikit-learn model."""
                    super().__init__(config)
                    self.model = model
                    self.model_type = _get_sklearn_model_type(model)
                    self.target_column = target_column
                    self.output_column = output_column or "prediction"
                    
                    # Determine if this is a transformer or predictor
                    self.is_transformer = hasattr(model, "transform")
                    self.is_predictor = hasattr(model, "predict")
                
                def fit(self, data: DataFrameContainer) -> None:
                    """
                    Fit is a no-op since we assume the model is already trained.
                    """
                    pass
                
                def transform(self, data: DataFrameContainer) -> DataFrameContainer:
                    """
                    Apply the scikit-learn model to the data.
                    
                    For transformers, uses model.transform()
                    For predictors, uses model.predict()
                    """
                    df = data.data.copy()
                    
                    # Prepare the input features
                    adapter = SKLearnAdapter()
                    X, _ = adapter.prepare_data(df, target_column=self.target_column)
                    
                    try:
                        if self.is_transformer:
                            # Apply transform method
                            result = self.model.transform(X)
                        elif self.is_predictor:
                            # Apply predict method
                            result = self.model.predict(X)
                        else:
                            raise ValueError("Model has no transform or predict method")
                            
                        # Handle different output formats
                        if isinstance(result, np.ndarray):
                            if len(result.shape) == 1:
                                # 1D array, add as a single column
                                df[self.output_column] = result
                            else:
                                # 2D array, add multiple columns
                                if result.shape[1] == 1:
                                    df[self.output_column] = result[:, 0]
                                else:
                                    for i in range(result.shape[1]):
                                        df[f"{self.output_column}_{i}"] = result[:, i]
                        elif isinstance(result, (pd.DataFrame, pd.Series)):
                            # If it's a pandas object, merge it into the original dataframe
                            if isinstance(result, pd.Series):
                                df[self.output_column] = result
                            else:
                                for col in result.columns:
                                    df[f"{self.output_column}_{col}"] = result[col]
                        else:
                            # Unknown result type
                            raise ValueError(f"Unsupported model output type: {type(result)}")
                        
                        # Add transformation metadata
                        model_info = adapter.get_model_info(self.model)
                        transform_metadata = {
                            "sklearn_model_applied": True,
                            "model_name": model_info.name,
                            "model_type": model_info.model_type.value,
                            "operation": "transform" if self.is_transformer else "predict",
                        }
                        
                        new_metadata = {**data.metadata, **transform_metadata}
                        return DataFrameContainer(df, new_metadata)
                        
                    except Exception as e:
                        if isinstance(e, ProcessingError):
                            raise
                        raise ProcessingError(
                            f"Error applying scikit-learn model: {str(e)}",
                            details={"original_error": str(e)}
                        )
            
            # Create and return the transformer
            return SKLearnModelTransformer(model, **kwargs)
            
        except Exception as e:
            if isinstance(e, ProcessingError):
                raise
            raise ProcessingError(
                f"Failed to create pipeline transformer: {str(e)}",
                details={"original_error": str(e)}
            )

) -> Any:
        """
        Train a scikit-learn model.
        
        Args:
            model: The scikit-learn model to train
            X: Feature data
            y: Target data (may be None for unsupervised models)
            **kwargs: Additional training parameters
            
        Returns:
            The trained model
            
        Raises:
            TrainingError: If training fails
        """
        try:
            # Validate inputs
            if not isinstance(model, BaseEstimator):
                raise ValueError("Model is not a scikit-learn estimator")
            
            # Determine if model is supervised or unsupervised
            from sklearn.base import ClassifierMixin, RegressorMixin
            is_supervised = isinstance(model, (ClassifierMixin, RegressorMixin))
            
            # Check if we have target data for supervised models
            if is_supervised and y is None:
                raise ValueError("Target data (y) is required for supervised models")
            
            # Apply fit method with appropriate arguments
            if is_supervised:
                model.fit(X, y, **kwargs)
            else:
                model.fit(X, **kwargs)
            
            return model
            
        except Exception as e:
            if isinstance(e, TrainingError):
                raise
            raise TrainingError(
                f"Failed to train model: {str(e)}",
                details={"original_error": str(e)}
            )
    
    def predict(
        self,
        model: Any,
        X: Any,
        **kwargs
    ) -> np.ndarray:
        """
        Make predictions with a scikit-learn model.
        
        Args:
            model: The trained scikit-learn model
            X: Feature data
            **kwargs: Additional prediction parameters
            
        Returns:
            Model predictions
            
        Raises:
            PredictionError: If prediction fails
        """
        try:
            # Validate inputs
            if not isinstance(model, BaseEstimator):
                raise ValueError("Model is not a scikit-learn estimator")
            
            # Determine prediction method to use
            from sklearn.base import ClassifierMixin
            
            # For classifiers, check if we want probabilities
            if isinstance(model, ClassifierMixin) and kwargs.get("return_probabilities", False):
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(X)
                else:
                    raise ValueError("Model does not support probability predictions")
            
            # Standard prediction
            return model.predict(X)
            
        except Exception as e:
            if isinstance(e, PredictionError):
                raise
            raise PredictionError(
                f"Failed to make predictions: {str(e)}",
                details={"original_error": str(e)}
            )
    
    def evaluate(
        self,
        model: Any,
        X: Any,
        y: Any,
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate a scikit-learn model's performance.
        
        Args:
            model: The trained scikit-learn model
            X: Feature data
            y: True target values
            metrics: List of metric names to compute
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary of metric names to values
            
        Raises:
            EvaluationError: If evaluation fails
        """
        try:
            # Validate inputs
            if not isinstance(model, BaseEstimator):
                raise ValueError("Model is not a scikit-learn estimator")
            
            # Make predictions
            y_pred = self.predict(model, X)
            
            # Determine model type to use appropriate metrics
            model_type = _get_sklearn_model_type(model)
            
            # Use provided metrics or default ones based on model type
            if metrics is None:
                if model_type == ModelType.CLASSIFIER:
                    metrics = ["accuracy", "precision", "recall", "f1"]
                elif model_type == ModelType.REGRESSOR:
                    metrics = ["r2", "mae", "mse", "rmse"]
                else:
                    metrics = []
            
            # Calculate all requested metrics
            results = {}
            
            for metric_name in metrics:
                metric_name = metric_name.lower()
                
                # Classification metrics
                if metric_name == "accuracy":
                    results[metric_name] = float(sklearn_metrics.accuracy_score(y, y_pred))
                elif metric_name in ("precision", "precision_weighted"):
                    results[metric_name] = float(sklearn_metrics.precision_score(y, y_pred, average="weighted"))
                elif metric_name in ("recall", "recall_weighted"):
                    results[metric_name] = float(sklearn_metrics.recall_score(y, y_pred, average="weighted"))
                elif metric_name in ("f1", "f1_weighted"):
                    results[metric_name] = float(sklearn_metrics.f1_score(y, y_pred, average="weighted"))
                
                # Regression metrics
                elif metric_name == "r2":
                    results[metric_name] = float(sklearn_metrics.r2_score(y, y_pred))
                elif metric_name in ("mae", "mean_absolute_error"):
                    results[metric_name] = float(sklearn_metrics.mean_absolute_error(y, y_pred))
                elif metric_name in ("mse", "mean_squared_error"):
                

