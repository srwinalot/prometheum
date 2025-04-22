import importlib
import inspect
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Type, cast

import numpy as np
import pandas as pd

# Ensure scikit-learn is available
try:
    import sklearn
    from sklearn.base import BaseEstimator
    from sklearn import metrics as sklearn_metrics
except ImportError:
    # If scikit-learn isn't installed, this module cannot function
    raise ImportError(
        "scikit-learn is required for this module. "
        "Install it with: pip install scikit-learn"
    )

# Import Prometheum core and ML base classes/exceptions
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
    Determine the type of a scikit-learn model (classifier, regressor, etc.).

    Args:
        model: A scikit-learn model instance.

    Returns:
        ModelType: The type of the model (e.g., CLASSIFIER, REGRESSOR, etc.).
    """
    from sklearn.base import ClassifierMixin, RegressorMixin, ClusterMixin, TransformerMixin

    # Identify ensemble models by module name
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
    # If none of the above, treat as custom model type
    return ModelType.CUSTOM

def _extract_model_params(model: Any) -> Dict[str, Any]:
    """
    Extract parameters from a scikit-learn model, converting non-serializable values to strings.

    Args:
        model: A scikit-learn model instance.

    Returns:
        Dict[str, Any]: The model's parameters (suitable for serialization).
    """
    if hasattr(model, "get_params"):
        params = model.get_params()
        # Convert any non-serializable parameter values to string representations
        for key, value in params.items():
            if hasattr(value, "__name__"):
                # If the value is a function or class, use its name
                params[key] = value.__name__
            elif not isinstance(value, (str, int, float, bool, list, dict, tuple, type(None))):
                # If it's an object not in basic types, convert to string
                params[key] = str(value)
        return params
    return {}

class SKLearnAdapter(MLAdapter):
    """
    Adapter for scikit-learn models within the Prometheum framework.

    Provides methods to create, train, predict, evaluate, and save/load scikit-learn models,
    integrating them with Prometheum's data structures and error handling.
    """

    @property
    def framework(self) -> FrameworkType:
        """Identify this adapter's ML framework (scikit-learn)."""
        return FrameworkType.SKLEARN

    def create_model(self, model_name: str, **kwargs) -> Any:
        """
        Create a new scikit-learn model instance by name.

        Args:
            model_name: Name of the model class (e.g., "RandomForestClassifier" or "sklearn.ensemble.RandomForestClassifier").
            **kwargs: Parameters to pass to the model's constructor.

        Returns:
            A new instance of the requested scikit-learn model.

        Raises:
            ModelCreationError: If the model class cannot be found or instantiation fails.
        """
        try:
            module_parts = model_name.split('.')
            if len(module_parts) == 1:
                # No module path specified; try common sklearn sub-modules
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
                for module in common_modules:
                    try:
                        mod = importlib.import_module(module)
                        if hasattr(mod, model_name):
                            model_cls = getattr(mod, model_name)
                            break
                    except (ImportError, AttributeError):
                        continue
                if model_cls is None:
                    # Model not found in common modules
                    raise ModelCreationError(
                        f"Could not find model class '{model_name}' in common scikit-learn modules",
                        details={"model_name": model_name}
                    )
            else:
                # If a full module path is provided, import directly
                module_name = '.'.join(module_parts[:-1])
                class_name = module_parts[-1]
                mod = importlib.import_module(module_name)
                model_cls = getattr(mod, class_name)

            # Instantiate the model class with provided parameters
            model = model_cls(**kwargs)
            # Verify the created object is a scikit-learn estimator
            if not isinstance(model, BaseEstimator):
                raise ModelCreationError(
                    f"Created object is not a scikit-learn estimator: {type(model)}",
                    details={"model_type": str(type(model))}
                )
            return model

        except (ImportError, AttributeError) as e:
            # Errors related to importing the module or attribute not found
            raise ModelCreationError(
                f"Failed to import model class '{model_name}': {e}",
                details={"original_error": str(e)}
            )
        except Exception as e:
            # Wrap any other exception as ModelCreationError (unless it's already that type)
            if isinstance(e, ModelCreationError):
                raise
            raise ModelCreationError(
                f"Failed to create model '{model_name}': {e}",
                details={"original_error": str(e)}
            )

    def get_model_info(self, model: Any) -> ModelInfo:
        """
        Collect metadata and parameters from a scikit-learn model.

        Args:
            model: The scikit-learn model instance.

        Returns:
            ModelInfo: An object containing information about the model (name, type, params, etc.).

        Raises:
            ValueError: If the provided object is not a scikit-learn estimator.
        """
        if not isinstance(model, BaseEstimator):
            # The model provided is not a sklearn estimator
            raise ValueError("Model is not a scikit-learn estimator")

        # Determine model type (classifier, regressor, etc.)
        model_type = _get_sklearn_model_type(model)
        model_name = type(model).__name__
        params = _extract_model_params(model)

        # Extract feature names if available (e.g., after model is fitted)
        feature_names: List[str] = []
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)

        # Timestamp for when info is gathered (as an ISO string)
        created_at = datetime.now().isoformat()

        # Check if the model has been fitted (trained) using sklearn's check_is_fitted
        is_fitted = False
        try:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(model)  # This will raise if not fitted
            is_fitted = True
        except Exception:
            # If check_is_fitted raises, model is not fitted; ignore exception
            is_fitted = False

        # Compile model info. Note: `is_fitted` could be stored or used if ModelInfo supports it.
        return ModelInfo(
            name=model_name,
            model_type=model_type,
            framework=FrameworkType.SKLEARN,
            params=params,
            description=(model.__doc__.split("\n")[0] if model.__doc__ else None),
            version=sklearn.__version__,
            created_at=created_at,
            metrics={},           # will be filled after evaluation (if any)
            feature_names=feature_names,
            target_name=None,
            serializable=True     # scikit-learn models are generally picklable
        )

    def prepare_data(
        self,
        data: Union[DataFrameContainer, pd.DataFrame],
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare input data for model training or prediction.

        Accepts a pandas DataFrame or Prometheum DataFrameContainer and extracts features (and target if provided),
        returning NumPy arrays suitable for scikit-learn.

        Args:
            data: Input data (pandas DataFrame or DataFrameContainer wrapping a DataFrame).
            target_column: Name of the target column for supervised learning (optional).
            feature_columns: Specific feature columns to select (optional; if not provided, use all except target).
            **kwargs: Additional parameters (not used here, but for interface compatibility).

        Returns:
            A tuple (X, y) where X is a NumPy array of features and y is a NumPy array of targets (or None if no target).

        Raises:
            ProcessingError: If data preparation fails (e.g., missing columns).
        """
        try:
            # Unwrap the DataFrame if a container is provided
            if isinstance(data, DataFrameContainer):
                df = data.data
            else:
                df = data

            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Expected pandas DataFrame, got {type(df)}")

            # Select specified feature columns, if provided
            if feature_columns:
                missing_cols = [col for col in feature_columns if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Feature columns not found in data: {missing_cols}")
                X_df = df[feature_columns]
            else:
                # Use all columns except the target (if target_column is given)
                if target_column:
                    if target_column not in df.columns:
                        raise ValueError(f"Target column '{target_column}' not found in data")
                    X_df = df.drop(columns=[target_column])
                else:
                    X_df = df  # No target specified, use all data as features

            # Convert feature DataFrame to NumPy array
            X = X_df.to_numpy()
            # Extract target array if target_column is provided (for supervised training)
            y: Optional[np.ndarray] = None
            if target_column:
                # By this point, we ensured target_column exists
                y = df[target_column].to_numpy()

            return X, y

        except Exception as e:
            # Wrap any error in ProcessingError (unless it's already that type)
            if isinstance(e, ProcessingError):
                raise
            raise ProcessingError(
                f"Failed to prepare data: {e}",
                details={"original_error": str(e)}
            )

    def train(self, model: Any, X: Any, y: Optional[Any] = None, **kwargs) -> Any:
        """
        Train a scikit-learn model on the given data.

        Args:
            model: The scikit-learn model (estimator) to train.
            X: Feature data (NumPy array or compatible).
            y: Target data (NumPy array), required for supervised models (None for unsupervised models).
            **kwargs: Additional parameters to pass to the model's fit method (e.g., epochs for neural nets, etc.).

        Returns:
            The trained model (same instance as provided, after fitting).

        Raises:
            TrainingError: If training fails (e.g., due to bad input or model issues).
        """
        try:
            if not isinstance(model, BaseEstimator):
                raise ValueError("Model is not a scikit-learn estimator")

            # Determine if the model expects a target (supervised) or not
            from sklearn.base import ClassifierMixin, RegressorMixin
            is_supervised = isinstance(model, (ClassifierMixin, RegressorMixin))
            if is_supervised and y is None:
                # For supervised models, y must be provided
                raise ValueError("Target data (y) is required for supervised models")

            # Train the model using the appropriate signature
            if is_supervised:
                model.fit(X, y, **kwargs)
            else:
                model.fit(X, **kwargs)
            return model

        except Exception as e:
            if isinstance(e, TrainingError):
                raise  # if a TrainingError was raised internally, propagate it
            # Wrap any other exception as TrainingError
            raise TrainingError(
                f"Failed to train model: {e}",
                details={"original_error": str(e)}
            )

    def predict(self, model: Any, X: Any, **kwargs) -> np.ndarray:
        """
        Generate predictions using a trained scikit-learn model.

        Args:
            model: The trained scikit-learn model.
            X: Feature data for prediction.
            **kwargs: Additional prediction options (e.g., return_probabilities=True for classifiers).

        Returns:
            A NumPy array of model predictions (or probabilities if requested for classifiers).

        Raises:
            PredictionError: If prediction fails.
        """
        try:
            if not isinstance(model, BaseEstimator):
                raise ValueError("Model is not a scikit-learn estimator")

            from sklearn.base import ClassifierMixin
            # If the model is a classifier and user wants probabilities:
            if isinstance(model, ClassifierMixin) and kwargs.get("return_probabilities", False):
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(X)
                else:
                    raise ValueError("Model does not support probability predictions")
            # Default: use standard predict
            return model.predict(X)

        except Exception as e:
            if isinstance(e, PredictionError):
                raise
            raise PredictionError(
                f"Failed to make predictions: {e}",
                details={"original_error": str(e)}
            )

    def evaluate(self, model: Any, X: Any, y: Any, metrics: Optional[List[str]] = None, **kwargs) -> Dict[str, float]:
        """
        Evaluate a trained scikit-learn model on a dataset using specified metrics.

        Args:
            model: The trained scikit-learn model.
            X: Feature data (for prediction).
            y: True target values (for calculating metrics).
            metrics: List of metric names to compute (e.g., ["accuracy","precision","recall","f1"] for classifiers).
                     If None, a default set is used based on model type.
            **kwargs: Additional evaluation parameters (not used here).

        Returns:
            Dictionary mapping each requested metric name to its computed value.

        Raises:
            EvaluationError: If evaluation fails.
        """
        try:
            if not isinstance(model, BaseEstimator):
                raise ValueError("Model is not a scikit-learn estimator")

            # Get model predictions for X
            y_pred = self.predict(model, X)
            # Determine default metrics if none provided, based on model type
            model_type = _get_sklearn_model_type(model)
            if metrics is None:
                if model_type == ModelType.CLASSIFIER:
                    metrics = ["accuracy", "precision", "recall", "f1"]
                elif model_type == ModelType.REGRESSOR:
                    metrics = ["r2", "mae", "mse", "rmse"]
                else:
                    metrics = []  # No default metrics for other types (cluster, transformer, etc.)

            results: Dict[str, float] = {}
            for metric_name in metrics:
                metric_name = metric_name.lower()
                # Compute classification metrics
                if metric_name == "accuracy":
                    results[metric_name] = float(sklearn_metrics.accuracy_score(y, y_pred))
                elif metric_name in ("precision", "precision_weighted"):
                    # Use weighted precision for classification (consider class imbalance)
                    results[metric_name] = float(sklearn_metrics.precision_score(y, y_pred, average="weighted"))
                elif metric_name in ("recall", "recall_weighted"):
                    results[metric_name] = float(sklearn_metrics.recall_score(y, y_pred, average="weighted"))
                elif metric_name in ("f1", "f1_score", "f1_weighted"):
                    # Treat "f1" or "f1_score" as weighted F1
                    results[metric_name] = float(sklearn_metrics.f1_score(y, y_pred, average="weighted"))
                # Compute regression metrics
                elif metric_name == "r2":
                    results[metric_name] = float(sklearn_metrics.r2_score(y, y_pred))
                elif metric_name in ("mae", "mean_absolute_error"):
                    results[metric_name] = float(sklearn_metrics.mean_absolute_error(y, y_pred))
                elif metric_name in ("mse", "mean_squared_error"):
                    results[metric_name] = float(sklearn_metrics.mean_squared_error(y, y_pred))
                elif metric_name == "rmse":
                    # Root Mean Squared Error as sqrt of MSE
                    results[metric_name] = float(sklearn_metrics.mean_squared_error(y, y_pred) ** 0.5)
                else:
                    # If an unknown metric name is provided, skip or handle as needed.
                    # (We choose to skip unknown metrics silently, but could log a warning.)
                    continue

            return results

        except Exception as e:
            if isinstance(e, EvaluationError):
                raise
            raise EvaluationError(
                f"Failed to evaluate model: {e}",
                details={"original_error": str(e)}
            )

    def save_model(self, model: Any, path: str) -> None:
        """
        Save a scikit-learn model to disk.

        Args:
            model: The scikit-learn model instance to save (should be picklable).
            path: Filesystem path where the model should be saved (e.g., "models/model.pkl").

        Raises:
            SerializationError: If an error occurs during saving (e.g., IO issues).
        """
        try:
            # Ensure the directory exists before saving
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(model, f)
        except Exception as e:
            # Wrap any exception as a SerializationError
            raise SerializationError(
                f"Failed to save model to '{path}': {e}",
                details={"original_error": str(e), "path": path}
            )

    def load_model(self, path: str) -> Any:
        """
        Load a scikit-learn model from disk.

        Args:
            path: Filesystem path from where to load the model.

        Returns:
            The loaded scikit-learn model instance.

        Raises:
            SerializationError: If loading fails or the object is not a valid scikit-learn model.
        """
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            # Optional: verify the loaded object is a scikit-learn estimator
            if not isinstance(model, BaseEstimator):
                raise ValueError("Loaded object is not a scikit-learn estimator")
            return model
        except Exception as e:
            raise SerializationError(
                f"Failed to load model from '{path}': {e}",
                details={"original_error": str(e), "path": path}
            )


