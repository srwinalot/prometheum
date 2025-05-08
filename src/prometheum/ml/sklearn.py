"""
Scikit-learn Adapter for Prometheum ML Framework.
"""

import os
import pickle
import importlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import sklearn

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn import metrics as sklearn_metrics

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
    from sklearn.base import ClusterMixin, TransformerMixin
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
    if hasattr(model, "get_params"):
        params = model.get_params()
        for key, value in params.items():
            if hasattr(value, "__name__"):
                params[key] = value.__name__
            elif not isinstance(value, (str, int, float, bool, list, dict, tuple, type(None))):
                params[key] = str(value)
        return params
    return {}


class SKLearnAdapter(MLAdapter):
    @property
    def framework(self) -> FrameworkType:
        return FrameworkType.SKLEARN

    def create_model(self, model_name: str, **kwargs) -> Any:
        try:
            module_parts = model_name.split(".")
            if len(module_parts) == 1:
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
                for mod in common_modules:
                    try:
                        module = importlib.import_module(mod)
                        if hasattr(module, model_name):
                            model_cls = getattr(module, model_name)
                            return model_cls(**kwargs)
                    except ImportError:
                        continue
                raise ModelCreationError(f"Model class '{model_name}' not found.")
            else:
                module = importlib.import_module(".".join(module_parts[:-1]))
                model_cls = getattr(module, module_parts[-1])
                return model_cls(**kwargs)
        except Exception as e:
            raise ModelCreationError(str(e), details={"model_name": model_name})

    def get_model_info(self, model: Any) -> ModelInfo:
        if not isinstance(model, BaseEstimator):
            raise ValueError("Not a scikit-learn estimator.")
        model_type = _get_sklearn_model_type(model)
        feature_names = list(getattr(model, "feature_names_in_", []))
        params = _extract_model_params(model)
        return ModelInfo(
            name=type(model).__name__,
            model_type=model_type,
            framework=FrameworkType.SKLEARN,
            params=params,
            description=(model.__doc__.split("\n")[0] if model.__doc__ else None),
            version=sklearn.__version__,
            created_at=datetime.now().isoformat(),
            metrics={},
            feature_names=feature_names,
            target_name=None,
            serializable=True,
        )

    def prepare_data(
        self,
        data: Union[DataFrameContainer, pd.DataFrame],
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        try:
            df = data.data if isinstance(data, DataFrameContainer) else data
            
            # Properly check if feature_columns exists and is not empty
            if feature_columns is not None and len(feature_columns) > 0:
                X = df[feature_columns].to_numpy()
            else:
                # Check if there will be any features left after removing target column
                if target_column and len(df.columns) <= 1:
                    raise ValueError("No feature columns remain after removing target column")
                
                X = df.drop(columns=[target_column]).to_numpy() if target_column else df.to_numpy()
                
                # Validate that we have features
                if X.shape[1] == 0:
                    raise ValueError("Dataset contains no features")
                
            y = df[target_column].to_numpy() if target_column else None
            return X, y
        except Exception as e:
            raise ProcessingError(f"Failed to prepare data: {e}")
    def train(self, model: Any, X: Any, y: Optional[Any] = None, **kwargs) -> Any:
        try:
            if isinstance(model, (ClassifierMixin, RegressorMixin)) and y is None:
                raise ValueError("Target data required for supervised models.")
            model.fit(X, y) if y is not None else model.fit(X)
            return model
        except Exception as e:
            raise TrainingError(f"Failed to train model: {e}")

    def predict(self, model: Any, X: Any, **kwargs) -> np.ndarray:
        try:
            if kwargs.get("return_probabilities") and hasattr(model, "predict_proba"):
                return model.predict_proba(X)
            return model.predict(X)
        except NotFittedError:
            # Let NotFittedError pass through directly (for proper testing)
            raise
        except Exception as e:
            raise PredictionError(f"Failed to predict: {e}")

    def evaluate(
        self, model: Any, X: Any, y: Any, metrics: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate a scikit-learn model using various metrics.

        Args:
            model: The scikit-learn model to evaluate
            X: Feature data for evaluation
            y: Ground truth labels/values
            metrics: List of metric names to compute (if None, defaults based on model type)
            **kwargs: Additional parameters for evaluation

        Returns:
            Dictionary of metric names to values

        Raises:
            EvaluationError: If evaluation fails
        """
        try:
            # Get model prediction
            y_pred = self.predict(model, X)
            model_type = _get_sklearn_model_type(model)
            
            # Determine if classifier and get class information
            is_classifier = model_type in (ModelType.CLASSIFIER, ModelType.ENSEMBLE) and isinstance(model, ClassifierMixin)
            
            # Determine default metrics based on model type
            if metrics is None:
                if is_classifier:
                    # Default classification metrics - include ROC AUC for all classification cases
                    metrics = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "roc_auc"]
                else:
                    metrics = ["r2", "adjusted_r2", "mae", "median_absolute_error", "mse", "rmse", "explained_variance", "max_error"]
            
            # Check classification type (binary vs multiclass)
            is_binary = False
            is_multiclass = False
            if is_classifier:
                unique_classes = np.unique(y)
                class_count = len(unique_classes)
                is_binary = class_count == 2
                is_multiclass = class_count > 2
                
                # Get probability predictions if needed for certain metrics
                if any(m.lower() in ["roc_auc", "log_loss"] for m in metrics) and hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X)
            
            # Calculate requested metrics
            # Calculate requested metrics
            results = {}
            for m in metrics:
                m = m.lower()
                
                # Group 1: Classification Metrics
                if is_classifier and m in [
                    "accuracy", "precision", "recall", "f1", 
                    "balanced_accuracy", "matthews_corrcoef", "jaccard_score"
                ]:
                    if m == "accuracy":
                        results[m] = float(sklearn_metrics.accuracy_score(y, y_pred))
                    elif m == "balanced_accuracy":
                        results[m] = float(sklearn_metrics.balanced_accuracy_score(y, y_pred))
                    elif m in ["precision", "recall", "f1", "jaccard_score"]:
                        try:
                            average = "binary" if is_binary else "weighted"
                            if m == "precision":
                                results[m] = float(sklearn_metrics.precision_score(y, y_pred, average=average))
                            elif m == "recall":
                                results[m] = float(sklearn_metrics.recall_score(y, y_pred, average=average))
                            elif m == "f1":
                                results[m] = float(sklearn_metrics.f1_score(y, y_pred, average=average))
                            elif m == "jaccard_score":
                                results[m] = float(sklearn_metrics.jaccard_score(y, y_pred, average=average))
                        except Exception as metric_err:
                            results[f"{m}_error"] = f"Failed to compute {m}: {str(metric_err)}"
                    elif m == "matthews_corrcoef":
                        try:
                            results[m] = float(sklearn_metrics.matthews_corrcoef(y, y_pred))
                        except Exception as metric_err:
                            results[f"{m}_error"] = f"Failed to compute Matthews correlation coefficient: {str(metric_err)}"
                elif m == "roc_auc" and is_classifier:
                    if is_multiclass:
                        # Explicitly treat multiclass as unsupported
                        results[f"{m}_error"] = "ROC AUC only supported for binary classification"
                    elif not hasattr(model, "predict_proba"):
                        # Model doesn't support probability predictions
                        results[f"{m}_error"] = "ROC AUC requires a model that supports probability predictions"
                    elif is_binary:
                        # Binary classification case
                        results[m] = float(sklearn_metrics.roc_auc_score(y, y_proba[:, 1]))
                    else:
                        results[f"{m}_error"] = "ROC AUC calculation error: Unknown classification type"
                elif m == "confusion_matrix" and is_classifier:
                    # Convert confusion matrix to a dictionary representation
                    cm = sklearn_metrics.confusion_matrix(y, y_pred)
                    results[m] = cm.tolist()  # Convert numpy array to list for serialization
                
                # Group 2: Regression Metrics
                elif m in ["r2", "adjusted_r2", "mae", "median_absolute_error", 
                          "mse", "rmse", "explained_variance", "max_error"]:
                    if m == "r2":
                        results[m] = float(sklearn_metrics.r2_score(y, y_pred))
                    elif m == "adjusted_r2":
                        # Calculate adjusted R2 from R2 and number of samples/features
                        r2 = float(sklearn_metrics.r2_score(y, y_pred))
                        n = len(y)  # Number of samples
                        if hasattr(X, "shape"):
                            p = X.shape[1]  # Number of features
                        else:
                            # If X is not a numpy array or doesn't have shape attribute
                            p = 1  # Default to 1 feature
                        # Formula: 1 - (1 - R2) * (n - 1) / (n - p - 1)
                        if n > p + 1:
                            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                        else:
                            adjusted_r2 = float('nan')  # Not enough samples for calculation
                        results[m] = float(adjusted_r2)
                    elif m == "mae":
                        results[m] = float(sklearn_metrics.mean_absolute_error(y, y_pred))
                    elif m == "median_absolute_error":
                        results[m] = float(sklearn_metrics.median_absolute_error(y, y_pred))
                    elif m == "mse":
                        results[m] = float(sklearn_metrics.mean_squared_error(y, y_pred))
                    elif m == "rmse":
                        results[m] = float(np.sqrt(sklearn_metrics.mean_squared_error(y, y_pred)))
                    elif m == "explained_variance":
                        results[m] = float(sklearn_metrics.explained_variance_score(y, y_pred))
                    elif m == "max_error":
                        results[m] = float(sklearn_metrics.max_error(y, y_pred))
                
                # Group 3: Metrics that are only applicable to certain model types
                elif (m in ["precision", "recall", "f1", "balanced_accuracy", 
                          "matthews_corrcoef", "jaccard_score", "roc_auc", "confusion_matrix"] 
                     and not is_classifier):
                    results[f"{m}_error"] = f"{m} only applicable for classification models"
                
                # Group 4: Try to use any available sklearn metric
                else:
                    # Try to use a metric function from sklearn.metrics
                    if hasattr(sklearn_metrics, m):
                        try:
                            metric_func = getattr(sklearn_metrics, m)
                            result = metric_func(y, y_pred)
                            if hasattr(result, "__iter__") and not isinstance(result, (str, dict)):
                                # If result is iterable but not a string or dict, convert to list
                                results[m] = [float(x) for x in result]
                            else:
                                results[m] = float(result)
                        except Exception as metric_err:
                            results[f"{m}_error"] = f"Failed to compute metric: {str(metric_err)}"
                    else:
                        results[f"{m}_error"] = f"Unknown metric: {m}"
            if hasattr(model, "_prometheum_info") and isinstance(model._prometheum_info, ModelInfo):
                model._prometheum_info.metrics.update(results)
                
            return results
        except Exception as e:
            raise EvaluationError(f"Failed to evaluate model: {e}")

    def save_model(self, model: Any, path: str) -> None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(model, f)
        except Exception as e:
            raise SerializationError(f"Failed to save model: {e}", details={"path": path})

    def load_model(self, path: str) -> Any:
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            if not isinstance(model, BaseEstimator):
                raise ValueError("Loaded object is not a valid scikit-learn model.")
            return model
        except Exception as e:
            raise SerializationError(f"Failed to load model: {e}", details={"path": path})
