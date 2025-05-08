"""
TensorFlow Adapter for Prometheum ML Framework.
(Cleaned and optimized)
"""

import os
import json
import importlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    raise ImportError("TensorFlow is required. Install it with: pip install tensorflow")

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


def _get_tf_model_type(model: Any) -> ModelType:
    if isinstance(model, keras.Sequential):
        last = model.layers[-1] if model.layers else None
        if last and hasattr(last, 'activation'):
            act = last.activation.__name__
            if act in ('sigmoid', 'softmax'):
                return ModelType.CLASSIFIER
            elif act in ('linear', 'relu') and getattr(last, 'units', None) == 1:
                return ModelType.REGRESSOR
    elif isinstance(model, keras.Model):
        out = model.layers[-1] if model.layers else None
        if out and hasattr(out, 'activation'):
            act = out.activation.__name__
            if act in ('sigmoid', 'softmax'):
                return ModelType.CLASSIFIER
            elif act in ('linear', 'relu') and getattr(out, 'units', None) == 1:
                return ModelType.REGRESSOR

    name = model.__class__.__name__.lower()
    if "classifier" in name:
        return ModelType.CLASSIFIER
    elif "regressor" in name:
        return ModelType.REGRESSOR
    elif "autoencoder" in name or "encoder" in name:
        return ModelType.TRANSFORMER
    return ModelType.NEURAL_NETWORK


def _extract_tf_model_params(model: Any) -> Dict[str, Any]:
    params = {}
    if hasattr(model, 'get_config'):
        try:
            params['config'] = str(model.get_config())
        except Exception:
            pass
    if hasattr(model, 'layers'):
        params['layers'] = [
            {
                "layer": i,
                "type": layer.__class__.__name__,
                "activation": getattr(layer.activation, '__name__', None),
                "units": getattr(layer, 'units', None)
            }
            for i, layer in enumerate(model.layers)
        ]
    if hasattr(model, 'optimizer'):
        try:
            opt = model.optimizer
            params['optimizer'] = {
                "name": opt.__class__.__name__,
                "learning_rate": float(opt.learning_rate.numpy()) if hasattr(opt, 'learning_rate') else None
            }
        except Exception:
            params['optimizer'] = str(model.optimizer)
    return params


class TensorFlowAdapter(MLAdapter):
    @property
    def framework(self) -> FrameworkType:
        return FrameworkType.TENSORFLOW

    def create_model(self, model_name: str, **kwargs) -> Any:
        try:
            if model_name == "Sequential":
                layers = kwargs.pop("layers", [])
                model = keras.Sequential(**kwargs)
                for layer in layers:
                    model.add(layer)
                return model
            elif model_name == "Functional":
                inputs = kwargs.pop("inputs", None)
                outputs = kwargs.pop("outputs", None)
                if not inputs or not outputs:
                    raise ModelCreationError("Functional model needs 'inputs' and 'outputs'")
                return keras.Model(inputs=inputs, outputs=outputs, **kwargs)
            elif model_name.startswith("keras.applications."):
                parts = model_name.split(".")
                module = importlib.import_module(".".join(parts[:-1]))
                model_fn = getattr(module, parts[-1])
                return model_fn(**kwargs)
            else:
                module = importlib.import_module(".".join(model_name.split(".")[:-1]))
                cls = getattr(module, model_name.split(".")[-1])
                return cls(**kwargs)
        except Exception as e:
            raise ModelCreationError(f"Could not create model: {e}")

    def get_model_info(self, model: Any) -> ModelInfo:
        model_type = _get_tf_model_type(model)
        params = _extract_tf_model_params(model)
        return ModelInfo(
            name=type(model).__name__,
            model_type=model_type,
            framework=FrameworkType.TENSORFLOW,
            params=params,
            description=(model.__doc__.split("\n")[0] if model.__doc__ else None),
            version=tf.__version__,
            created_at=datetime.now().isoformat(),
            metrics={},
            feature_names=[],
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
            if feature_columns:
                X = df[feature_columns].to_numpy()
            else:
                X = df.drop(columns=[target_column]).to_numpy() if target_column else df.to_numpy()
            y = df[target_column].to_numpy() if target_column else None
            return X, y
        except Exception as e:
            raise ProcessingError(f"prepare_data failed: {e}")

    def train(self, model: Any, X: Any, y: Optional[Any] = None, **kwargs) -> Any:
        try:
            if not model.compiled_loss:
                model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
            model.fit(X, y, **kwargs)
            return model
        except Exception as e:
            raise TrainingError(f"train failed: {e}")

    def predict(self, model: Any, X: Any, **kwargs) -> np.ndarray:
        try:
            preds = model.predict(X, **kwargs)
            return preds.numpy() if isinstance(preds, tf.Tensor) else preds
        except Exception as e:
            raise PredictionError(f"predict failed: {e}")

    def evaluate(
        self, model: Any, X: Any, y: Any, metrics: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, float]:
        try:
            results = model.evaluate(X, y, return_dict=True, **kwargs)
            return {k: float(v) for k, v in results.items()}
        except Exception as e:
            raise EvaluationError(f"evaluate failed: {e}")

    def save_model(self, model: Any, path: str) -> None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            model.save(path)
        except Exception as e:
            raise SerializationError(f"save_model failed: {e}")

    def load_model(self, path: str) -> Any:
        try:
            return keras.models.load_model(path)
        except Exception as e:
            raise SerializationError(f"load_model failed: {e}")
