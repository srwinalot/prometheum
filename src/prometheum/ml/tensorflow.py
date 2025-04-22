"""
TensorFlow Adapter for Prometheum ML Framework.

This module provides an adapter for integrating TensorFlow/Keras models with
Prometheum data processing pipelines. It handles model creation, training,
evaluation, and serialization.
"""

import os
import json
import importlib
import inspect
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Type, cast

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    raise ImportError(
        "TensorFlow is required for this module. "
        "Install it with: pip install tensorflow"
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


def _get_tf_model_type(model: Any) -> ModelType:
    """
    Determine the type of a TensorFlow model.
    
    Args:
        model: A TensorFlow/Keras model instance
        
    Returns:
        ModelType: The type of the model
    """
    # Check if it's a Sequential model (common case)
    if isinstance(model, keras.Sequential):
        # Check the last layer to determine model type
        last_layer = model.layers[-1] if model.layers else None
        
        if last_layer:
                if y is not None:
                    # Supervised learning
                    dataset = tf.data.Dataset.from_tensor_slices((X, y))
                else:
                    # Unsupervised learning
                    dataset = tf.data.Dataset.from_tensor_slices(X)
                
                if shuffle:
                    # Buffer size as large as the dataset, or 1000 if dataset is larger
                    buffer_size = min(len(X), 1000)
                    dataset = dataset.shuffle(buffer_size=buffer_size)
                
                dataset = dataset.batch(batch_size)
                dataset = dataset.prefetch(tf.data.AUTOTUNE)
                
                # Return the dataset and feature names (for reference)
                return dataset, feature_names
            
            # Save feature names on the arrays for optional use later
            X._feature_names = feature_names
            
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
        Train a TensorFlow/Keras model.
        
        Args:
            model: The TensorFlow/Keras model to train
            X: Feature data (numpy array or tf.data.Dataset)
            y: Target data (may be None if X is a Dataset or for unsupervised models)
            **kwargs: Additional training parameters, passed to model.fit()
                - epochs: Number of training epochs (default: 10)
                - batch_size: Batch size for training (default: 32, ignored if X is a Dataset)
                - validation_split: Fraction of data to use for validation (default: 0.2)
                - callbacks: List of Keras callbacks
                - verbose: Verbosity level for training (default: 1)
            
        Returns:
            The trained model
            
        Raises:
            TrainingError: If training fails
        """
        try:
            # Validate inputs
            if not isinstance(model, (keras.Model, keras.Sequential)):
                raise ValueError("Model is not a TensorFlow/Keras model")
            
            # Extract and handle kwargs with defaults
            epochs = kwargs.pop('epochs', 10)
            batch_size = kwargs.pop('batch_size', 32)
            validation_split = kwargs.pop('validation_split', 0.2)
            callbacks = kwargs.pop('callbacks', None)
            verbose = kwargs.pop('verbose', 1)
            
            # Check if the model is compiled
            if not model.compiled_loss:
                # Default compilation if not already compiled
                optimizer = kwargs.pop('optimizer', 'adam')
                loss = kwargs.pop('loss', 'mse')  # Default loss
                metrics_list = kwargs.pop('metrics', ['accuracy'])
                
                model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=metrics_list
                )
            
            # Save feature names if available
            if hasattr(X, '_feature_names'):
                model._feature_names = X._feature_names
            
            # Handle different input types
            if isinstance(X, tf.data.Dataset):
                # If X is already a dataset, use it directly
                history = model.fit(
                    X,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=verbose,
                    **kwargs
                )
            else:
                # Standard numpy arrays
                history = model.fit(
                    X, y,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    verbose=verbose,
                    **kwargs
                )
            
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
        Make predictions with a TensorFlow/Keras model.
        
        Args:
            model: The trained TensorFlow/Keras model
            X: Feature data (numpy array or tf.data.Dataset)
            **kwargs: Additional prediction parameters
                - batch_size: Batch size for prediction (default: 32, ignored if X is a Dataset)
                - verbose: Verbosity level (default: 0)
                - return_probabilities: If True, return probabilities for classification models
            
        Returns:
            Model predictions as numpy array
            
        Raises:
            PredictionError: If prediction fails
        """
        try:
            # Validate inputs
            if not isinstance(model, (keras.Model, keras.Sequential)):
                raise ValueError("Model is not a TensorFlow/Keras model")
            
            # Extract prediction parameters
            batch_size = kwargs.pop('batch_size', 32)
            verbose = kwargs.pop('verbose', 0)
            return_probs = kwargs.pop('return_probabilities', False)
            
            # Determine if this is a classification model
            model_type = _get_tf_model_type(model)
            is_classifier = model_type == ModelType.CLASSIFIER
            
            # For classification models with probability output request
            if is_classifier and return_probs and hasattr(model, 'predict_proba'):
                # Some models have predict_proba
                predictions = model.predict_proba(X, batch_size=batch_size, verbose=verbose, **kwargs)
            elif is_classifier and return_probs:
                # Use standard predict for models without predict_proba
                predictions = model.predict(X, batch_size=batch_size, verbose=verbose, **kwargs)
            else:
                # Standard prediction
                predictions = model.predict(X, batch_size=batch_size, verbose=verbose, **kwargs)
            
            # Ensure we return numpy arrays
            if isinstance(predictions, tf.Tensor):
                predictions = predictions.numpy()
            
            return predictions
            
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
        Evaluate a TensorFlow/Keras model's performance.
        
        Args:
            model: The trained TensorFlow/Keras model
            X: Feature data (numpy array or tf.data.Dataset)
            y: True target values (may be None if X is a Dataset)
            metrics: List of metric names to compute (if None, use model's metrics)
            **kwargs: Additional evaluation parameters
                - batch_size: Batch size for evaluation (default: 32, ignored if X is a Dataset)
                - verbose: Verbosity level (default: 0)
            
        Returns:
            Dictionary of metric names to values
            
        Raises:
            EvaluationError: If evaluation fails
        """
        try:
            # Validate inputs
            if not isinstance(model, (keras.Model, keras.Sequential)):
                raise ValueError("Model is not a TensorFlow/Keras model")
            
            # Extract evaluation parameters
            batch_size = kwargs.pop('batch_size', 32)
            verbose = kwargs.pop('verbose', 0)
            
            # Get evaluation results
            if isinstance(X, tf.data.Dataset):
                # Dataset already contains features and targets
                evaluation = model.evaluate(X, verbose=verbose, return_dict=True, **kwargs)
            else:
                # Separate feature and target arrays
                evaluation = model.evaluate(X, y, batch_size=batch_size, verbose=verbose, return_dict=True, **kwargs)
            
            # Convert to float values (some metrics might be tf.Tensor)
            results = {k: float(v) if hasattr(v, 'numpy') else float(v) for k, v in evaluation.items()}
            
            # If specific metrics were requested but not in standard evaluation
            if metrics:
                # Get predictions for custom metrics
                y_pred = self.predict(model, X, batch_size=batch_size)
                
                # TensorFlow metrics need tensors
                y_true_tensor = tf.convert_to_tensor(y)
                y_pred_tensor = tf.convert_to_tensor(y_pred)
                
                # Calculate requested metrics if they're not already in results
                for metric_name in metrics:
                    if metric_name not in results:
                        # Try to find metric in keras.metrics
                        try:
                            metric_fn = getattr(keras.metrics, metric_name)
                            if callable(metric_fn):
                                metric_obj = metric_fn()
                                metric_obj.update_state(y_true_tensor, y_pred_tensor)
                                results[metric_name] = float(metric_obj.result().numpy())
                        except (AttributeError, ValueError) as e:
                            results[f"{metric_name}_error"] = f"Could not compute: {str(e)}"
            
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
        Save a TensorFlow/Keras model to disk.
        
        Args:
            model: The TensorFlow/Keras model to save
            path: Path where the model should be saved
            
        Raises:
            SerializationError: If saving fails
        """
        try:
            # Validate inputs
            if not isinstance(model, (keras.Model, keras.Sequential)):
                raise ValueError("Model is not a TensorFlow/Keras model")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            # Save model info as JSON
            model_info = self.get_model_info(model)
            model_info_json = {k: v for k, v in model_info.to_dict().items() if k != 'params'}
            model_info_json['framework'] = 'tensorflow'
            
            # Extract feature names if present
            feature_names = getattr(model, '_feature_names', [])
            model_info_json['feature_names'] = feature_names
            
            # Save model in SavedModel format
            if not path.endswith('/'):
                model_path = path
                info_path = f"{path}_info.json"
            else:
                model_path = f"{path}model"
                info_path = f"{path}model_info.json"
                
            # Save the model
            model.save(model_path)
            
            # Save the model info
            with open(info_path, 'w') as f:
                json.dump(model_info_json, f)
                
        except Exception as e:
            if isinstance(e, SerializationError):
                raise
            raise SerializationError(
                f"Failed to save model: {str(e)}",
                details={"original_error": str(e), "path": path}
            )
    
    def load_model(self, path: str) -> Any:
        """
        Load a TensorFlow/Keras model from disk.
        
        Args:
            path: Path from which to load the model
            
        Returns:
            The loaded TensorFlow/Keras model
            
        Raises:
            SerializationError: If loading fails
        """
        try:
            # Validate path
            if not os.path.exists(path):
                if os.path.exists(f"{path}_info.json"):
                    # Path exists with _info suffix, seems valid
                    info_path = f"{path}_info.json"
                    model_path = path
                else:
                    raise ValueError(f"Model file does not exist: {path}")
            elif os.path.isdir(path):
                # Directory path - check for model_info.json
                info_path = os.path.join(path, "model_info.json")
                model_path = os.path.join(path, "model")
                if not os.path.exists(info_path):
                    info_path = None
                    model_path = path
            else:
                # Assume the path is direct to the model
                info_path = f"{path}_info.json" if os.path.exists(f"{path}_info.json") else None
                model_path = path
            
            # Load the model using keras.models.load_model
            model = keras.models.load_model(model_path)
            
            # Load the model info if available
            if info_path and os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
                
                # Restore feature names if available
                if 'feature_names' in model_info:
                    model._feature_names = model_info['feature_names']
            
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
        Create a Prometheum transformer from a TensorFlow/Keras model.
        
        This allows TensorFlow models to be used within Prometheum pipelines.
        The resulting transformer will apply the model's predict method to the input data.
        
        Args:
            model: The trained TensorFlow/Keras model
            **kwargs: Additional parameters for the transformer
                - target_column: Name of the target column (for supervised models)
                - output_column: Name for the output column (default: "prediction")
                - batch_size: Batch size for prediction (default: 32)
            
        Returns:
            A transformer that can be used in a Prometheum pipeline
            
        Raises:
            ProcessingError: If transformer creation fails
        """
        try:
            # Validate the model
            if not isinstance(model, (keras.Model, keras.Sequential)):
                raise ValueError("Model is not a TensorFlow/Keras model")
            
            # Create a transformer wrapper for the TensorFlow model
            class TensorFlowModelTransformer(DataTransformer):
                """Transformer that applies a TensorFlow model."""
                
                def __init__(self, model, target_column=None, output_column=None, batch_size=32, **config):
                    """Initialize with a TensorFlow model."""
                    super().__init__(config)
                    self.model = model
                    return ModelType.CLASSIFIER
                elif activation_name in ('linear', 'relu'):
                    # Could be regressor or feature extractor
                    if last_layer.units == 1 if hasattr(last_layer, 'units') else False:
                        return ModelType.REGRESSOR
            
            # Check output shape for transformers
            if hasattr(last_layer, 'units') and last_layer.units > 1:
                return ModelType.TRANSFORMER
                
    # For pre-built models, check class name
    model_class_name = model.__class__.__name__.lower()
    if any(term in model_class_name for term in ('classifier', 'classification')):
        return ModelType.CLASSIFIER
    elif any(term in model_class_name for term in ('regressor', 'regression')):
        return ModelType.REGRESSOR
    elif any(term in model_class_name for term in ('autoencoder', 'encoder', 'embedding')):
        return ModelType.TRANSFORMER
    elif 'gan' in model_class_name:
        return ModelType.NEURAL_NETWORK
    
    # Default to neural network
    return ModelType.NEURAL_NETWORK


def _extract_tf_model_params(model: Any) -> Dict[str, Any]:
    """
    Extract parameters from a TensorFlow model.
    
    Args:
        model: A TensorFlow/Keras model instance
        
    Returns:
        Dict[str, Any]: The model's parameters
    """
    params = {}
    
    # Extract basic model configuration
    if hasattr(model, 'get_config'):
        try:
            config = model.get_config()
            # Simplify config to avoid serialization issues
            params['config'] = str(config)
        except Exception:
            pass
    
    # Extract layer information
    if hasattr(model, 'layers'):
        layer_info = []
        for i, layer in enumerate(model.layers):
            layer_type = layer.__class__.__name__
            layer_config = {}
            
            # Get basic layer properties
            if hasattr(layer, 'units'):
                layer_config['units'] = layer.units
            if hasattr(layer, 'activation') and hasattr(layer.activation, '__name__'):
                layer_config['activation'] = layer.activation.__name__
            if hasattr(layer, 'input_shape'):
                layer_config['input_shape'] = str(layer.input_shape)
            if hasattr(layer, 'output_shape'):
                layer_config['output_shape'] = str(layer.output_shape)
                
            layer_info.append({
                'layer_idx': i,
                'layer_type': layer_type,
                'config': layer_config
            })
            
        params['layers'] = layer_info
        params['num_layers'] = len(layer_info)
    
    # Extract optimizer information
    if hasattr(model, 'optimizer') and model.optimizer is not None:
        try:
            optimizer = model.optimizer
            params['optimizer'] = {
                'name': optimizer.__class__.__name__,
                'learning_rate': float(optimizer.learning_rate.numpy()) if hasattr(optimizer, 'learning_rate') else None
            }
        except Exception:
            params['optimizer'] = {'name': str(model.optimizer)}
    
    return params


class TensorFlowAdapter(MLAdapter):
    """
    Adapter for TensorFlow/Keras models.
    
    This adapter provides methods to create, train, evaluate, and save TensorFlow
    models within the Prometheum ecosystem.
    """
    
    @property
    def framework(self) -> FrameworkType:
        """Get the ML framework type."""
        return FrameworkType.TENSORFLOW
    
    def create_model(self, model_name: str, **kwargs) -> Any:
        """
        Create a TensorFlow/Keras model instance.
        
        Args:
            model_name: Name of the model class or architecture
                (e.g., 'Sequential', 'keras.applications.ResNet50')
            **kwargs: Parameters to pass to the model constructor
            
        Returns:
            A new model instance
            
        Raises:
            ModelCreationError: If model creation fails
        """
        try:
            # Case 1: Built-in Keras model types
            if model_name == 'Sequential':
                # If layers are provided, add them to the model
                layers = kwargs.pop('layers', [])
                model = keras.Sequential(**kwargs)
                for layer in layers:
                    model.add(layer)
                return model
            
            # Case 2: Functional API
            elif model_name == 'Functional':
                inputs = kwargs.pop('inputs', None)
                outputs = kwargs.pop('outputs', None)
                
                if inputs is None or outputs is None:
                    raise ModelCreationError(
                        "Functional API requires 'inputs' and 'outputs' parameters",
                        details={"inputs_provided": inputs is not None, "outputs_provided": outputs is not None}
                    )
                
                return keras.Model(inputs=inputs, outputs=outputs, **kwargs)
            
            # Case 3: Pre-built Keras applications
            elif model_name.startswith('keras.applications.'):
                module_parts = model_name.split('.')
                if len(module_parts) < 3:
                    raise ModelCreationError(
                        f"Invalid model name format: {model_name}. Expected 'keras.applications.ModelName'",
                        details={"model_name": model_name}
                    )
                
                model_class_name = module_parts[-1]
                module_name = '.'.join(module_parts[:-1])
                
                try:
                    module = importlib.import_module(module_name)
                    model_class = getattr(module, model_class_name)
                    return model_class(**kwargs)
                except (ImportError, AttributeError) as e:
                    raise ModelCreationError(
                        f"Failed to import model class '{model_name}': {str(e)}",
                        details={"model_name": model_name, "original_error": str(e)}
                    )
            
            # Case 4: Fully custom architecture defined by string
            elif '.' in model_name:
                module_parts = model_name.split('.')
                model_class_name = module_parts[-1]
                module_name = '.'.join(module_parts[:-1])
                
                try:
                    module = importlib.import_module(module_name)
                    model_class = getattr(module, model_class_name)
                    return model_class(**kwargs)
                except (ImportError, AttributeError) as e:
                    raise ModelCreationError(
                        f"Failed to import model class '{model_name}': {str(e)}",
                        details={"model_name": model_name, "original_error": str(e)}
                    )
            
            # Case 5: Loading a pre-defined model architecture
            else:
                try:
                    # Try to find the model in keras.applications
                    applications = importlib.import_module('tensorflow.keras.applications')
                    if hasattr(applications, model_name):
                        model_fn = getattr(applications, model_name)
                        return model_fn(**kwargs)
                except (ImportError, AttributeError):
                    pass
                
                # If we get here, we couldn't find the model
                raise ModelCreationError(
                    f"Could not find model architecture '{model_name}'",
                    details={"model_name": model_name}
                )
                
        except Exception as e:
            if isinstance(e, ModelCreationError):
                raise
            raise ModelCreationError(
                f"Failed to create TensorFlow model '{model_name}': {str(e)}",
                details={"original_error": str(e)}
            )
    
    def get_model_info(self, model: Any) -> ModelInfo:
        """
        Get information about a TensorFlow model.
        
        Args:
            model: The TensorFlow/Keras model instance
            
        Returns:
            ModelInfo: Information about the model
        """
        if not isinstance(model, (keras.Model, keras.Sequential)):
            raise ValueError("Model is not a TensorFlow/Keras model")
        
        model_type = _get_tf_model_type(model)
        model_name = model.__class__.__name__
        params = _extract_tf_model_params(model)
        
        # Extract feature names if the model has been trained on labeled data
        feature_names = []
        if hasattr(model, '_feature_names') and model._feature_names:
            feature_names = model._feature_names
        
        # Extract created timestamp
        created_at = datetime.now().isoformat()
        
        # Check if model has been trained
        is_fitted = False
        if hasattr(model, 'history') and model.history is not None:
            is_fitted = True
        
        # Get model metrics
        metrics = {}
        if hasattr(model, 'history') and model.history and hasattr(model.history, 'history'):
            # Extract the last epoch's metrics
            for metric_name, values in model.history.history.items():
                if values and len(values) > 0:
                    metrics[metric_name] = float(values[-1])
        
        # Get TensorFlow version
        tf_version = tf.__version__
        
        return ModelInfo(
            name=model_name,
            model_type=model_type,
            framework=FrameworkType.TENSORFLOW,
            params=params,
            description=model.__doc__.split("\n")[0] if model.__doc__ else None,
            version=tf_version,
            created_at=created_at,
            metrics=metrics,
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
        Prepare data for TensorFlow model training or prediction.
        
        Args:
            data: Input data (DataFrame or DataFrameContainer)
            target_column: Name of target column (for supervised learning)
            feature_columns: List of feature column names to use
            **kwargs: Additional preparation parameters
                - batch_size: Size of batches for TF Dataset (if created)
                - shuffle: Whether to shuffle the data
                - create_dataset: If True, return tf.data.Dataset instead of numpy arrays
                - categorical_target: If True, convert target to categorical
                - num_classes: Number of classes for categorical conversion
            
        Returns:
            Tuple of (X, y) where X is features and y is target (y may be None for unsupervised)
            The return types can be numpy arrays or tf.data.Dataset depending on kwargs
            
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
                
                # Convert target to categorical if requested
                if kwargs.get('categorical_target', False) and y is not None:
                    num_classes = kwargs.get('num_classes', None)
                    y = keras.utils.to_categorical(y, num_classes=num_classes)
            
            # Store feature names for later use
            feature_names = X_df.columns.tolist()
            
            # Create TensorFlow Dataset if requested
            if kwargs.get('create_dataset', False):
                batch_size = kwargs.get('batch_size', 32)
                shuffle = kwargs.get('shuffle', True)
                
                if y is not None:
                    # Supervised learning
                    dataset = tf.data.Dataset.from_tensor_slices((X, y))
                else:
                    #

