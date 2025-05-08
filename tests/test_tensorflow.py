"""
Tests for the TensorFlowAdapter class and TensorFlow integration.

This module provides comprehensive tests for the TensorFlow adapter implementation,
covering model creation, data preparation, training, evaluation, and serialization.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, optimizers, losses, metrics
    from tensorflow.keras.applications import MobileNetV2
except ImportError:
    pytest.skip("TensorFlow not installed", allow_module_level=True)

from prometheum.core.base import DataFrameContainer
from prometheum.ml.base import ModelType, FrameworkType, PredictionError, TrainingError
from prometheum.ml.tensorflow import TensorFlowAdapter


# Custom TensorFlow Models for Testing
def create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=2):
    """Create a simple classification model with Sequential API."""
    model = models.Sequential([
        layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
        layers.Dense(output_dim, activation='softmax')
    ])
    return model


def create_simple_regressor(input_dim=4, hidden_dim=10):
    """Create a simple regression model with Sequential API."""
    model = models.Sequential([
        layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,)),
        layers.Dense(1)
    ])
    return model


def create_functional_model(input_dim=4, hidden_dim=10, output_dim=2):
    """Create a model using the Functional API."""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(hidden_dim, activation='relu')(inputs)
    outputs = layers.Dense(output_dim, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs)


def create_autoencoder(input_dim=4, encoding_dim=2):
    """Create a simple autoencoder with Functional API."""
    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation='relu')(inputs)
    
    # Decoder
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    
    # Autoencoder model
    autoencoder = models.Model(inputs=inputs, outputs=decoded)
    
    # Encoder model
    encoder = models.Model(inputs=inputs, outputs=encoded)
    
    return autoencoder, encoder


class CustomCallback(callbacks.Callback):
    """A simple callback for testing."""
    def __init__(self):
        super().__init__()
        self.called = 0
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.called += 1
        self.epochs.append(epoch)
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


class CustomLayer(layers.Layer):
    """Custom layer for testing."""
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


@pytest.fixture
def adapter():
    """Return a fresh TensorFlowAdapter instance."""
    return TensorFlowAdapter()


@pytest.fixture
def binary_classification_data():
    """Generate a simple binary classification dataset."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)  # Simple binary decision
    # Convert to dataframe for easier testing
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


@pytest.fixture
def multiclass_data():
    """Generate a simple multiclass classification dataset."""
    np.random.seed(42)
    X = np.random.randn(150, 4)
    # Generate 3 classes based on simple rules
    y = np.zeros(150)
    y[(X[:, 0] > 0.5)] = 1
    y[(X[:, 1] > 0.7)] = 2
    # Convert to dataframe
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


@pytest.fixture
def regression_data():
    """Generate a simple regression dataset."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    # Simple linear relationship with noise
    y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + 0.5*X[:, 3] + np.random.randn(100)*0.1
    # Convert to dataframe
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


@pytest.fixture
def single_class_data():
    """Generate a dataset with only one class."""
    # Create a dataset with a single class (all 0s)
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = np.zeros(50)  # All samples are class 0
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


@pytest.fixture
def empty_feature_data():
    """Generate a dataset with no features (just target)."""
    # Create a simple dataset with just a target column
    np.random.seed(42)
    y = np.random.randint(0, 2, 30)
    df = pd.DataFrame({"target": y})
    return df


@pytest.fixture
def simple_classifier():
    """Return a simple untrained neural network classifier."""
    return create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=2)


@pytest.fixture
def simple_regressor():
    """Return a simple untrained neural network regressor."""
    return create_simple_regressor(input_dim=4, hidden_dim=10)


@pytest.fixture
def functional_model():
    """Return a model created with the Functional API."""
    return create_functional_model(input_dim=4, hidden_dim=10, output_dim=2)


@pytest.fixture
def autoencoder_models():
    """Return autoencoder and encoder models."""
    return create_autoencoder(input_dim=4, encoding_dim=2)


class TestTensorFlowModelCreation:
    """Test TensorFlow model creation functionality."""

    def test_create_sequential_model(self, adapter):
        """Test creating a Sequential model."""
        model = adapter.create_model("Sequential", 
                                     layers=[
                                         layers.Dense(10, activation='relu', input_shape=(4,)),
                                         layers.Dense(2, activation='softmax')
                                     ])
        assert model is not None
        assert isinstance(model, models.Sequential)
        
        # Validate layer structure
        assert len(model.layers) == 2
        assert isinstance(model.layers[0], layers.Dense)
        assert model.layers[0].units == 10
        assert isinstance(model.layers[1], layers.Dense)
        assert model.layers[1].units == 2
        
    def test_create_functional_model(self, adapter):
        """Test creating a model with the Functional API."""
        inputs = layers.Input(shape=(4,))
        x = layers.Dense(10, activation='relu')(inputs)
        outputs = layers.Dense(2, activation='softmax')(x)
        
        model = adapter.create_model("Model", inputs=inputs, outputs=outputs)
        assert model is not None
        assert isinstance(model, models.Model)
        
        # Validate input/output shapes
        assert model.input_shape == (None, 4)
        assert model.output_shape == (None, 2)
        
    def test_create_from_predefined_function(self, adapter):
        """Test creating a model using a predefined function."""
        model = adapter.create_model("create_simple_classifier", 
                                     model_fn=create_simple_classifier,
                                     input_dim=4, 
                                     hidden_dim=12, 
                                     output_dim=3)
        assert model is not None
        assert isinstance(model, models.Sequential)
        
        # Validate layer structure
        assert len(model.layers) == 2
        assert model.layers[0].units == 12
        assert model.layers[1].units == 3
        
    def test_create_from_keras_applications(self, adapter):
        """Test creating a model from Keras applications."""
        model = adapter.create_model("MobileNetV2", 
                                     include_top=False,
                                     weights=None,
                                     input_shape=(96, 96, 3),
                                     pooling='avg')
        assert model is not None
        assert isinstance(model, models.Model)
        
        # Check that it's a headless MobileNetV2
        assert model.name.startswith('mobilenetv2')
        
    def test_create_with_custom_layer(self, adapter):
        """Test creating a model with a custom layer."""
        model = adapter.create_model("Sequential", 
                                     layers=[
                                         CustomLayer(units=10, input_shape=(4,)),
                                         layers.Activation('relu'),
                                         layers.Dense(2, activation='softmax')
                                     ])
        assert model is not None
        assert isinstance(model.layers[0], CustomLayer)
        assert model.layers[0].units == 10
        
    def test_invalid_model_name(self, adapter):
        """Test error handling for invalid model name."""
        with pytest.raises(Exception):
            adapter.create_model("NonExistentModel")
    
    def test_invalid_layers_argument(self, adapter):
        """Test error handling for invalid layers argument."""
        with pytest.raises(Exception):
            # Not valid layers
            adapter.create_model("Sequential", layers=["not", "layers"])


class TestTensorFlowModelInfo:
    """Test model info extraction."""
    
    def test_get_model_info_classifier(self, adapter, simple_classifier):
        """Test getting info for classifier model."""
        # Compile the model first
        simple_classifier.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        info = adapter.get_model_info(simple_classifier)
        
        assert info.name.lower() in ("sequential", "model")
        assert info.model_type in (ModelType.CLASSIFIER, ModelType.NEURAL_NETWORK)
        assert info.framework == FrameworkType.TENSORFLOW
        
        # Verify layer structure is captured
        assert "layers" in info.params
        assert isinstance(info.params["layers"], list)
        assert len(info.params["layers"]) == 2
        
        # Verify optimizer and loss are captured
        assert "optimizer" in info.params
        assert info.params["optimizer"]["name"] == "Adam"
        assert "loss" in info.params
        assert info.params["loss"] == "sparse_categorical_crossentropy"
    
    def test_get_model_info_regressor(self, adapter, simple_regressor):
        """Test getting info for regressor model."""
        # Compile the model first
        simple_regressor.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        info = adapter.get_model_info(simple_regressor)
        
        assert info.name.lower() in ("sequential", "model")
        assert info.model_type in (ModelType.REGRESSOR, ModelType.NEURAL_NETWORK)
        assert info.framework == FrameworkType.TENSORFLOW
        
        # Verify loss is mse
        assert "loss" in info.params
        assert info.params["loss"] == "mse"
    
    def test_get_model_info_functional(self, adapter, functional_model):
        """Test getting info for a Functional API model."""
        # Compile the model first
        functional_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        info = adapter.get_model_info(functional_model)
        
        assert info.name.lower() in ("functional", "model")
        assert info.model_type in (ModelType.CLASSIFIER, ModelType.NEURAL_NETWORK)
        assert info.framework == FrameworkType.TENSORFLOW
        
        # Verify input and output shapes
        assert "input_shape" in info.params
        assert "output_shape" in info.params
    
    def test_get_model_info_autoencoder(self, adapter, autoencoder_models):
        """Test getting info for autoencoder model."""
        autoencoder, _ = autoencoder_models
        
        # Compile the model first
        autoencoder.compile(
            optimizer='adam',
            loss='mse'
        )
        
        info = adapter.get_model_info(autoencoder)
        
        assert info.name.lower() in ("model", "functional")
        assert info.model_type in (ModelType.TRANSFORMER, ModelType.NEURAL_NETWORK)
        assert info.framework == FrameworkType.TENSORFLOW
    
    def test_get_model_info_with_custom_optimizer(self, adapter, simple_classifier):
        """Test getting info for model with custom optimizer."""
        # Compile with custom optimizer
        optimizer = optimizers.Adam(learning_rate=0.01)
        simple_classifier.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        info = adapter.get_model_info(simple_classifier)
        
        assert "optimizer" in info.params
        assert info.params["optimizer"]["name"] == "Adam"
        assert info.params["optimizer"]["learning_rate"] == 0.01
    
    def test_get_model_info_with_custom_metrics(self, adapter, simple_regressor):
        """Test getting info for model with custom metrics."""
        # Define a custom metric
        @tf.function
        def custom_metric(y_true, y_pred):
            return tf.reduce_mean(tf.abs(y_true - y_pred)) / tf.reduce_mean(tf.abs(y_true))
        
        # Compile with custom metric
        simple_regressor.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', custom_metric]
        )
        
        info = adapter.get_model_info(simple_regressor)
        
        assert "metrics" in info.params
        assert len(info.params["metrics"]) >= 2
        assert "mae" in info.params["metrics"]
    
    def test_uncompiled_model_info(self, adapter, simple_classifier):
        """Test getting info for an uncompiled model."""
        # Model is not compiled yet
        info = adapter.get_model_info(simple_classifier)
        
        assert info.name.lower() in ("sequential", "model")
        assert info.framework == FrameworkType.TENSORFLOW
        
        # Verify basic model structure is captured even without compilation
        assert "layers" in info.params
        assert isinstance(info.params["layers"], list)
        assert len(info.params["layers"]) == 2


class TestTensorFlowDataPreparation:
    """Test data preparation functionality."""
    
    def test_prepare_data_with_dataframe(self, adapter, binary_classification_data):
        """Test preparing data from DataFrame."""
        X, y = adapter.prepare_data(
            binary_classification_data, 
            target_column="target"
        )
        
        assert isinstance(X, np.ndarray) or isinstance(X, tf.Tensor)
        assert isinstance(y, np.ndarray) or isinstance(y, tf.Tensor)
        assert X.shape[0] == 100
        assert X.shape[1] == 4
        assert y.shape[0] == 100
        assert hasattr(X, "_feature_names")
        assert len(X._feature_names) == 4
    
    def test_prepare_data_with_container(self, adapter, regression_data):
        """Test preparing data from DataFrameContainer."""
        container = DataFrameContainer(regression_data, {"source": "test"})
        X, y = adapter.prepare_data(
            container, 
            target_column="target"
        )
        
        assert isinstance(X, np.ndarray) or isinstance(X, tf.Tensor)
        assert isinstance(y, np.ndarray) or isinstance(y, tf.Tensor)
        assert X.shape[0] == regression_data.shape[0]
        assert X.shape[1] == regression_data.shape[1] - 1
    
    def test_prepare_data_with_feature_selection(self, adapter, multiclass_data):
        """Test preparing data with specific feature columns."""
        feature_cols = multiclass_data.columns[:-1][:2]  # Just first two features
        X, y = adapter.prepare_data(
            multiclass_data,
            target_column="target",
            feature_columns=feature_cols
        )
        
        assert X.shape[1] == 2
        assert y.shape[0] == multiclass_data.shape[0]
        assert hasattr(X, "_feature_names")
        assert X._feature_names == list(feature_cols)
    
    def test_prepare_data_with_tf_dataset(self, adapter, binary_classification_data):
        """Test preparing data with TensorFlow Dataset creation."""
        X, y = adapter.prepare_data(
            binary_classification_data, 
            target_column="target",
            create_dataset=True,
            batch_size=16
        )
        
        assert isinstance(X, tf.data.Dataset)
        assert y is None  # Dataset returns None for y
        
        # Check if dataset contains the correct elements
        for batch_x, batch_y in X.take(1):
            assert batch_x.shape[1] == 4
            assert batch_y.shape[0] <= 16  # Batch size
    
    def test_prepare_data_with_custom_dtype(self, adapter, regression_data):
        """Test preparing data with custom data type."""
        X, y = adapter.prepare_data(
            regression_data,
            target_column="target",
            dtype=tf.float64
        )
        
        if isinstance(X, tf.Tensor):
            assert X.dtype == tf.float64
            assert y.dtype == tf.float64
        else:
            # If numpy array
            assert X.dtype == np.float64
            assert y.dtype == np.float64
    
    def test_prepare_data_error_handling(self, adapter, empty_feature_data):
        """Test error handling with invalid data."""
        # Should raise an exception for data with no features
        with pytest.raises(Exception):
            X, y = adapter.prepare_data(empty_feature_data, target_column="target")
        
        # Should raise exception for missing features
        with pytest.raises(Exception):
            # Request a feature that doesn't exist
            X, y = adapter.prepare_data(
                binary_classification_data, 
                target_column="target",
                feature_columns=["feature_0", "nonexistent_feature"]
            )
    
    def test_prepare_data_with_categorical_encoding(self, adapter):
        """Test preparing data with categorical encoding."""
        # Create dataset with categorical features
        np.random.seed(42)
        df = pd.DataFrame({
            'feature_0': np.random.randn(100),
            'feature_1': np.random.choice(['A', 'B', 'C'], 100),
            'feature_2': np.random.choice(['X', 'Y'], 100),
            'target': np.random.randint(0, 2, 100)
        })
        
        X, y = adapter.prepare_data(
            df,
            target_column="target",
            categorical_columns=["feature_1", "feature_2"]
        )
        
        # Feature count should increase due to one-hot encoding
        assert X.shape[1] > 3  # Original 3 features + one-hot encoded columns - removed categoricals


class TestTensorFlowTrainingAndPrediction:
    """Test model training and prediction."""
    
    def test_train_binary_classifier(self, adapter, binary_classification_data):
        """Test training a binary classifier."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=1)
        
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=5,
            batch_size=16,
            verbose=0
        )
        
        assert hasattr(trained_model, "optimizer")
        assert hasattr(trained_model, "loss")
        
        # Should have training history
        assert hasattr(trained_model, "history")
        assert "loss" in trained_model.history.history
        assert len(trained_model.history.history["loss"]) == 5  # 5 epochs
        
        # Test prediction capability
        y_pred = adapter.predict(trained_model, X)
        assert y_pred.shape == (100,)  # 100 samples
    
    def test_train_with_validation_split(self, adapter, binary_classification_data):
        """Test training with validation split."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=1)
        
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=3,
            validation_split=0.2,
            verbose=0
        )
        
        assert "val_loss" in trained_model.history.history
        assert len(trained_model.history.history["val_loss"]) == 3  # 3 epochs
    
    def test_train_with_callbacks(self, adapter, binary_classification_data):
        """Test training with callbacks."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=1)
        callback = CustomCallback()
        
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=3,
            callbacks=[callback],
            verbose=0
        )
        
        # Verify callback was called for each epoch
        assert callback.called == 3
        assert len(callback.epochs) == 3
        assert all(loss is not None for loss in callback.train_losses)
    
    def test_train_with_tf_dataset(self, adapter, binary_classification_data):
        """Test training with TensorFlow Dataset."""
        dataset = adapter.prepare_data(
            binary_classification_data, 
            target_column="target",
            create_dataset=True,
            batch_size=16
        )[0]
        
        model = create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=1)
        
        trained_model = adapter.train(
            model, 
            dataset,
            epochs=3,
            verbose=0
        )
        
        assert hasattr(trained_model, "history")
        assert "loss" in trained_model.history.history
    
    def test_train_with_custom_metrics(self, adapter, regression_data):
        """Test training with custom metrics."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        model = create_simple_regressor(input_dim=4, hidden_dim=10)
        
        # Create custom metric
        @tf.function
        def mean_abs_percent_error(y_true, y_pred):
            return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + 1e-7))) * 100
        
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=3,
            metrics=['mae', mean_abs_percent_error],
            verbose=0
        )
        
        assert "mae" in trained_model.history.history
        assert "mean_abs_percent_error" in trained_model.history.history
    
    def test_train_with_early_stopping(self, adapter, regression_data):
        """Test training with early stopping."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        model = create_simple_regressor(input_dim=4, hidden_dim=10)
        
        # Create early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )
        
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=10,  # More epochs than needed
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Early stopping should prevent all 10 epochs from running
        assert len(trained_model.history.history["loss"]) < 10
    
    def test_train_multiclass_classifier(self, adapter, multiclass_data):
        """Test training a multiclass classifier."""
        X, y = adapter.prepare_data(multiclass_data, target_column="target")
        model = create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=3)
        
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=3,
            verbose=0
        )
        
        # Test prediction capability
        y_pred = adapter.predict(trained_model, X)
        assert y_pred.shape == (150,)  # 150 samples
        assert np.all(np.isin(np.unique(y_pred), [0, 1, 2]))  # Classes 0, 1, 2
    
    def test_train_regressor(self, adapter, regression_data):
        """Test training a regression model."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        model = create_simple_regressor(input_dim=4, hidden_dim=10)
        
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=3,
            verbose=0
        )
        
        # Test prediction capability
        y_pred = adapter.predict(trained_model, X)
        assert y_pred.shape == (100,)
        assert np.issubdtype(y_pred.dtype, np.floating)  # Ensure floating point predictions
    
    def test_prediction_with_preprocessing(self, adapter, regression_data):
        """Test prediction with preprocessing layers."""
        X_np, y_np = adapter.prepare_data(regression_data, target_column="target")
        
        # Create model with preprocessing layer
        inputs = layers.Input(shape=(4,))
        norm = layers.Normalization(axis=-1)(inputs)
        hidden = layers.Dense(10, activation='relu')(norm)
        outputs = layers.Dense(1)(hidden)
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Adapt normalization layer
        norm_layer = model.layers[1]
        norm_layer.adapt(X_np)
        
        trained_model = adapter.train(
            model, 
            X_np, 
            y_np,
            epochs=3,
            verbose=0
        )
        
        # Test prediction capability
        y_pred = adapter.predict(trained_model, X_np)
        assert y_pred.shape == (100,)
        
    def test_untrained_model_prediction(self, adapter, binary_classification_data):
        """Test prediction with untrained model raises error."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=1)
        
        # Trying to predict with untrained model should raise exception
        with pytest.raises(PredictionError):
            adapter.predict(model, X)


class TestTensorFlowEvaluation:
    """Test model evaluation functionality."""
    
    def test_binary_classification_metrics(self, adapter, binary_classification_data):
        """Test evaluation metrics for binary classification."""
        # Split data for proper evaluation
        train_df = binary_classification_data.sample(frac=0.7, random_state=42)
        test_df = binary_classification_data.drop(train_df.index)
        
        X_train, y_train = adapter.prepare_data(train_df, target_column="target")
        X_test, y_test = adapter.prepare_data(test_df, target_column="target")
        
        model = create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=1)
        model = adapter.train(
            model, 
            X_train, 
            y_train,
            epochs=5,
            verbose=0
        )
        
        # Test default metrics
        metrics = adapter.evaluate(model, X_test, y_test)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        
        # Verify metrics are in valid ranges
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        
        # Test specific metrics
        specific_metrics = ["accuracy", "confusion_matrix"]
        metrics = adapter.evaluate(model, X_test, y_test, metrics=specific_metrics)
        
        assert "accuracy" in metrics
        assert "confusion_matrix" in metrics
        assert isinstance(metrics["confusion_matrix"], np.ndarray) or isinstance(metrics["confusion_matrix"], list)
    
    def test_multiclass_classification_metrics(self, adapter, multiclass_data):
        """Test evaluation metrics for multiclass classification."""
        # Split data for proper evaluation
        train_df = multiclass_data.sample(frac=0.7, random_state=42)
        test_df = multiclass_data.drop(train_df.index)
        
        X_train, y_train = adapter.prepare_data(train_df, target_column="target")
        X_test, y_test = adapter.prepare_data(test_df, target_column="target")
        
        model = create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=3)
        model = adapter.train(
            model, 
            X_train, 
            y_train,
            epochs=5,
            verbose=0
        )
        
        # Test default metrics
        metrics = adapter.evaluate(model, X_test, y_test)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        
        # For multiclass, precision/recall/f1 should be macro-averaged
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
    
    def test_regression_metrics(self, adapter, regression_data):
        """Test evaluation metrics for regression."""
        # Split data for proper evaluation
        train_df = regression_data.sample(frac=0.7, random_state=42)
        test_df = regression_data.drop(train_df.index)
        
        X_train, y_train = adapter.prepare_data(train_df, target_column="target")
        X_test, y_test = adapter.prepare_data(test_df, target_column="target")
        
        model = create_simple_regressor(input_dim=4, hidden_dim=10)
        model = adapter.train(
            model, 
            X_train, 
            y_train,
            epochs=5,
            verbose=0
        )
        
        # Test default metrics
        metrics = adapter.evaluate(model, X_test, y_test)
        
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        
        # Verify metrics are reasonable
        assert metrics["mse"] >= 0
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["r2"] <= 1
        
        # Verify RMSE is square root of MSE
        assert np.isclose(metrics["rmse"], np.sqrt(metrics["mse"]), rtol=1e-5)
    
    def test_custom_metric(self, adapter, binary_classification_data):
        """Test using a custom metric function."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=1)
        model = adapter.train(model, X, y, epochs=3, verbose=0)
        
        # Define a custom metric function
        def balanced_accuracy(y_true, y_pred):
            from sklearn.metrics import balanced_accuracy_score
            return balanced_accuracy_score(y_true, y_pred)
        
        # Test with custom metric function
        metrics = adapter.evaluate(
            model, 
            X, 
            y, 
            metrics=["accuracy", balanced_accuracy]
        )
        
        assert "accuracy" in metrics
        assert "balanced_accuracy" in metrics
        assert 0 <= metrics["balanced_accuracy"] <= 1
    
    def test_evaluation_with_bad_metric(self, adapter, binary_classification_data):
        """Test evaluation with an invalid metric."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=1)
        model = adapter.train(model, X, y, epochs=3, verbose=0)
        
        # Test with invalid metric
        metrics = adapter.evaluate(
            model, 
            X, 
            y, 
            metrics=["accuracy", "nonexistent_metric"]
        )
        
        assert "accuracy" in metrics
        assert "nonexistent_metric_error" in metrics
    
    def test_evaluate_with_tf_dataset(self, adapter, binary_classification_data):
        """Test evaluation with TensorFlow Dataset."""
        # Create a dataset for training
        dataset = adapter.prepare_data(
            binary_classification_data, 
            target_column="target",
            create_dataset=True,
            batch_size=16
        )[0]
        
        model = create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=1)
        model = adapter.train(model, dataset, epochs=3, verbose=0)
        
        # Prepare test data as dataset
        test_df = binary_classification_data.sample(frac=0.3, random_state=24)
        test_dataset = adapter.prepare_data(
            test_df, 
            target_column="target",
            create_dataset=True,
            batch_size=8
        )[0]
        
        # Evaluate using dataset
        metrics = adapter.evaluate(model, test_dataset)
        
        assert "accuracy" in metrics


class TestTensorFlowSerialization:
    """Test model serialization functionality."""
    
    def test_save_and_load_model(self, adapter, binary_classification_data):
        """Test saving and loading a model."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=1)
        trained_model = adapter.train(model, X, y, epochs=3, verbose=0)
        
        # Get predictions from original model
        original_preds = adapter.predict(trained_model, X)
        
        # Save the model to a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model")
            adapter.save_model(trained_model, model_path)
            
            # Ensure file/directory exists
            assert os.path.exists(model_path)
            
            # Load the model
            loaded_model = adapter.load_model(model_path)
            
            # Verify loaded model type
            assert isinstance(loaded_model, models.Model)
            
            # Get predictions from loaded model
            loaded_preds = adapter.predict(loaded_model, X)
            
            # Verify predictions match
            np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5)
    
    def test_save_and_load_h5_format(self, adapter, regression_data):
        """Test saving and loading in HDF5 format."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        model = create_simple_regressor(input_dim=4, hidden_dim=10)
        trained_model = adapter.train(model, X, y, epochs=3, verbose=0)
        
        # Get predictions from original model
        original_preds = adapter.predict(trained_model, X)
        
        # Save the model to H5 format
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.h5")
            adapter.save_model(trained_model, model_path)
            
            # Ensure file exists
            assert os.path.exists(model_path)
            
            # Load the model
            loaded_model = adapter.load_model(model_path)
            
            # Get predictions from loaded model
            loaded_preds = adapter.predict(loaded_model, X)
            
            # Verify predictions match
            np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5)
    
    def test_save_and_load_with_custom_objects(self, adapter, binary_classification_data):
        """Test saving and loading models with custom layers."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        
        # Create model with custom layer
        inputs = layers.Input(shape=(4,))
        custom = CustomLayer(units=8)(inputs)
        hidden = layers.Activation('relu')(custom)
        outputs = layers.Dense(1, activation='sigmoid')(hidden)
        model = models.Model(inputs=inputs, outputs=outputs)
        
        trained_model = adapter.train(model, X, y, epochs=3, verbose=0)
        
        # Get predictions from original model
        original_preds = adapter.predict(trained_model, X)
        
        # Save the model to a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "custom_model")
            adapter.save_model(trained_model, model_path)
            
            # Ensure file/directory exists
            assert os.path.exists(model_path)
            
            # Load the model
            loaded_model = adapter.load_model(model_path, custom_objects={"CustomLayer": CustomLayer})
            
            # Verify model type and structure
            assert isinstance(loaded_model, models.Model)
            assert isinstance(loaded_model.layers[1], CustomLayer)
            assert loaded_model.layers[1].units == 8
            
            # Get predictions from loaded model
            loaded_preds = adapter.predict(loaded_model, X)
            
            # Verify predictions match
            np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5)
    
    def test_save_with_optimizer_state(self, adapter, regression_data):
        """Test that optimizer state is preserved during serialization."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        model = create_simple_regressor(input_dim=4, hidden_dim=10)
        
        # Use an optimizer with a learning rate
        optimizer = optimizers.Adam(learning_rate=0.01)
        
        # Train model with custom optimizer
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=3,
            optimizer=optimizer,
            verbose=0
        )
        
        # Save the model to a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model_with_optimizer")
            adapter.save_model(trained_model, model_path)
            
            # Load the model
            loaded_model = adapter.load_model(model_path)
            
            # Verify optimizer was preserved
            assert hasattr(loaded_model, 'optimizer')
            assert isinstance(loaded_model.optimizer, optimizers.Adam)
            
            # Verify optimizer learning rate
            assert loaded_model.optimizer.learning_rate.numpy() == 0.01
    
    def test_load_nonexistent_model(self, adapter):
        """Test error handling when loading from a non-existent path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nonexistent_path = os.path.join(tmp_dir, "nonexistent_model")
            
            with pytest.raises(Exception):
                adapter.load_model(nonexistent_path)
    
    def test_saved_model_format(self, adapter, binary_classification_data):
        """Test saving and loading in TensorFlow SavedModel format."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=1)
        trained_model = adapter.train(model, X, y, epochs=3, verbose=0)
        
        # Get predictions from original model
        original_preds = adapter.predict(trained_model, X)
        
        # Save the model to a temporary directory as SavedModel format
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "saved_model")
            adapter.save_model(trained_model, model_path, save_format="tf")
            
            # Ensure directory exists
            assert os.path.exists(model_path)
            assert os.path.exists(os.path.join(model_path, "saved_model.pb"))
            
            # Load the model
            loaded_model = adapter.load_model(model_path)
            
            # Get predictions from loaded model
            loaded_preds = adapter.predict(loaded_model, X)
            
            # Verify predictions match
            np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5)
    
    def test_handle_model_with_preprocessing(self, adapter, multiclass_data):
        """Test serialization of models with preprocessing layers."""
        X, y = adapter.prepare_data(multiclass_data, target_column="target")
        
        # Create model with normalization layer
        inputs = layers.Input(shape=(4,))
        norm = layers.Normalization(axis=-1)(inputs)
        hidden = layers.Dense(10, activation='relu')(norm)
        outputs = layers.Dense(3, activation='softmax')(hidden)
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Adapt the normalization layer
        norm_layer = model.layers[1]
        norm_layer.adapt(X)
        
        # Train the model
        trained_model = adapter.train(model, X, y, epochs=3, verbose=0)
        
        # Get predictions
        original_preds = adapter.predict(trained_model, X)
        
        # Save and load the model
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "preprocessing_model")
            adapter.save_model(trained_model, model_path)
            
            loaded_model = adapter.load_model(model_path)
            
            # Get predictions from loaded model
            loaded_preds = adapter.predict(loaded_model, X)
            
            # Verify predictions match
            np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5)


class TestTensorFlowEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_sample_data(self, adapter):
        """Test handling of single sample datasets."""
        # Create a dataset with just one sample
        X = np.random.randn(1, 4)
        y = np.array([1.0])
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y
        
        # Prepare data
        X_tensor, y_tensor = adapter.prepare_data(df, target_column="target")
        assert X_tensor.shape == (1, 4)
        assert y_tensor.shape == (1,)
        
        # Create and train model (should work with a single sample)
        model = create_simple_regressor(input_dim=4, hidden_dim=5)
        
        # Should work with batch_size=1
        trained_model = adapter.train(
            model,
            X_tensor,
            y_tensor,
            epochs=3,
            batch_size=1,
            verbose=0
        )
        
        # Should be able to predict for a single sample
        y_pred = adapter.predict(trained_model, X_tensor)
        assert y_pred.shape == (1,)
    
    def test_single_class_data(self, adapter, single_class_data):
        """Test handling of single-class classification datasets."""
        X, y = adapter.prepare_data(single_class_data, target_column="target")
        
        # Create and train a classifier on single-class data
        model = create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=1)
        trained_model = adapter.train(model, X, y, epochs=3, verbose=0)
        
        # Predictions should all be the same class
        y_pred = adapter.predict(trained_model, X)
        unique_classes = np.unique(y_pred)
        assert len(unique_classes) == 1
        
        # Metrics should handle single class appropriately
        metrics = adapter.evaluate(trained_model, X, y)
        
        # Check that we get appropriate metrics
        assert "accuracy" in metrics
    
    def test_nan_data_handling(self, adapter):
        """Test handling of NaN values in training data."""
        # Create a dataset with some NaN values
        X = np.random.randn(50, 4)
        # Insert NaN values in random positions
        X[np.random.randint(0, 50, 5), np.random.randint(0, 4, 5)] = np.nan
        y = np.random.randn(50)
        
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y
        
        # This should raise an exception during data preparation or training
        with pytest.raises(Exception):
            X_tensor, y_tensor = adapter.prepare_data(df, target_column="target")
            model = create_simple_regressor(input_dim=4, hidden_dim=10)
            adapter.train(model, X_tensor, y_tensor, epochs=3, verbose=0)
    
    def test_empty_batch_handling(self, adapter, binary_classification_data):
        """Test handling of potential empty batches."""
        # Create a very small dataset
        small_df = binary_classification_data.iloc[:3]
        X, y = adapter.prepare_data(small_df, target_column="target")
        
        model = create_simple_classifier(input_dim=4, hidden_dim=5, output_dim=1)
        
        # Train with batch size larger than dataset (should handle this gracefully)
        trained_model = adapter.train(
            model,
            X,
            y,
            epochs=3,
            batch_size=5,  # Larger than dataset size
            verbose=0
        )
        
        # Should still be trained and able to predict
        assert trained_model is not None
        
        # Test prediction
        y_pred = adapter.predict(trained_model, X)
        assert y_pred.shape == (3,)
    
    def test_zero_epoch_training(self, adapter, binary_classification_data):
        """Test training with zero epochs."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=1)
        
        # Compile the model first
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Train with zero epochs (should do nothing but not error)
        trained_model = adapter.train(
            model,
            X,
            y,
            epochs=0,
            verbose=0
        )
        
        # Should be the same as input model
        assert trained_model is not None
        
        # Prediction should raise an error if adapter prevents untrained model prediction
        with pytest.raises(Exception):
            adapter.predict(trained_model, X)
    def test_custom_training_loop(self, adapter, regression_data):
        """Test using a custom training loop."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        
        # Create a model
        model = create_simple_regressor(input_dim=4, hidden_dim=10)
        optimizer = optimizers.Adam(learning_rate=0.01)
        loss_fn = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer=optimizer, loss=loss_fn)
        
        # Define a custom training loop function
        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss_value = loss_fn(y, y_pred)
            
            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss_value
        
        # Run custom training loop for a few steps
        batch_size = 32
        dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)
        
        for epoch in range(3):
            for step, (x_batch, y_batch) in enumerate(dataset):
                loss = train_step(x_batch, y_batch)
                if step >= 3:  # Just do a few steps for the test
                    break
        
        # Model should be trained
        y_pred = adapter.predict(model, X)
        assert y_pred.shape == (100,)
    
    def test_custom_model_class(self, adapter, binary_classification_data):
        """Test training with a custom model subclass."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        
        # Define a custom model class
        class CustomModel(models.Model):
            def __init__(self):
                super().__init__()
                self.dense1 = layers.Dense(10, activation='relu')
                self.dense2 = layers.Dense(1, activation='sigmoid')
                
            def call(self, inputs, training=None):
                x = self.dense1(inputs)
                return self.dense2(x)
            
            def get_config(self):
                return {}
                
            @classmethod
            def from_config(cls, config):
                return cls()
        
        # Create an instance of the custom model
        model = CustomModel()
        
        # Train the model
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=3,
            verbose=0
        )
        
        # Test prediction capability
        y_pred = adapter.predict(trained_model, X)
        assert y_pred.shape == (100,)
    
    def test_custom_loss_function(self, adapter, regression_data):
        """Test training with custom loss function."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        model = create_simple_regressor(input_dim=4, hidden_dim=10)
        
        # Define a custom loss function
        @tf.function
        def custom_huber_loss(y_true, y_pred, delta=1.0):
            error = y_true - y_pred
            is_small_error = tf.abs(error) <= delta
            squared_loss = 0.5 * tf.square(error)
            linear_loss = delta * (tf.abs(error) - 0.5 * delta)
            return tf.where(is_small_error, squared_loss, linear_loss)
        
        # Train with custom loss
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=3,
            loss=custom_huber_loss,
            verbose=0
        )
        
        # Test prediction capability
        y_pred = adapter.predict(trained_model, X)
        assert y_pred.shape == (100,)
        
        # Verify loss was correctly used
        assert "loss" in trained_model.history.history
    
    def test_custom_callback_tracking(self, adapter, regression_data):
        """Test using complex custom callback with state tracking."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        model = create_simple_regressor(input_dim=4, hidden_dim=10)
        
        # Create a stateful custom callback
        class LearningRateScheduler(callbacks.Callback):
            def __init__(self, initial_lr=0.01, decay_factor=0.5):
                super().__init__()
                self.initial_lr = initial_lr
                self.decay_factor = decay_factor
                self.lr_history = []
                self.updated_epochs = []
                
            def on_epoch_begin(self, epoch, logs=None):
                if epoch > 0:
                    current_lr = self.initial_lr * (self.decay_factor ** epoch)
                    self.model.optimizer.lr.assign(current_lr)
                    self.updated_epochs.append(epoch)
                
            def on_epoch_end(self, epoch, logs=None):
                self.lr_history.append(float(self.model.optimizer.lr.numpy()))
        
        # Initialize the callback
        lr_scheduler = LearningRateScheduler(initial_lr=0.01, decay_factor=0.5)
        
        # Train model with callback
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=3,
            callbacks=[lr_scheduler],
            verbose=0
        )
        
        # Verify learning rate was updated correctly
        assert len(lr_scheduler.lr_history) == 3
        assert len(lr_scheduler.updated_epochs) == 2  # No update on first epoch
        assert lr_scheduler.lr_history[0] > lr_scheduler.lr_history[1]  # Learning rate decreased
        assert lr_scheduler.lr_history[1] > lr_scheduler.lr_history[2]  # Learning rate decreased more
        
        # Test prediction capability
        y_pred = adapter.predict(trained_model, X)
        assert y_pred.shape == (100,)


class TestTensorFlowGPUSupport:
    """Test GPU-specific functionality (if GPU is available)."""
    
    @pytest.mark.skipif(not tf.test.is_gpu_available(), 
                        reason="GPU not available for testing")
    def test_gpu_training(self, adapter, regression_data):
        """Test training on GPU."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        model = create_simple_regressor(input_dim=4, hidden_dim=10)
        
        # Train model on GPU
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=3,
            device="GPU",  # Request GPU explicitly
            verbose=0
        )
        
        # Verify model was trained on GPU
        # We know it's on GPU if either:
        # 1. The model's device is a GPU device
        # 2. TensorFlow shows GPU memory allocation
        
        # Get device of first layer weights
        first_layer_weights = trained_model.layers[0].weights[0]
        
        # Check if weights are on GPU (device name contains "GPU")
        assert "GPU" in first_layer_weights.device.upper()
        
        # Test prediction capability
        y_pred = adapter.predict(trained_model, X)
        assert y_pred.shape == (100,)
    
    @pytest.mark.skipif(not tf.test.is_gpu_available(), 
                        reason="GPU not available for testing")
    def test_mixed_precision_training(self, adapter, binary_classification_data):
        """Test training with mixed precision."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        
        # Create a model
        inputs = layers.Input(shape=(4,))
        hidden = layers.Dense(16, activation='relu')(inputs)
        outputs = layers.Dense(1, activation='sigmoid')(hidden)
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Enable mixed precision
        policy = 'mixed_float16'
        
        # Train with mixed precision
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=3,
            mixed_precision=policy,
            device="GPU",
            verbose=0
        )
        
        # Test prediction capability
        y_pred = adapter.predict(trained_model, X)
        assert y_pred.shape == (100,)
        
        # Verify model has mixed precision policy
        from tensorflow.keras import mixed_precision
        assert mixed_precision.global_policy().name == policy
    
    @pytest.mark.skipif(not tf.test.is_gpu_available(), 
                        reason="GPU not available for testing")
    def test_multi_gpu_model(self, adapter, regression_data):
        """Test training with multiple GPUs if available."""
        # Skip if multiple GPUs aren't available
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) < 2:
            pytest.skip("Multiple GPUs not available")
        
        X, y = adapter.prepare_data(regression_data, target_column="target")
        
        # Create a model for multi-GPU training
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = create_simple_regressor(input_dim=4, hidden_dim=10)
        
        # Train the model
        trained_model = adapter.train(
            model,
            X,
            y,
            epochs=2,
            verbose=0
        )
        
        # Verify it uses distributed strategy
        assert hasattr(trained_model, 'distribute_strategy')
        
        # Test prediction
        y_pred = adapter.predict(trained_model, X)
        assert y_pred.shape == (100,)


class TestTensorFlowErrorConditions:
    """Test various error conditions and exception handling."""
    
    def test_incompatible_input_shape(self, adapter, binary_classification_data):
        """Test handling incompatible input shapes during prediction."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=1)
        trained_model = adapter.train(model, X, y, epochs=2, verbose=0)
        
        # Create data with wrong input shape
        wrong_X = np.random.randn(10, 5)  # 5 features instead of 4
        
        # Should raise an exception
        with pytest.raises(Exception):
            adapter.predict(trained_model, wrong_X)
    
    def test_invalid_loss_function(self, adapter, regression_data):
        """Test handling invalid loss function."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        model = create_simple_regressor(input_dim=4, hidden_dim=10)
        
        # Invalid loss function
        with pytest.raises(Exception):
            adapter.train(
                model,
                X,
                y,
                loss="nonexistent_loss",
                epochs=2,
                verbose=0
            )
    
    def test_incompatible_data_target_shape(self, adapter):
        """Test handling incompatible data and target shapes."""
        # Create a dataset with mismatched shapes
        X = np.random.randn(50, 4)
        y = np.random.randn(60)  # Target has 60 samples, X has 50
        
        # Should raise an exception during training
        model = create_simple_regressor(input_dim=4, hidden_dim=10)
        
        with pytest.raises(Exception):
            adapter.train(model, X, y, epochs=1, verbose=0)
    
    def test_invalid_device_specification(self, adapter, binary_classification_data):
        """Test handling invalid device specification."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=1)
        
        # Invalid device name should raise an exception
        with pytest.raises(Exception):
            adapter.train(
                model,
                X,
                y,
                device="nonexistent_device",
                epochs=1,
                verbose=0
            )
    
    def test_invalid_optimizer_configuration(self, adapter, regression_data):
        """Test handling invalid optimizer configuration."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        model = create_simple_regressor(input_dim=4, hidden_dim=10)
        
        # Invalid optimizer parameters
        with pytest.raises(Exception):
            adapter.train(
                model,
                X,
                y,
                optimizer="adam",
                optimizer_params={"learning_rate": -0.1},  # Negative learning rate
                epochs=1,
                verbose=0
            )
        
        # Invalid optimizer name
        with pytest.raises(Exception):
            adapter.train(
                model,
                X,
                y,
                optimizer="nonexistent_optimizer",
                epochs=1,
                verbose=0
            )
    
    def test_invalid_callback_usage(self, adapter, binary_classification_data):
        """Test handling invalid callback usage."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=1)
        
        # Invalid callback (not a proper callback object)
        invalid_callback = {"this_is": "not a callback"}
        
        with pytest.raises(Exception):
            adapter.train(
                model,
                X,
                y,
                callbacks=[invalid_callback],
                epochs=1,
                verbose=0
            )
    
    def test_invalid_metric_specification(self, adapter, regression_data):
        """Test handling invalid metric specification."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        model = create_simple_regressor(input_dim=4, hidden_dim=10)
        
        # Invalid metric object (not a proper metric or string)
        invalid_metric = {"this_is": "not a metric"}
        
        with pytest.raises(Exception):
            adapter.train(
                model,
                X,
                y,
                metrics=[invalid_metric],
                epochs=1,
                verbose=0
            )
    
    def test_model_compilation_errors(self, adapter, binary_classification_data):
        """Test handling model compilation errors."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        
        # Create a model with incompatible output shape for the loss function
        # (e.g., using MSE loss with a model that outputs probabilities)
        model = create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=2)
        
        # Explicitly compile with incompatible loss
        with pytest.raises(Exception):
            adapter.train(
                model,
                X,
                y,
                loss="binary_crossentropy",  # Incompatible with 2-class output
                epochs=1,
                verbose=0
            )
    
    def test_predict_without_compile(self, adapter, binary_classification_data):
        """Test prediction with uncompiled model."""
        X, _ = adapter.prepare_data(binary_classification_data, target_column="target")
        
        # Create model but don't compile or train it
        model = create_simple_classifier(input_dim=4, hidden_dim=10, output_dim=1)
        
        # Should raise exception when trying to predict without compilation/training
        with pytest.raises(Exception):
            adapter.predict(model, X)


# Add documentation about the test suite
"""
The TensorFlow adapter test suite provides comprehensive coverage of the adapter's functionality:

1. Core TensorFlow Functionality:
   - Model creation (Sequential, Functional API, Subclassing)
   - Data preparation with tensors and tf.data.Dataset
   - Training with various configurations
   - Model evaluation and metrics
   - Model serialization and loading

2. TensorFlow-Specific Features:
   - GPU support and device management
   - Custom training loops
   - Custom callbacks
   - Custom loss functions
   - Preprocessing layers
   - Mixed precision training

3. Error Handling:
   - Invalid inputs
   - Device errors
   - Shape mismatches
   - Compilation errors
   - Optimizer configuration errors

4. Edge Cases:
   - Single sample training
   - Single class classification
   - NaN data handling
   - Empty batch handling
   - Zero epoch training
"""
