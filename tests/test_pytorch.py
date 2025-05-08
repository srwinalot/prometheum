"""
Tests for the PyTorchAdapter class and PyTorch integration.

This module provides comprehensive tests for the PyTorch adapter implementation,
covering model creation, data preparation, training, evaluation, and serialization.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from prometheum.core.base import DataFrameContainer
from prometheum.ml.base import ModelType, FrameworkType, PredictionError, TrainingError
from prometheum.ml.pytorch import PyTorchAdapter


# Custom Neural Network Models for Testing
class SimpleClassifier(nn.Module):
    """A simple neural network classifier for testing."""
    def __init__(self, input_dim=4, hidden_dim=10, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SimpleRegressor(nn.Module):
    """A simple neural network regressor for testing."""
    def __init__(self, input_dim=4, hidden_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze(-1)  # Remove last dimension for regression target


class AutoEncoder(nn.Module):
    """A simple autoencoder for testing transformation models."""
    def __init__(self, input_dim=4, encoding_dim=2):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


class CustomCallback:
    """A simple callback for testing."""
    def __init__(self):
        self.called = 0
        self.epochs = []
        
    def __call__(self, model=None, epoch=None, train_loss=None, val_loss=None):
        self.called += 1
        self.epochs.append(epoch)
        self.last_train_loss = train_loss
        self.last_val_loss = val_loss


@pytest.fixture
def adapter():
    """Return a fresh PyTorchAdapter instance."""
    return PyTorchAdapter()


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
    return SimpleClassifier(input_dim=4, hidden_dim=10, output_dim=2)


@pytest.fixture
def simple_regressor():
    """Return a simple untrained neural network regressor."""
    return SimpleRegressor(input_dim=4, hidden_dim=10)


@pytest.fixture
def autoencoder():
    """Return a simple untrained autoencoder."""
    return AutoEncoder(input_dim=4, encoding_dim=2)


class TestPyTorchModelCreation:
    """Test PyTorch model creation functionality."""

    def test_create_simple_module(self, adapter):
        """Test creating a simple PyTorch module."""
        model = adapter.create_model("SimpleClassifier", 
                                     model_class=SimpleClassifier, 
                                     input_dim=4, 
                                     hidden_dim=8, 
                                     output_dim=2)
        assert model is not None
        assert isinstance(model, SimpleClassifier)
        assert isinstance(model, nn.Module)
        
        # Validate architecture parameters
        assert model.fc1.in_features == 4
        assert model.fc1.out_features == 8
        assert model.fc2.in_features == 8
        assert model.fc2.out_features == 2
        
    def test_create_with_layers(self, adapter):
        """Test creating a model from layers."""
        layers = [
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        ]
        model = adapter.create_model("Sequential", layers=layers)
        assert model is not None
        assert isinstance(model, nn.Sequential)
        assert len(list(model.children())) == 3
        
        # Validate layer structure
        assert isinstance(model[0], nn.Linear)
        assert isinstance(model[1], nn.ReLU)
        assert isinstance(model[2], nn.Linear)
        assert model[0].in_features == 4
        assert model[0].out_features == 10
        assert model[2].in_features == 10
        assert model[2].out_features == 2
    
    def test_create_from_torch_modules(self, adapter):
        """Test creating standard torch models."""
        model = adapter.create_model("Linear", in_features=4, out_features=2)
        assert model is not None
        assert isinstance(model, nn.Linear)
        assert model.in_features == 4
        assert model.out_features == 2
    
    def test_import_from_module_path(self, adapter):
        """Test creating a model by full module path."""
        model = adapter.create_model("torch.nn.Sequential", 
                                    layers=[nn.Linear(4, 2), nn.ReLU()])
        assert model is not None
        assert isinstance(model, nn.Sequential)
        assert len(list(model.children())) == 2
    
    def test_invalid_model_name(self, adapter):
        """Test error handling for invalid model name."""
        with pytest.raises(Exception):
            adapter.create_model("NonExistentModel")
    
    def test_invalid_layers_argument(self, adapter):
        """Test error handling for invalid layers argument."""
        with pytest.raises(Exception):
            # Not a list of nn.Module objects
            adapter.create_model("Sequential", layers=["not", "modules"])


class TestPyTorchModelInfo:
    """Test model info extraction."""
    
    def test_get_model_info_classifier(self, adapter, simple_classifier):
        """Test getting info for classifier model."""
        info = adapter.get_model_info(simple_classifier)
        
        assert info.name == "SimpleClassifier"
        assert info.model_type in (ModelType.CLASSIFIER, ModelType.NEURAL_NETWORK)
        assert info.framework == FrameworkType.PYTORCH
        assert "architecture" in info.params
        assert info.params["architecture"] == "SimpleClassifier"
        assert "num_parameters" in info.params
        
        # Verify parameter counts
        num_params = sum(p.numel() for p in simple_classifier.parameters())
        assert info.params["num_parameters"] == num_params
        assert info.params["num_trainable_parameters"] == num_params
        
        # Validate module structure is captured
        assert "modules" in info.params
        assert isinstance(info.params["modules"], list)
        assert len(info.params["modules"]) > 0
    
    def test_get_model_info_regressor(self, adapter, simple_regressor):
        """Test getting info for regressor model."""
        info = adapter.get_model_info(simple_regressor)
        
        assert info.name == "SimpleRegressor"
        assert info.model_type in (ModelType.REGRESSOR, ModelType.NEURAL_NETWORK)
        assert info.framework == FrameworkType.PYTORCH
    
    def test_get_model_info_transformer(self, adapter, autoencoder):
        """Test getting info for transformer-type model."""
        info = adapter.get_model_info(autoencoder)
        
        assert info.name == "AutoEncoder"
        assert info.model_type in (ModelType.TRANSFORMER, ModelType.NEURAL_NETWORK)
        assert info.framework == FrameworkType.PYTORCH
    
    def test_get_model_info_with_optimizer(self, adapter, simple_classifier):
        """Test getting info for model with optimizer."""
        simple_classifier.optimizer = optim.Adam(simple_classifier.parameters(), lr=0.01)
        info = adapter.get_model_info(simple_classifier)
        
        assert "optimizer" in info.params
        assert info.params["optimizer"]["name"] == "Adam"
        assert info.params["optimizer"]["lr"] == 0.01
    
    def test_get_model_info_with_loss(self, adapter, simple_classifier):
        """Test getting info for model with loss function."""
        simple_classifier.loss_fn = nn.CrossEntropyLoss()
        info = adapter.get_model_info(simple_classifier)
        
        assert "loss_function" in info.params
        assert info.params["loss_function"] == "CrossEntropyLoss"


class TestPyTorchDataPreparation:
    """Test data preparation functionality."""
    
    def test_prepare_data_with_dataframe(self, adapter, binary_classification_data):
        """Test preparing data from DataFrame."""
        X, y = adapter.prepare_data(
            binary_classification_data, 
            target_column="target"
        )
        
        assert isinstance(X, torch.Tensor)
        assert isinstance(y, torch.Tensor)
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
        
        assert isinstance(X, torch.Tensor)
        assert isinstance(y, torch.Tensor)
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
    
    def test_prepare_data_with_dataloader(self, adapter, binary_classification_data):
        """Test preparing data with DataLoader creation."""
        X, y = adapter.prepare_data(
            binary_classification_data, 
            target_column="target",
            create_dataloader=True,
            batch_size=16
        )
        
        assert hasattr(X, "dataloader")
        assert isinstance(X.dataloader, DataLoader)
        assert hasattr(X, "feature_names")
        assert len(X.feature_names) == 4
        assert y is None  # DataLoader returns None for y
    
    def test_prepare_data_with_custom_tensor_dtype(self, adapter, regression_data):
        """Test preparing data with custom tensor dtype."""
        X, y = adapter.prepare_data(
            regression_data,
            target_column="target",
            tensor_dtype="float64"
        )
        
        assert X.dtype == torch.float64
        assert y.dtype == torch.float64
    
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


class TestPyTorchTrainingAndPrediction:
    """Test model training and prediction."""
    
    def test_train_binary_classifier(self, adapter, binary_classification_data):
        """Test training a binary classifier."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = SimpleClassifier(input_dim=4, hidden_dim=10, output_dim=2)
        
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=5,
            batch_size=16,
            verbose=0
        )
        
        assert hasattr(trained_model, "_is_fitted")
        assert trained_model._is_fitted is True
        
        # Should have training history
        assert hasattr(trained_model, "training_metrics")
        assert "train_loss" in trained_model.training_metrics
        assert len(trained_model.training_metrics["train_loss"]) == 5  # 5 epochs
        
        # Verify model has optimizer and loss function
        assert hasattr(trained_model, "optimizer")
        assert hasattr(trained_model, "loss_fn")
        
        # Test prediction capability
        y_pred = adapter.predict(trained_model, X)
        assert y_pred.shape == (100,)  # 100 samples
        assert np.all(np.isin(y_pred, [0, 1]))  # Binary predictions
    
    def test_train_regressor(self, adapter, regression_data):
        """Test training a regression model."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        model = SimpleRegressor(input_dim=4, hidden_dim=10)
        
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=5,
            batch_size=16,
            loss="mse",
            verbose=0
        )
        
        assert hasattr(trained_model, "_is_fitted")
        assert trained_model._is_fitted is True
        
        # Test prediction capability
        y_pred = adapter.predict(trained_model, X)
        assert y_pred.shape == (100,)  # 100 samples
        assert y_pred.dtype.kind == 'f'  # Floating point predictions
    
    def test_train_with_validation_split(self, adapter, binary_classification_data):
        """Test training with validation split."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = SimpleClassifier(input_dim=4, hidden_dim=10, output_dim=2)
        
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=3,
            validation_split=0.2,
            verbose=0
        )
        
        assert "val_loss" in trained_model.training_metrics
        assert len(trained_model.training_metrics["val_loss"]) == 3  # 3 epochs
    
    def test_train_with_callbacks(self, adapter, binary_classification_data):
        """Test training with callbacks."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = SimpleClassifier(input_dim=4, hidden_dim=10, output_dim=2)
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
        assert callback.epochs == [1, 2, 3]
        assert hasattr(callback, "last_train_loss")
    
    def test_train_with_custom_optimizer(self, adapter, regression_data):
        """Test training with custom optimizer."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        model = SimpleRegressor(input_dim=4, hidden_dim=10)
        
        # Create custom optimizer
        custom_opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=3,
            optimizer=custom_opt,
            verbose=0
        )
        
        # Verify optimizer was used
        assert trained_model.optimizer is custom_opt
        assert isinstance(trained_model.optimizer, optim.SGD)
    
    def test_train_with_custom_loss_function(self, adapter, regression_data):
        """Test training with custom loss function."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        model = SimpleRegressor(input_dim=4, hidden_dim=10)
        
        # Create custom loss function
        def custom_l1_loss(y_pred, y_true):
            return torch.mean(torch.abs(y_pred - y_true))
        
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=3,
            loss=custom_l1_loss,
            verbose=0
        )
        
        # Test prediction capability
        y_pred = adapter.predict(trained_model, X)
        assert y_pred.shape == (100,)
    
    def test_train_with_dataloader(self, adapter, binary_classification_data):
        """Test training with a DataLoader."""
        data_loader, _ = adapter.prepare_data(
            binary_classification_data, 
            target_column="target",
            create_dataloader=True,
            batch_size=16
        )
        
        model = SimpleClassifier(input_dim=4, hidden_dim=10, output_dim=2)
        
        trained_model = adapter.train(
            model, 
            data_loader,
            epochs=3,
            verbose=0
        )
        
        assert hasattr(trained_model, "_is_fitted")
        assert trained_model._is_fitted is True
        
        # Get regular tensors for prediction testing
        X_test, y_test = adapter.prepare_data(binary_classification_data, target_column="target")
        
        # Test prediction capability
        y_pred = adapter.predict(trained_model, X_test)
        assert y_pred.shape == (100,)
    
    def test_untrained_model_prediction(self, adapter, binary_classification_data):
        """Test prediction with untrained model raises error."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = SimpleClassifier(input_dim=4, hidden_dim=10, output_dim=2)
        
        # Trying to predict with untrained model should raise exception
        with pytest.raises(PredictionError):
            adapter.predict(model, X)
    
    def test_gradient_calculation(self, adapter, regression_data):
        """Test gradient calculation during training."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        model = SimpleRegressor(input_dim=4, hidden_dim=10)
        
        # Register hooks to check for gradient calculation
        gradients_seen = {'count': 0}
        def gradient_hook(grad):
            gradients_seen['count'] += 1
        
        # Attach hooks to track gradients
        handles = []
        for param in model.parameters():
            handles.append(param.register_hook(gradient_hook))
        
        # Train model
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=2,
            batch_size=32,
            verbose=0
        )
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Verify gradients were calculated
        assert gradients_seen['count'] > 0


class TestPyTorchEvaluation:
    """Test model evaluation functionality."""
    
    def test_binary_classification_metrics(self, adapter, binary_classification_data):
        """Test evaluation metrics for binary classification."""
        # Split data for proper evaluation
        train_df = binary_classification_data.sample(frac=0.7, random_state=42)
        test_df = binary_classification_data.drop(train_df.index)
        
        X_train, y_train = adapter.prepare_data(train_df, target_column="target")
        X_test, y_test = adapter.prepare_data(test_df, target_column="target")
        
        model = SimpleClassifier(input_dim=4, hidden_dim=10, output_dim=2)
        trained_model = adapter.train(
            model, 
            X_train, 
            y_train,
            epochs=5,
            verbose=0
        )
        
        # Test default metrics
        metrics = adapter.evaluate(trained_model, X_test, y_test)
        
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
        metrics = adapter.evaluate(trained_model, X_test, y_test, metrics=specific_metrics)
        
        assert "accuracy" in metrics
        assert "confusion_matrix" in metrics
        assert isinstance(metrics["confusion_matrix"], np.ndarray) or isinstance(metrics["confusion_matrix"], list)
    
    def test_regression_metrics(self, adapter, regression_data):
        """Test evaluation metrics for regression."""
        # Split data for proper evaluation
        train_df = regression_data.sample(frac=0.7, random_state=42)
        test_df = regression_data.drop(train_df.index)
        
        X_train, y_train = adapter.prepare_data(train_df, target_column="target")
        X_test, y_test = adapter.prepare_data(test_df, target_column="target")
        
        model = SimpleRegressor(input_dim=4, hidden_dim=10)
        trained_model = adapter.train(
            model, 
            X_train, 
            y_train,
            epochs=5,
            verbose=0
        )
        
        # Test default metrics
        metrics = adapter.evaluate(trained_model, X_test, y_test)
        
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
        model = SimpleClassifier(input_dim=4, hidden_dim=10, output_dim=2)
        trained_model = adapter.train(model, X, y, epochs=3, verbose=0)
        
        # Define a custom metric function
        def balanced_accuracy(y_true, y_pred):
            from sklearn.metrics import balanced_accuracy_score
            return balanced_accuracy_score(y_true, y_pred)
        
        # Test with custom metric function
        metrics = adapter.evaluate(
            trained_model, 
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
        model = SimpleClassifier(input_dim=4, hidden_dim=10, output_dim=2)
        trained_model = adapter.train(model, X, y, epochs=3, verbose=0)
        
        # Test with invalid metric
        metrics = adapter.evaluate(
            trained_model, 
            X, 
            y, 
            metrics=["accuracy", "nonexistent_metric"]
        )
        
        assert "accuracy" in metrics
        assert "nonexistent_metric_error" in metrics


class TestPyTorchDeviceManagement:
    """Test device management functionality."""
    
    def test_model_cpu_device(self, adapter, binary_classification_data):
        """Test model training and inference on CPU."""
        X, y = adapter.prepare_data(
            binary_classification_data, 
            target_column="target",
            device="cpu"
        )
        
        model = SimpleClassifier(input_dim=4, hidden_dim=10, output_dim=2)
        
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=2,
            device="cpu",
            verbose=0
        )
        
        # Check model device
        assert next(trained_model.parameters()).device.type == "cpu"
        
        # Predict on same device
        y_pred = adapter.predict(trained_model, X)
        assert y_pred.shape == (100,)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), 
                        reason="CUDA not available for testing")
    def test_model_cuda_device(self, adapter, binary_classification_data):
        """Test model training and inference on CUDA (if available)."""
        X, y = adapter.prepare_data(
            binary_classification_data, 
            target_column="target",
            device="cuda"
        )
        
        model = SimpleClassifier(input_dim=4, hidden_dim=10, output_dim=2)
        
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=2,
            device="cuda",
            verbose=0
        )
        
        # Check model device
        assert next(trained_model.parameters()).device.type == "cuda"
        
        # Predict on same device
        y_pred = adapter.predict(trained_model, X)
        assert y_pred.shape == (100,)
    
    def test_device_mismatch_handling(self, adapter, binary_classification_data):
        """Test handling of device mismatches during prediction."""
        # Train on CPU
        X_cpu, y_cpu = adapter.prepare_data(
            binary_classification_data, 
            target_column="target",
            device="cpu"
        )
        
        model = SimpleClassifier(input_dim=4, hidden_dim=10, output_dim=2)
        trained_model = adapter.train(model, X_cpu, y_cpu, epochs=2, device="cpu", verbose=0)
        
        # Test prediction with input data on CPU
        y_pred_cpu = adapter.predict(trained_model, X_cpu)
        assert y_pred_cpu.shape == (100,)
        
        # Now prepare data without specifying device
        X_default, _ = adapter.prepare_data(binary_classification_data, target_column="target")
        
        # Should handle device differences transparently
        y_pred_default = adapter.predict(trained_model, X_default)
        assert y_pred_default.shape == (100,)
        
        # Results should be the same regardless of initial device
        assert np.allclose(y_pred_cpu.cpu().numpy(), y_pred_default.cpu().numpy())
    
    def test_data_loader_device_management(self, adapter, binary_classification_data):
        """Test device management with DataLoader."""
        X, _ = adapter.prepare_data(
            binary_classification_data, 
            target_column="target",
            create_dataloader=True,
            batch_size=16,
            device="cpu"
        )
        
        model = SimpleClassifier(input_dim=4, hidden_dim=10, output_dim=2)
        
        # Should accept DataLoader and handle device management
        trained_model = adapter.train(model, X, epochs=2, device="cpu", verbose=0)
        assert hasattr(trained_model, "_is_fitted")
        assert trained_model._is_fitted is True


class TestPyTorchSerialization:
    """Test model serialization functionality."""
    
    def test_save_and_load_model(self, adapter, binary_classification_data):
        """Test saving and loading a model."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = SimpleClassifier(input_dim=4, hidden_dim=10, output_dim=2)
        trained_model = adapter.train(model, X, y, epochs=3, verbose=0)
        
        # Get predictions from original model
        original_preds = adapter.predict(trained_model, X)
        
        # Save the model to a temporary file
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.pt")
            adapter.save_model(trained_model, model_path)
            
            # Ensure file exists
            assert os.path.exists(model_path)
            
            # Load the model
            loaded_model = adapter.load_model(model_path)
            
            # Verify loaded model type
            assert isinstance(loaded_model, SimpleClassifier)
            
            # Verify model has required attributes
            assert hasattr(loaded_model, "_is_fitted")
            assert hasattr(loaded_model, "loss_fn")
            assert hasattr(loaded_model, "optimizer")
            
            # Get predictions from loaded model
            loaded_preds = adapter.predict(loaded_model, X)
            
            # Verify predictions match
            np.testing.assert_allclose(
                original_preds.cpu().numpy(), 
                loaded_preds.cpu().numpy(),
                rtol=1e-5
            )
    
    def test_save_and_load_with_custom_objects(self, adapter, regression_data):
        """Test saving and loading models with custom components."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        
        # Custom model with custom attributes
        class CustomRegressor(SimpleRegressor):
            def __init__(self, input_dim=4, hidden_dim=10, custom_param="test"):
                super().__init__(input_dim, hidden_dim)
                self.custom_param = custom_param
        
        model = CustomRegressor(input_dim=4, hidden_dim=10, custom_param="custom_value")
        trained_model = adapter.train(model, X, y, epochs=3, verbose=0)
        
        # Verify custom attribute
        assert hasattr(trained_model, "custom_param")
        assert trained_model.custom_param == "custom_value"
        
        # Save and load model
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "custom_model.pt")
            adapter.save_model(trained_model, model_path)
            
            loaded_model = adapter.load_model(model_path)
            
            # Verify custom attribute was preserved
            assert hasattr(loaded_model, "custom_param")
            assert loaded_model.custom_param == "custom_value"
            
            # Test prediction capability
            y_pred = adapter.predict(loaded_model, X)
            assert y_pred.shape == (100,)
    
    def test_save_model_with_training_history(self, adapter, binary_classification_data):
        """Test that training history is preserved during serialization."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = SimpleClassifier(input_dim=4, hidden_dim=10, output_dim=2)
        
        trained_model = adapter.train(
            model, 
            X, 
            y,
            epochs=5,
            validation_split=0.2,
            verbose=0
        )
        
        # Verify training metrics exist
        assert hasattr(trained_model, "training_metrics")
        assert "train_loss" in trained_model.training_metrics
        assert "val_loss" in trained_model.training_metrics
        
        # Save and load model
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "history_model.pt")
            adapter.save_model(trained_model, model_path)
            
            loaded_model = adapter.load_model(model_path)
            
            # Verify training metrics were preserved
            assert hasattr(loaded_model, "training_metrics")
            assert "train_loss" in loaded_model.training_metrics
            assert "val_loss" in loaded_model.training_metrics
            assert len(loaded_model.training_metrics["train_loss"]) == 5
            assert len(loaded_model.training_metrics["val_loss"]) == 5
    
    def test_load_nonexistent_model(self, adapter):
        """Test error handling when loading from a non-existent path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nonexistent_path = os.path.join(tmp_dir, "nonexistent_model.pt")
            
            with pytest.raises(Exception):
                adapter.load_model(nonexistent_path)
    
    def test_load_corrupted_model(self, adapter):
        """Test error handling when loading a corrupted model file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            corrupt_path = os.path.join(tmp_dir, "corrupt_model.pt")
            
            # Create a corrupted model file
            with open(corrupt_path, 'wb') as f:
                f.write(b'not a valid pytorch model')
            
            with pytest.raises(Exception):
                adapter.load_model(corrupt_path)


class TestPyTorchEdgeCases:
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
        model = SimpleRegressor(input_dim=4, hidden_dim=10)
        
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
        model = SimpleClassifier(input_dim=4, hidden_dim=10, output_dim=2)
        trained_model = adapter.train(model, X, y, epochs=3, verbose=0)
        
        # Predictions should all be the same class
        y_pred = adapter.predict(trained_model, X)
        assert len(np.unique(y_pred.cpu().numpy())) == 1
        
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
        
        # This should raise an exception during data preparation
        with pytest.raises(Exception):
            X_tensor, y_tensor = adapter.prepare_data(df, target_column="target")
        
        # Create dataset with NaN values in target
        X_clean = np.random.randn(50, 4)
        y_with_nan = np.random.randn(50)
        y_with_nan[np.random.randint(0, 50, 3)] = np.nan
        
        df_clean_X_nan_y = pd.DataFrame(X_clean, columns=[f"feature_{i}" for i in range(X_clean.shape[1])])
        df_clean_X_nan_y["target"] = y_with_nan
        
        # This should also raise an exception
        with pytest.raises(Exception):
            X_tensor, y_tensor = adapter.prepare_data(df_clean_X_nan_y, target_column="target")
    
    def test_empty_batch_handling(self, adapter, binary_classification_data):
        """Test handling of potential empty batches."""
        # Create a very small dataset
        small_df = binary_classification_data.iloc[:3]
        X, y = adapter.prepare_data(small_df, target_column="target")
        
        model = SimpleClassifier(input_dim=4, hidden_dim=10, output_dim=2)
        
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
        assert hasattr(trained_model, "_is_fitted")
        assert trained_model._is_fitted is True
        
        # Test prediction
        y_pred = adapter.predict(trained_model, X)
        assert y_pred.shape == (3,)
    
    def test_zero_epoch_training(self, adapter, binary_classification_data):
        """Test training with zero epochs."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = SimpleClassifier(input_dim=4, hidden_dim=10, output_dim=2)
        
        # Train with zero epochs (should do nothing but not error)
        with pytest.warns(UserWarning, match="No training will be performed"):
            trained_model = adapter.train(
                model,
                X,
                y,
                epochs=0,
                verbose=0
            )
        
        # Should be marked as not fitted
        assert not hasattr(trained_model, "_is_fitted") or not trained_model._is_fitted
        
        # Prediction should raise an error
        with pytest.raises(Exception):
            adapter.predict(trained_model, X)
    
    def test_model_with_custom_predict_method(self, adapter, regression_data):
        """Test model with custom predict method."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        
        # Create a model with custom predict method
        class ModelWithCustomPredict(SimpleRegressor):
            def predict(self, x):
                # Just a constant prediction for testing
                return torch.ones(x.shape[0])
        
        model = ModelWithCustomPredict(input_dim=4, hidden_dim=10)
        trained_model = adapter.train(model, X, y, epochs=2, verbose=0)
        
        # The adapter should use the model's custom predict method
        y_pred = adapter.predict(trained_model, X)
        
        # All predictions should be 1.0
        assert torch.allclose(y_pred, torch.ones(X.shape[0]))
    
    def test_training_with_uninitialized_weights(self, adapter, regression_data):
        """Test training a model with uninitialized weights."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        
        # Create a model with explicitly uninitialized weights
        class UninitializedModel(nn.Module):
            def __init__(self, input_dim=4, hidden_dim=10):
                super().__init__()
                self.linear1 = nn.Linear(input_dim, hidden_dim)
                self.linear2 = nn.Linear(hidden_dim, 1)
                # Don't initialize weights
                
            def forward(self, x):
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.linear2(x)
                return x.squeeze(-1)
        
        model = UninitializedModel(input_dim=4, hidden_dim=10)
        
        # Should train without issues
        trained_model = adapter.train(model, X, y, epochs=3, verbose=0)
        assert hasattr(trained_model, "_is_fitted")
        assert trained_model._is_fitted is True
        
        # Predict
        y_pred = adapter.predict(trained_model, X)
        assert y_pred.shape == (100,)


class TestPyTorchErrorConditions:
    """Test various error conditions and exception handling."""
    
    def test_incompatible_input_shape(self, adapter, binary_classification_data):
        """Test handling incompatible input shapes during prediction."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = SimpleClassifier(input_dim=4, hidden_dim=10, output_dim=2)
        trained_model = adapter.train(model, X, y, epochs=2, verbose=0)
        
        # Create data with wrong input shape
        wrong_X = torch.randn(10, 5)  # 5 features instead of 4
        
        # Should raise an exception
        with pytest.raises(Exception):
            adapter.predict(trained_model, wrong_X)
    
    def test_invalid_optimizer_params(self, adapter, regression_data):
        """Test handling invalid optimizer parameters."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        model = SimpleRegressor(input_dim=4, hidden_dim=10)
        
        # Invalid learning rate
        with pytest.raises(Exception):
            adapter.train(
                model,
                X,
                y,
                optimizer="adam",
                optimizer_params={"lr": -0.1},  # Negative learning rate
                epochs=2,
                verbose=0
            )
        
        # Invalid optimizer name
        with pytest.raises(Exception):
            adapter.train(
                model,
                X,
                y,
                optimizer="nonexistent_optimizer",
                epochs=2,
                verbose=0
            )
    
    def test_invalid_loss_function(self, adapter, regression_data):
        """Test handling invalid loss function."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        model = SimpleRegressor(input_dim=4, hidden_dim=10)
        
        # Invalid loss function name
        with pytest.raises(Exception):
            adapter.train(
                model,
                X,
                y,
                loss="nonexistent_loss",
                epochs=2,
                verbose=0
            )
    
    def test_invalid_device(self, adapter, binary_classification_data):
        """Test handling invalid device specification."""
        # Should raise an exception for invalid device
        with pytest.raises(Exception):
            X, y = adapter.prepare_data(
                binary_classification_data, 
                target_column="target",
                device="invalid_device"
            )
        
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = SimpleClassifier(input_dim=4, hidden_dim=10, output_dim=2)
        
        # Invalid device during training
        with pytest.raises(Exception):
            adapter.train(
                model,
                X,
                y,
                device="invalid_device",
                epochs=2,
                verbose=0
            )
    
    def test_corrupted_model_state(self, adapter, binary_classification_data):
        """Test handling corrupted model state."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = SimpleClassifier(input_dim=4, hidden_dim=10, output_dim=2)
        trained_model = adapter.train(model, X, y, epochs=2, verbose=0)
        
        # Corrupt the model by removing some parameters
        for name, param in list(trained_model.named_parameters())[:1]:
            delattr(trained_model, name.split('.')[0])
        
        # Should raise an exception during prediction
        with pytest.raises(Exception):
            adapter.predict(trained_model, X)
    
    def test_model_not_module(self, adapter, binary_classification_data):
        """Test handling when model is not a nn.Module."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        
        # Create something that's not a PyTorch model
        not_a_model = {"This is": "not a model"}
        
        # Should raise an exception during training
        with pytest.raises(Exception):
            adapter.train(not_a_model, X, y, epochs=2, verbose=0)
        
        # Should raise an exception during get_model_info
        with pytest.raises(Exception):
            adapter.get_model_info(not_a_model)
        
        # Should raise an exception during prediction
        with pytest.raises(Exception):
            adapter.predict(not_a_model, X)


# Only run this test class if CUDA is available
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for testing")
class TestPyTorchGPUSupport:
    """Test CUDA-specific functionality."""
    
    def test_gpu_device_allocation(self, adapter, regression_data):
        """Test explicit GPU device allocation."""
        X, y = adapter.prepare_data(
            regression_data, 
            target_column="target",
            device="cuda:0"  # Explicitly use first GPU
        )
        
        assert X.device.type == "cuda"
        assert X.device.index == 0
        assert y.device.type == "cuda"
        assert y.device.index == 0
    
    def test_multi_gpu_model(self, adapter, regression_data):
        """Test multi-GPU model if multiple GPUs are available."""
        # Skip if multiple GPUs aren't available
        if torch.cuda.device_count() < 2:
            pytest.skip("Multiple GPUs not available")
        
        X, y = adapter.prepare_data(regression_data, target_column="target")
        
        # Create a model suitable for DataParallel
        model = SimpleRegressor(input_dim=4, hidden_dim=20)
        model = nn.DataParallel(model)
        
        # Train the model
        trained_model = adapter.train(
            model,
            X,
            y,
            epochs=2,
            device="cuda",
            verbose=0
        )
        
        # Verify it's still a DataParallel model
        assert isinstance(trained_model, nn.DataParallel)
        
        
