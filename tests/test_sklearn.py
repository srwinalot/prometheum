"""
Tests for the SKLearnAdapter class and scikit-learn integration.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes, make_classification
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

from prometheum.core.base import DataFrameContainer
from prometheum.ml.base import ModelType, FrameworkType
from prometheum.ml.sklearn import SKLearnAdapter


@pytest.fixture
def adapter():
    """Return a fresh SKLearnAdapter instance."""
    return SKLearnAdapter()


@pytest.fixture
def binary_classification_data():
    """Generate a simple binary classification dataset."""
    X, y = make_classification(
        n_samples=100, n_features=4, n_informative=2, n_redundant=0, 
        random_state=42, n_classes=2
    )
    # Convert to dataframe for easier testing
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


@pytest.fixture
def multiclass_data():
    """Load the Iris dataset for multiclass classification."""
    iris = load_iris()
    df = pd.DataFrame(
        data=np.c_[iris.data, iris.target],
        columns=[*iris.feature_names, "target"]
    )
    return df


@pytest.fixture
def regression_data():
    """Load the diabetes dataset for regression testing."""
    diabetes = load_diabetes()
    df = pd.DataFrame(
        data=np.c_[diabetes.data, diabetes.target],
        columns=[*[f"feature_{i}" for i in range(diabetes.data.shape[1])], "target"]
    )
    return df


@pytest.fixture
def single_class_data():
    """Generate a dataset with only one class."""
    # Create a dataset with a single class (all 0s)
    X = np.random.randn(50, 4)
    y = np.zeros(50)  # All samples are class 0
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


@pytest.fixture
def empty_feature_data():
    """Generate a dataset with no features (just target)."""
    # Create a simple dataset with just a target column
    y = np.random.randint(0, 2, 30)
    df = pd.DataFrame({"target": y})
    return df


class TestSKLearnModelCreation:
    """Test scikit-learn model creation functionality."""

    def test_create_classifier(self, adapter):
        """Test creating a classifier model."""
        model = adapter.create_model("LogisticRegression", random_state=42)
        assert model is not None
        assert isinstance(model, LogisticRegression)
    
    def test_create_regressor(self, adapter):
        """Test creating a regression model."""
        model = adapter.create_model("LinearRegression")
        assert model is not None
        assert isinstance(model, LinearRegression)
    
    def test_create_with_module_path(self, adapter):
        """Test creating a model with full module path."""
        model = adapter.create_model("sklearn.ensemble.RandomForestClassifier", n_estimators=10)
        assert model is not None
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 10
    
    def test_invalid_model_name(self, adapter):
        """Test error handling for invalid model name."""
        with pytest.raises(Exception):
            adapter.create_model("NonExistentModel")


class TestSKLearnModelInfo:
    """Test model info extraction."""
    
    def test_get_model_info_classifier(self, adapter):
        """Test getting info for classifier model."""
        model = adapter.create_model("RandomForestClassifier", n_estimators=50)
        info = adapter.get_model_info(model)
        
        assert info.name == "RandomForestClassifier"
        assert info.model_type == ModelType.ENSEMBLE
        assert info.framework == FrameworkType.SKLEARN
        assert "n_estimators" in info.params
        assert info.params["n_estimators"] == 50
    
    def test_get_model_info_regressor(self, adapter):
        """Test getting info for regressor model."""
        model = adapter.create_model("LinearRegression")
        info = adapter.get_model_info(model)
        
        assert info.name == "LinearRegression"
        assert info.model_type == ModelType.REGRESSOR
        assert info.framework == FrameworkType.SKLEARN


class TestSKLearnDataPreparation:
    """Test data preparation functionality."""
    
    def test_prepare_data_with_dataframe(self, adapter, binary_classification_data):
        """Test preparing data from DataFrame."""
        X, y = adapter.prepare_data(
            binary_classification_data, 
            target_column="target"
        )
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == 100
        assert X.shape[1] == 4
        assert y.shape[0] == 100
    
    def test_prepare_data_with_container(self, adapter, regression_data):
        """Test preparing data from DataFrameContainer."""
        container = DataFrameContainer(regression_data, {"source": "test"})
        X, y = adapter.prepare_data(
            container, 
            target_column="target"
        )
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
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


class TestSKLearnTrainingAndPrediction:
    """Test model training and prediction."""
    
    def test_train_binary_classifier(self, adapter, binary_classification_data):
        """Test training a binary classifier."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = adapter.create_model("LogisticRegression", random_state=42)
        
        trained_model = adapter.train(model, X, y)
        assert hasattr(trained_model, "coef_")  # Model has been fitted
        
        y_pred = adapter.predict(trained_model, X)
        assert y_pred.shape == y.shape
        assert np.all(np.isin(y_pred, [0, 1]))  # Binary predictions
    
    def test_train_multiclass_classifier(self, adapter, multiclass_data):
        """Test training a multiclass classifier."""
        X, y = adapter.prepare_data(multiclass_data, target_column="target")
        model = adapter.create_model("RandomForestClassifier", random_state=42)
        
        trained_model = adapter.train(model, X, y)
        assert hasattr(trained_model, "estimators_")  # Model has been fitted
        
        y_pred = adapter.predict(trained_model, X)
        assert y_pred.shape == y.shape
        assert len(np.unique(y_pred)) > 1  # Multiple class predictions
    
    def test_train_regressor(self, adapter, regression_data):
        """Test training a regression model."""
        X, y = adapter.prepare_data(regression_data, target_column="target")
        model = adapter.create_model("RandomForestRegressor", random_state=42)
        
        trained_model = adapter.train(model, X, y)
        assert hasattr(trained_model, "estimators_")  # Model has been fitted
        
        y_pred = adapter.predict(trained_model, X)
        assert y_pred.shape == y.shape
        assert y_pred.dtype.kind == 'f'  # Floating point predictions


class TestSKLearnEvaluation:
    """Test model evaluation functionality."""
    
    def test_binary_classification_metrics(self, adapter, binary_classification_data):
        """Test evaluation metrics for binary classification."""
        # Split data for proper evaluation
        train_df, test_df = train_test_split(
            binary_classification_data, test_size=0.3, random_state=42
        )
        
        X_train, y_train = adapter.prepare_data(train_df, target_column="target")
        X_test, y_test = adapter.prepare_data(test_df, target_column="target")
        
        model = adapter.create_model("LogisticRegression", random_state=42)
        trained_model = adapter.train(model, X_train, y_train)
        
        # Test default metrics
        metrics = adapter.evaluate(trained_model, X_test, y_test)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        
        # Test specific metrics including confusion matrix
        specific_metrics = ["accuracy", "confusion_matrix"]
        metrics = adapter.evaluate(trained_model, X_test, y_test, metrics=specific_metrics)
        
        assert "accuracy" in metrics
        assert "confusion_matrix" in metrics
        assert isinstance(metrics["confusion_matrix"], list)
        
    def test_multiclass_classification_metrics(self, adapter, multiclass_data):
        """Test evaluation metrics for multiclass classification."""
        train_df, test_df = train_test_split(
            multiclass_data, test_size=0.3, random_state=42
        )
        
        X_train, y_train = adapter.prepare_data(train_df, target_column="target")
        X_test, y_test = adapter.prepare_data(test_df, target_column="target")
        
        model = adapter.create_model("RandomForestClassifier", random_state=42)
        trained_model = adapter.train(model, X_train, y_train)
        
        metrics = adapter.evaluate(trained_model, X_test, y_test)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc_error" in metrics  # Should have error for multiclass
    
    def test_regression_metrics(self, adapter, regression_data):
        """Test evaluation metrics for regression."""
        train_df, test_df = train_test_split(
            regression_data, test_size=0.3, random_state=42
        )
        
        X_train, y_train = adapter.prepare_data(train_df, target_column="target")
        X_test, y_test = adapter.prepare_data(test_df, target_column="target")
        
        model = adapter.create_model("LinearRegression")
        trained_model = adapter.train(model, X_train, y_train)
        
        metrics = adapter.evaluate(trained_model, X_test, y_test)
        
        assert "r2" in metrics
        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "explained_variance" in metrics
        assert "max_error" in metrics
        
        # Ensure all metrics are floating point values
        for key, value in metrics.items():
            if not key.endswith("_error"):
                assert isinstance(value, float)
    
    def test_advanced_classification_metrics(self, adapter, binary_classification_data):
        """Test advanced classification metrics."""
        # Split data for proper evaluation
        train_df, test_df = train_test_split(
            binary_classification_data, test_size=0.3, random_state=42
        )
        
        X_train, y_train = adapter.prepare_data(train_df, target_column="target")
        X_test, y_test = adapter.prepare_data(test_df, target_column="target")
        
        model = adapter.create_model("RandomForestClassifier", random_state=42)
        trained_model = adapter.train(model, X_train, y_train)
        
        # Test specific advanced metrics
        metrics = adapter.evaluate(
            trained_model, 
            X_test, 
            y_test, 
            metrics=["balanced_accuracy", "matthews_corrcoef", "jaccard_score"]
        )
        
        # Verify all advanced metrics are present
        assert "balanced_accuracy" in metrics
        assert "matthews_corrcoef" in metrics
        assert "jaccard_score" in metrics
        
        # Verify values are valid
        assert 0 <= metrics["balanced_accuracy"] <= 1
        assert -1 <= metrics["matthews_corrcoef"] <= 1
        assert 0 <= metrics["jaccard_score"] <= 1
    
    def test_advanced_regression_metrics(self, adapter, regression_data):
        """Test advanced regression metrics."""
        # Split data for proper evaluation
        train_df, test_df = train_test_split(
            regression_data, test_size=0.3, random_state=42
        )
        
        X_train, y_train = adapter.prepare_data(train_df, target_column="target")
        X_test, y_test = adapter.prepare_data(test_df, target_column="target")
        
        model = adapter.create_model("LinearRegression")
        trained_model = adapter.train(model, X_train, y_train)
        
        # Test specific advanced metrics
        metrics = adapter.evaluate(
            trained_model, 
            X_test, 
            y_test, 
            metrics=["r2", "adjusted_r2", "median_absolute_error"]
        )
        
        # Verify all advanced metrics are present
        assert "r2" in metrics
        assert "adjusted_r2" in metrics
        assert "median_absolute_error" in metrics
        
        # Verify adjusted_r2 is related to r2
        assert metrics["adjusted_r2"] <= metrics["r2"]
        
        # Verify median_absolute_error is non-negative
        assert metrics["median_absolute_error"] >= 0
    
    def test_custom_metric(self, adapter, binary_classification_data):
        """Test using a custom metric by name."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = adapter.create_model("LogisticRegression", random_state=42)
        trained_model = adapter.train(model, X, y)
        
        # Test with a valid sklearn metric not built into the adapter
        metrics = adapter.evaluate(trained_model, X, y, metrics=["accuracy", "jaccard_score"])
        
        assert "accuracy" in metrics
        assert "jaccard_score" in metrics
        
        # Test with invalid metric
        metrics = adapter.evaluate(trained_model, X, y, metrics=["accuracy", "nonexistent_metric"])
        
        assert "accuracy" in metrics
        assert "nonexistent_metric_error" in metrics


class TestSKLearnSerialization:
    """Test model serialization functionality."""
    
    def test_save_and_load_model(self, adapter, binary_classification_data):
        """Test saving and loading a model."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = adapter.create_model("LogisticRegression", random_state=42)
        trained_model = adapter.train(model, X, y)
        
        # Get predictions from original model
        original_preds = adapter.predict(trained_model, X)
        
        # Save the model to a temporary file
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.pkl")
            adapter.save_model(trained_model, model_path)
            
            # Ensure file exists
            assert os.path.exists(model_path)
            
            # Load the model
            loaded_model = adapter.load_model(model_path)
            
            # Verify loaded model type
            assert isinstance(loaded_model, LogisticRegression)
            
            # Get predictions from loaded model
            loaded_preds = adapter.predict(loaded_model, X)
            
            # Verify predictions match
            np.testing.assert_array_equal(original_preds, loaded_preds)
    
    def test_load_invalid_model(self, adapter):
        """Test error handling when loading an invalid model."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            invalid_path = os.path.join(tmp_dir, "nonexistent.pkl")
            
            # Try to load a non-existent file
            with pytest.raises(Exception):
                adapter.load_model(invalid_path)


class TestSKLearnEdgeCases:
    """Test scikit-learn adapter edge cases."""
    
    def test_single_class_classification(self, adapter, single_class_data):
        """Test handling of single-class classification datasets."""
        X, y = adapter.prepare_data(single_class_data, target_column="target")
        
        # Create and train a classifier on single-class data
        model = adapter.create_model("RandomForestClassifier", random_state=42)
        trained_model = adapter.train(model, X, y)
        
        # Predictions should all be the same class
        y_pred = adapter.predict(trained_model, X)
        assert len(np.unique(y_pred)) == 1
        
        # Metrics should handle single class appropriately
        metrics = adapter.evaluate(trained_model, X, y)
        
        # Accuracy should be 1.0 for perfect prediction of single class
        assert metrics["accuracy"] == 1.0
        
        # Some metrics will have errors due to single class limitations
        assert any("_error" in key for key in metrics.keys())
    
    def test_empty_feature_set(self, adapter, empty_feature_data):
        """Test handling of datasets with no features."""
        # This should raise an exception during preparation
        with pytest.raises(Exception):
            X, y = adapter.prepare_data(empty_feature_data, target_column="target")
    
    def test_missing_features(self, adapter, binary_classification_data):
        """Test handling of missing features during prediction."""
        # Train on full feature set
        X_full, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = adapter.create_model("RandomForestClassifier", random_state=42)
        trained_model = adapter.train(model, X_full, y)
        
        # Create subset with missing features
        subset_df = binary_classification_data.drop(columns=["feature_0"])
        
        # This should raise an exception during prediction
        with pytest.raises(Exception):
            X_subset, _ = adapter.prepare_data(subset_df, target_column="target")
            adapter.predict(trained_model, X_subset)
    
    def test_untrained_model(self, adapter, binary_classification_data):
        """Test handling of untrained models."""
        X, y = adapter.prepare_data(binary_classification_data, target_column="target")
        model = adapter.create_model("RandomForestClassifier", random_state=42)
        
        # Prediction with untrained model should raise exception
        with pytest.raises(NotFittedError):
            adapter.predict(model, X)
        
        # Evaluation with untrained model should raise exception
        with pytest.raises(Exception):
            adapter.evaluate(model, X, y)

