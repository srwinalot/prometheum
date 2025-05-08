"""
Integration tests for interoperability between ML frameworks.

This module tests the integration between different ML frameworks (PyTorch, TensorFlow)
and the Prometheum data processing pipeline, including model conversion, shared data
handling, and mixed framework pipelines.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd

# Skip tests if ML libraries are not available
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Core components
from prometheum.core.base import DataFrameContainer, Pipeline, DataTransformer
from prometheum.core.exceptions import ProcessingError

# ML adapters
from prometheum.ml.base import ModelType, FrameworkType
from prometheum.ml.pytorch import PyTorchAdapter
from prometheum.ml.tensorflow import TensorFlowAdapter

# Model conversion utilities
from prometheum.ml.converters import ModelConverter


# Skip all tests if either framework is missing
pytestmark = pytest.mark.skipif(
    not (PYTORCH_AVAILABLE and TENSORFLOW_AVAILABLE),
    reason="PyTorch and TensorFlow are required for framework interoperability tests"
)


class SimpleDataPreprocessor(DataTransformer):
    """Simple transformer that standardizes numeric data."""
    
    def __init__(self, **kwargs):
        """Initialize the preprocessor."""
        super().__init__(kwargs)
        self.means = None
        self.stds = None
        self.fitted = False
    
    def fit(self, data):
        """Compute means and standard deviations."""
        if isinstance(data, DataFrameContainer):
            df = data.data
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ProcessingError("Expected DataFrame or DataFrameContainer")
        
        # Compute statistics for numeric columns
        numerics = df.select_dtypes(include=[np.number])
        self.means = numerics.mean()
        self.stds = numerics.std().replace(0, 1)  # Avoid division by zero
        self.fitted = True
    
    def transform(self, data):
        """Standardize data using computed statistics."""
        if not self.fitted:
            raise ProcessingError("Preprocessor not fitted")
        
        if isinstance(data, DataFrameContainer):
            df = data.data.copy()
            
            # Standardize numeric columns
            for col in self.means.index:
                if col in df.columns:
                    df[col] = (df[col] - self.means[col]) / self.stds[col]
            
            # Return a new container with the same metadata
            result = DataFrameContainer(df, data.metadata.copy())
            result.add_metadata("preprocessed", True)
            return result
        
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            
            # Standardize numeric columns
            for col in self.means.index:
                if col in df.columns:
                    df[col] = (df[col] - self.means[col]) / self.stds[col]
            
            return df
        
        else:
            raise ProcessingError("Expected DataFrame or DataFrameContainer")


@pytest.fixture
def sample_data():
    """Create sample data suitable for both classification and regression."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    # Binary target for classification
    y_binary = (X[:, 0] + X[:, 1] > 0).astype(float)
    # Continuous target for regression
    y_continuous = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + 0.5*X[:, 3] + np.random.randn(100)*0.1
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target_binary"] = y_binary
    df["target_continuous"] = y_continuous
    
    return df


@pytest.fixture
def data_container(sample_data):
    """Create a DataFrameContainer with metadata."""
    return DataFrameContainer(sample_data, {
        "source": "test",
        "description": "Sample data for ML framework interoperability tests",
        "version": "1.0.0"
    })


@pytest.fixture
def pytorch_adapter():
    """Create a PyTorch adapter instance."""
    return PyTorchAdapter()


@pytest.fixture
def tensorflow_adapter():
    """Create a TensorFlow adapter instance."""
    return TensorFlowAdapter()


@pytest.fixture
def preprocessor():
    """Create a fitted data preprocessor."""
    return SimpleDataPreprocessor()


@pytest.fixture
def model_converter():
    """Create a model converter instance."""
    return ModelConverter()


@pytest.fixture
def simple_pytorch_classifier():
    """Create a simple PyTorch classifier."""
    class SimpleClassifier(nn.Module):
        def __init__(self, input_dim=4, hidden_dim=10, output_dim=1):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            return x
    
    return SimpleClassifier()


@pytest.fixture
def simple_tensorflow_classifier():
    """Create a simple TensorFlow classifier."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


class TestModelConversion:
    """Test converting models between different ML frameworks."""
    
    def test_pytorch_to_tensorflow_conversion(self, pytorch_adapter, tensorflow_adapter, 
                                             model_converter, sample_data, simple_pytorch_classifier):
        """Test converting a PyTorch model to TensorFlow."""
        # Prepare data
        X, y = pytorch_adapter.prepare_data(sample_data, target_column="target_binary")
        
        # Train PyTorch model
        pytorch_model = pytorch_adapter.train(simple_pytorch_classifier, X, y, epochs=3)
        
        # Get PyTorch predictions
        pytorch_predictions = pytorch_adapter.predict(pytorch_model, X)
        
        # Convert model to TensorFlow
        tensorflow_model = model_converter.pytorch_to_tensorflow(pytorch_model)
        
        # Ensure the converted model has the correct structure
        assert isinstance(tensorflow_model, tf.keras.Model)
        assert len(tensorflow_model.layers) >= 2  # At least input and output layers
        
        # Prepare TensorFlow data
        X_tf, y_tf = tensorflow_adapter.prepare_data(sample_data, target_column="target_binary")
        
        # Get TensorFlow predictions
        tensorflow_predictions = tensorflow_adapter.predict(tensorflow_model, X_tf)
        
        # Compare predictions (should be close but not identical)
        np_pytorch_predictions = pytorch_predictions.detach().cpu().numpy().flatten()
        np_tensorflow_predictions = tensorflow_predictions.flatten()
        
        # Basic shape validation
        assert np_pytorch_predictions.shape == np_tensorflow_predictions.shape
        
        # Check correlation between predictions (should be high for equivalent models)
        correlation = np.corrcoef(np_pytorch_predictions, np_tensorflow_predictions)[0, 1]
        assert correlation > 0.8, "Predictions between converted models should be highly correlated"
    
    def test_tensorflow_to_pytorch_conversion(self, pytorch_adapter, tensorflow_adapter,
                                            model_converter, sample_data, simple_tensorflow_classifier):
        """Test converting a TensorFlow model to PyTorch."""
        # Prepare data
        X, y = tensorflow_adapter.prepare_data(sample_data, target_column="target_binary")
        
        # Train TensorFlow model
        tensorflow_model = tensorflow_adapter.train(simple_tensorflow_classifier, X, y, epochs=3)
        
        # Get TensorFlow predictions
        tensorflow_predictions = tensorflow_adapter.predict(tensorflow_model, X)
        
        # Convert model to PyTorch
        pytorch_model = model_converter.tensorflow_to_pytorch(tensorflow_model)
        
        # Ensure the converted model has the correct structure
        assert isinstance(pytorch_model, nn.Module)
        
        # Prepare PyTorch data
        X_pt, y_pt = pytorch_adapter.prepare_data(sample_data, target_column="target_binary")
        
        # Get PyTorch predictions
        pytorch_predictions = pytorch_adapter.predict(pytorch_model, X_pt)
        
        # Compare predictions
        np_tensorflow_predictions = tensorflow_predictions.flatten()
        np_pytorch_predictions = pytorch_predictions.detach().cpu().numpy().flatten()
        
        # Basic shape validation
        assert np_pytorch_predictions.shape == np_tensorflow_predictions.shape
        
        # Check correlation between predictions
        correlation = np.corrcoef(np_pytorch_predictions, np_tensorflow_predictions)[0, 1]
        assert correlation > 0.8, "Predictions between converted models should be highly correlated"
    
    def test_convert_with_weights_preservation(self, pytorch_adapter, tensorflow_adapter,
                                             model_converter, sample_data, simple_pytorch_classifier):
        """Test weight preservation during model conversion."""
        # Prepare data
        X, y = pytorch_adapter.prepare_data(sample_data, target_column="target_binary")
        
        # Train PyTorch model
        pytorch_model = pytorch_adapter.train(simple_pytorch_classifier, X, y, epochs=3)
        
        # Extract some key weights from PyTorch model
        pytorch_weight_sum = sum(p.sum().item() for p in pytorch_model.parameters())
        
        # Convert to TensorFlow
        tensorflow_model = model_converter.pytorch_to_tensorflow(pytorch_model)
        
        # Extract weights from TensorFlow model
        tensorflow_weight_sum = sum(np.sum(w) for w in tensorflow_model.get_weights())
        
        # Compare weight sums - should be close (not exact due to format differences)
        assert abs(pytorch_weight_sum - tensorflow_weight_sum) < 1e-2, \
            "Weight values should be preserved during conversion"


class TestSharedDataHandling:
    """Test handling the same datasets across different ML frameworks."""
    
    def test_shared_dataset_preparation(self, pytorch_adapter, tensorflow_adapter, data_container):
        """Test preparing the same dataset for different frameworks."""
        # Prepare data for PyTorch
        X_pt, y_pt = pytorch_adapter.prepare_data(
            data_container,
            target_column="target_binary"
        )
        
        # Prepare data for TensorFlow
        X_tf, y_tf = tensorflow_adapter.prepare_data(
            data_container,
            target_column="target_binary"
        )
        
        # Verify PyTorch data format
        assert isinstance(X_pt, torch.Tensor)
        assert isinstance(y_pt, torch.Tensor)
        assert X_pt.shape[0] == 100
        assert X_pt.shape[1] == 4
        
        # Verify TensorFlow data format
        assert isinstance(X_tf, np.ndarray) or isinstance(X_tf, tf.Tensor)
        assert isinstance(y_tf, np.ndarray) or isinstance(y_tf, tf.Tensor)
        assert X_tf.shape[0] == 100
        assert X_tf.shape[1] == 4
        
        # Ensure the actual data values are the same (within format differences)
        pt_numpy = X_pt.detach().cpu().numpy() if isinstance(X_pt, torch.Tensor) else X_pt
        tf_numpy = X_tf.numpy() if isinstance(X_tf, tf.Tensor) else X_tf
        
        np.testing.assert_allclose(pt_numpy, tf_numpy, rtol=1e-5)
    
    def test_shared_dataset_with_preprocessing(self, pytorch_adapter, tensorflow_adapter, 
                                            data_container, preprocessor):
        """Test preprocessing a dataset and using it with different frameworks."""
        # Fit the preprocessor
        preprocessor.fit(data_container)
        
        # Transform the data
        preprocessed_data = preprocessor.transform(data_container)
        
        # Verify preprocessing was applied
        assert "preprocessed" in preprocessed_data.metadata
        assert preprocessed_data.metadata["preprocessed"] is True
        
        # Prepare data for PyTorch
        X_pt, y_pt = pytorch_adapter.prepare_data(
            preprocessed_data,
            target_column="target_binary"
        )
        
        # Prepare data for TensorFlow
        X_tf, y_tf = tensorflow_adapter.prepare_data(
            preprocessed_data,
            target_column="target_binary"
        )
        
        # Train a model with each framework
        pytorch_model = pytorch_adapter.create_model("SimpleClassifier", 
                                                  model_class=simple_pytorch_classifier,
                                                  input_dim=4, 
                                                  hidden_dim=10, 
                                                  output_dim=1)
        
        tensorflow_model = tensorflow_adapter.create_model("Sequential", 
                                                        layers=[
                                                            tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
                                                            tf.keras.layers.Dense(1, activation='sigmoid')
                                                        ])
        
        # Train both models
        trained_pt_model = pytorch_adapter.train(pytorch_model, X_pt, y_pt, epochs=3)
        trained_tf_model = tensorflow_adapter.train(tensorflow_model, X_tf, y_tf, epochs=3)
        
        # Both models should train successfully on the preprocessed data
        assert hasattr(trained_pt_model, '_is_fitted')
        assert hasattr(trained_tf_model, 'history')
    
    def test_cross_framework_evaluation(self, pytorch_adapter, tensorflow_adapter, 
                                     sample_data, simple_pytorch_classifier, simple_tensorflow_classifier):
        """Test evaluating models with datasets prepared for different frameworks."""
        # Prepare the same dataset for both frameworks
        X_pt, y_pt = pytorch_adapter.prepare_data(sample_data, target_column="target_binary")
        X_tf, y_tf = tensorflow_adapter.prepare_data(sample_data, target_column="target_binary")
        
        # Train both models
        trained_pt_model = pytorch_adapter.train(simple_pytorch_classifier, X_pt, y_pt, epochs=3)
        trained_tf_model = tensorflow_adapter.train(simple_tensorflow_classifier, X_tf, y_tf, epochs=3)
        
        # Evaluate PyTorch model with both frameworks' evaluation
        pt_metrics_from_pt = pytorch_adapter.evaluate(trained_pt_model, X_pt, y_pt)
        
        # Convert PyTorch data to format TensorFlow can use (numpy)
        pt_data_as_np = X_pt.detach().cpu().numpy()
        pt_target_as_np = y_pt.detach().cpu().numpy()
        
        # Evaluate PyTorch model with TensorFlow's evaluation metrics
        # This requires a model converter or adapter to get predictions in TensorFlow format
        pt_converted_model = model_converter.pytorch_to_tensorflow(trained_pt_model)
        pt_metrics_from_tf = tensorflow_adapter.evaluate(pt_converted_model, pt_data_as_np, pt_target_as_np)
        
        # Similarly, evaluate TensorFlow model with both frameworks
        tf_metrics_from_tf = tensorflow_adapter.evaluate(trained_tf_model, X_tf, y_tf)
        
        # Convert TensorFlow model to PyTorch
        tf_converted_model = model_converter.tensorflow_to_pytorch(trained_tf_model)
        tf_metrics_from_pt = pytorch_adapter.evaluate(tf_converted_model, X_pt, y_pt)
        
        # Verify same metrics exist across frameworks
        assert "accuracy" in pt_metrics_from_pt
        assert "accuracy" in pt_metrics_from_tf
        assert "accuracy" in tf_metrics_from_tf
        assert "accuracy" in tf_metrics_from_pt
        
        # Verify metric values are reasonably close
        # Metrics might not be identical due to implementation differences
        assert abs(pt_metrics_from_pt["accuracy"] - pt_metrics_from_tf["accuracy"]) < 0.1
        assert abs(tf_metrics_from_tf["accuracy"] - tf_metrics_from_pt["accuracy"]) < 0.1


class PyTorchModelProcessor(DataTransformer):
    """Data transformer that uses a PyTorch model for predictions."""
    
    def __init__(self, model=None, adapter=None, **kwargs):
        """Initialize with a PyTorch model and adapter."""
        super().__init__(kwargs)
        self.model = model
        self.adapter = adapter or PyTorchAdapter()
        self.fitted = model is not None
    
    def fit(self, data):
        """Fit or further train the PyTorch model."""
        if not self.model:
            # Create a default model if none was provided
            self.model = nn.Sequential(
                nn.Linear(4, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
                nn.Sigmoid()
            )
        
        if isinstance(data, DataFrameContainer):
            X, y = self.adapter.prepare_data(data.data, target_column="target_binary")
        else:
            X, y = self.adapter.prepare_data(data, target_column="target_binary")
        
        self.model = self.adapter.train(self.model, X, y, epochs=3)
        self.fitted = True
    
    def transform(self, data):
        """Transform data using PyTorch model predictions."""
        if not self.fitted:
            raise ProcessingError("Model not fitted")
        
        if isinstance(data, DataFrameContainer):
            df = data.data.copy()
            X, _ = self.adapter.prepare_data(df.drop(columns=["target_binary", "target_continuous"], errors='ignore'))
            
            # Get predictions
            predictions = self.adapter.predict(self.model, X)
            
            # Add predictions as a new column
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.detach().cpu().numpy()
            
            df["pytorch_predictions"] = predictions.flatten()
            
            # Return a new container with the same metadata
            result = DataFrameContainer(df, data.metadata.copy())
            result.add_metadata("pytorch_processed", True)
            return result
        
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            X, _ = self.adapter.prepare_data(df.drop(columns=["target_binary", "target_continuous"], errors='ignore'))
            
            # Get predictions
            predictions = self.adapter.predict(self.model, X)
            
            # Add predictions as a new column
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.detach().cpu().numpy()
                
            df["pytorch_predictions"] = predictions.flatten()
            return df
        
        else:
            raise ProcessingError("Expected DataFrame or DataFrameContainer")


class TensorFlowModelProcessor(DataTransformer):
    """Data transformer that uses a TensorFlow model for predictions."""
    
    def __init__(self, model=None, adapter=None, **kwargs):
        """Initialize with a TensorFlow model and adapter."""
        super().__init__(kwargs)
        self.model = model
        self.adapter = adapter or TensorFlowAdapter()
        self.fitted = model is not None and hasattr(model, 'predict')
    
    def fit(self, data):
        """Fit or further train the TensorFlow model."""
        if not self.model:
            # Create a default model if none was provided
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        
        if isinstance(data, DataFrameContainer):
            X, y = self.adapter.prepare_data(data.data, target_column="target_binary")
        else:
            X, y = self.adapter.prepare_data(data, target_column="target_binary")
        
        self.model = self.adapter.train(self.model, X, y, epochs=3)
        self.fitted = True
    
    def transform(self, data):
        """Transform data using TensorFlow model predictions."""
        if not self.fitted:
            raise ProcessingError("Model not fitted")
        
        if isinstance(data, DataFrameContainer):
            df = data.data.copy()
            X, _ = self.adapter.prepare_data(df.drop(columns=["target_binary", "target_continuous"], errors='ignore'))
            
            # Get predictions
            predictions = self.adapter.predict(self.model, X)
            
            # Add predictions as a new column
            df["tensorflow_predictions"] = predictions.flatten()
            
            # Return a new container with the same metadata
            result = DataFrameContainer(df, data.metadata.copy())
            result.add_metadata("tensorflow_processed", True)
            return result
        
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            X, _ = self.adapter.prepare_data(df.drop(columns=["target_binary", "target_continuous"], errors='ignore'))
            
            # Get predictions
            predictions = self.adapter.predict(self.model, X)
            
            # Add predictions as a new column
            df["tensorflow_predictions"] = predictions.flatten()
            return df
        
        else:
            raise ProcessingError("Expected DataFrame or DataFrameContainer")


class TestMixedFrameworkPipeline:
    """Test pipelines that combine multiple ML frameworks."""
    
    def test_pytorch_tensorflow_pipeline(self, pytorch_adapter, tensorflow_adapter, 
                                         data_container, preprocessor):
        """Test a pipeline that combines PyTorch and TensorFlow models."""
        # Create model processors
        pytorch_processor = PyTorchModelProcessor(adapter=pytorch_adapter)
        tensorflow_processor = TensorFlowModelProcessor(adapter=tensorflow_adapter)
        
        # Create a pipeline with both frameworks
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("pytorch", pytorch_processor),
            ("tensorflow", tensorflow_processor)
        ])
        
        # Fit the pipeline components
        preprocessor.fit(data_container)
        pytorch_processor.fit(data_container)
        tensorflow_processor.fit(data_container)
        
        # Process the data through the pipeline
        result = pipeline.process(data_container)
        
        # Verify result has predictions from both frameworks
        assert "pytorch_predictions" in result.data.columns
        assert "tensorflow_predictions" in result.data.columns
        
        # Verify metadata contains processing flags
        assert "preprocessed" in result.metadata
        assert "pytorch_processed" in result.metadata
        assert "tensorflow_processed" in result.metadata
        
        # Verify original metadata is preserved
        assert result.metadata["source"] == "test"
        assert result.metadata["version"] == "1.0.0"
        
        # Verify predictions are correlated (both models should have learned similar patterns)
        pt_preds = result.data["pytorch_predictions"]
        tf_preds = result.data["tensorflow_predictions"]
        correlation = np.corrcoef(pt_preds, tf_preds)[0, 1]
        assert correlation > 0.5, "Predictions should be correlated"
    
    def test_framework_specific_pipelines(self, pytorch_adapter, tensorflow_adapter, 
                                         sample_data, preprocessor):
        """Test framework-specific pipelines with shared preprocessing."""
        # Fit the preprocessor
        preprocessor.fit(sample_data)
        
        # Create framework-specific pipelines with shared preprocessing
        pytorch_pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("pytorch", PyTorchModelProcessor(adapter=pytorch_adapter))
        ])
        
        tensorflow_pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("tensorflow", TensorFlowModelProcessor(adapter=tensorflow_adapter))
        ])
        
        # Process the same data through both pipelines
        pytorch_result = pytorch_pipeline.process(sample_data)
        tensorflow_result = tensorflow_pipeline.process(sample_data)
        
        # Verify each pipeline added its framework-specific predictions
        assert "pytorch_predictions" in pytorch_result.columns
        assert "tensorflow_predictions" in tensorflow_result.columns
        
        # Both should be binary classification predictions between 0 and 1
        assert all(0 <= pred <= 1 for pred in pytorch_result["pytorch_predictions"])
        assert all(0 <= pred <= 1 for pred in tensorflow_result["tensorflow_predictions"])
    
    def test_error_handling_across_frameworks(self, pytorch_adapter, tensorflow_adapter,
                                            sample_data):
        """Test error handling and propagation across frameworks."""
        # Create a processor that will raise an error
        class ErrorProcessor(DataTransformer):
            def fit(self, data):
                pass
                
            def transform(self, data):
                raise ProcessingError("Intentional error")
        
        # Create pipeline with multiple frameworks and an error processor
        pipeline = Pipeline([
            ("pytorch", PyTorchModelProcessor(adapter=pytorch_adapter)),
            ("error", ErrorProcessor()),
            ("tensorflow", TensorFlowModelProcessor(adapter=tensorflow_adapter))
        ])
        
        # Fit the PyTorch processor
        pipeline.processors[0].fit(sample_data)
        
        # The pipeline should propagate the error from the error processor
        with pytest.raises(Exception) as exc_info:
            pipeline.process(sample_data)
        
        # Verify error details are preserved
        assert "Intentional error" in str(exc_info.value)


class TestEndToEndWorkflow:
    """Test complete end-to-end ML workflows."""
    
    def test_train_convert_evaluate_workflow(self, pytorch_adapter, tensorflow_adapter,
                                           model_converter, sample_data):
        """Test a complete workflow: train, convert, evaluate, save."""
        # 1. Create a PyTorch model
        pytorch_model = nn.Sequential(
            nn.Linear(4, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
            nn.Sigmoid()
        )
        
        # 2. Prepare data
        X_pt, y_pt = pytorch_adapter.prepare_data(sample_data, target_column="target_binary")
        
        # 3. Train the PyTorch model
        trained_pt_model = pytorch_adapter.train(pytorch_model, X_pt, y_pt, epochs=5)
        
        # 4. Evaluate the PyTorch model
        pt_metrics = pytorch_adapter.evaluate(trained_pt_model, X_pt, y_pt)
        assert "accuracy" in pt_metrics
        # 5. Convert to TensorFlow
        tensorflow_model = model_converter.pytorch_to_tensorflow(trained_pt_model)
        
        # 6. Prepare data for TensorFlow
        X_tf, y_tf = tensorflow_adapter.prepare_data(sample_data, target_column="target_binary")
        
        # 7. Evaluate the TensorFlow model
        tf_metrics = tensorflow_adapter.evaluate(tensorflow_model, X_tf, y_tf)
        assert "accuracy" in tf_metrics
        
        # 8. Verify metrics are similar between frameworks
        assert abs(pt_metrics["accuracy"] - tf_metrics["accuracy"]) < 0.1
        
        # 9. Save both models
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save PyTorch model
            pt_path = os.path.join(tmp_dir, "pytorch_model.pt")
            pytorch_adapter.save_model(trained_pt_model, pt_path)
            
            # Save TensorFlow model
            tf_path = os.path.join(tmp_dir, "tensorflow_model")
            tensorflow_adapter.save_model(tensorflow_model, tf_path)
            
            # 10. Load both models
            loaded_pt_model = pytorch_adapter.load_model(pt_path)
            loaded_tf_model = tensorflow_adapter.load_model(tf_path)
            
            # 11. Verify loaded models make similar predictions
            pt_preds = pytorch_adapter.predict(loaded_pt_model, X_pt)
            tf_preds = tensorflow_adapter.predict(loaded_tf_model, X_tf)
            
            # Convert predictions to numpy for comparison
            pt_preds_np = pt_preds.detach().cpu().numpy().flatten()
            tf_preds_np = tf_preds.flatten()
            
            # Verify correlation between predictions
            correlation = np.corrcoef(pt_preds_np, tf_preds_np)[0, 1]
            assert correlation > 0.8, "Predictions from loaded models should be correlated"
    
    def test_mixed_framework_pipeline_workflow(self, pytorch_adapter, tensorflow_adapter,
                                             model_converter, sample_data, preprocessor):
        """Test a complete workflow with a mixed-framework pipeline."""
        # 1. Create data container
        container = DataFrameContainer(sample_data, {
            "source": "test",
            "created_at": "2023-01-01",
            "version": "1.0.0"
        })
        
        # 2. Fit preprocessor
        preprocessor.fit(container)
        
        # 3. Create model processors
        pt_processor = PyTorchModelProcessor(adapter=pytorch_adapter)
        tf_processor = TensorFlowModelProcessor(adapter=tensorflow_adapter)
        
        # 4. Create ensemble processor that combines predictions
        class EnsembleProcessor(DataTransformer):
            def fit(self, data):
                pass  # No fitting needed
            
            def transform(self, data):
                if isinstance(data, DataFrameContainer):
                    df = data.data.copy()
                else:
                    df = data.copy()
                
                # Create ensemble prediction (average of both models)
                if "pytorch_predictions" in df.columns and "tensorflow_predictions" in df.columns:
                    df["ensemble_predictions"] = (df["pytorch_predictions"] + df["tensorflow_predictions"]) / 2
                
                # Create a result container if input was a container
                if isinstance(data, DataFrameContainer):
                    result = DataFrameContainer(df, data.metadata.copy())
                    result.add_metadata("ensembled", True)
                    return result
                return df
        
        # 5. Create and train the mixed-framework pipeline
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("pytorch", pt_processor),
            ("tensorflow", tf_processor),
            ("ensemble", EnsembleProcessor())
        ])
        
        # 6. Fit individual processors
        pt_processor.fit(container)
        tf_processor.fit(container)
        
        # 7. Process data through pipeline
        result = pipeline.process(container)
        
        # 8. Verify results and metadata
        assert "pytorch_predictions" in result.data.columns
        assert "tensorflow_predictions" in result.data.columns
        assert "ensemble_predictions" in result.data.columns
        
        assert "preprocessed" in result.metadata
        assert "pytorch_processed" in result.metadata
        assert "tensorflow_processed" in result.metadata
        assert "ensembled" in result.metadata
        
        # 9. Verify ensemble predictions are between framework predictions
        pt_preds = result.data["pytorch_predictions"]
        tf_preds = result.data["tensorflow_predictions"]
        ensemble_preds = result.data["ensemble_predictions"]
        
        # For each row, ensemble should be between PyTorch and TensorFlow
        assert all(
            min(pt, tf) <= ensemble <= max(pt, tf)
            for pt, tf, ensemble in zip(pt_preds, tf_preds, ensemble_preds)
        )
        
        # 10. Save and load the pipeline result
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save result to CSV
            result_path = os.path.join(tmp_dir, "ensemble_results.csv")
            result.data.to_csv(result_path, index=False)
            
            # Reload and verify
            loaded_df = pd.read_csv(result_path)
            assert "ensemble_predictions" in loaded_df.columns
            
            # Verify values match
            np.testing.assert_allclose(
                ensemble_preds, 
                loaded_df["ensemble_predictions"],
                rtol=1e-5
            )
    
    def test_model_deployment_workflow(self, pytorch_adapter, tensorflow_adapter,
                                     model_converter, sample_data):
        """Test a deployment workflow with model conversion and serving."""
        # 1. Create and train a PyTorch model
        class DeploymentModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(8, 1)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                x = self.sigmoid(x)
                return x
        
        pytorch_model = DeploymentModel()
        X_pt, y_pt = pytorch_adapter.prepare_data(sample_data, target_column="target_binary")
        trained_model = pytorch_adapter.train(pytorch_model, X_pt, y_pt, epochs=5)
        
        # 2. Convert to TensorFlow for deployment
        tf_model = model_converter.pytorch_to_tensorflow(trained_model)
        
        # 3. Save the deployed model in TensorFlow format
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create deployment directory
            deploy_dir = os.path.join(tmp_dir, "deployment")
            os.makedirs(deploy_dir, exist_ok=True)
            
            # Save model in SavedModel format
            model_path = os.path.join(deploy_dir, "model")
            tensorflow_adapter.save_model(tf_model, model_path, save_format="tf")
            
            # Save sample input for testing
            sample_input = sample_data.iloc[:5].drop(columns=["target_binary", "target_continuous"])
            sample_input.to_csv(os.path.join(deploy_dir, "sample_input.csv"), index=False)
            
            # 4. Simulate deployment environment
            # Load model in deployment environment
            deployed_model = tensorflow_adapter.load_model(model_path)
            
            # Load sample input
            deployed_input = pd.read_csv(os.path.join(deploy_dir, "sample_input.csv"))
            
            # 5. Make predictions in deployment environment
            X_deploy, _ = tensorflow_adapter.prepare_data(deployed_input)
            deployed_predictions = tensorflow_adapter.predict(deployed_model, X_deploy)
            
            # Make predictions with original model for comparison
            X_original = X_pt[:5]  # First 5 samples
            original_predictions = pytorch_adapter.predict(trained_model, X_original)
            
            # Convert to numpy for comparison
            np_original = original_predictions.detach().cpu().numpy().flatten()
            np_deployed = deployed_predictions.flatten()
            
            # 6. Verify deployment predictions match original
            np.testing.assert_allclose(np_original, np_deployed, rtol=1e-4)


# Documentation about the integration test suite
"""
The framework interoperability test suite provides comprehensive coverage of:

1. Model Conversion:
   - PyTorch to TensorFlow conversion
   - TensorFlow to PyTorch conversion
   - Weights and architecture preservation
   - Prediction consistency across frameworks

2. Shared Data Handling:
   - Dataset preparation for multiple frameworks
   - Consistent preprocessing across frameworks
   - Cross-framework evaluation with shared datasets

3. Mixed Framework Pipelines:
   - Pipelines combining PyTorch and TensorFlow components
   - Framework-specific data processing
   - Error handling across framework boundaries
   - Ensemble methods combining multiple frameworks

4. End-to-End Workflows:
   - Complete ML workflows across frameworks
   - Model training, conversion, and evaluation
   - Deployment scenarios with model serving
   - Cross-framework metrics comparison

These tests ensure smooth interoperability between PyTorch and TensorFlow
within the Prometheum framework, enabling users to leverage the strengths
of both ecosystems in their ML pipelines.
"""
