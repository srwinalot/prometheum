"""Storage health prediction model trainer."""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from .predictor import HealthPredictor
from .features import HealthFeatureExtractor

logger = logging.getLogger(__name__)

class HealthModelTrainer:
    """Trains models for storage health prediction."""
    
    def __init__(self, predictor: HealthPredictor):
        self.predictor = predictor
        self.default_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

    async def train_model(
        self,
        devices: List[str],
        window_hours: int = 24,
        history_days: int = 30,
        test_size: float = 0.2,
        model_params: Optional[Dict] = None,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """Train a new prediction model.
        
        Args:
            devices: List of devices to use for training
            window_hours: Hours of history for feature windows
            history_days: Days of history to use for training
            test_size: Proportion of data to use for testing
            model_params: GridSearchCV parameters (uses default if None)
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing:
            - Training metrics
            - Cross-validation results
            - Best parameters
            - Feature importance
        """
        # Collect training data
        logger.info("Collecting training data...")
        data = await self.predictor.collect_training_data(
            devices,
            window_hours,
            history_days
        )
        
        X, y = data["X"], data["y"]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )
        
        # Set up model and parameters
        base_model = RandomForestClassifier(random_state=42)
        param_grid = model_params or self.default_params
        
        # Create and configure grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        # Train model
        logger.info("Training model...")
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Generate feature names
        feature_names = self._generate_feature_names()
        
        # Update predictor
        self.predictor.model = best_model
        self.predictor.feature_names = feature_names
        
        # Evaluate model
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "best_parameters": grid_search.best_params_,
            "cv_results": {
                "mean_test_score": float(grid_search.cv_results_["mean_test_score"][grid_search.best_index_]),
                "std_test_score": float(grid_search.cv_results_["std_test_score"][grid_search.best_index_])
            },
            "feature_importance": self._get_feature_importance(best_model, feature_names)
        }
        
        logger.info("Model training completed")
        
        return metrics

    def _generate_feature_names(self) -> List[str]:
        """Generate feature names based on extractor configuration."""
        feature_names = []
        
        # Performance metric features
        for metric in self.predictor.feature_extractor.performance_features:
            # Basic statistics
            feature_names.extend([
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_max"
            ])
            # Rate of change features
            feature_names.extend([
                f"{metric}_change_mean",
                f"{metric}_change_max"
            ])
        
        # Error metric features
        for error in self.predictor.feature_extractor.error_features:
            feature_names.extend([
                f"{error}_total",
                f"{error}_rate",
                f"{error}_acceleration"
            ])
        
        # Alert features
        feature_names.extend([
            "total_alerts",
            "active_alerts",
            "avg_resolution_minutes"
        ])
        
        # Status changes
        feature_names.append("status_changes")
        
        return feature_names

    def _get_feature_importance(
        self,
        model: RandomForestClassifier,
        feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Get sorted feature importance with names."""
        importances = list(zip(feature_names, model.feature_importances_))
        sorted_importances = sorted(importances, key=lambda x: x[1], reverse=True)
        
        return [
            {
                "feature": name,
                "importance": float(importance),
                "description": self._get_feature_description(name)
            }
            for name, importance in sorted_importances
        ]

    def _get_feature_description(self, feature_name: str) -> str:
        """Get human-readable description of a feature."""
        descriptions = {
            "mean": "Average value over time window",
            "std": "Standard deviation (variability)",
            "max": "Maximum value in window",
            "change_mean": "Average rate of change",
            "change_max": "Maximum rate of change",
            "total": "Total count in window",
            "rate": "Occurrences per hour",
            "acceleration": "Rate of increase/decrease",
            "total_alerts": "Total number of alerts",
            "active_alerts": "Currently active alerts",
            "avg_resolution_minutes": "Average time to resolve alerts",
            "status_changes": "Number of status changes"
        }
        
        # Find matching description parts
        parts = []
        for key, desc in descriptions.items():
            if key in feature_name:
                parts.append(desc)
        
        if not parts:
            return "No description available"
        
        return " - ".join(parts)

    def cross_validate_thresholds(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """Find optimal prediction thresholds using cross-validation."""
        from sklearn.model_selection import KFold
        
        # Initialize arrays for probabilities
        y_probs = np.zeros(len(y))
        
        # Perform cross-validation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X):
            # Train model on fold
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            model = RandomForestClassifier(
                **self.predictor.model.get_params()
            )
            model.fit(X_train, y_train)
            
            # Get probabilities for validation set
            y_probs[val_idx] = model.predict_proba(X_val)[:, 1]
        
        # Find optimal thresholds
        thresholds = np.arange(0.1, 1.0, 0.1)
        best_f1 = 0
        best_thresholds = {
            "failure_probability": 0.7,
            "warning_probability": 0.4
        }
        
        for failure_thresh in thresholds:
            for warning_thresh in thresholds:
                if warning_thresh >= failure_thresh:
                    continue
                
                # Create predictions using these thresholds
                y_pred = np.zeros_like(y)
                y_pred[y_probs >= failure_thresh] = 2  # failure
                y_pred[(y_probs >= warning_thresh) & (y_probs < failure_thresh)] = 1  # warning
                
                # Calculate F1 score
                from sklearn.metrics import f1_score
                f1 = f1_score(y, y_pred, average='weighted')
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresholds = {
                        "failure_probability": float(failure_thresh),
                        "warning_probability": float(warning_thresh)
                    }
        
        return best_thresholds

