"""Storage health prediction module."""

import logging
import pickle
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .features import HealthFeatureExtractor

logger = logging.getLogger(__name__)

class HealthPredictor:
    """Predicts storage device health issues using ML."""
    
    def __init__(self, feature_extractor: HealthFeatureExtractor):
        self.feature_extractor = feature_extractor
        self.model = None
        self.feature_names = []
        self.prediction_thresholds = {
            "failure_probability": 0.7,  # Probability threshold for failure prediction
            "warning_probability": 0.4   # Probability threshold for warning
        }

    def _prepare_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to model input vector."""
        vector = []
        
        # Performance metrics
        perf = features.get("performance", {})
        for metric in self.feature_extractor.performance_features:
            stats = perf.get(metric, {})
            vector.extend([
                stats.get("mean", 0),
                stats.get("std", 0),
                stats.get("max", 0)
            ])
            
            # Add rate of change features
            changes = perf.get(f"{metric}_change", {})
            vector.extend([
                changes.get("mean", 0),
                changes.get("max", 0)
            ])

        # Error metrics
        errors = features.get("errors", {})
        for error_type in self.feature_extractor.error_features:
            vector.extend([
                errors.get(f"{error_type}_total", 0),
                errors.get(f"{error_type}_rate", 0),
                errors.get(f"{error_type}_acceleration", 0)
            ])

        # Alert metrics
        alerts = features.get("alerts", {})
        vector.extend([
            alerts.get("total_alerts", 0),
            alerts.get("active_alerts", 0),
            alerts.get("avg_resolution_minutes", 0)
        ])
        
        # Status changes
        vector.append(features.get("status_changes", 0))
        
        return np.array(vector).reshape(1, -1)

    async def predict_device_health(
        self,
        device: str,
        window_hours: int = 24
    ) -> Dict[str, Any]:
        """Predict health issues for a device.
        
        Args:
            device: Device name
            window_hours: Hours of history to analyze
            
        Returns:
            Dictionary containing:
            - Prediction (healthy, warning, failure)
            - Confidence score
            - Contributing factors
            - Recommended actions
        """
        if not self.model:
            return {
                "status": "error",
                "error": "Model not trained"
            }
        
        # Get features
        try:
            features = await self.feature_extractor.get_device_features(
                device, window_hours
            )
        except Exception as e:
            logger.error(f"Error extracting features for {device}: {e}")
            return {
                "status": "error",
                "error": f"Failed to extract features: {str(e)}"
            }
        
        # Prepare feature vector
        X = self._prepare_feature_vector(features)
        
        # Get prediction probabilities
        try:
            probs = self.model.predict_proba(X)[0]
            pred_class = self.model.predict(X)[0]
        except Exception as e:
            logger.error(f"Error making prediction for {device}: {e}")
            return {
                "status": "error",
                "error": f"Failed to make prediction: {str(e)}"
            }
        
        # Determine prediction details
        status = "healthy"
        if probs[1] >= self.prediction_thresholds["failure_probability"]:
            status = "failure"
        elif probs[1] >= self.prediction_thresholds["warning_probability"]:
            status = "warning"
        
        # Get feature importance for explanation
        importances = list(zip(self.feature_names, self.model.feature_importances_))
        top_factors = sorted(importances, key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "device": device,
            "status": status,
            "confidence": float(max(probs)),
            "prediction_time": datetime.now().isoformat(),
            "window_hours": window_hours,
            "contributing_factors": [
                {"feature": name, "importance": float(importance)}
                for name, importance in top_factors
            ],
            "recommendations": self._get_recommendations(status, features)
        }

    def _get_recommendations(
        self,
        status: str,
        features: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on prediction."""
        recommendations = []
        
        if status == "failure":
            recommendations.extend([
                "Backup data immediately",
                "Prepare replacement device",
                "Schedule maintenance window"
            ])
        elif status == "warning":
            recommendations.append("Monitor device more frequently")
            
            # Check specific metrics
            perf = features.get("performance", {})
            errors = features.get("errors", {})
            
            if any(errors.get(f"{e}_rate", 0) > 0 for e in self.feature_extractor.error_features):
                recommendations.append("Run extended device diagnostics")
            
            utilization = perf.get("utilization_percent", {}).get("mean", 0)
            if utilization > 80:
                recommendations.append("Consider load balancing")
        
        return recommendations

    def save_model(self, path: str) -> None:
        """Save the trained model."""
        if self.model:
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_names': self.feature_names,
                    'thresholds': self.prediction_thresholds
                }, f)

    def load_model(self, path: str) -> None:
        """Load a trained model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.prediction_thresholds = data.get(
                'thresholds',
                self.prediction_thresholds
            )

    async def collect_training_data(
        self,
        devices: List[str],
        window_hours: int = 24,
        history_days: int = 30
    ) -> Dict[str, Any]:
        """Collect training data from device history."""
        training_data = []
        labels = []
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=history_days)
        
        for device in devices:
            # Get device history
            smart_history = await self.feature_extractor.history_db.get_smart_history(
                device, start_time, end_time
            )
            
            # Create training examples from historical windows
            current_time = start_time + timedelta(hours=window_hours)
            while current_time <= end_time:
                window_start = current_time - timedelta(hours=window_hours)
                
                # Get features for this window
                features = await self.feature_extractor.get_device_features(
                    device,
                    window_hours=window_hours,
                    start_time=window_start,
                    end_time=current_time
                )
                
                # Determine if failure occurred within 24 hours
                next_day = current_time + timedelta(hours=24)
                failed = any(
                    entry["health_status"] == "failed"
                    for entry in smart_history
                    if window_start <= datetime.fromisoformat(entry["timestamp"]) <= next_day
                )
                
                # Add to training data
                training_data.append(self._prepare_feature_vector(features)[0])
                labels.append(1 if failed else 0)
                
                current_time += timedelta(hours=1)
        
        return {
            "X": np.array(training_data),
            "y": np.array(labels),
            "devices": devices,
            "window_hours": window_hours,
            "history_days": history_days
        }

