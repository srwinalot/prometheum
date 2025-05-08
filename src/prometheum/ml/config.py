"""Configuration management for ML components."""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class MLConfig:
    """Configuration for ML components."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize ML configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        
        # Ensure required paths exist
        self._ensure_paths()

    def _get_default_config_path(self) -> str:
        """Get default configuration path."""
        return os.path.join(
            os.path.expanduser("~"),
            ".prometheum",
            "ml_config.yaml"
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"Loaded ML configuration from {self.config_path}")
                    return config
            except Exception as e:
                logger.error(f"Error loading ML config: {e}")
        
        # Create default configuration
        config = self._create_default_config()
        self._save_config(config)
        return config

    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        base_dir = os.path.join(
            os.path.expanduser("~"),
            ".prometheum",
            "ml"
        )
        
        return {
            "paths": {
                "models_dir": os.path.join(base_dir, "models"),
                "training_data": os.path.join(base_dir, "training_data"),
                "default_model": os.path.join(base_dir, "models", "default_model.pkl")
            },
            "feature_extraction": {
                "default_window_hours": 24,
                "max_window_hours": 168,  # 1 week
                "min_data_points": 100,
                "sampling_rate_minutes": 5
            },
            "prediction": {
                "thresholds": {
                    "failure_probability": 0.7,
                    "warning_probability": 0.4
                },
                "confidence_threshold": 0.8,
                "prediction_window_hours": 24
            },
            "training": {
                "default_history_days": 30,
                "max_history_days": 365,
                "test_size": 0.2,
                "cv_folds": 5,
                "random_state": 42,
                "model_parameters": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [10, 20, 30, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            },
            "monitoring": {
                "prediction_interval_minutes": 60,
                "retraining_interval_days": 7,
                "min_training_samples": 1000,
                "alert_on_prediction_failure": True
            }
        }

    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        
        logger.info(f"Saved ML configuration to {self.config_path}")

    def _ensure_paths(self) -> None:
        """Ensure required paths exist."""
        paths = self.config.get("paths", {})
        for path in paths.values():
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)

    def get_path(self, path_name: str) -> str:
        """Get a configured path."""
        return self.config.get("paths", {}).get(path_name)

    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature extraction configuration."""
        return self.config.get("feature_extraction", {})

    def get_prediction_config(self) -> Dict[str, Any]:
        """Get prediction configuration."""
        return self.config.get("prediction", {})

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.get("training", {})

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return self.config.get("monitoring", {})

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        def deep_update(d: Dict, u: Dict) -> Dict:
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = deep_update(d[k], v)
                else:
                    d[k] = v
            return d
        
        self.config = deep_update(self.config, updates)
        self._save_config(self.config)
        self._ensure_paths()

    def get_value(self, *keys: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        value = self.config
        for key in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(key, default)
            if value is None:
                return default
        return value

    def set_value(self, value: Any, *keys: str) -> None:
        """Set a configuration value using dot notation."""
        if not keys:
            return
        
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        self._save_config(self.config)

