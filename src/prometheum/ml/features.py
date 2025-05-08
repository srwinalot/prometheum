"""Feature extraction for storage health prediction."""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from prometheum.storage.history import HealthHistoryDB

logger = logging.getLogger(__name__)

class HealthFeatureExtractor:
    """Extract features from health monitoring data for ML."""
    
    def __init__(self, history_db: HealthHistoryDB):
        self.history_db = history_db
        
        # Feature definitions
        self.performance_features = [
            "read_iops",
            "write_iops",
            "read_throughput_mbps",
            "write_throughput_mbps",
            "read_latency_ms",
            "write_latency_ms",
            "utilization_percent"
        ]
        
        self.error_features = [
            "read_errors",
            "write_errors",
            "checksum_errors"
        ]

    async def get_device_features(
        self,
        device: str,
        window_hours: int = 24,
        resolution_minutes: int = 5
    ) -> Dict[str, Any]:
        """Get ML features for a device over a time window.
        
        Args:
            device: Device name
            window_hours: Hours of history to include
            resolution_minutes: Time resolution in minutes
        
        Returns:
            Dictionary of features including:
            - Performance metrics statistics
            - Error counts and rates
            - Alert patterns
            - Status changes
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=window_hours)
        
        # Get historical data
        smart_history = await self.history_db.get_smart_history(
            device, start_time, end_time
        )
        
        performance_history = await self.history_db.get_performance_history(
            device, start_time, end_time
        )
        
        alerts = await self.history_db.get_alerts(
            device=device,
            include_resolved=True,
            start_time=start_time
        )
        
        # Extract performance metric statistics
        perf_stats = self._calculate_performance_stats(performance_history)
        
        # Calculate error metrics
        error_stats = self._calculate_error_stats(smart_history)
        
        # Analyze alert patterns
        alert_stats = self._analyze_alerts(alerts)
        
        # Combine all features
        return {
            "device": device,
            "window_hours": window_hours,
            "timestamp": end_time.isoformat(),
            "performance": perf_stats,
            "errors": error_stats,
            "alerts": alert_stats,
            "status_changes": self._count_status_changes(smart_history)
        }

    def _calculate_performance_stats(
        self,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate statistics from performance metrics."""
        stats = {}
        
        for feature in self.performance_features:
            values = [
                entry.get(feature, 0)
                for entry in history
                if feature in entry
            ]
            
            if values:
                stats[feature] = {
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "latest": float(values[-1]) if values else 0.0
                }
                
                # Calculate rates of change
                if len(values) > 1:
                    changes = np.diff(values)
                    stats[f"{feature}_change"] = {
                        "min": float(np.min(changes)),
                        "max": float(np.max(changes)),
                        "mean": float(np.mean(changes))
                    }
        
        return stats

    def _calculate_error_stats(
        self,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate error-related statistics."""
        stats = {}
        
        for feature in self.error_features:
            values = [
                entry.get("errors", {}).get(feature, 0)
                for entry in history
            ]
            
            if values:
                # Calculate total errors
                stats[f"{feature}_total"] = sum(values)
                
                # Calculate error rate (errors per hour)
                if len(history) > 1:
                    first_time = datetime.fromisoformat(history[0]["timestamp"])
                    last_time = datetime.fromisoformat(history[-1]["timestamp"])
                    hours = (last_time - first_time).total_seconds() / 3600
                    if hours > 0:
                        stats[f"{feature}_rate"] = sum(values) / hours
                
                # Track error acceleration
                if len(values) > 1:
                    changes = np.diff(values)
                    stats[f"{feature}_acceleration"] = float(np.mean(changes))
        
        return stats

    def _analyze_alerts(
        self,
        alerts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze alert patterns."""
        stats = {
            "total_alerts": len(alerts),
            "active_alerts": len([a for a in alerts if not a.get("resolved", False)]),
            "severity_counts": {},
            "resolution_times": []
        }
        
        for alert in alerts:
            # Count by severity
            severity = alert.get("severity", "unknown")
            stats["severity_counts"][severity] = \
                stats["severity_counts"].get(severity, 0) + 1
            
            # Calculate resolution times for resolved alerts
            if alert.get("resolved", False) and "resolution_time" in alert:
                created = datetime.fromisoformat(alert["timestamp"])
                resolved = datetime.fromisoformat(alert["resolution_time"])
                resolution_minutes = (resolved - created).total_seconds() / 60
                stats["resolution_times"].append(resolution_minutes)
        
        # Calculate average resolution time
        if stats["resolution_times"]:
            stats["avg_resolution_minutes"] = np.mean(stats["resolution_times"])
        
        return stats

    def _count_status_changes(
        self,
        history: List[Dict[str, Any]]
    ) -> int:
        """Count how many times the health status changed."""
        if not history:
            return 0
            
        changes = 0
        prev_status = history[0].get("health_status")
        
        for entry in history[1:]:
            curr_status = entry.get("health_status")
            if curr_status != prev_status:
                changes += 1
                prev_status = curr_status
        
        return changes

