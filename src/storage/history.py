"""
Health history database module for Prometheum.

This module provides classes for storing and retrieving health monitoring data,
including SMART metrics, performance data, and alert history.
"""

import logging
import sqlite3
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class HealthHistoryDB:
    """Database for storing health monitoring history."""
    
    def __init__(self, db_path: str = "/var/lib/prometheum/storage/health_history.db"):
        """Initialize the health history database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._ensure_db()
        
    def _ensure_db(self) -> None:
        """Ensure the database and tables exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS smart_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device TEXT NOT NULL,
            model TEXT,
            serial TEXT,
            firmware TEXT,
            temperature REAL,
            power_on_hours INTEGER,
            health_status TEXT,
            data JSON,
            timestamp TEXT NOT NULL
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device TEXT NOT NULL,
            iops_read REAL,
            iops_write REAL,
            throughput_read REAL,
            throughput_write REAL,
            latency_read REAL,
            latency_write REAL,
            cpu_usage REAL,
            memory_usage REAL,
            temperature REAL,
            timestamp TEXT NOT NULL
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS health_alerts (
            id TEXT PRIMARY KEY,
            device TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            details JSON,
            acknowledged INTEGER DEFAULT 0,
            resolved INTEGER DEFAULT 0,
            resolution_time TEXT,
            timestamp TEXT NOT NULL
        )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_smart_device ON smart_data(device)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_smart_timestamp ON smart_data(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_device ON performance_metrics(device)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_device ON health_alerts(device)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON health_alerts(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON health_alerts(resolved)")
        
        conn.commit()
        conn.close()
        
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(self.db_path)
    
    async def store_smart_data(self, smart_data: Dict[str, Any]) -> None:
        """Store SMART data in the database.
        
        Args:
            smart_data: Dictionary of SMART data
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO smart_data 
                (device, model, serial, firmware, temperature, power_on_hours, health_status, data, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    smart_data["device"],
                    smart_data.get("model"),
                    smart_data.get("serial"),
                    smart_data.get("firmware"),
                    smart_data.get("temperature"),
                    smart_data.get("power_on_hours"),
                    smart_data.get("health_status"),
                    smart_data.get("attributes"),
                    smart_data.get("timestamp") or datetime.now().isoformat()
                )
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing SMART data: {e}")
            raise e
        finally:
            conn.close()
    
    async def store_performance_metrics(self, device: str, metrics: Dict[str, float]) -> None:
        """Store performance metrics in the database.
        
        Args:
            device: Device name
            metrics: Dictionary of performance metrics
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO performance_metrics 
                (device, iops_read, iops_write, throughput_read, throughput_write, 
                latency_read, latency_write, cpu_usage, memory_usage, temperature, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    device,
                    metrics.get("iops_read"),
                    metrics.get("iops_write"),
                    metrics.get("throughput_read"),
                    metrics.get("throughput_write"),
                    metrics.get("latency_read"),
                    metrics.get("latency_write"),
                    metrics.get("cpu_usage"),
                    metrics.get("memory_usage"),
                    metrics.get("temperature"),
                    datetime.now().isoformat()
                )
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing performance metrics: {e}")
            raise e
        finally:
            conn.close()
    
    async def store_alert(self, alert: Dict[str, Any]) -> None:
        """Store an alert in the database.
        
        Args:
            alert: Alert dictionary containing:
                - id: Unique alert ID
                - device: Device name
                - message: Alert message
                - severity: Alert severity level
                - timestamp: Alert creation time
                - acknowledged: Whether alert is acknowledged
                - resolved: Whether alert is resolved
                - resolution_time: When alert was resolved (if resolved)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO health_alerts 
                (id, device, severity, message, details, acknowledged, resolved, resolution_time, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert["id"],
                    alert["device"],
                    alert.get("severity", alert.get("level")), # Support both "severity" and "level" for compatibility
                    alert["message"],
                    json.dumps({"raw_data": alert.get("details", {})}),  # Store additional details as JSON
                    1 if alert.get("acknowledged", False) else 0,
                    1 if alert.get("resolved", False) else 0,
                    alert.get("resolution_time", alert.get("resolved_timestamp")),  # Support both field names
                    alert["timestamp"]
                )
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing alert: {e}")
            raise e
        finally:
            conn.close()
    
    async def update_alert(self, alert_id: str, updates: Dict[str, Any]) -> None:
        """Update an existing alert.
        
        Args:
            alert_id: Alert ID
            updates: Dictionary of fields to update
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Build update query dynamically
            set_clauses = []
            params = []
            
            for key, value in updates.items():
                if key in ["acknowledged", "resolved"]:
                    set_clauses.append(f"{key} = ?")
                    params.append(1 if value else 0)
                elif key == "resolution_time" and updates.get("resolved"):
                    set_clauses.append("resolution_time = ?")
                    params.append(datetime.now().isoformat())
                elif key not in ["id", "timestamp"]:  # Don't update primary key or timestamp
                    set_clauses.append(f"{key} = ?")
                    params.append(value)
            
            if not set_clauses:
                return
                
            params.append(alert_id)
            query = f"UPDATE health_alerts SET {', '.join(set_clauses)} WHERE id = ?"
            
            cursor.execute(query, params)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating alert: {e}")
            raise e
        finally:
            conn.close()
    
    async def get_smart_history(self, 
                             device: str, 
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get SMART data history for a device.
        
        Args:
            device: Device name
            start_time: Start time for history query
            end_time: End time for history query
            
        Returns:
            List of SMART data records
        """
        query = "SELECT * FROM smart_data WHERE device = ?"
        params = [device]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
            
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
            
        query += " ORDER BY timestamp DESC"
        
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        conn.close()
        
        return [
            {columns[i]: value for i, value in enumerate(row)}
            for row in rows
        ]
    
    async def get_performance_history(self,
                                   device: str,
                                   start_time: Optional[datetime] = None,
                                   end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get performance metrics history for a device.
        
        Args:
            device: Device name
            start_time: Start time for history query
            end_time: End time for history query
            
        Returns:
            List of performance metric records
        """
        query = "SELECT * FROM performance_metrics WHERE device = ?"
        params = [device]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
            
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
            
        query += " ORDER BY timestamp DESC"
        
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        conn.close()
        
        return [
            {columns[i]: value for i, value in enumerate(row)}
            for row in rows
        ]
    
    async def get_alerts(self, 
                      device: Optional[str] = None,
                      include_resolved: bool = False,
                      start_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get health alerts.
        
        Args:
            device: Optional device filter
            include_resolved: Whether to include resolved alerts
            start_time: Optional start time filter
            
        Returns:
            List of alert records
        """
        query = "SELECT * FROM health_alerts WHERE 1=1"
        params = []
        
        if device:
            query += " AND device = ?"
            params.append(device)
        
        if not include_resolved:
            query += " AND resolved = 0"
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        query += " ORDER BY timestamp DESC"
        
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        return [
            {
                col: (row[i] == 1 if col in ["acknowledged", "resolved"] else row[i])
                for i, col in enumerate(columns)
            }
            for row in rows
        ]

    async def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """Clean up data older than specified days.
        
        Args:
            days_to_keep: Number of days of data to retain
        """
        cutoff = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            # Clean up old SMART data
            cursor.execute(
                "DELETE FROM smart_data WHERE timestamp < ?",
                (cutoff,)
            )
            
            # Clean up old performance metrics
            cursor.execute(
                "DELETE FROM performance_metrics WHERE timestamp < ?",
                (cutoff,)
            )
            
            # Clean up old resolved alerts
            cursor.execute("""
                DELETE FROM health_alerts 
                WHERE resolved = 1 
                AND resolution_time < ?
            """, (cutoff,))
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    async def get_device_health_summary(self, device: str) -> Dict[str, Any]:
        """Get a summary of device health history.
        
        Args:
            device: Device name
            
        Returns:
            Health summary dictionary
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get latest SMART data
        cursor.execute(
            "SELECT health_status, timestamp FROM smart_data WHERE device = ? ORDER BY timestamp DESC LIMIT 1",
            (device,)
        )
        smart_row = cursor.fetchone()
        
        # Get alert counts
        cursor.execute(
            "SELECT severity, COUNT(*) FROM health_alerts WHERE device = ? AND resolved = 0 GROUP BY severity",
            (device,)
        )
        alert_counts = {severity: count for severity, count in cursor.fetchall()}
        
        # Get historical health status
        seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
        cursor.execute(
            "SELECT health_status, COUNT(*) FROM smart_data WHERE device = ? AND timestamp > ? GROUP BY health_status",
            (device, seven_days_ago)
        )
        health_history = {status: count for status, count in cursor.fetchall()}
        
        return {
            "device": device,
            "current_health": smart_row[0] if smart_row else "unknown",
            "last_update": smart_row[1] if smart_row else None,
            "active_alerts": sum(alert_counts.values()),
            "alert_counts": alert_counts,
            "health_history": health_history
        }

