"""Health history storage module for testing."""

import logging
import sqlite3
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

class HealthHistoryDB:
    """Health history database."""
    
    def __init__(self, db_path: str):
        """Initialize database."""
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS smart_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device TEXT NOT NULL,
                    health_status TEXT NOT NULL,
                    data JSON NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    device TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    acknowledged INTEGER DEFAULT 0,
                    resolved INTEGER DEFAULT 0,
                    resolution_time TEXT
                )
            """)
            
            conn.commit()
        finally:
            conn.close()

    async def store_smart_data(self, data: Dict[str, Any]) -> None:
        """Store SMART data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO smart_data (device, health_status, data, timestamp)
                VALUES (?, ?, ?, ?)
            """, (
                data["device"],
                data["health_status"],
                json.dumps(data),
                data["timestamp"]
            ))
            conn.commit()
        finally:
            conn.close()

    async def store_alert(self, alert: Dict[str, Any]) -> None:
        """Store an alert."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO alerts (id, device, message, severity, timestamp, acknowledged, resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                alert["id"],
                alert["device"],
                alert["message"],
                alert["severity"],
                alert["timestamp"],
                1 if alert.get("acknowledged", False) else 0,
                1 if alert.get("resolved", False) else 0
            ))
            conn.commit()
        finally:
            conn.close()

    async def get_alerts(self, device: Optional[str] = None, include_resolved: bool = False) -> List[Dict[str, Any]]:
        """Get alerts from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            query = "SELECT * FROM alerts WHERE 1=1"
            params = []
            
            if device:
                query += " AND device = ?"
                params.append(device)
            
            if not include_resolved:
                query += " AND resolved = 0"
            
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
        finally:
            conn.close()

    async def get_device_health_summary(self, device: str) -> Dict[str, Any]:
        """Get device health summary."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get latest health status
            cursor.execute("""
                SELECT health_status, timestamp FROM smart_data
                WHERE device = ? ORDER BY timestamp DESC LIMIT 1
            """, (device,))
            latest = cursor.fetchone()
            
            # Get active alerts count
            cursor.execute("""
                SELECT COUNT(*) FROM alerts
                WHERE device = ? AND resolved = 0
            """, (device,))
            active_alerts = cursor.fetchone()[0]
            
            return {
                "device": device,
                "current_health": latest[0] if latest else "unknown",
                "last_update": latest[1] if latest else None,
                "active_alerts": active_alerts
            }
        finally:
            conn.close()
