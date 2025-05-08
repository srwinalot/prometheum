"""
Storage performance metrics collection module.

This module provides functionality for collecting and analyzing storage device
performance metrics, including I/O statistics, utilization, and latency.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import platform
import statistics

logger = logging.getLogger(__name__)

class IOStats:
    """Container for I/O statistics."""
    
    def __init__(self):
        self.reads = 0
        self.writes = 0
        self.read_bytes = 0
        self.write_bytes = 0
        self.read_time = 0  # in milliseconds
        self.write_time = 0  # in milliseconds
        self.io_time = 0    # in milliseconds
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert I/O stats to dictionary."""
        return {
            "reads": self.reads,
            "writes": self.writes,
            "read_bytes": self.read_bytes,
            "write_bytes": self.write_bytes,
            "read_time": self.read_time,
            "write_time": self.write_time,
            "io_time": self.io_time,
            "timestamp": self.timestamp.isoformat()
        }

class DeviceMetrics:
    """Container for device performance metrics."""
    
    def __init__(self, device: str):
        self.device = device
        self.current: Optional[IOStats] = None
        self.previous: Optional[IOStats] = None
        self.history: List[Dict[str, Any]] = []
        self.history_limit = 3600  # Keep 1 hour of metrics (with 1s intervals)

    def update(self, stats: IOStats) -> None:
        """Update metrics with new I/O stats."""
        self.previous = self.current
        self.current = stats
        
        if self.previous and self.current:
            # Calculate performance metrics
            metrics = self._calculate_metrics()
            self.history.append(metrics)
            
            # Maintain history limit
            if len(self.history) > self.history_limit:
                self.history.pop(0)

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from current and previous stats."""
        if not (self.current and self.previous):
            return {"status": "initializing"}
        
        interval = (self.current.timestamp - self.previous.timestamp).total_seconds()
        if interval <= 0:
            return {"status": "error", "error": "Invalid time interval"}
        
        # Calculate IOPS
        read_iops = (self.current.reads - self.previous.reads) / interval
        write_iops = (self.current.writes - self.previous.writes) / interval
        
        # Calculate throughput (bytes/sec)
        read_throughput = (self.current.read_bytes - self.previous.read_bytes) / interval
        write_throughput = (self.current.write_bytes - self.previous.write_bytes) / interval
        
        # Calculate latency (ms)
        read_latency = 0
        write_latency = 0
        if read_iops > 0:
            read_latency = (self.current.read_time - self.previous.read_time) / (self.current.reads - self.previous.reads)
        if write_iops > 0:
            write_latency = (self.current.write_time - self.previous.write_time) / (self.current.writes - self.previous.writes)
        
        # Calculate utilization
        util = ((self.current.io_time - self.previous.io_time) / (interval * 1000)) * 100
        
        return {
            "timestamp": self.current.timestamp.isoformat(),
            "read_iops": round(read_iops, 2),
            "write_iops": round(write_iops, 2),
            "read_throughput_mbps": round(read_throughput / 1024 / 1024, 2),
            "write_throughput_mbps": round(write_throughput / 1024 / 1024, 2),
            "read_latency_ms": round(read_latency, 2),
            "write_latency_ms": round(write_latency, 2),
            "utilization_percent": round(min(util, 100), 2)
        }

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.history:
            return {"status": "no_data"}
        return self.history[-1]

    def get_average_metrics(self, interval: int = 300) -> Dict[str, Any]:
        """Get average metrics over the specified interval (seconds)."""
        if not self.history:
            return {"status": "no_data"}
        
        cutoff = datetime.fromisoformat(self.history[-1]["timestamp"]) - timedelta(seconds=interval)
        recent = [
            m for m in self.history 
            if datetime.fromisoformat(m["timestamp"]) >= cutoff
        ]
        
        if not recent:
            return {"status": "no_data"}
        
        metrics = {}
        for key in ["read_iops", "write_iops", "read_throughput_mbps", 
                   "write_throughput_mbps", "read_latency_ms", 
                   "write_latency_ms", "utilization_percent"]:
            values = [m[key] for m in recent if key in m]
            if values:
                metrics[key] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": round(statistics.mean(values), 2),
                    "median": round(statistics.median(values), 2)
                }
        
        metrics["interval_seconds"] = interval
        metrics["samples"] = len(recent)
        return metrics

class MetricsCollector:
    """Collects performance metrics for storage devices."""
    
    def __init__(self):
        self.os_type = platform.system().lower()
        self.devices: Dict[str, DeviceMetrics] = {}
        self._sector_size = 512  # Default sector size

    async def collect_metrics(self, devices: List[str]) -> Dict[str, Dict[str, Any]]:
        """Collect current metrics for specified devices."""
        results = {}
        
        for device in devices:
            try:
                if device not in self.devices:
                    self.devices[device] = DeviceMetrics(device)
                
                stats = await self._get_device_stats(device)
                if stats:
                    self.devices[device].update(stats)
                    results[device] = self.devices[device].get_current_metrics()
                else:
                    results[device] = {"status": "error", "error": "Could not collect stats"}
            except Exception as e:
                logger.error(f"Error collecting metrics for {device}: {e}")
                results[device] = {"status": "error", "error": str(e)}
        
        return results

    async def _get_device_stats(self, device: str) -> Optional[IOStats]:
        """Get I/O statistics for a device."""
        if self.os_type == "linux":
            return await self._get_linux_stats(device)
        elif self.os_type == "darwin":  # macOS
            return await self._get_macos_stats(device)
        else:
            # Use simulated stats for non-supported platforms or when testing
            return self._get_simulated_stats(device)

    async def _get_linux_stats(self, device: str) -> Optional[IOStats]:
        """Get device stats from /proc/diskstats."""
        try:
            stats = IOStats()
            diskstats_path = Path("/proc/diskstats")
            
            with open(diskstats_path, 'r') as f:
                content = f.read()
                for line in content.splitlines():
                    if device in line:
                        parts = line.split()
                        stats.reads = int(parts[3])
                        stats.writes = int(parts[7])
                        stats.read_bytes = int(parts[5]) * self._sector_size
                        stats.write_bytes = int(parts[9]) * self._sector_size
                        stats.read_time = int(parts[6])
                        stats.write_time = int(parts[10])
                        stats.io_time = int(parts[12])
                        return stats
            
            return None
        except Exception as e:
            logger.error(f"Error reading Linux stats for {device}: {e}")
            return None

    async def _get_macos_stats(self, device: str) -> Optional[IOStats]:
        """Get device stats using iostat on macOS."""
        try:
            stats = IOStats()
            cmd = ["iostat", "-d", "-c2", device]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"iostat error for {device}: {stderr.decode()}")
                return None
            
            lines = stdout.decode().splitlines()
            if len(lines) > 3:  # Skip headers and first sample
                parts = lines[3].split()
                stats.reads = int(float(parts[3]))
                stats.writes = int(float(parts[4]))
                # macOS iostat doesn't provide as detailed stats as Linux
                # We can only get operations count
                return stats
            
            return None
        except Exception as e:
            logger.error(f"Error reading macOS stats for {device}: {e}")
            return None
    
    def _get_simulated_stats(self, device: str) -> IOStats:
        """Get simulated stats for testing or unsupported platforms."""
        import random
        
        stats = IOStats()
        # Get last stats if available
        if device in self.devices and self.devices[device].current:
            last = self.devices[device].current
            # Simulate some activity
            stats.reads = last.reads + random.randint(0, 100)
            stats.writes = last.writes + random.randint(0, 50)
            stats.read_bytes = last.read_bytes + random.randint(0, 1024*1024*10)
            stats.write_bytes = last.write_bytes + random.randint(0, 1024*1024*5)
            stats.read_time = last.read_time + random.randint(0, 100)
            stats.write_time = last.write_time + random.randint(0, 50)
            stats.io_time = last.io_time + random.randint(0, 100)
        else:
            # Initial values
            stats.reads = random.randint(10000, 20000)
            stats.writes = random.randint(5000, 10000)
            stats.read_bytes = random.randint(1024*1024*100, 1024*1024*200)
            stats.write_bytes = random.randint(1024*1024*50, 1024*1024*100)
            stats.read_time = random.randint(1000, 2000)
            stats.write_time = random.randint(500, 1000)
            stats.io_time = random.randint(1000, 2000)
        
        return stats

async def get_device_metrics(device: str, collector: Optional[MetricsCollector] = None) -> Dict[str, Any]:
    """Get current metrics for a device."""
    if collector is None:
        collector = MetricsCollector()
    
    metrics = await collector.collect_metrics([device])
    return metrics.get(device, {"status": "error", "error": "Failed to collect metrics"})

