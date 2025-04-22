"""
System management routes.
"""

import os
import platform
import time
import datetime
import shutil
import psutil
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks

from ..dependencies import get_current_user, get_admin_user
from ...utils import run_command, load_config, save_config

router = APIRouter()

# Models
class SystemConfig(BaseModel):
    """System configuration model."""
    hostname: Optional[str] = None
    timezone: Optional[str] = None
    ntp_servers: Optional[List[str]] = None
    dns_servers: Optional[List[str]] = None

class NetworkInterface(BaseModel):
    """Network interface configuration."""
    name: str
    dhcp: bool = True
    ipv4_address: Optional[str] = None
    ipv4_netmask: Optional[str] = None
    ipv4_gateway: Optional[str] = None
    dns_servers: Optional[List[str]] = None

class UpdateConfig(BaseModel):
    """System update configuration."""
    auto_update: bool = False
    update_channel: str = "stable"
    update_time: str = "03:00"  # 3 AM

# Helper functions
def get_system_info() -> Dict[str, Any]:
    """Get basic system information."""
    try:
        boot_time = datetime.datetime.fromtimestamp(psutil.boot_time()).isoformat()
        
        # Get memory info
        mem = psutil.virtual_memory()
        mem_total = mem.total
        mem_available = mem.available
        mem_used = mem.used
        mem_percent = mem.percent
        
        # Get disk info
        disk = psutil.disk_usage('/')
        disk_total = disk.total
        disk_used = disk.used
        disk_free = disk.free
        disk_percent = disk.percent
        
        # Get CPU info
        cpu_percent = psutil.cpu_percent(interval=0.5)
        cpu_count = psutil.cpu_count()
        cpu_stats = psutil.cpu_stats()
        
        # Load average (Linux-specific, fallback for other platforms)
        try:
            load_avg = os.getloadavg()
        except AttributeError:
            load_avg = (0, 0, 0)  # Fallback for non-UNIX platforms
        
        return {
            "hostname": platform.node(),
            "os": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "architecture": platform.machine()
            },
            "uptime": {
                "boot_time": boot_time,
                "uptime_seconds": time.time() - psutil.boot_time()
            },
            "cpu": {
                "count": cpu_count,
                "percent": cpu_percent,
                "load_avg": load_avg
            },
            "memory": {
                "total": mem_total,
                "available": mem_available,
                "used": mem_used,
                "percent": mem_percent
            },
            "disk": {
                "total": disk_total,
                "used": disk_used,
                "free": disk_free,
                "percent": disk_percent
            },
            "time": {
                "current_time": datetime.datetime.now().isoformat(),
                "timezone": time.tzname
            }
        }
    except Exception as e:
        return {"error": f"Failed to get system info: {str(e)}"}

def get_network_info() -> List[Dict[str, Any]]:
    """Get network interfaces information."""
    try:
        interfaces = []
        net_io = psutil.net_io_counters(pernic=True)
        net_if = psutil.net_if_addrs()
        
        for interface_name, addresses in net_if.items():
            interface_info = {
                "name": interface_name,
                "addresses": [],
                "traffic": {}
            }
            
            for addr in addresses:
                address_info = {
                    "family": str(addr.family),
                    "address": addr.address,
                    "netmask": addr.netmask,
                    "broadcast": addr.broadcast
                }
                interface_info["addresses"].append(address_info)
            
            # Add traffic stats if available
            if interface_name in net_io:
                stats = net_io[interface_name]
                interface_info["traffic"] = {
                    "bytes_sent": stats.bytes_sent,
                    "bytes_recv": stats.bytes_recv,
                    "packets_sent": stats.packets_sent,
                    "packets_recv": stats.packets_recv,
                    "errin": stats.errin,
                    "errout": stats.errout,
                    "dropin": stats.dropin,
                    "dropout": stats.dropout
                }
            
            interfaces.append(interface_info)
        
        return interfaces
    except Exception as e:
        return [{"error": f"Failed to get network info: {str(e)}"}]

# Routes
@router.get("/info")
async def get_system_status(user = Depends(get_current_user)):
    """Get system status and information."""
    return get_system_info()

@router.get("/metrics")
async def get_system_metrics(user = Depends(get_current_user)):
    """Get system performance metrics."""
    try:
        # Get current metrics
        cpu_percent = psutil.cpu_percent(interval=0.5, percpu=True)
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk_io = psutil.disk_io_counters()
        net_io = psutil.net_io_counters()
        
        return {
            "cpu": {
                "percent_per_core": cpu_percent,
                "average": sum(cpu_percent) / len(cpu_percent) if cpu_percent else 0
            },
            "memory": {
                "used_percent": mem.percent,
                "used": mem.used,
                "available": mem.available
            },
            "swap": {
                "used_percent": swap.percent,
                "used": swap.used,
                "free": swap.free
            },
            "disk_io": {
                "read_bytes": disk_io.read_bytes,
                "write_bytes": disk_io.write_bytes,
                "read_count": disk_io.read_count,
                "write_count": disk_io.write_count
            },
            "net_io": {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system metrics: {str(e)}"
        )

@router.get("/processes")
async def list_processes(user = Depends(get_current_user)):
    """List running processes."""
    try:
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'username', 'memory_percent', 'cpu_percent']):
            processes.append(proc.info)
        
        # Sort by CPU usage (descending)
        processes.sort(key=lambda p: p.get('cpu_percent', 0), reverse=True)
        
        return processes[:50]  # Return top 50 processes
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list processes: {str(e)}"
        )

@router.post("/processes/{pid}/kill")
async def kill_process(pid: int, user = Depends(get_admin_user)):
    """Kill a process by PID (admin only)."""
    try:
        process = psutil.Process(pid)
        process.terminate()
        
        # Wait for termination
        gone, still_alive = psutil.wait_procs([process], timeout=3)
        
        if process in still_alive:
            # Force kill if still running
            process.kill()
        
        return {"success": True, "message": f"Process {pid} terminated"}
    except psutil.NoSuchProcess:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Process {pid} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to kill process: {str(e)}"
        )

@router.get("/network")
async def get_network_status(user = Depends(get_current_user)):
    """Get network configuration and status."""
    return get_network_info()

@router.post("/network/interfaces/{interface}")
async def configure_network_interface(
    interface: str,
    config: NetworkInterface,
    user = Depends(get_admin_user)
):
    """Configure a network interface (admin only)."""
    try:
        # Check if interface exists
        interfaces = get_network_info()
        interface_exists = any(i.get("name") == interface for i in interfaces)
        
        if not interface_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Interface {interface} not found"
            )
        
        # Configure interface based on dhcp setting
        if config.dhcp:
            cmd = f"ip addr flush dev {interface} && dhclient {interface}"
        else:
            # Static IP configuration
            if not config.ipv4_address or not config.ipv4_netmask:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Static IP configuration requires address and netmask"
                )
            
            cmd = f"ip addr flush dev {interface} && ip addr add {config.ipv4_address}/{config.ipv4_netmask} dev {interface}"
            if config.ipv4_gateway:
                cmd += f" && ip route add default via {config.ipv4_gateway} dev {interface}"
        
        # Execute network configuration command
        result = run_command(cmd)
        
        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to configure interface: {result.stderr}"
            )
        
        # Configure DNS if provided
        if config.dns_servers:
            # Update resolv.conf
            with open("/etc/resolv.conf", "w") as f:
                for server in config.dns_servers:
                    f.write(f"nameserver {server}\n")
        
        return {"success": True, "message": f"Interface {interface} configured"}
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to configure interface: {str(e)}"
        )

@router.get("/config")
async def get_system_config(user = Depends(get_current_user)):
    """Get system configuration."""
    try:
        # Load system configuration
        config = load_config("/var/lib/prometheum/system_config.json", {
            "hostname": platform.node(),
            "timezone": time.tzname[0],
            "ntp_servers": ["pool.ntp.org"],
            "dns_servers": ["8.8.8.8", "8.8.4.4"]
        })
        
        return config
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system configuration: {str(e)}"
        )

@router.post("/config")
async def update_system_config(config: SystemConfig, user = Depends(get_admin_user)):
    """Update system configuration (admin only)."""
    try:
        # Load current config
        current_config = load_config("/var/lib/prometheum/system_config.json", {})
        
        # Update fields that are provided
        if config.hostname is not None:
            current_config["hostname"] = config.hostname
            # Apply hostname change
            run_command(f"hostname {config.hostname}")
            with open("/etc/hostname", "w") as f:
                f.write(config.hostname)
        
        if config.timezone is not None:
            current_config["timezone"] = config.timezone
            # Apply timezone change
            run_command(f"timedatectl set-timezone {config.timezone}")
        
        if config.ntp_servers is not None:
            current_config["ntp_servers"] = config.ntp_servers
            # Apply NTP server change
            with open("/etc/ntp.conf", "w") as f:
                for server in config.ntp_servers:
                    f.write(f"server {server}\n")
            run_command("systemctl restart ntp")
        
        if config.dns_servers is not None:
            current_config["dns_servers"] = config.dns_servers
            # Apply DNS server change
            with open("/etc/resolv.conf", "w") as f:
                for server in config.dns_servers:
                    f.write(f"nameserver {server}\n")
        
        # Save updated config
        save_config("/var/lib/prometheum/system_config.json", current_config)
        
        return {"success": True, "message": "System configuration updated"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update system configuration: {str(e)}"
        )

@router.post("/backup")
async def create_system_backup(user = Depends(get_admin_user)):
    """Create a system configuration backup (admin only)."""
    try:
        # Generate backup filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = "/var/lib/prometheum/backups"
        os.makedirs(backup_dir, exist_ok=True)
        backup_file = f"{backup_dir}/system_backup_{timestamp}.tar.gz"
        
        # Create backup of configuration files
        cmd = f"tar -czvf {backup_file} /var/lib/prometheum/*.json /etc/hostname /etc/resolv.conf"
        result = run_command(cmd)
        
        if not result.success:
            raise Exception(f"Backup command failed: {result.stderr}")
        
        return {
            "success": True, 
            "message": "System backup created",
            "backup_file": backup_file
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create backup: {str(e)}"
        )
@router.post("/restore")
async def restore_system_backup(
    backup_file: str,
    user = Depends(get_admin_user)
):
    """Restore a system configuration backup (admin only)."""
    try:
        # Check if backup file exists
        if not os.path.exists(backup_file):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Backup file not found: {backup_file}"
            )
        
        # Extract backup to a temporary directory
        temp_dir = "/tmp/prometheum_restore"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Extract backup
        cmd = f"tar -xzvf {backup_file} -C {temp_dir}"
        result = run_command(cmd)
        
        if not result.success:
            raise Exception(f"Restore extraction failed: {result.stderr}")
        
        # Copy configuration files back
        conf_files = [
            "system_config.json",
            "users.json",
            "storage/pools.json",
            "storage/volumes.json"
        ]
        
        for conf_file in conf_files:
            src_path = os.path.join(temp_dir, "var", "lib", "prometheum", conf_file)
            dest_path = f"/var/lib/prometheum/{conf_file}"
            
            if os.path.exists(src_path):
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(src_path, dest_path)
        
        # Restore system files
        system_files = [
            ("etc/hostname", "/etc/hostname"),
            ("etc/resolv.conf", "/etc/resolv.conf")
        ]
        
        for src_rel, dest_abs in system_files:
            src_path = os.path.join(temp_dir, src_rel)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dest_abs)
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return {
            "success": True,
            "message": "System configuration restored successfully"
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restore backup: {str(e)}"
        )

@router.get("/update/check")
async def check_for_updates(user = Depends(get_current_user)):
    """Check for system updates."""
    try:
        # Update package lists
        result = run_command("apk update")
        
        if not result.success:
            raise Exception(f"Package update check failed: {result.stderr}")
        
        # Get list of upgradable packages
        result = run_command("apk upgrade --simulate")
        
        upgradable_packages = []
        for line in result.stdout.splitlines():
            if "Upgrading" in line and "=>" in line:
                parts = line.split()
                if len(parts) >= 4:
                    package = parts[1]
                    old_version = parts[2]
                    new_version = parts[4]
                    upgradable_packages.append({
                        "package": package,
                        "current_version": old_version,
                        "new_version": new_version
                    })
        
        return {
            "updates_available": len(upgradable_packages) > 0,
            "packages": upgradable_packages
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check for updates: {str(e)}"
        )

@router.post("/update/apply")
async def apply_updates(
    background_tasks: BackgroundTasks,
    user = Depends(get_admin_user)
):
    """Apply system updates (admin only)."""
    def perform_update():
        try:
            # Update package lists
            run_command("apk update")
            
            # Apply updates
            run_command("apk upgrade")
            
            # Log update completion
            with open("/var/lib/prometheum/update_log.txt", "a") as f:
                f.write(f"{datetime.datetime.now().isoformat()}: Update completed successfully\n")
        except Exception as e:
            # Log update failure
            with open("/var/lib/prometheum/update_log.txt", "a") as f:
                f.write(f"{datetime.datetime.now().isoformat()}: Update failed: {str(e)}\n")
    
    # Start update in background
    background_tasks.add_task(perform_update)
    
    return {
        "success": True,
        "message": "System update started in background"
    }

@router.get("/update/status")
async def get_update_status(user = Depends(get_current_user)):
    """Get system update status."""
    try:
        log_file = "/var/lib/prometheum/update_log.txt"
        
        if not os.path.exists(log_file):
            return {
                "last_update": None,
                "status": "No updates have been performed"
            }
        
        # Read last 5 lines of update log
        result = run_command(f"tail -n 5 {log_file}")
        log_entries = result.stdout.splitlines()
        
        # Parse update status from log
        last_update = None
        update_status = "Unknown"
        
        for line in reversed(log_entries):
            if ": Update completed successfully" in line:
                last_update = line.split(":")[0]
                update_status = "Success"
                break
            elif ": Update failed" in line:
                last_update = line.split(":")[0]
                update_status = "Failed"
                break
        
        return {
            "last_update": last_update,
            "status": update_status,
            "recent_log": log_entries
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get update status: {str(e)}"
        )

@router.post("/maintenance/reboot")
async def reboot_system(
    background_tasks: BackgroundTasks,
    user = Depends(get_admin_user)
):
    """Reboot the system (admin only)."""
    def perform_reboot():
        # Wait a few seconds to allow API response to return
        time.sleep(3)
        # Execute reboot command
        os.system("reboot")
    
    # Start reboot in background
    background_tasks.add_task(perform_reboot)
    
    return {
        "success": True,
        "message": "System reboot initiated"
    }

@router.post("/maintenance/shutdown")
async def shutdown_system(
    background_tasks: BackgroundTasks,
    user = Depends(get_admin_user)
):
    """Shutdown the system (admin only)."""
    def perform_shutdown():
        # Wait a few seconds to allow API response to return
        time.sleep(3)
        # Execute shutdown command
        os.system("poweroff")
    
    # Start shutdown in background
    background_tasks.add_task(perform_shutdown)
    
    return {
        "success": True,
        "message": "System shutdown initiated"
    }

@router.post("/maintenance/services/{service}")
async def manage_service(
    service: str,
    action: str,
    user = Depends(get_admin_user)
):
    """Manage system services (admin only)."""
    valid_actions = ["start", "stop", "restart", "status"]
    valid_services = ["networking", "dhcpcd", "sshd", "ntp", "storage", "prometheum"]
    
    if action not in valid_actions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid action. Must be one of: {valid_actions}"
        )
    
    if service not in valid_services:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid service. Must be one of: {valid_services}"
        )
    
    try:
        # Map service name to actual service if needed
        service_map = {
            "networking": "networking",
            "prometheum": "prometheum"
        }
        
        actual_service = service_map.get(service, service)
        
        # Execute service management command
        cmd = f"systemctl {action} {actual_service}"
        result = run_command(cmd)
        
        if not result.success and action != "status":
            raise Exception(f"Service command failed: {result.stderr}")
        
        # For status action, parse output to determine status
        if action == "status":
            is_active = "Active: active" in result.stdout
            status_msg = "running" if is_active else "stopped"
            return {
                "service": service,
                "status": status_msg,
                "details": result.stdout
            }
        
        return {
            "success": True,
            "message": f"Service {service} {action} completed successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to {action} service {service}: {str(e)}"
        )
