"""
Router-specific configuration for Dismetheum.

This module defines the router configuration for the Dismetheum variant
of the Prometheum personal cloud storage system, including network interfaces,
DHCP/DNS settings, and traffic routing rules.
"""

import os
import json
import socket
import subprocess
import platform
import netifaces
from typing import Dict, List, Optional, Any, Union, Set

# Import system configuration if available
try:
    from config.system import system_config
except ImportError:
    # If system_config is not available, use default values
    class DummyConfig:
        def get(self, section, key, default=None):
            return default
    system_config = DummyConfig()

# Router base directories
ROUTER_CONFIG_DIR = "/etc/dismetheum"
ROUTER_DATA_DIR = "/var/lib/dismetheum"
ROUTER_LOG_DIR = "/var/log/dismetheum"
ROUTER_RUN_DIR = "/var/run/dismetheum"

# Default network interface settings
DEFAULT_WAN_INTERFACE = "eth0"  # External interface (internet)
DEFAULT_LAN_INTERFACE = "eth1"  # Internal interface (local network)
DEFAULT_WIFI_INTERFACE = "wlan0"  # WiFi interface
DEFAULT_BRIDGE_INTERFACE = "br0"  # Bridge interface for combined networks

# Default IP address settings
DEFAULT_LAN_IP = "192.168.1.1"
DEFAULT_LAN_NETMASK = "255.255.255.0"
DEFAULT_LAN_NETWORK = "192.168.1.0/24"

# DHCP server settings
DHCP_ENABLED = True
DHCP_RANGE_START = "192.168.1.100"
DHCP_RANGE_END = "192.168.1.200"
DHCP_LEASE_TIME = "24h"
DHCP_DNS_SERVER = DEFAULT_LAN_IP  # Local DNS server

# DNS server settings
DNS_ENABLED = True
DNS_UPSTREAM_SERVERS = ["1.1.1.1", "8.8.8.8"]  # Cloudflare and Google DNS
DNS_CACHE_SIZE = 1000
DNS_INTERCEPT_REQUESTS = True  # Used for cloud service interception

# NAS settings
NAS_ENABLED = True
NAS_SHARE_PATH = os.path.join(ROUTER_DATA_DIR, "shares")
NAS_PROTOCOLS = ["SMB", "NFS", "AFP"]  # Protocols to enable
NAS_DISCOVERY_ENABLED = True  # Enable mDNS/Bonjour discovery
NAS_GUEST_ACCESS = False  # Disable guest access by default
NAS_SHADOW_COPY = True  # Enable shadow copies for file versioning

# Traffic interception settings
TRAFFIC_INTERCEPTION_ENABLED = True
TRAFFIC_INTERCEPTION_MODE = "transparent_proxy"  # transparent_proxy, dns_redirect, arp_spoof
TRAFFIC_INTERCEPTION_PORTS = [80, 443]  # Ports to intercept
TRAFFIC_INTERCEPTION_SERVICES = ["icloud", "onedrive", "gdrive", "dropbox"]

# Firewall settings
FIREWALL_ENABLED = True
DEFAULT_INPUT_POLICY = "DROP"
DEFAULT_FORWARD_POLICY = "DROP"
DEFAULT_OUTPUT_POLICY = "ACCEPT"
ALLOWED_INPUT_PORTS = [22, 80, 443, 137, 138, 139, 445, 548]  # SSH, HTTP, HTTPS, SMB, AFP


class RouterConfig:
    """Router configuration manager for Dismetheum."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the router configuration.
        
        Args:
            config_path: Optional path to the configuration file
        """
        self.config_path = config_path or os.path.join(ROUTER_CONFIG_DIR, "router_config.json")
        self.config = self._load_defaults()
        self._load_from_file()
        self._detect_network_interfaces()
    
    def _load_defaults(self) -> Dict[str, Any]:
        """Load default router configuration settings."""
        return {
            "network": {
                "wan_interface": DEFAULT_WAN_INTERFACE,
                "lan_interface": DEFAULT_LAN_INTERFACE,
                "wifi_interface": DEFAULT_WIFI_INTERFACE,
                "bridge_interface": DEFAULT_BRIDGE_INTERFACE,
                "lan_ip": DEFAULT_LAN_IP,
                "lan_netmask": DEFAULT_LAN_NETMASK,
                "lan_network": DEFAULT_LAN_NETWORK
            },
            "dhcp": {
                "enabled": DHCP_ENABLED,
                "range_start": DHCP_RANGE_START,
                "range_end": DHCP_RANGE_END,
                "lease_time": DHCP_LEASE_TIME,
                "dns_server": DHCP_DNS_SERVER,
                "static_leases": {}  # MAC -> IP mappings
            },
            "dns": {
                "enabled": DNS_ENABLED,
                "upstream_servers": DNS_UPSTREAM_SERVERS,
                "cache_size": DNS_CACHE_SIZE,
                "intercept_requests": DNS_INTERCEPT_REQUESTS,
                "local_domain": "dismetheum.local",
                "custom_records": {}  # Hostname -> IP mappings
            },
            "nas": {
                "enabled": NAS_ENABLED,
                "share_path": NAS_SHARE_PATH,
                "protocols": NAS_PROTOCOLS,
                "discovery_enabled": NAS_DISCOVERY_ENABLED,
                "guest_access": NAS_GUEST_ACCESS,
                "shadow_copy": NAS_SHADOW_COPY,
                "shares": [
                    {
                        "name": "Public",
                        "path": "public",
                        "access": "read_write",
                        "guest_ok": True
                    },
                    {
                        "name": "Media",
                        "path": "media",
                        "access": "read_only",
                        "guest_ok": True
                    },
                    {
                        "name": "Private",
                        "path": "private",
                        "access": "read_write",
                        "guest_ok": False
                    }
                ]
            },
            "traffic_interception": {
                "enabled": TRAFFIC_INTERCEPTION_ENABLED,
                "mode": TRAFFIC_INTERCEPTION_MODE,
                "ports": TRAFFIC_INTERCEPTION_PORTS,
                "services": TRAFFIC_INTERCEPTION_SERVICES,
                "excluded_ips": []  # IPs to exclude from interception
            },
            "firewall": {
                "enabled": FIREWALL_ENABLED,
                "default_input_policy": DEFAULT_INPUT_POLICY,
                "default_forward_policy": DEFAULT_FORWARD_POLICY,
                "default_output_policy": DEFAULT_OUTPUT_POLICY,
                "allowed_input_ports": ALLOWED_INPUT_PORTS,
                "port_forwarding": [],  # Port forwarding rules
                "custom_rules": []  # Custom iptables rules
            }
        }
    
    def _load_from_file(self) -> None:
        """Load configuration from file if it exists."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                
                # Merge with defaults, preserving file values
                self._merge_configs(self.config, file_config)
        except Exception as e:
            print(f"Warning: Could not load router configuration from {self.config_path}: {str(e)}")
    
    def _merge_configs(self, default_config: Dict[str, Any], file_config: Dict[str, Any]) -> None:
        """
        Merge file configuration with defaults recursively.
        
        Args:
            default_config: Default configuration dictionary
            file_config: Configuration from file
        """
        for key, value in file_config.items():
            if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                self._merge_configs(default_config[key], value)
            else:
                default_config[key] = value
    
    def _detect_network_interfaces(self) -> None:
        """
        Detect available network interfaces and update configuration.
        
        This tries to identify the WAN (internet-connected) and LAN interfaces.
        """
        try:
            # Get all available interfaces
            interfaces = netifaces.interfaces()
            
            # Skip loopback and virtual interfaces
            valid_interfaces = [iface for iface in interfaces if not (
                iface == 'lo' or 
                iface.startswith('veth') or 
                iface.startswith('docker') or 
                iface.startswith('br-') or
                iface.startswith('vir')
            )]
            
            if not valid_interfaces:
                print("Warning: No valid network interfaces detected")
                return
            
            # Try to determine WAN interface (the one with a default route)
            wan_interface = None
            try:
                gateways = netifaces.gateways()
                if 'default' in gateways and netifaces.AF_INET in gateways['default']:
                    default_gw, wan_interface = gateways['default'][netifaces.AF_INET]
            except Exception:
                pass
            
            # If we found a WAN interface, update the config
            if wan_interface and wan_interface in valid_interfaces:
                self.config["network"]["wan_interface"] = wan_interface
                valid_interfaces.remove(wan_interface)
            
            # Try to determine LAN interface (first remaining interface)
            if valid_interfaces:
                lan_interface = valid_interfaces[0]
                self.config["network"]["lan_interface"] = lan_interface
                
                # If there's a second interface, use it as WiFi
                if len(valid_interfaces) > 1:
                    wifi_interface = valid_interfaces[1]
                    if wifi_interface.startswith(('wl', 'wlan')):
                        self.config["network"]["wifi_interface"] = wifi_interface
        
        except Exception as e:
            print(f"Warning: Error detecting network interfaces: {str(e)}")
    
    def save(self) -> bool:
        """
        Save configuration to file.
        
        Returns:
            bool: True if successful
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving router configuration: {str(e)}")
            return False
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if not found
            
        Returns:
            The configuration value or default
        """
        return self.config.get(section, {}).get(key, default)
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Value to set
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
    
    def get_wan_ip(self) -> Optional[str]:
        """
        Get the current WAN IP address.
        
        Returns:
            str: WAN IP address or None if not found
        """
        try:
            wan_interface = self.get("network", "wan_interface")
            if not wan_interface:
                return None
            
            addresses = netifaces.ifaddresses(wan_interface)
            if netifaces.AF_INET in addresses:
                return addresses[netifaces.AF_INET][0]['addr']
        except Exception:
            pass
        
        return None
    
    def generate_dhcpd_config(self) -> str:
        """
        Generate DHCP server configuration for dnsmasq.
        
        Returns:
            str: DHCP configuration
        """
        dhcp_config = []
        
        if self.get("dhcp", "enabled"):
            dhcp_config.append(f"dhcp-range={self.get('dhcp', 'range_start')},{self.get('dhcp', 'range_end')},{self.get('dhcp', 'lease_time')}")
            dhcp_config.append(f"dhcp-option=option:router,{self.get('network', 'lan_ip')}")
            dhcp_config.append(f"dhcp-option=option:dns-server,{self.get('dhcp', 'dns_server')}")
            
            # Add static leases
            static_leases = self.get("dhcp", "static_leases", {})
            for mac, ip in static_leases.items():
                dhcp_config.append(f"dhcp-host={mac},{ip}")
        
        return "\n".join(dhcp_config)
    
    def generate_dns_config(self) -> str:
        """
        Generate DNS server configuration for dnsmasq.
        
        Returns:
            str: DNS configuration
        """
        dns_config = []
        
        if self.get("dns", "enabled"):
            dns_config.append(f"cache-size={self.get('dns', 'cache_size')}")
            dns_config.append(f"domain={self.get('dns', 'local_domain')}")
            
            # Add upstream DNS servers
            for server in self.get("dns", "upstream_servers", []):
                dns_config.append(f"server={server}")
            
            # Add custom DNS records
            custom_records = self.get("dns", "custom_records", {})
            for hostname, ip in custom_records.items():
                dns_config.append(f"address=/{hostname}/{ip}")
            
            # Add cloud service interception if enabled
            if self.get("dns", "intercept_requests") and self.get("traffic_interception", "enabled"):
                for service in self.get("traffic_interception", "services", []):
                    if service == "icloud":
                        dns_config.append("address=/icloud.com/" + self.get("network", "lan_ip"))
                        dns_config.append("address=/apple-cloudkit.com/" + self.get("network", "lan_ip"))
                    elif service == "onedrive":
                        dns_config.append("address=/onedrive.live.com/" + self.get("network", "lan_ip"))
                        dns_config.append("address=/1drv.ms/" + self.get("network", "lan_ip"))
                    elif service == "gdrive":
                        dns_config.append("address=/drive.google.com/" + self.get("network", "lan_ip"))
                        dns_config.append("address=/docs.google.com/" + self.get("network", "

