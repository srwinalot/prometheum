#!/bin/bash
#
# Dismetheum Router Setup Script
# This script configures a system to function as a router/NAS with
# cloud service interception capability.
#

set -e  # Exit on any error

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CONFIG_DIR="/etc/dismetheum"
DATA_DIR="/var/lib/dismetheum"
LOG_DIR="/var/log/dismetheum"
FORCE=false
VERBOSE=false

# Network settings (these can be overridden by config file)
WAN_INTERFACE=""
LAN_INTERFACE=""
BRIDGE_INTERFACE="br0"
LAN_IP="192.168.1.1"
LAN_NETMASK="255.255.255.0"
LAN_NETWORK="192.168.1.0/24"
DHCP_RANGE_START="192.168.1.100"
DHCP_RANGE_END="192.168.1.200"
DHCP_LEASE_TIME="24h"

# Log file
LOGFILE="$LOG_DIR/router_setup.log"

# Display help message
function show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help             Display this help message"
    echo "  -c, --config DIR       Set configuration directory (default: $CONFIG_DIR)"
    echo "  --wan INTERFACE        Specify WAN (internet-facing) interface"
    echo "  --lan INTERFACE        Specify LAN (local network) interface"
    echo "  --ip IP                Set LAN IP address (default: $LAN_IP)"
    echo "  --netmask MASK         Set LAN network mask (default: $LAN_NETMASK)"
    echo "  -f, --force            Force setup even if risks detected"
    echo "  -v, --verbose          Enable verbose output"
    echo
    echo "Example:"
    echo "  $0 --wan eth0 --lan eth1 --ip 192.168.0.1"
    echo
}

# Log message to console and log file
function log() {
    local level="$1"
    local message="$2"
    local color=""
    
    # Set color based on level
    case "$level" in
        "INFO")
            color="$BLUE"
            ;;
        "SUCCESS")
            color="$GREEN"
            ;;
        "WARNING")
            color="$YELLOW"
            ;;
        "ERROR")
            color="$RED"
            ;;
        *)
            color="$NC"
            ;;
    esac
    
    # Log to console
    echo -e "${color}[$level] $message${NC}"
    
    # Log to file
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message" >> "$LOGFILE"
}

# Parse command line arguments
function parse_args() {
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--config)
                CONFIG_DIR="$2"
                shift
                shift
                ;;
            --wan)
                WAN_INTERFACE="$2"
                shift
                shift
                ;;
            --lan)
                LAN_INTERFACE="$2"
                shift
                shift
                ;;
            --ip)
                LAN_IP="$2"
                shift
                shift
                ;;
            --netmask)
                LAN_NETMASK="$2"
                shift
                shift
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            *)
                log "ERROR" "Unknown option: $key"
                show_help
                exit 1
                ;;
        esac
    done
}

# Check if running as root
function check_root() {
    if [ "$(id -u)" -ne 0 ]; then
        log "ERROR" "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Load configuration from file
function load_config() {
    local config_file="$CONFIG_DIR/router_config.json"
    
    if [ -f "$config_file" ]; then
        log "INFO" "Loading configuration from $config_file"
        
        # This is a simple implementation that assumes the config file is structured
        # as key=value pairs for simplicity. In a real implementation, you would use a
        # proper JSON parser.
        if command -v jq >/dev/null 2>&1; then
            # Use jq if available
            WAN_INTERFACE=$(jq -r '.network.wan_interface // empty' "$config_file") || WAN_INTERFACE=""
            LAN_INTERFACE=$(jq -r '.network.lan_interface // empty' "$config_file") || LAN_INTERFACE=""
            BRIDGE_INTERFACE=$(jq -r '.network.bridge_interface // empty' "$config_file") || BRIDGE_INTERFACE="br0"
            LAN_IP=$(jq -r '.network.lan_ip // empty' "$config_file") || LAN_IP="192.168.1.1"
            LAN_NETMASK=$(jq -r '.network.lan_netmask // empty' "$config_file") || LAN_NETMASK="255.255.255.0"
            LAN_NETWORK=$(jq -r '.network.lan_network // empty' "$config_file") || LAN_NETWORK="192.168.1.0/24"
            DHCP_RANGE_START=$(jq -r '.dhcp.range_start // empty' "$config_file") || DHCP_RANGE_START="192.168.1.100"
            DHCP_RANGE_END=$(jq -r '.dhcp.range_end // empty' "$config_file") || DHCP_RANGE_END="192.168.1.200"
            DHCP_LEASE_TIME=$(jq -r '.dhcp.lease_time // empty' "$config_file") || DHCP_LEASE_TIME="24h"
        else
            # Fallback to grep/sed if jq is not available
            log "WARNING" "jq not found, using limited config parsing"
            # These are basic grep/sed operations, not a full JSON parser
            WAN_INTERFACE=$(grep -o '"wan_interface"[[:space:]]*:[[:space:]]*"[^"]*"' "$config_file" | sed 's/.*"wan_interface"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')
            LAN_INTERFACE=$(grep -o '"lan_interface"[[:space:]]*:[[:space:]]*"[^"]*"' "$config_file" | sed 's/.*"lan_interface"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')
            # Similar for other values
        fi
    else
        log "WARNING" "Configuration file not found, using default values"
    fi
    
    if [ "$VERBOSE" = true ]; then
        log "INFO" "WAN Interface: $WAN_INTERFACE"
        log "INFO" "LAN Interface: $LAN_INTERFACE"
        log "INFO" "Bridge Interface: $BRIDGE_INTERFACE"
        log "INFO" "LAN IP: $LAN_IP"
        log "INFO" "LAN Netmask: $LAN_NETMASK"
        log "INFO" "LAN Network: $LAN_NETWORK"
    fi
}

# Detect network interfaces if not specified
function detect_interfaces() {
    if [ -z "$WAN_INTERFACE" ] || [ -z "$LAN_INTERFACE" ]; then
        log "INFO" "Detecting network interfaces..."
        
        # Get all network interfaces
        local interfaces=$(ip -o link show | awk -F': ' '{print $2}' | grep -v "lo\|docker\|veth\|br-\|vir")
        
        if [ -z "$interfaces" ]; then
            log "ERROR" "No network interfaces detected"
            exit 1
        fi
        
        # Try to determine WAN interface (the one with a default route)
        if [ -z "$WAN_INTERFACE" ]; then
            WAN_INTERFACE=$(ip route | grep default | awk '{print $5}' | head -n1)
            
            if [ -z "$WAN_INTERFACE" ]; then
                # If no default route, use the first interface
                WAN_INTERFACE=$(echo "$interfaces" | head -n1)
                log "WARNING" "No default route found, guessing WAN interface: $WAN_INTERFACE"
            else
                log "INFO" "Detected WAN interface: $WAN_INTERFACE"
            fi
        fi
        
        # Choose LAN interface (a different interface than WAN)
        if [ -z "$LAN_INTERFACE" ]; then
            LAN_INTERFACE=$(echo "$interfaces" | grep -v "$WAN_INTERFACE" | head -n1)
            
            if [ -z "$LAN_INTERFACE" ]; then
                log "ERROR" "Could not detect a second network interface for LAN"
                exit 1
            else
                log "INFO" "Detected LAN interface: $LAN_INTERFACE"
            fi
        fi
    fi
    
    # Verify interfaces exist
    if ! ip link show "$WAN_INTERFACE" &>/dev/null; then
        log "ERROR" "WAN interface $WAN_INTERFACE does not exist"
        exit 1
    fi
    
    if ! ip link show "$LAN_INTERFACE" &>/dev/null; then
        log "ERROR" "LAN interface $LAN_INTERFACE does not exist"
        exit 1
    fi
}

# Set up IP forwarding
function setup_ip_forwarding() {
    log "INFO" "Setting up IP forwarding..."
    
    # Enable IP forwarding in sysctl
    echo "net.ipv4.ip_forward=1" > /etc/sysctl.d/99-dismetheum.conf
    sysctl -p /etc/sysctl.d/99-dismetheum.conf
    
    # Make sure it's also set for the current session
    echo 1 > /proc/sys/net/ipv4/ip_forward
    
    log "SUCCESS" "IP forwarding enabled"
}

# Configure network interfaces
function configure_interfaces() {
    log "INFO" "Configuring network interfaces..."
    
    # Check if the WAN interface is used for internet access
    if ! ping -c 1 -W 2 8.8.8.8 &>/dev/null; then
        log "WARNING" "Internet connectivity check failed. Make sure WAN interface has internet access."
        if [ "$FORCE" != true ]; then
            log "ERROR" "Use --force to continue anyway"
            exit 1
        fi
    fi
    
    # Create network interface configuration
    local netplan_config="/etc/netplan/01-dismetheum.yaml"
    
    # Check if we're using netplan (Ubuntu) or interfaces (Debian)
    if command -v netplan &>/dev/null; then
        log "INFO" "Using netplan for network configuration"
        
        # Create netplan configuration
        cat > "$netplan_config" <<EOF
network:
  version: 2
  renderer: networkd
  ethernets:
    $WAN_INTERFACE:
      dhcp4: true
      dhcp6: false
    $LAN_INTERFACE:
      addresses:
        - $LAN_IP/$LAN_NETMASK
      dhcp4: false
      dhcp6: false
EOF
        
        # Apply network configuration
        netplan apply
        
    else
        log "INFO" "Using interfaces for network configuration"
        
        # Create interfaces configuration
        cat > /etc/network/interfaces.d/dismetheum <<EOF
# WAN interface
auto $WAN_INTERFACE
iface $WAN_INTERFACE inet dhcp

# LAN interface
auto $LAN_INTERFACE
iface $LAN_INTERFACE inet static
    address $LAN_IP
    netmask $LAN_NETMASK
EOF
        
        # Restart networking
        systemctl restart networking
    fi
    
    log "SUCCESS" "Network interfaces configured"
}

# Set up DHCP server
function setup_dhcp_server() {
    log "INFO" "Setting up DHCP server..."
    
    # Install dnsmasq if not already installed
    if ! command -v dnsmasq &>/dev/null; then
        log "INFO" "Installing dnsmasq..."
        apt-get update
        apt-get install -y dnsmasq
    fi
    
    # Create dnsmasq configuration for DHCP
    cat > /etc/dnsmasq.d/dismetheum-dhcp.conf <<EOF
# DHCP configuration
interface=$LAN_INTERFACE
dhcp-range=$DHCP_RANGE_START,$DHCP_RANGE_END,$LAN_NETMASK,$DHCP_LEASE_TIME
dhcp-option=option:router,$LAN_IP
dhcp-option=option:dns-server,$LAN_IP
dhcp-option=option:domain-name,dismetheum.local

# DHCP lease file
dhcp-leasefile=$DATA_DIR/dhcp/leases
EOF
    
    # Create directory for leases
    mkdir -p "$DATA_DIR/dhcp"
    
    # Restart dnsmasq
    systemctl restart dnsmasq
    
    log "SUCCESS" "DHCP server configured"
}

# Set up DNS server
function setup_dns_server() {
    log "INFO" "Setting up DNS server..."
    
    # Create dnsmasq configuration for DNS
    cat > /etc/dnsmasq.d/dismetheum-dns.conf <<EOF
# DNS configuration
no-resolv
server=1.1.1.1
server=8.8.8.8
cache-size=1000
domain=dismetheum.local
expand-hosts
local=/dismetheum.local/
address=/dismetheum.local/$LAN_IP
EOF
    
    # Add example cloud interception rules
    cat > /etc/dnsmasq.d/dismetheum-interception.conf <<EOF
# Cloud service interception
# Uncomment lines below to enable interception
#address=/icloud.com/$LAN_IP
#address=/apple-cloudkit.com/$LAN_IP
#address=/onedrive.live.com/$LAN_IP
#address=/1drv.ms/$LAN_IP
#address=/drive.google.com/$LAN_IP
#address=/docs.google.com/$LAN_IP
#address=/dropbox.com/$LAN_IP
EOF
    
    # Restart dnsmasq
    systemctl restart dnsmasq
    
    log "SUCCESS" "DNS server configured"
}

# Set up NAT and firewall
function setup_firewall() {
    log "INFO" "Setting up firewall and NAT..."
    
    # Install iptables-persistent if not already installed
    if ! dpkg -l | grep -q iptables-persistent; then
        log "INFO" "Installing iptables-persistent..."
        DEBIAN_FRONTEND=noninteractive apt-get install -y iptables-persistent
    fi
    
    # Flush existing rules
    iptables -F
    iptables -t nat -F
    iptables -X
    
    # Default policies
    iptables -P INPUT DROP
    iptables -P FORWARD DROP
    iptables -P OUTPUT ACCEPT
    
    # Allow established connections
    iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
    iptables -A FORWARD -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
    
    # Allow local loopback
    iptables -A INPUT -i lo -j ACCEPT
    
    # Allow SSH for administration
    iptables -A INPUT -p tcp --dport 22 -j ACCEPT
    
    # Allow HTTP/HTTPS for web UI
    iptables -A INPUT -p tcp --dport 80 -j ACCEPT
    iptables -A INPUT -p tcp --dport 443 -j ACCEPT
    
    # Allow DNS
    iptables -A INPUT -p udp --dport 53 -j ACCEPT
    iptables -A INPUT -p tcp --dport 53 -j ACCEPT
    
    # Allow DHCP
    iptables -A INPUT -p udp --dport 67:68 -j ACCEPT
    
    # Allow NAS protocols from LAN
    # Samba/CIFS
    iptables -A INPUT -i $LAN_INTERFACE -p tcp --dport 137:139 -j ACCEPT
    iptables -A INPUT -i $LAN_INTERFACE -p udp --dport 137:139 -j ACCEPT
    iptables -A INPUT -i $LAN_INTERFACE -p tcp --dport 445 -j ACCEPT
    
    # NFS
    iptables -A INPUT -i $LAN_INTERFACE -p tcp --dport 111 -j ACCEPT
    iptables -A INPUT -i $LAN_INTERFACE -p udp --dport 111 -j ACCEPT
    iptables -A INPUT -i $LAN_INTERFACE -p tcp --dport 2049 -j ACCEPT
    iptables -A INPUT -i $LAN_INTERFACE -p udp --dport 2049 -j ACCEPT
    
    # AFP (Apple Filing Protocol)
    iptables -A INPUT -i $LAN_INTERFACE -p tcp --dport 548 -j ACCEPT
    
    # mDNS/Bonjour discovery
    iptables -A INPUT -i $LAN_INTERFACE -p udp --dport 5353 -j ACCEPT
    
    # Allow traffic from LAN to WAN (masquerading/NAT)
    iptables -A FORWARD -i $LAN_INTERFACE -o $WAN_INTERFACE -j ACCEPT
    
    # NAT configuration to allow internet access from LAN
    iptables -t nat -A POSTROUTING -o $WAN_INTERFACE -j MASQUERADE
    
    # Cloud service interception rules
    # These rules redirect traffic to the Prometheum proxy
    if [ -f "$CONFIG_DIR/interception_enabled" ]; then
        log "INFO" "Setting up cloud service interception rules..."
        
        # HTTP traffic redirection
        iptables -t nat -A PREROUTING -i $LAN_INTERFACE -p tcp --dport 80 -m mark ! --mark 0x1 -j REDIRECT --to-port 8080
        
        # HTTPS traffic redirection
        iptables -t nat -A PREROUTING -i $LAN_INTERFACE -p tcp --dport 443 -m mark ! --mark 0x1 -j REDIRECT --to-port 8443
        
        # Mark our own traffic to avoid redirect loops
        iptables -t mangle -A OUTPUT -p tcp -m owner --uid-owner prometheum -j MARK --set-mark 0x1
    fi
    
    # Save rules to persist across reboots
    if command -v netfilter-persistent &>/dev/null; then
        netfilter-persistent save
    else
        # Fallback method
        mkdir -p /etc/iptables
        iptables-save > /etc/iptables/rules.v4
        
        # Ensure rules are restored on boot
        cat > /etc/network/if-pre-up.d/iptables <<EOF
#!/bin/sh
/sbin/iptables-restore < /etc/iptables/rules.v4
exit 0
EOF
        chmod +x /etc/network/if-pre-up.d/iptables
    fi
    
    log "SUCCESS" "Firewall and NAT configured"
}

# Set up NAS services (Samba, NFS, AFP)
function setup_nas() {
    log "INFO" "Setting up NAS services..."
    
    # Create share directories if they don't exist
    mkdir -p "$DATA_DIR/shares/public"
    mkdir -p "$DATA_DIR/shares/media"
    mkdir -p "$DATA_DIR/shares/private"
    
    # Set proper permissions
    chmod 0770 "$DATA_DIR/shares/private"
    chmod 0775 "$DATA_DIR/shares/public"
    chmod 0775 "$DATA_DIR/shares/media"
    
    # Setup Samba (SMB/CIFS)
    if [ -z "$(which smbd)" ]; then
        log "INFO" "Installing Samba..."
        apt-get update
        apt-get install -y samba samba-common-bin
    fi
    
    # Create Samba configuration
    cat > /etc/samba/smb.conf <<EOF
[global]
   workgroup = WORKGROUP
   server string = Dismetheum NAS
   netbios name = DISMETHEUM
   security = user
   map to guest = bad user
   dns proxy = no
   # Disable legacy protocols
   client min protocol = SMB2
   server min protocol = SMB2

[public]
   path = $DATA_DIR/shares/public
   browsable = yes
   writable = yes
   guest ok = yes
   read only = no

[media]
   path = $DATA_DIR/shares/media
   browsable = yes
   writable = no
   guest ok = yes
   read only = yes

[private]
   path = $DATA_DIR/shares/private
   browsable = yes
   writable = yes
   guest ok = no
   read only = no
   valid users = @prometheum
EOF
    
    # Setup NFS
    if [ -z "$(which nfsd)" ]; then
        log "INFO" "Installing NFS server..."
        apt-get update
        apt-get install -y nfs-kernel-server
    fi
    
    # Create NFS exports
    cat > /etc/exports <<EOF
$DATA_DIR/shares/public *(rw,sync,no_subtree_check,no_root_squash)
$DATA_DIR/shares/media *(ro,sync,no_subtree_check,no_root_squash)
$DATA_DIR/shares/private 192.168.1.0/24(rw,sync,no_subtree_check)
EOF
    
    # Setup AFP (Apple Filing Protocol)
    if [ -z "$(which afpd)" ]; then
        log "INFO" "Installing AFP server..."
        apt-get update
        apt-get install -y netatalk avahi-daemon
    fi
    
    # Create AFP configuration
    cat > /etc/netatalk/afp.conf <<EOF
[Global]
mimic model = RackMac
hostname = Dismetheum
zeroconf = yes
log file = /var/log/dismetheum/afpd.log

[Public]
path = $DATA_DIR/shares/public
rwlist = @everyone

[Media]
path = $DATA_DIR/shares/media
rolist = @everyone

[Private]
path = $DATA_DIR/shares/private
valid users = prometheum
EOF
    
    # Setup mDNS/Bonjour
    cat > /etc/avahi/services/smb.service <<EOF
<?xml version="1.0" standalone='no'?>
<!DOCTYPE service-group SYSTEM "avahi-service.dtd">
<service-group>
  <name replace-wildcards="yes">%h</name>
  <service>
    <type>_smb._tcp</type>
    <port>445</port>
  </service>
</service-group>
EOF
    
    cat > /etc/avahi/services/afp.service <<EOF
<?xml version="1.0" standalone='no'?>
<!DOCTYPE service-group SYSTEM "avahi-service.dtd">
<service-group>
  <name replace-wildcards="yes">%h</name>
  <service>
    <type>_afpovertcp._tcp</type>
    <port>548</port>
  </service>
</service-group>
EOF
    
    # Create user/group for NAS access
    if ! getent group prometheum >/dev/null; then
        groupadd -r prometheum
    fi
    
    if ! id -u prometheum >/dev/null 2>&1; then
        useradd -r -g prometheum -s /usr/sbin/nologin -d "$DATA_DIR" prometheum
    fi
    
    # Set permissions
    chown -R prometheum:prometheum "$DATA_DIR/shares"
    
    # Create samba user
    (echo "prometheum"; echo "prometheum") | smbpasswd -a -s prometheum
    
    # Restart services
    systemctl restart smbd nmbd nfs-kernel-server netatalk avahi-daemon
    
    log "SUCCESS" "NAS services configured"
}

# Main function
function main() {
    # Create log directory
    mkdir -p "$(dirname "$LOGFILE")"
    touch "$LOGFILE"  # Create log file if it doesn't exist
    
    # Display banner
    log "INFO" "Dismetheum Router Setup"
    
    # Check if running as root
    check_root
    
    # Parse command-line arguments
    parse_args "$@"
    
    # Load configuration
    load_config
    
    # Detect network interfaces
    detect_interfaces
    
    # Set up IP forwarding
    setup_ip_forwarding
    
    # Configure network interfaces
    configure_interfaces
    
    # Set up DHCP server
    setup_dhcp_server
    
    # Set up DNS server
    setup_dns_server
    
    # Set up NAT and firewall
    setup_firewall
    
    # Set up NAS services
    setup_nas
    
    # Configure monitoring
    setup_monitoring
    
    # Validate setup
    validate_setup
    
    # Backup configurations
    backup_configurations
    
    log "SUCCESS" "Dismetheum router setup completed successfully"
}

# Set up system monitoring
function setup_monitoring() {
    log "INFO" "Setting up system monitoring..."
    
    # Install monitoring tools
    apt-get update
    apt-get install -y vnstat iftop htop iotop
    
    # Create monitoring script
    cat > "$CONFIG_DIR/scripts/monitor.sh" <<EOF
#!/bin/bash

# Get network stats
wan_iface="$WAN_INTERFACE"
lan_iface="$LAN_INTERFACE"

# Network throughput
wan_rx=\$(cat /proc/net/dev | grep \$wan_iface | awk '{print \$2}')
wan_tx=\$(cat /proc/net/dev | grep \$wan_iface | awk '{print \$10}')
lan_rx=\$(cat /proc/net/dev | grep \$lan_iface | awk '{print \$2}')
lan_tx=\$(cat /proc/net/dev | grep \$lan_iface | awk '{print \$10}')

# CPU load
cpu_load=\$(uptime | awk -F'load average:' '{print \$2}' | awk -F, '{print \$1}' | tr -d ' ')

# Memory usage
mem_total=\$(free -m | grep Mem | awk '{print \$2}')
mem_used=\$(free -m | grep Mem | awk '{print \$3}')
mem_percent=\$(echo "scale=2; \$mem_used / \$mem_total * 100" | bc)

# Disk usage
disk_total=\$(df -h "$DATA_DIR" | tail -1 | awk '{print \$2}')
disk_used=\$(df -h "$DATA_DIR" | tail -1 | awk '{print \$3}')
disk_percent=\$(df -h "$DATA_DIR" | tail -1 | awk '{print \$5}')

# Output in JSON format
cat << EOJ
{
  "timestamp": "\$(date +%s)",
  "network": {
    "wan": {
      "rx_bytes": \$wan_rx,
      "tx_bytes": \$wan_tx
    },
    "lan": {
      "rx_bytes": \$lan_rx,
      "tx_bytes": \$lan_tx
    }
  },
  "system": {
    "cpu_load": \$cpu_load,
    "memory": {
      "total_mb": \$mem_total,
      "used_mb": \$mem_used,
      "percent": \$mem_percent
    },
    "storage": {
      "total": "\$disk_total",
      "used": "\$disk_used",
      "percent": "\$disk_percent"
    }
  }
}
EOJ
EOF
    
    # Make script executable
    chmod +x "$CONFIG_DIR/scripts/monitor.sh"
    
    # Create a cron job to run monitoring every 5 minutes
    cat > /etc/cron.d/dismetheum-monitor <<EOF
*/5 * * * * root $CONFIG_DIR/scripts/monitor.sh > $DATA_DIR/monitor/latest.json 2>/dev/null
0 * * * * root find $DATA_DIR/monitor -type f -name "*.json" -mtime +7 -delete > /dev/null 2>&1
EOF
    
    # Create monitoring directory
    mkdir -p "$DATA_DIR/monitor"
    
    log "SUCCESS" "System monitoring configured"
}

# Validate the router setup
function validate_setup() {
    log "INFO" "Validating setup..."
    
    local errors=0
    
    # Check if services are running
    local services=("dnsmasq" "smbd" "nfsd" "netatalk" "avahi-daemon")
    for service in "${services[@]}"; do
        if ! systemctl is-active --quiet $service; then
            log "ERROR" "Service $service is not running"
            errors=$((errors + 1))
        fi
    done
    
    # Check if interfaces are properly configured
    if ! ip addr show $LAN_INTERFACE | grep -q "$LAN_IP"; then
        log "ERROR" "LAN interface not properly configured"
        errors=$((errors + 1))
    fi
    
    # Check IP forwarding
    if [ "$(cat /proc/sys/net/ipv4/ip_forward)" != "1" ]; then
        log "ERROR" "IP forwarding is not enabled"
        errors=$((errors + 1))
    fi
    
    # Check if firewall rules exist
    if ! iptables -L -n | grep -q "MASQUERADE"; then
        log "ERROR" "NAT rules not properly configured"
        errors=$((errors + 1))
    fi
    
    # Check if NAS shares are accessible
    if ! grep -q "$DATA_DIR/shares/public" /etc/exports; then
        log "ERROR" "NFS exports not properly configured"
        errors=$((errors + 1))
    fi
    
    if [ $errors -eq 0 ]; then
        log "SUCCESS" "All system checks passed"
    else
        log "WARNING" "$errors issues found during validation"
        if [ "$FORCE" != true ]; then
            log "INFO" "Recommend running with --force to ignore these issues or fix them manually"
        fi
    fi
}

# Backup configuration files
function backup_configurations() {
    log "INFO" "Backing up configuration files..."
    
    local backup_dir="$DATA_DIR/backups/config_$(date +%Y%m%d%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup network configs
    if [ -f /etc/netplan/01-dismetheum.yaml ]; then
        cp /etc/netplan/01-dismetheum.yaml "$backup_dir/"
    fi
    
    if [ -f /etc/network/interfaces.d/dismetheum ]; then
        cp /etc/network/interfaces.d/dismetheum "$backup_dir/"
    fi
    
    # Backup DHCP/DNS configs
    cp -r /etc/dnsmasq.d "$backup_dir/"
    
    # Backup firewall rules
    iptables-save > "$backup_dir/iptables.rules"
    
    # Backup NAS configs
    cp /etc/samba/smb.conf "$backup_dir/"
    cp /etc/exports "$backup_dir/"
    cp /etc/netatalk/afp.conf "$backup_dir/"
    
    # Backup router config
    cp "$CONFIG_DIR/router_config.json" "$backup_dir/"
    
    # Create a dated symlink to the latest backup
    ln -sf "$backup_dir" "$DATA_DIR/backups/latest"
    
    log "SUCCESS" "Configuration files backed up to $backup_dir"
}

# Run the main function with all arguments
main "$@"

