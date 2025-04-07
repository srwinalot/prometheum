# Dismetheum - Router-based Personal Cloud Storage

Dismetheum is a router-based variant of Prometheum that combines NAS capabilities with cloud service interception, providing a seamless self-hosted cloud storage solution.

## Features

- **Router functionality** with integrated DHCP, DNS, and firewall management
- **NAS capabilities** with support for SMB, NFS, and AFP protocols
- **Transparent cloud service interception** for iCloud, Google Drive, OneDrive, and Dropbox
- **Automatic device synchronization** across your local network
- **Multi-protocol file sharing** for cross-platform compatibility
- **Secure data storage** with end-to-end encryption
- **Web-based management interface** for easy administration
- **Storage pooling** to combine multiple drives into a unified storage solution
- **Versioning and backup** for critical data protection
- **Access control** with user and group permissions

## Hardware Requirements

- **CPU**: Dual-core 1.5GHz+ (x86_64 or ARM64)
- **RAM**: 2GB minimum, 4GB+ recommended
- **Storage**: 5GB+ for system, additional storage for data
- **Network**: Minimum 2 physical network interfaces
- **Compatible OS**: Debian/Ubuntu Linux

## Installation

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/prometheum.git
cd prometheum

# Run the installation script in router mode
sudo ./scripts/install.sh --router

# Run router setup (configures network interfaces, DHCP, DNS, etc.)
sudo ./scripts/router_setup.sh
```

### Advanced Installation Options

```bash
# Custom installation directories
sudo ./scripts/install.sh --router --directory /opt/custom --config /etc/custom

# Skip dependency installation (if you've installed them manually)
sudo ./scripts/install.sh --router --no-deps

# Configure specific network interfaces
sudo ./scripts/router_setup.sh --wan eth0 --lan eth1 --ip 192.168.2.1
```

## Network Setup

After installation, Dismetheum will configure:

1. **WAN Interface**: Connected to your internet source (cable modem, existing router)
2. **LAN Interface**: For your local devices to connect to Dismetheum
3. **DHCP Server**: Assigns IP addresses to your devices (default range: 192.168.1.100-200)
4. **DNS Server**: Handles name resolution and cloud service interception
5. **Firewall**: Protects your network with secure default rules

### Physical Setup

```
Internet Source (Modem/Router) <---> Dismetheum WAN port
                                     Dismetheum LAN port <---> Your Devices
```

For best results, configure your internet modem to "bridge mode" if possible.

## Configuration

### Web Interface

The router can be configured through the web interface:
- URL: `http://<router-ip>:8080` (default: http://192.168.1.1:8080)
- Default credentials:
  - Username: `admin`
  - Password: (shown during installation)

### SSH Access

SSH access is enabled by default for advanced configuration:
```bash
ssh admin@<router-ip>
```

### File Sharing Access

- **SMB/CIFS (Windows)**: `\\<router-ip>\share`
- **NFS (Linux)**: `mount <router-ip>:/var/lib/dismetheum/shares/public /mnt`
- **AFP (macOS)**: Connect via Finder → Go → Connect to Server → `afp://<router-ip>/`

## Cloud Service Interception

Dismetheum can intercept and store data from popular cloud services:

1. In the web interface, navigate to Settings → Cloud Services
2. Enable interception for specific services
3. Configure any specific interception rules
4. Your devices will continue to use their normal cloud apps, but data will be stored locally

## Troubleshooting

### Network Issues

- **No internet connection**: Verify WAN interface is properly connected
- **Can't access web interface**: Check LAN IP configuration
- **DHCP not working**: Restart the dnsmasq service (`sudo systemctl restart dnsmasq`)

### Storage Issues

- **Can't access shared folders**: Verify SMB/NFS services are running
- **Permission denied**: Check user permissions in the web interface
- **Disk full errors**: Add additional storage or clean up unused files

### Service Logs

Log files are located in `/var/log/dismetheum/`:
```bash
# View main service logs
sudo tail -f /var/log/dismetheum/service.log

# View DHCP/DNS logs
sudo tail -f /var/log/dismetheum/dnsmasq.log
```

## Security Considerations

- **Change default passwords** immediately after installation
- **Keep the system updated** regularly
- **Review firewall rules** to ensure they meet your security requirements
- **Enable HTTPS** for the web interface in production environments
- **Configure automatic backups** for important data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# Prometheum

A powerful and flexible self-hosted cloud storage solution that transforms your own device into a personal iCloud-like system. Prometheum gives you complete control over your data while providing the convenience of commercial cloud services.

## Features

- **Seamless File Synchronization** across all your devices
- **Secure Device Management** with easy addition and removal of trusted devices
- **Automated Backup System** with configurable retention policies
- **Media Organization** with photo/video libraries and smart albums
- **End-to-End Encryption** for maximum data privacy
- **Sharing Controls** to securely share files with others

## Installation

```bash
pip install prometheum
```

## Quick Start

```python
from prometheum import cloud_manager

# Initialize your personal cloud
manager = cloud_manager.setup(storage_path="/path/to/storage")

# Add a sync directory
manager.add_sync_directory(
    local_path="/Users/username/Documents",
    sync_policy="two-way",
    auto_backup=True
)

# Start the sync service
manager.start_sync_service()
```

## Documentation

Full documentation can be found in the `docs` directory or at [documentation URL].

## Why Self-Hosted Cloud?

- **Complete Privacy**: Your data never leaves your control
- **No Subscription Fees**: Pay once for the software, use your own hardware
- **Unlimited Storage**: Only limited by your own hardware
- **Customizable**: Adapt to your specific needs and workflow
- **Open Source**: Community-driven development and security auditing

## System Requirements

- **Server Device**: Any computer that can run Python 3.8+ (Windows, macOS, Linux)
- **Storage**: Sufficient disk space for your files (external drives supported)
- **Network**: Local network for home use, or public IP/domain for remote access
- **Client Devices**: Any device that supports standard protocols (WebDAV, CalDAV, etc.)

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone git@github.com:username/prometheum.git
cd prometheum

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

## License

[License information]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
