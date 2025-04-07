#!/bin/bash
#
# Prometheum/Dismetheum Installation Script
# This script installs the Prometheum personal cloud storage system or
# the Dismetheum router-based version depending on the installation mode.
#

set -e  # Exit on any error

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
INSTALL_DIR="/opt/prometheum"
CONFIG_DIR="/etc/prometheum"
DATA_DIR="/var/lib/prometheum"
LOG_DIR="/var/log/prometheum"
ROUTER_MODE=false
USER_MODE=false
PYTHON_MIN_VERSION="3.8"
INSTALL_DEPS=true
FORCE_INSTALL=false
GENERATE_SSL=true
DEBUG=false
VERSION="1.0.0"

# Display banner
function show_banner() {
    echo -e "${BLUE}"
    echo "╔═════════════════════════════════════════════════════════════╗"
    echo "║                                                             ║"
    echo "║  █▀█ █▀█ █▀█ █▀▄▀█ █▀▀ ▀█▀ █ █ █▀▀ █ █ █▀▄▀█               ║"
    echo "║  █▀▀ █▀▄ █▄█ █ ▀ █ ██▄  █  █▀█ ██▄ █▄█ █ ▀ █               ║"
    echo "║                                                             ║"
    if [ "$ROUTER_MODE" = true ]; then
        echo "║  ▄▄ █▀▄ █ █▀ █▀▄▀█ █▀▀ ▀█▀ █ █ █▀▀ █ █ █▀▄▀█ (Router Mode) ║"
        echo "║  ── █▄▀ █ ▄█ █ ▀ █ ██▄  █  █▀█ ██▄ █▄█ █ ▀ █             ║"
    fi
    echo "║                                                             ║"
    echo "║  Personal Cloud Storage System                              ║"
    echo "║  Version: $VERSION                                         ║"
    echo "╚═════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Display help message
function show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help             Display this help message"
    echo "  -d, --directory DIR    Set installation directory (default: $INSTALL_DIR)"
    echo "  -c, --config DIR       Set configuration directory (default: $CONFIG_DIR)"
    echo "  --data-dir DIR         Set data directory (default: $DATA_DIR)"
    echo "  --log-dir DIR          Set log directory (default: $LOG_DIR)"
    echo "  -r, --router           Install in router mode (Dismetheum)"
    echo "  -u, --user             Install in user mode (no system service)"
    echo "  -n, --no-deps          Skip dependency installation"
    echo "  -f, --force            Force installation even if requirements not met"
    echo "  --no-ssl               Skip SSL certificate generation"
    echo "  --debug                Enable verbose debugging output"
    echo
    echo "Example:"
    echo "  $0 --router --config /etc/dismetheum"
    echo
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
            -d|--directory)
                INSTALL_DIR="$2"
                shift
                shift
                ;;
            -c|--config)
                CONFIG_DIR="$2"
                shift
                shift
                ;;
            --data-dir)
                DATA_DIR="$2"
                shift
                shift
                ;;
            --log-dir)
                LOG_DIR="$2"
                shift
                shift
                ;;
            -r|--router)
                ROUTER_MODE=true
                shift
                ;;
            -u|--user)
                USER_MODE=true
                shift
                ;;
            -n|--no-deps)
                INSTALL_DEPS=false
                shift
                ;;
            -f|--force)
                FORCE_INSTALL=true
                shift
                ;;
            --no-ssl)
                GENERATE_SSL=false
                shift
                ;;
            --debug)
                DEBUG=true
                shift
                ;;
            *)
                echo -e "${RED}Error: Unknown option $key${NC}"
                show_help
                exit 1
                ;;
        esac
    done

    # Validation
    if [ "$ROUTER_MODE" = true ] && [ "$USER_MODE" = true ]; then
        echo -e "${RED}Error: Cannot use both --router and --user options${NC}"
        exit 1
    fi

    # Update directories if in router mode
    if [ "$ROUTER_MODE" = true ]; then
        CONFIG_DIR="/etc/dismetheum"
        DATA_DIR="/var/lib/dismetheum"
        LOG_DIR="/var/log/dismetheum"
    fi

    # Update directories if in user mode
    if [ "$USER_MODE" = true ]; then
        CONFIG_DIR="$HOME/.prometheum"
        DATA_DIR="$CONFIG_DIR/data"
        LOG_DIR="$CONFIG_DIR/logs"
    fi
}

# Check system requirements
function check_requirements() {
    echo -e "${BLUE}Checking system requirements...${NC}"
    
    # Check if running as root when not in user mode
    if [ "$USER_MODE" = false ] && [ $(id -u) -ne 0 ]; then
        echo -e "${RED}Error: This script must be run as root (use sudo) unless using --user mode${NC}"
        exit 1
    fi

    # Check operating system
    if [ "$ROUTER_MODE" = true ]; then
        if ! grep -qE "Debian|Ubuntu" /etc/os-release 2>/dev/null; then
            echo -e "${RED}Error: Router mode is only supported on Debian/Ubuntu Linux${NC}"
            if [ "$FORCE_INSTALL" = false ]; then
                exit 1
            else
                echo -e "${YELLOW}Warning: Unsupported OS detected, but continuing due to --force option${NC}"
            fi
        fi
    else
        if [ "$(uname)" != "Linux" ] && [ "$(uname)" != "Darwin" ]; then
            echo -e "${RED}Error: Unsupported operating system: $(uname)${NC}"
            if [ "$FORCE_INSTALL" = false ]; then
                exit 1
            else
                echo -e "${YELLOW}Warning: Unsupported OS detected, but continuing due to --force option${NC}"
            fi
        fi
    fi

    # Check Python version
    if ! command -v python3 >/dev/null 2>&1; then
        echo -e "${RED}Error: Python 3 is not installed${NC}"
        exit 1
    fi

    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if ! awk -v ver="$python_version" -v min="$PYTHON_MIN_VERSION" 'BEGIN { exit !(ver >= min) }'; then
        echo -e "${RED}Error: Python $PYTHON_MIN_VERSION or higher is required (found $python_version)${NC}"
        if [ "$FORCE_INSTALL" = false ]; then
            exit 1
        else
            echo -e "${YELLOW}Warning: Unsupported Python version, but continuing due to --force option${NC}"
        fi
    fi

    # Check free disk space (minimum 5GB for router mode, 1GB for standard)
    min_space=1048576  # 1GB in KB
    if [ "$ROUTER_MODE" = true ]; then
        min_space=5242880  # 5GB in KB
    fi

    free_space=$(df -k "$INSTALL_DIR" 2>/dev/null | tail -1 | awk '{print $4}')
    if [ -z "$free_space" ] || [ "$free_space" -lt "$min_space" ]; then
        echo -e "${RED}Error: Insufficient disk space. Required: $(($min_space/1024))MB, Available: $(($free_space/1024))MB${NC}"
        if [ "$FORCE_INSTALL" = false ]; then
            exit 1
        else
            echo -e "${YELLOW}Warning: Insufficient disk space, but continuing due to --force option${NC}"
        fi
    fi

    # Extra checks for router mode
    if [ "$ROUTER_MODE" = true ]; then
        # Check if at least 2 network interfaces are available
        if [ $(ip -o link show | grep -v "lo" | wc -l) -lt 2 ]; then
            echo -e "${RED}Error: Router mode requires at least 2 network interfaces${NC}"
            if [ "$FORCE_INSTALL" = false ]; then
                exit 1
            else
                echo -e "${YELLOW}Warning: Insufficient network interfaces, but continuing due to --force option${NC}"
            fi
        fi
    fi

    echo -e "${GREEN}All system requirements met.${NC}"
}

# Install dependencies
function install_dependencies() {
    if [ "$INSTALL_DEPS" = false ]; then
        echo -e "${YELLOW}Skipping dependency installation as requested.${NC}"
        return
    fi

    echo -e "${BLUE}Installing required dependencies...${NC}"
    
    # General dependencies for all modes
    local packages="python3 python3-pip python3-venv python3-dev build-essential openssl libssl-dev"
    
    # Additional dependencies for router mode
    if [ "$ROUTER_MODE" = true ]; then
        packages="$packages iptables dnsmasq hostapd bridge-utils iproute2 isc-dhcp-server nftables"
        packages="$packages samba nfs-kernel-server netatalk ufw sqlite3 avahi-daemon netfilter-persistent"
    else
        packages="$packages sqlite3 avahi-daemon"
    fi

    # Install packages based on OS
    if [ "$(uname)" = "Linux" ]; then
        if command -v apt-get >/dev/null 2>&1; then
            # Debian/Ubuntu
            apt-get update
            apt-get install -y $packages
        elif command -v dnf >/dev/null 2>&1; then
            # Fedora/RHEL/CentOS
            dnf install -y $packages
        elif command -v pacman >/dev/null 2>&1; then
            # Arch Linux
            pacman -Sy --noconfirm $packages
        else
            echo -e "${YELLOW}Warning: Unsupported package manager. Please install required packages manually.${NC}"
            echo "Required packages: $packages"
        fi
    elif [ "$(uname)" = "Darwin" ]; then
        # macOS using Homebrew
        if ! command -v brew >/dev/null 2>&1; then
            echo -e "${RED}Error: Homebrew is required for macOS installation${NC}"
            echo "Install Homebrew from https://brew.sh/"
            exit 1
        fi
        brew update
        brew install openssl python3 sqlite
    fi

    # Install Python packages
    echo -e "${BLUE}Installing Python dependencies...${NC}"
    python3 -m pip install --upgrade pip
    python3 -m pip install virtualenv

    echo -e "${GREEN}Dependencies installed successfully.${NC}"
}

# Create directories
function create_directories() {
    echo -e "${BLUE}Creating required directories...${NC}"
    
    # Create the main directories
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$DATA_DIR"
    mkdir -p "$LOG_DIR"
    
    # Create sub-directories
    if [ "$ROUTER_MODE" = true ]; then
        # Router-specific directories
        mkdir -p "$DATA_DIR/shares/public"
        mkdir -p "$DATA_DIR/shares/media"
        mkdir -p "$DATA_DIR/shares/private"
        mkdir -p "$DATA_DIR/dhcp"
        mkdir -p "$DATA_DIR/dns"
        mkdir -p "$CONFIG_DIR/network"
    fi
    
    # Common sub-directories
    mkdir -p "$DATA_DIR/db"
    mkdir -p "$CONFIG_DIR/ssl"
    mkdir -p "$DATA_DIR/backups"
    mkdir -p "$DATA_DIR/temp"
    
    # Set appropriate permissions
    if [ "$USER_MODE" = false ]; then
        chown -R root:root "$INSTALL_DIR" "$CONFIG_DIR"
        chmod -R 755 "$INSTALL_DIR" "$CONFIG_DIR"
        
        # Make sure data directory is accessible
        chmod -R 770 "$DATA_DIR"
        
        # Create a prometheum group if it doesn't exist
        if ! getent group prometheum >/dev/null; then
            groupadd -r prometheum
        fi
        
        # Change group ownership of data directory
        chgrp -R prometheum "$DATA_DIR"
        chgrp -R prometheum "$LOG_DIR"
    fi
    
    echo -e "${GREEN}Directories created successfully.${NC}"
}

# Generate SSL certificates
function generate_ssl_certificates() {
    if [ "$GENERATE_SSL" = false ]; then
        echo -e "${YELLOW}Skipping SSL certificate generation as requested.${NC}"
        return
    fi

    echo -e "${BLUE}Generating SSL certificates...${NC}"
    
    SSL_DIR="$CONFIG_DIR/ssl"
    SSL_CERT="$SSL_DIR/cert.pem"
    SSL_KEY="$SSL_DIR/key.pem"
    
    # Check if certificates already exist
    if [ -f "$SSL_CERT" ] && [ -f "$SSL_KEY" ]; then
        echo -e "${YELLOW}SSL certificates already exist. Use --force to regenerate.${NC}"
        if [ "$FORCE_INSTALL" = false ]; then
            return
        fi
    fi
    
    # Generate self-signed certificate
    hostname=$(hostname)
    openssl req -x509 -newkey rsa:4096 -keyout "$SSL_KEY" -out "$SSL_CERT" -days 365 -nodes \
        -subj "/CN=$hostname" -addext "subjectAltName=DNS:$hostname,DNS:localhost,IP:127.0.0.1"
    
    # Set proper permissions
    chmod 600 "$SSL_KEY"
    chmod 644 "$SSL_CERT"
    
    echo -e "${GREEN}SSL certificates generated successfully.${NC}"
}

# Initialize database
function initialize_database() {
    echo -e "${BLUE}Initializing database...${NC}"
    
    DB_PATH="$DATA_DIR/db/prometheum.db"
    
    # Create database directory if it doesn't exist
    mkdir -p "$(dirname "$DB_PATH")"
    
    # Create database schema using SQLite
    sqlite3 "$DB_PATH" <<EOF
    -- Create users table
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        email TEXT UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP,
        is_admin BOOLEAN DEFAULT FALSE
    );

    -- Create devices table
    CREATE TABLE IF NOT EXISTS devices (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        os_type TEXT,
        owner_id TEXT NOT NULL,
        public_key TEXT,
        registered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_seen TIMESTAMP,
        is_trusted BOOLEAN DEFAULT FALSE,
        status TEXT DEFAULT 'active',
        FOREIGN KEY (owner_id) REFERENCES users(id)
    );

    -- Create files table
    CREATE TABLE IF NOT EXISTS files (
        id TEXT PRIMARY KEY,
        path TEXT NOT NULL,
        size INTEGER NOT NULL,
        checksum TEXT,
        mime_type TEXT,
        owner_id TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        modified_at TIMESTAMP,
        is_encrypted BOOLEAN DEFAULT FALSE,
        FOREIGN KEY (owner_id) REFERENCES users(id)
    );

    -- Create file_versions table
    CREATE TABLE IF NOT EXISTS file_versions (
        id TEXT PRIMARY KEY,
        file_id TEXT NOT NULL,
        version_number INTEGER NOT NULL,
        path TEXT NOT NULL,
        size INTEGER NOT NULL,
        checksum TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (file_id) REFERENCES files(id)
    );

    -- Create sync_directories table
    CREATE TABLE IF NOT EXISTS sync_directories (
        id TEXT PRIMARY KEY,
        local_path TEXT NOT NULL,
        remote_path TEXT,
        owner_id TEXT NOT NULL,
        sync_policy TEXT DEFAULT 'two-way',
        auto_backup BOOLEAN DEFAULT FALSE,
        last_sync TIMESTAMP,
        status TEXT DEFAULT 'active',
        FOREIGN KEY (owner_id) REFERENCES users(id)
    );

    -- Create shares table
    CREATE TABLE IF NOT EXISTS shares (
        id TEXT PRIMARY KEY,
        file_id TEXT NOT NULL,
        owner_id TEXT NOT NULL,
        recipient_id TEXT NOT NULL,
        permissions TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP,
        FOREIGN KEY (file_id) REFERENCES files(id),
        FOREIGN KEY (owner_id) REFERENCES users(id),
        FOREIGN KEY (recipient_id) REFERENCES users(id)
    );

    -- Create cloud_auth table
    CREATE TABLE IF NOT EXISTS cloud_auth (
        id TEXT PRIMARY KEY,
        service TEXT NOT NULL,
        user_id TEXT NOT NULL,
        auth_data TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP,
        status TEXT DEFAULT 'active',
        FOREIGN KEY (user_id) REFERENCES users(id)
    );

    -- Create interception_rules table
    CREATE TABLE IF NOT EXISTS interception_rules (
        id TEXT PRIMARY KEY,
        service TEXT NOT NULL,
        pattern TEXT NOT NULL,
        action TEXT NOT NULL,
        enabled BOOLEAN DEFAULT TRUE,
        priority INTEGER DEFAULT 100,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
EOF

    # Set proper permissions
    if [ "$USER_MODE" = false ]; then
        chown prometheum:prometheum "$DB_PATH"
        chmod 600 "$DB_PATH"
    fi

    # Create default admin user if router mode is enabled
    if [ "$ROUTER_MODE" = true ]; then
        # Generate random password for first-time setup
        local ADMIN_PASSWORD=$(openssl rand -base64 12)
        local ADMIN_PASSWORD_HASH=$(echo -n "$ADMIN_PASSWORD" | sha256sum | awk '{print $1}')
        
        # Insert admin user
        sqlite3 "$DB_PATH" <<EOF
        INSERT OR IGNORE INTO users (id, username, password_hash, email, is_admin)
        VALUES ('admin-$(uuidgen)', 'admin', '$ADMIN_PASSWORD_HASH', 'admin@localhost', 1);
EOF
        
        echo -e "${YELLOW}Default admin credentials:${NC}"
        echo -e "  Username: ${GREEN}admin${NC}"
        echo -e "  Password: ${GREEN}$ADMIN_PASSWORD${NC}"
        echo -e "${YELLOW}Please change this password after first login!${NC}"
    fi

    echo -e "${GREEN}Database initialized successfully.${NC}"
}

# Configure system service
function setup_system_service() {
    if [ "$USER_MODE" = true ]; then
        echo -e "${YELLOW}Skipping system service setup in user mode.${NC}"
        return
    fi

    echo -e "${BLUE}Setting up system service...${NC}"
    
    # Determine which service file to use
    local SERVICE_NAME="prometheum"
    local SERVICE_FILE="systemd/prometheum.service"
    
    if [ "$ROUTER_MODE" = true ]; then
        SERVICE_NAME="dismetheum"
        SERVICE_FILE="systemd/dismetheum.service"
    fi
    
    # Create prometheum user if it doesn't exist
    if ! id -u prometheum &>/dev/null; then
        useradd -r -s /bin/false -d "$DATA_DIR" -M prometheum
    fi
    
    # Copy service file to systemd directory
    cp "$SERVICE_FILE" "/etc/systemd/system/${SERVICE_NAME}.service"
    
    # Update paths in service file if installation paths were customized
    sed -i "s|/opt/prometheum|$INSTALL_DIR|g" "/etc/systemd/system/${SERVICE_NAME}.service"
    
    if [ "$ROUTER_MODE" = true ]; then
        sed -i "s|/etc/dismetheum|$CONFIG_DIR|g" "/etc/systemd/system/${SERVICE_NAME}.service"
        sed -i "s|/var/lib/dismetheum|$DATA_DIR|g" "/etc/systemd/system/${SERVICE_NAME}.service"
        sed -i "s|/var/log/dismetheum|$LOG_DIR|g" "/etc/systemd/system/${SERVICE_NAME}.service"
    else
        sed -i "s|/etc/prometheum|$CONFIG_DIR|g" "/etc/systemd/system/${SERVICE_NAME}.service"
        sed -i "s|/var/lib/prometheum|$DATA_DIR|g" "/etc/systemd/system/${SERVICE_NAME}.service"
        sed -i "s|/var/log/prometheum|$LOG_DIR|g" "/etc/systemd/system/${SERVICE_NAME}.service"
    fi
    
    # Reload systemd to recognize the new service
    systemctl daemon-reload
    
    # Enable and start the service
    systemctl enable "$SERVICE_NAME"
    
    # Start the service if requested
    echo -e "${YELLOW}To start the service, run:${NC}"
    echo -e "  ${GREEN}sudo systemctl start $SERVICE_NAME${NC}"
    
    echo -e "${GREEN}System service setup completed.${NC}"
}

# Main function
function main() {
    # Display banner
    show_banner
    
    # Parse command line arguments
    parse_args "$@"
    
    # Check system requirements
    check_requirements
    
    # Install dependencies
    install_dependencies
    
    # Create directories
    create_directories
    
    # Generate SSL certificates
    generate_ssl_certificates
    
    # Initialize database
    initialize_database
    
    # Configure system service
    setup_system_service
    
    # Router-specific setup
    if [ "$ROUTER_MODE" = true ]; then
        echo -e "${BLUE}Configuring router functionality...${NC}"
        echo -e "${YELLOW}Run the router setup script with:${NC}"
        echo -e "  ${GREEN}sudo $INSTALL_DIR/scripts/router_setup.sh${NC}"
    fi
    
    echo -e "${GREEN}Prometheum installed successfully!${NC}"
    echo -e "${BLUE}Open your browser and go to http://localhost:8080${NC}"
    
    if [ "$ROUTER_MODE" = true ]; then
        echo -e "${BLUE}For router configuration, go to http://$LAN_IP:8080${NC}"
    fi
}

# Run the main function with all arguments
main "$@"

