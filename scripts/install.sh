#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log functions
log() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

error() {
  echo -e "${RED}[ERROR]${NC} $1"
  exit 1
}

warn() {
  echo -e "${YELLOW}[WARNING]${NC} $1"
}

status() {
  echo -e "${BLUE}[STATUS]${NC} $1"
}

# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
  error "This script must be run as root"
fi

# Change to script directory
cd "$(dirname "$0")/.." || error "Failed to change to project directory"
PROJECT_ROOT="$(pwd)"

# Display welcome message
log "Welcome to the Prometheum NAS Router OS installer"
log "This script will install Prometheum on your system"
echo ""

# Check system requirements
status "Checking system requirements..."

# Check Linux distribution
if [ ! -f /etc/os-release ]; then
  error "Could not determine Linux distribution"
fi

source /etc/os-release
log "Detected OS: $NAME $VERSION_ID"

# Check for Python 3.9+
if ! command -v python3 &> /dev/null; then
  error "Python 3 not found. Please install Python 3.9 or later"
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
  error "Python 3.9 or later is required, found $PYTHON_VERSION"
fi

log "Python version $PYTHON_VERSION - OK"

# Check for required packages
for pkg in pip systemd postgresql openssl; do
  if ! command -v $pkg &> /dev/null; then
    warn "$pkg not found. Some features may not work properly"
  else
    log "$pkg - OK"
  fi
done

# Create prometheum user and group
status "Setting up user and groups..."
if ! getent group prometheum &> /dev/null; then
  groupadd --system prometheum
  log "Created prometheum group"
else
  log "Group prometheum already exists"
fi

if ! getent passwd prometheum &> /dev/null; then
  useradd --system --gid prometheum --no-create-home --shell /sbin/nologin prometheum
  log "Created prometheum user"
else
  log "User prometheum already exists"
fi

# Create directory structure
status "Creating directory structure..."
mkdir -p /opt/prometheum
mkdir -p /etc/prometheum
mkdir -p /var/lib/prometheum/{storage,backup,data}
mkdir -p /var/log/prometheum
mkdir -p /var/run/prometheum

# Copy application files
status "Installing application files..."
cp -r "$PROJECT_ROOT/src" /opt/prometheum/
cp -r "$PROJECT_ROOT/resources" /opt/prometheum/ 2>/dev/null || true
cp -r "$PROJECT_ROOT/README.md" /opt/prometheum/ 2>/dev/null || true

# Create initial configuration files
status "Creating configuration files..."
if [ ! -f /etc/prometheum/api.env ]; then
  cat > /etc/prometheum/api.env << EOF
# Prometheum API Configuration
LOG_LEVEL=INFO
DEBUG=false
ALLOWED_ORIGINS=http://localhost:8000
DATABASE_URL=postgresql://postgres:postgres@localhost/prometheum
JWT_SECRET_KEY=change_this_to_a_secure_random_string
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
EOF
  log "Created API configuration file"
else
  log "API configuration file already exists, not overwriting"
fi

if [ ! -f /etc/prometheum/storage.env ]; then
  cat > /etc/prometheum/storage.env << EOF
# Prometheum Storage Configuration
LOG_LEVEL=INFO
STORAGE_PATH=/var/lib/prometheum/storage
BACKUP_PATH=/var/lib/prometheum/backup
ENABLE_AUTO_MOUNT=true
EOF
  log "Created Storage configuration file"
else
  log "Storage configuration file already exists, not overwriting"
fi

if [ ! -f /etc/prometheum/health.env ]; then
  cat > /etc/prometheum/health.env << EOF
# Prometheum Health Configuration
LOG_LEVEL=INFO
CHECK_INTERVAL=60
NOTIFY_EMAIL=admin@example.com
EOF
  log "Created Health configuration file"
else
  log "Health configuration file already exists, not overwriting"
fi

# Install systemd services
status "Installing systemd services..."
cp "$PROJECT_ROOT/systemd/"*.service /etc/systemd/system/
systemctl daemon-reload

# Set proper permissions
status "Setting permissions..."
chown -R prometheum:prometheum /opt/prometheum
chown -R prometheum:prometheum /var/lib/prometheum
chown -R prometheum:prometheum /var/log/prometheum
chown -R prometheum:prometheum /var/run/prometheum
chown -R prometheum:prometheum /etc/prometheum

chmod 750 /opt/prometheum
chmod 750 /var/lib/prometheum
chmod 750 /var/log/prometheum
chmod 750 /var/run/prometheum
chmod 750 /etc/prometheum

# Make the storage directory accessible to the root user (for the storage service)
chown root:root /var/lib/prometheum/storage
chmod 755 /var/lib/prometheum/storage

# Make configuration files read-only except for root
chmod 640 /etc/prometheum/*.env

# Create a random JWT secret key
JWT_SECRET=$(openssl rand -hex 32)
sed -i "s/change_this_to_a_secure_random_string/$JWT_SECRET/" /etc/prometheum/api.env

# Install Python dependencies
status "Installing Python dependencies..."
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
  pip3 install -r "$PROJECT_ROOT/requirements.txt"
else
  warn "requirements.txt not found, skipping dependency installation"
fi

log "Installation complete!"
log "To start the services, run:"
echo "  systemctl enable --now prometheum-storage.service"
echo "  systemctl enable --now prometheum-api.service"
echo "  systemctl enable --now prometheum-health.service"
echo ""
log "To check status, run:"
echo "  systemctl status prometheum-api.service"
echo ""
log "View logs with:"
echo "  journalctl -u prometheum-api.service"
echo ""
log "Thank you for installing Prometheum NAS Router OS!"

