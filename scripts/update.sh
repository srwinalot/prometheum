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
  if [ "$BACKUP_CREATED" = true ]; then
    log "Attempting to restore from backup..."
    restore_from_backup
  fi
  exit 1
}

warn() {
  echo -e "${YELLOW}[WARNING]${NC} $1"
}

status() {
  echo -e "${BLUE}[STATUS]${NC} $1"
}

# Initialize variables
BACKUP_CREATED=false
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="/var/lib/prometheum/backup/update_$TIMESTAMP"

# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
  error "This script must be run as root"
fi

# Change to script directory
cd "$(dirname "$0")/.." || error "Failed to change to project directory"
PROJECT_ROOT="$(pwd)"

# Display welcome message
log "Welcome to the Prometheum NAS Router OS updater"
log "This script will update your Prometheum installation"
echo ""

# Check if Prometheum is installed
status "Checking current installation..."
if [ ! -d "/opt/prometheum" ] || [ ! -d "/etc/prometheum" ]; then
  error "Prometheum installation not found. Please run install.sh first"
fi

# Function to create backup
create_backup() {
  status "Creating backup of current installation..."
  
  # Create backup directory
  mkdir -p "$BACKUP_DIR"
  mkdir -p "$BACKUP_DIR/opt"
  mkdir -p "$BACKUP_DIR/etc"
  
  # Backup application files
  cp -r /opt/prometheum "$BACKUP_DIR/opt/" || warn "Failed to backup application files"
  
  # Backup configuration files
  cp -r /etc/prometheum "$BACKUP_DIR/etc/" || warn "Failed to backup configuration files"
  
  # Create backup info file
  cat > "$BACKUP_DIR/backup_info.txt" << EOF
Backup created on: $(date)
Hostname: $(hostname)
System: $(uname -a)
Prometheum version: $(cat /opt/prometheum/src/VERSION 2>/dev/null || echo "Unknown")
EOF
  
  log "Backup created at $BACKUP_DIR"
  BACKUP_CREATED=true
}

# Function to restore from backup
restore_from_backup() {
  if [ ! -d "$BACKUP_DIR" ]; then
    error "Backup directory not found. Cannot restore"
  fi
  
  status "Restoring from backup..."
  
  # Stop services
  systemctl stop prometheum-health.service prometheum-api.service prometheum-storage.service || warn "Failed to stop some services"
  
  # Restore application files
  if [ -d "$BACKUP_DIR/opt/prometheum" ]; then
    rm -rf /opt/prometheum
    cp -r "$BACKUP_DIR/opt/prometheum" /opt/ || warn "Failed to restore application files"
  fi
  
  # Restore configuration files
  if [ -d "$BACKUP_DIR/etc/prometheum" ]; then
    rm -rf /etc/prometheum
    cp -r "$BACKUP_DIR/etc/prometheum" /etc/ || warn "Failed to restore configuration files"
  fi
  
  # Restart services
  systemctl start prometheum-storage.service || warn "Failed to start storage service"
  sleep 3
  systemctl start prometheum-api.service || warn "Failed to start API service"
  sleep 2
  systemctl start prometheum-health.service || warn "Failed to start health service"
  
  log "Restoration from backup completed"
}

# Create backup
create_backup

# Stop services in reverse order
status "Stopping services..."
systemctl stop prometheum-health.service || warn "Failed to stop health service"
systemctl stop prometheum-api.service || warn "Failed to stop API service"
systemctl stop prometheum-storage.service || warn "Failed to stop storage service"

# Update application files
status "Updating application files..."
rm -rf /opt/prometheum/src
cp -r "$PROJECT_ROOT/src" /opt/prometheum/ || error "Failed to copy source files"
cp -r "$PROJECT_ROOT/resources" /opt/prometheum/ 2>/dev/null || warn "No resources directory found"
cp -r "$PROJECT_ROOT/README.md" /opt/prometheum/ 2>/dev/null || warn "No README.md found"

# Update systemd service files
status "Updating systemd service files..."
cp "$PROJECT_ROOT/systemd/"*.service /etc/systemd/system/ || warn "Failed to update some service files"
systemctl daemon-reload

# Update Python dependencies
status "Updating Python dependencies..."
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
  pip3 install --upgrade -r "$PROJECT_ROOT/requirements.txt" || error "Failed to update Python dependencies"
else
  warn "requirements.txt not found, skipping dependency update"
fi

# Handle database migrations
status "Running database migrations..."
if [ -f "/opt/prometheum/src/db/migrations.py" ]; then
  cd /opt/prometheum
  python3 -m src.db.migrations || error "Database migration failed"
  log "Database migrations completed successfully"
else
  warn "No migrations found, skipping"
fi

# Set proper permissions
status "Updating permissions..."
chown -R prometheum:prometheum /opt/prometheum
chown -R prometheum:prometheum /var/log/prometheum
chown -R prometheum:prometheum /var/run/prometheum
chown -R prometheum:prometheum /etc/prometheum

# Make the storage directory accessible to the root user (for the storage service)
chown root:root /var/lib/prometheum/storage
chmod 755 /var/lib/prometheum/storage

# Make configuration files read-only except for root
chmod 640 /etc/prometheum/*.env

# Start services in order
status "Starting services..."
systemctl start prometheum-storage.service || error "Failed to start storage service"
sleep 3
systemctl start prometheum-api.service || error "Failed to start API service"
sleep 2
systemctl start prometheum-health.service || error "Failed to start health service"

# Verify services are running
status "Verifying services..."
sleep 5

if ! systemctl is-active --quiet prometheum-storage.service; then
  error "Storage service failed to start properly"
fi

if ! systemctl is-active --quiet prometheum-api.service; then
  error "API service failed to start properly"
fi

if ! systemctl is-active --quiet prometheum-health.service; then
  warn "Health service failed to start properly"
fi

log "Update completed successfully!"
log "Backup saved at: $BACKUP_DIR"
log "To check status, run:"
echo "  systemctl status prometheum-api.service"
echo ""
log "View logs with:"
echo "  journalctl -u prometheum-api.service"
echo ""
log "Thank you for updating Prometheum NAS Router OS!"

