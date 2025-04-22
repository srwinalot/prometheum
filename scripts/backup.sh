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

# Default settings
BACKUP_ROOT="/var/lib/prometheum/backup"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_TYPE="full"
MAX_BACKUPS=10
COMPRESS=true
VERIFY=true
BACKUP_DB=true
QUIET=false
RESTORE_MODE=false
RESTORE_ID=""
INCREMENTAL_BASE=""

# Print usage
usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Backup and restore utility for Prometheum NAS Router OS"
  echo ""
  echo "Options:"
  echo "  -t, --type TYPE       Backup type: full or incremental (default: full)"
  echo "  -b, --base ID         Base backup ID for incremental backups"
  echo "  -d, --dir DIR         Backup directory (default: $BACKUP_ROOT)"
  echo "  -m, --max NUM         Maximum number of backups to keep (default: 10)"
  echo "  -n, --no-compress     Don't compress backup files"
  echo "  -s, --skip-verify     Skip backup verification"
  echo "  -D, --skip-database   Skip database backup"
  echo "  -q, --quiet           Quiet mode (minimal output)"
  echo "  -r, --restore ID      Restore from backup with given ID"
  echo "  -l, --list            List available backups"
  echo "  -h, --help            Show this help message"
  echo ""
  echo "Examples:"
  echo "  $0 --type full                     # Create a full backup"
  echo "  $0 --type incremental --base ID    # Create an incremental backup"
  echo "  $0 --restore ID                    # Restore from a backup"
  echo "  $0 --list                          # List available backups"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--type)
      BACKUP_TYPE="$2"
      shift 2
      ;;
    -b|--base)
      INCREMENTAL_BASE="$2"
      shift 2
      ;;
    -d|--dir)
      BACKUP_ROOT="$2"
      shift 2
      ;;
    -m|--max)
      MAX_BACKUPS="$2"
      shift 2
      ;;
    -n|--no-compress)
      COMPRESS=false
      shift
      ;;
    -s|--skip-verify)
      VERIFY=false
      shift
      ;;
    -D|--skip-database)
      BACKUP_DB=false
      shift
      ;;
    -q|--quiet)
      QUIET=true
      shift
      ;;
    -r|--restore)
      RESTORE_MODE=true
      RESTORE_ID="$2"
      shift 2
      ;;
    -l|--list)
      LIST_MODE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      warn "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
  error "This script must be run as root"
fi

# Check backup type
if [ "$BACKUP_TYPE" != "full" ] && [ "$BACKUP_TYPE" != "incremental" ]; then
  error "Invalid backup type: $BACKUP_TYPE. Must be 'full' or 'incremental'"
fi

# Check if incremental backup has a base
if [ "$BACKUP_TYPE" = "incremental" ] && [ -z "$INCREMENTAL_BASE" ]; then
  error "Incremental backup requires a base backup ID (--base ID)"
fi

# Function to list available backups
list_backups() {
  if [ ! -d "$BACKUP_ROOT" ]; then
    log "No backups found in $BACKUP_ROOT"
    return
  fi
  
  log "Available backups in $BACKUP_ROOT:"
  echo "-----------------------------------"
  echo "ID                 | Type       | Date                | Size    "
  echo "-------------------+------------+---------------------+---------"
  
  for backup_dir in "$BACKUP_ROOT"/prometheum_*; do
    if [ -d "$backup_dir" ]; then
      backup_id=$(basename "$backup_dir" | cut -d'_' -f2-)
      
      # Get backup type
      if [ -f "$backup_dir/backup_info.txt" ]; then
        backup_type=$(grep "Backup type:" "$backup_dir/backup_info.txt" | cut -d':' -f2 | tr -d ' ' || echo "Unknown")
      else
        backup_type="Unknown"
      fi
      
      # Get backup date
      if [ -f "$backup_dir/backup_info.txt" ]; then
        backup_date=$(grep "Backup created on:" "$backup_dir/backup_info.txt" | cut -d':' -f2- || echo "Unknown")
      else
        backup_date=$(stat -c %y "$backup_dir" 2>/dev/null || date -r "$backup_dir" 2>/dev/null || echo "Unknown")
      fi
      
      # Get backup size
      backup_size=$(du -sh "$backup_dir" | cut -f1)
      
      printf "%-19s | %-10s | %-19s | %-8s\n" "$backup_id" "$backup_type" "$backup_date" "$backup_size"
    fi
  done
  
  echo "-----------------------------------"
}

# Function to validate a backup
validate_backup() {
  local backup_dir="$1"
  status "Validating backup..."
  
  # Check basic backup structure
  if [ ! -f "$backup_dir/backup_info.txt" ]; then
    warn "Backup validation failed: Missing backup_info.txt"
    return 1
  fi
  
  if [ ! -d "$backup_dir/etc" ]; then
    warn "Backup validation failed: Missing configuration files"
    return 1
  fi
  
  if [ ! -d "$backup_dir/data" ]; then
    warn "Backup validation failed: Missing data directory"
    return 1
  fi
  
  # Check backup info file
  if ! grep -q "Backup created on:" "$backup_dir/backup_info.txt"; then
    warn "Backup validation failed: Invalid backup_info.txt"
    return 1
  fi
  
  # Check config files
  if [ ! -d "$backup_dir/etc/prometheum" ]; then
    warn "Backup validation failed: Missing Prometheum configuration"
    return 1
  fi
  
  # Verify database backup if applicable
  if [ -f "$backup_dir/database.sql.gz" ]; then
    # Test if the gzip file is valid
    if ! gzip -t "$backup_dir/database.sql.gz" 2>/dev/null; then
      warn "Backup validation failed: Corrupted database backup"
      return 1
    fi
  elif [ -f "$backup_dir/database.sql" ]; then
    # Check if file is empty
    if [ ! -s "$backup_dir/database.sql" ]; then
      warn "Backup validation failed: Empty database backup"
      return 1
    fi
  elif grep -q "Database: Yes" "$backup_dir/backup_info.txt"; then
    warn "Backup validation failed: Missing database backup"
    return 1
  fi
  
  log "Backup validation successful"
  return 0
}

# Function to create a backup
create_backup() {
  local backup_dir="$BACKUP_ROOT/prometheum_${TIMESTAMP}"
  local base_dir=""
  
  # Create backup directory
  mkdir -p "$backup_dir"
  mkdir -p "$backup_dir/etc"
  mkdir -p "$backup_dir/data"
  mkdir -p "$backup_dir/opt"
  
  # If incremental, ensure base backup exists
  if [ "$BACKUP_TYPE" = "incremental" ]; then
    base_dir="$BACKUP_ROOT/prometheum_${INCREMENTAL_BASE}"
    if [ ! -d "$base_dir" ]; then
      error "Base backup not found: $INCREMENTAL_BASE"
    fi
    log "Using base backup: $INCREMENTAL_BASE"
  fi

  # Backup configuration files
  status "Backing up configuration files..."
  cp -r /etc/prometheum "$backup_dir/etc/" || warn "Failed to backup some configuration files"
  
  # Backup user data
  status "Backing up user data..."
  cp -a /var/lib/prometheum/data "$backup_dir/" || warn "Failed to backup some user data"
  
  # Backup database if enabled
  if [ "$BACKUP_DB" = true ]; then
    status "Backing up database..."
    if command -v pg_dump &> /dev/null; then
      if [ "$COMPRESS" = true ]; then
        sudo -u postgres pg_dump prometheum | gzip > "$backup_dir/database.sql.gz" || warn "Failed to backup database"
      else
        sudo -u postgres pg_dump prometheum > "$backup_dir/database.sql" || warn "Failed to backup database"
      fi
    else
      warn "pg_dump not found, skipping database backup"
    fi
  fi
  
  # Backup system state
  status "Backing up system state..."
  cp -a /opt/prometheum/src/VERSION "$backup_dir/opt/" 2>/dev/null || true
  systemctl status prometheum-*.service > "$backup_dir/service_status.txt" 2>&1 || warn "Failed to capture service status"
  
  # Create backup info file
  status "Creating backup metadata..."
  cat > "$backup_dir/backup_info.txt" << EOF
Backup created on: $(date)
Backup type: $BACKUP_TYPE
Hostname: $(hostname)
System: $(uname -a)
Prometheum version: $(cat /opt/prometheum/src/VERSION 2>/dev/null || echo "Unknown")
Database: $([ "$BACKUP_DB" = true ] && echo "Yes" || echo "No")
EOF

  if [ "$BACKUP_TYPE" = "incremental" ]; then
    echo "Base backup: $INCREMENTAL_BASE" >> "$backup_dir/backup_info.txt"
  fi
  
  # Add list of backed up files
  find "$backup_dir" -type f | sort > "$backup_dir/file_manifest.txt"
  
  # Compress the backup if requested
  if [ "$COMPRESS" = true ] && [ "$BACKUP_TYPE" = "full" ]; then
    status "Compressing backup..."
    tar -czf "$backup_dir.tar.gz" -C "$BACKUP_ROOT" "$(basename "$backup_dir")" || error "Failed to compress backup"
    rm -rf "$backup_dir"
    backup_dir="$backup_dir.tar.gz"
    log "Backup compressed to $(basename "$backup_dir")"
  fi
  
  # Verify the backup if requested
  if [ "$VERIFY" = true ]; then
    if [ "$COMPRESS" = true ] && [ "$BACKUP_TYPE" = "full" ]; then
      status "Verifying compressed backup..."
      if ! tar -tzf "$backup_dir" > /dev/null; then
        error "Backup verification failed"
      fi
    else
      validate_backup "$backup_dir" || error "Backup verification failed"
    fi
  fi
  
  # Cleanup old backups
  if [ "$MAX_BACKUPS" -gt 0 ]; then
    status "Cleaning up old backups..."
    # Count backups
    backup_count=$(find "$BACKUP_ROOT" -maxdepth 1 -name "prometheum_*" | wc -l)
    
    # Remove oldest backups if we have too many
    if [ "$backup_count" -gt "$MAX_BACKUPS" ]; then
      excess=$((backup_count - MAX_BACKUPS))
      log "Removing $excess old backup(s)"
      find "$BACKUP_ROOT" -maxdepth 1 -name "prometheum_*" | sort | head -n "$excess" | xargs rm -rf
    fi
  fi
  
  log "Backup completed successfully: $(basename "$backup_dir")"
  echo "Backup location: $backup_dir"
}

# Function to restore from a backup
restore_backup() {
  local backup_id="$1"
  local backup_dir="$BACKUP_ROOT/prometheum_${backup_id}"
  local compressed=false
  
  # Check if backup exists or if it's a compressed backup
  if [ ! -d "$backup_dir" ]; then
    if [ -f "${backup_dir}.tar.gz" ]; then
      compressed=true
      log "Found compressed backup"
    else
      error "Backup not found: $backup_id"
    fi
  fi
  
  # Confirm restore operation
  echo "WARNING: This will overwrite your current Prometheum installation!"
  read -p "Are you sure you want to restore from backup $backup_id? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log "Restore operation cancelled"
    exit 0
  fi
  
  # Extract compressed backup if needed
  if [ "$compressed" = true ]; then
    status "Extracting backup..."
    temp_dir=$(mktemp -d)
    tar -xzf "${backup_dir}.tar.gz" -C "$temp_dir" || error "Failed to extract backup"
    backup_dir="${temp_dir}/$(basename "$backup_dir")"
  fi
  
  # Validate backup before restoring
  status "Validating backup before restore..."
  validate_backup "$backup_dir" || error "Invalid backup, aborting restore"
  
  # Stop services
  status "Stopping services..."
  systemctl stop prometheum-health.service prometheum-api.service prometheum-storage.service || warn "Failed to stop some services"
  
  # Restore user data
  status "Restoring user data..."
  if [ -d "$backup_dir/data" ]; then
    rm -rf /var/lib/prometheum/data
    mkdir -p /var/lib/prometheum/data
    cp -a "$backup_dir/data/." /var/lib/prometheum/data/ || error "Failed to restore user data"
    log "User data restored successfully"
  else
    warn "Backup does not contain user data, skipping"
  fi
  
  # Restore database if available
  if [ -f "$backup_dir/database.sql.gz" ]; then
    status "Restoring database..."
    if command -v psql &> /dev/null; then
      systemctl is-active --quiet postgresql || systemctl start postgresql
      gzip -dc "$backup_dir/database.sql.gz" | sudo -u postgres psql -c "DROP DATABASE IF EXISTS prometheum;" && \
      sudo -u postgres psql -c "CREATE DATABASE prometheum;" && \
      gzip -dc "$backup_dir/database.sql.gz" | sudo -u postgres psql prometheum || error "Failed to restore database"
      log "Database restored successfully"
    else
      warn "psql not found, skipping database restore"
    fi
  elif [ -f "$backup_dir/database.sql" ]; then
    status "Restoring database..."
    if command -v psql &> /dev/null; then
      systemctl is-active --quiet postgresql || systemctl start postgresql
      sudo -u postgres psql -c "DROP DATABASE IF EXISTS prometheum;" && \
      sudo -u postgres psql -c "CREATE DATABASE prometheum;" && \
      sudo -u postgres psql prometheum < "$backup_dir/database.sql" || error "Failed to restore database"
      log "Database restored successfully"
    else
      warn "psql not found, skipping database restore"
    fi
  else
    warn "No database backup found, skipping database restore"
  fi
  
  # Set proper permissions
  status "Setting permissions..."
  chown -R prometheum:prometheum /opt/prometheum
  chown -R prometheum:prometheum /var/lib/prometheum
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
  systemctl daemon-reload
  systemctl start prometheum-storage.service || error "Failed to start storage service"
  sleep 3
  systemctl start prometheum-api.service || error "Failed to start API service"
  sleep 2
  systemctl start prometheum-health.service || warn "Failed to start health service"
  
  # Verify services are running
  status "Verifying services..."
  sleep 5
  
  service_issues=0
  if ! systemctl is-active --quiet prometheum-storage.service; then
    warn "Storage service failed to start properly"
    service_issues=$((service_issues + 1))
  fi
  
  if ! systemctl is-active --quiet prometheum-api.service; then
    warn "API service failed to start properly"
    service_issues=$((service_issues + 1))
  fi
  
  if ! systemctl is-active --quiet prometheum-health.service; then
    warn "Health service failed to start properly"
    service_issues=$((service_issues + 1))
  fi
  
  # Clean up temporary files if any
  if [ "$compressed" = true ] && [ -d "$temp_dir" ]; then
    status "Cleaning up temporary files..."
    rm -rf "$temp_dir"
  fi
  
  # Final status
  if [ $service_issues -eq 0 ]; then
    log "Restore completed successfully!"
  else
    warn "Restore completed with $service_issues service issues. Check service logs for details."
  fi
}

# Main execution
if [ "$LIST_MODE" = true ]; then
  list_backups
  exit 0
elif [ "$RESTORE_MODE" = true ]; then
  if [ -z "$RESTORE_ID" ]; then
    error "No backup ID specified for restore"
  fi
  restore_backup "$RESTORE_ID"
else
  create_backup
fi

