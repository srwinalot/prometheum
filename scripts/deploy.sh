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

# Check if podman is installed
if ! command -v podman &> /dev/null; then
  error "Podman is not installed. Please install Podman first."
fi

# Function to check if a URL is reachable
check_url() {
  local url=$1
  local max_attempts=$2
  local delay=$3
  local attempt=1
  
  while [ $attempt -le $max_attempts ]; do
    status "Checking $url (attempt $attempt/$max_attempts)..."
    if curl -s -f "$url" > /dev/null 2>&1; then
      log "Service at $url is up!"
      return 0
    fi
    
    attempt=$((attempt + 1))
    sleep $delay
  done
  
  warn "Service at $url could not be reached after $max_attempts attempts"
  return 1
}

# Change to project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || error "Failed to change to project root directory"

# Create data directories for volumes
log "Setting up volume directories..."
mkdir -p data/{volumes,backup,config,models}

# Check if pod already exists
POD_EXISTS=$(podman pod exists prometheum-pod && echo "true" || echo "false")

if [ "$POD_EXISTS" = "true" ]; then
  warn "Pod 'prometheum-pod' already exists"
  
  read -p "Do you want to stop and remove the existing pod? (y/n) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "Stopping pod 'prometheum-pod'..."
    podman pod stop prometheum-pod || warn "Failed to stop pod, it may not be running"
    
    log "Removing pod 'prometheum-pod'..."
    podman pod rm prometheum-pod || error "Failed to remove pod"
  else
    error "Deployment aborted. Please remove the existing pod before proceeding."
  fi
fi

# Create the pod
log "Creating pod from configuration..."
podman play kube config/pod.yaml || error "Failed to create pod"

# Check if the pod is running
status "Verifying pod status..."
podman pod ps | grep prometheum-pod || error "Pod is not running"

# Health checks
log "Performing health checks..."

# Wait for services to be ready
sleep 10

# Check storage service
check_url "http://localhost:8001/health" 5 3

# Check API service
check_url "http://localhost:8000/api/health" 5 3

# Check AI service
check_url "http://localhost:8002/health" 5 3

# Check Documentation service
check_url "http://localhost:8080" 5 3

log "Deployment summary:"
echo "-----------------------------"
echo "API Service: http://localhost:8000"
echo "Storage Service: http://localhost:8001"
echo "AI Service: http://localhost:8002"
echo "Documentation: http://localhost:8080"
echo "-----------------------------"

log "Pod 'prometheum-pod' deployed successfully!"
log "To stop the pod, run: podman pod stop prometheum-pod"
log "To start the pod again, run: podman pod start prometheum-pod"

