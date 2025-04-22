#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Log function
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

# Check if podman is installed
if ! command -v podman &> /dev/null; then
  error "Podman is not installed. Please install Podman first."
fi

# Change to project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || error "Failed to change to project root directory"

log "Building Prometheum containers..."

# Build API container
log "Building API container..."
podman build -t localhost/prometheum-api:latest -f containers/api/Containerfile . || error "Failed to build API container"

# Build Storage container
log "Building Storage container..."
podman build -t localhost/prometheum-storage:latest -f containers/storage/Containerfile . || error "Failed to build Storage container"

# Build AI container
log "Building AI container..."
podman build -t localhost/prometheum-ai:latest -f containers/ai/Containerfile . || error "Failed to build AI container"

# Build Documentation container
log "Building Documentation container..."
podman build -t localhost/prometheum-docs:latest -f containers/docs/Containerfile . || error "Failed to build Documentation container"

# List all images
log "Container images built successfully:"
podman images | grep prometheum

log "All containers built successfully!"
log "To deploy the pod, run: ./scripts/deploy.sh"

