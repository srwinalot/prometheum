# Prometheum Containerized Deployment

This directory contains the containerization setup for Prometheum, using Podman for container management. The application is broken down into separate containers to manage resource usage and dependencies more effectively.

## Container Architecture

Prometheum is divided into four main containers:

1. **API Container** (`api/Containerfile`)
   - Runs the FastAPI application
   - Handles HTTP requests and client interactions
   - Connects to other services internally

2. **Storage Container** (`storage/Containerfile`) 
   - Manages persistent storage operations
   - Handles volume mounting and data persistence
   - Provides storage-related APIs

3. **AI Processing Container** (`ai/Containerfile`)
   - Runs the LLaMA and sentence transformers models
   - Handles computationally intensive AI operations
   - Optimized for ML workloads

4. **Documentation Container** (`docs/Containerfile`)
   - Serves the project documentation
   - Generates documentation from source files
   - Provides a web interface for viewing docs

## Prerequisites

- [Podman](https://podman.io/getting-started/installation) must be installed on your system
- At least 4GB of RAM for running all containers (8GB recommended)
- ~2GB of free disk space for container images and volumes

## Building and Deploying

### Building Container Images

To build all container images:

```bash
./scripts/build.sh
```

This will build all four containers and tag them appropriately. The script includes error handling and will provide detailed output during the build process.

### Deploying the Pod

To deploy the pod with all containers:

```bash
./scripts/deploy.sh
```

This script will:
1. Create necessary volume directories
2. Check for and handle existing pod instances
3. Deploy the pod using the configuration in `config/pod.yaml`
4. Perform health checks to verify all services are running
5. Display connection information for all services

## Resource Requirements

Container | Memory | CPU | Disk
--- | --- | --- | ---
API | 256MB-512MB | 0.25-0.5 cores | ~100MB
Storage | 256MB-512MB | 0.25-0.5 cores | Depends on data volume
AI | 1GB-2GB | 0.5-1 cores | ~500MB + models
Documentation | 128MB-256MB | 0.1-0.2 cores | ~50MB

## Service Endpoints

Once deployed, services are available at:

- **API**: http://localhost:8000
- **Storage**: http://localhost:8001
- **AI Processing**: http://localhost:8002
- **Documentation**: http://localhost:8080

## Common Operations

### Starting and Stopping the Pod

```bash
# Start the pod
podman pod start prometheum-pod

# Stop the pod
podman pod stop prometheum-pod
```

### Checking Container Logs

```bash
# View logs for a specific container
podman logs prometheum-pod-api
podman logs prometheum-pod-storage
podman logs prometheum-pod-ai
podman logs prometheum-pod-docs
```

### Accessing Container Shell

```bash
# Get a shell inside a container
podman exec -it prometheum-pod-api /bin/bash
```

### Updating Container Images

To update after code changes:

1. Rebuild the affected container(s) using `./scripts/build.sh`
2. Redeploy using `./scripts/deploy.sh`

## Development Workflow

### Local Development with Containers

1. Make code changes in your local environment
2. Rebuild affected container(s): `podman build -t localhost/prometheum-api:latest -f containers/api/Containerfile .`
3. Restart the specific container: `podman restart prometheum-pod-api`
4. Test your changes

### Running Tests in Containers

```bash
# Execute tests inside a container
podman exec prometheum-pod-api python -m pytest
```

## Troubleshooting

### Common Issues

#### Container Fails to Start

Check the logs for detailed error messages:
```bash
podman logs prometheum-pod-api
```

#### Services Unreachable

Verify the pod is running and the containers are in the correct state:
```bash
podman pod ps
podman ps -a --pod
```

#### Resource Constraints

If containers are being killed due to insufficient resources:
1. Modify resource limits in `config/pod.yaml`
2. Redeploy the pod

#### Volume Permissions

If encountering permission issues with volumes:
```bash
# Fix permissions on data directories
sudo chown -R $(id -u):$(id -g) ./data
```

## Advanced Configuration

### Environment Variables

Each container loads configuration from environment files in the `config/` directory:
- `config/api.env`
- `config/storage.env`
- `config/ai.env`

### Custom Pod Configuration

To modify the pod configuration:
1. Edit `config/pod.yaml`
2. Re-run `./scripts/deploy.sh`

## Performance Optimization

- For production deployments, consider adjusting resource limits in `config/pod.yaml`
- The AI container can be tuned for specific hardware by modifying `OMP_NUM_THREADS` and `MKL_NUM_THREADS` in its Containerfile
- For larger deployments, consider using Podman Compose or upgrading to Kubernetes

