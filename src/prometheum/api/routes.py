"""
API routes for Prometheum cloud service.

This module defines the REST API endpoints that enable clients 
to interact with the Prometheum personal cloud system, including
file operations, sync management, and system status.
"""

import os
import io
import json
import time
import uuid
import logging
import hashlib
import mimetypes
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi import Query, Body, Response, status, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, validator, EmailStr, AnyHttpUrl
import jwt
from jwt.exceptions import InvalidTokenError

# For rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    HAVE_RATE_LIMIT = True
except ImportError:
    HAVE_RATE_LIMIT = False

# These imports would be used in a real implementation
# from ..core.cloud_manager import CloudManager
# from ..devices.device_manager import Device
# from ..storage.file_manager import FileMetadata
# from ..network.interceptor import NetworkInterceptor, ServiceProvider

# For annotation purposes only
class CloudManager:
    pass

class Device:
    pass

class FileMetadata:
    pass

class NetworkInterceptor:
    pass

class ServiceProvider:
    APPLE_ICLOUD = "APPLE_ICLOUD"
    MICROSOFT_ONEDRIVE = "MICROSOFT_ONEDRIVE"
    GOOGLE_DRIVE = "GOOGLE_DRIVE"
    DROPBOX = "DROPBOX"
    BOX = "BOX"

# Configure logging
logger = logging.getLogger(__name__)

# Initialize rate limiter if available
if HAVE_RATE_LIMIT:
    limiter = Limiter(key_func=get_remote_address)
else:
    # Dummy limiter for when slowapi is not available
    class DummyLimiter:
        def limit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    limiter = DummyLimiter()

# Create FastAPI app
app = FastAPI(
    title="Prometheum API",
    description="API for personal cloud storage system with iCloud-like synchronization",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add rate limiter error handler if available
if HAVE_RATE_LIMIT:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, this should be restricted to your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In a real implementation, this would be set during application startup
cloud_manager: Optional[CloudManager] = None
network_interceptor: Optional[NetworkInterceptor] = None

# JWT settings (should be moved to secure configuration in production)
JWT_SECRET_KEY = "CHANGE_THIS_TO_SECURE_KEY_IN_PRODUCTION"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = 60

# OAuth2 token URL for authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


# ---- Pydantic Models for Request/Response Data ----

class ErrorResponse(BaseModel):
    """Standard error response model."""
    status: str = "error"
    message: str
    code: int = 400
    details: Optional[Dict[str, Any]] = None


class SuccessResponse(BaseModel):
    """Standard success response model."""
    status: str = "success"
    message: str = "Operation completed successfully"
    data: Optional[Dict[str, Any]] = None


class TokenResponse(BaseModel):
    """Response model for authentication token."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    device_id: str


class UserCredentials(BaseModel):
    """User credentials for authentication."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    device_id: Optional[str] = None


class DeviceRegistrationRequest(BaseModel):
    """Request model for device registration."""
    name: str = Field(..., min_length=1, max_length=100)
    device_type: str = Field(..., min_length=1, max_length=50)
    os_type: str = Field(..., min_length=1, max_length=50)
    owner: str = Field(..., min_length=1, max_length=100)
    public_key: Optional[str] = None


class DeviceUpdateRequest(BaseModel):
    """Request model for updating device properties."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    is_trusted: Optional[bool] = None
    is_primary: Optional[bool] = None


class DeviceResponse(BaseModel):
    """Response model for device information."""
    device_id: str
    name: str
    type: str
    os_type: str
    registered_date: str
    last_seen: str
    is_trusted: bool
    status: str


class SyncDirectoryRequest(BaseModel):
    """Request model for adding a sync directory."""
    local_path: str = Field(..., min_length=1)
    remote_path: Optional[str] = None
    sync_policy: str = Field("two-way", regex=r"^(one-way-upload|one-way-download|two-way|mirror)$")
    auto_backup: bool = False
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    service: Optional[str] = None  # For cloud service integration


class SyncStatusResponse(BaseModel):
    """Response model for sync status."""
    status: str = "success"
    sync_status: Dict[str, Any]
    device_status: Dict[str, Any]
    storage_status: Dict[str, Any]


class ConflictResolutionRequest(BaseModel):
    """Request model for resolving sync conflicts."""
    conflict_id: str
    resolution: str = Field(..., regex=r"^(local|remote|both|newest|manual)$")
    manual_choice_file_id: Optional[str] = None  # Only used with manual resolution


class FileBatchRequest(BaseModel):
    """Request model for batch file operations."""
    file_paths: List[str] = Field(..., min_items=1)


class FileOperationRequest(BaseModel):
    """Request model for file operations."""
    source_path: str
    destination_path: str
    overwrite: bool = False


class FileMetadataResponse(BaseModel):
    """Response model for file metadata."""
    id: str
    name: str
    path: str
    size: int
    created: str
    modified: str
    mime_type: str
    version: int
    is_directory: bool
    is_favorite: Optional[bool] = None
    is_shared: Optional[bool] = None
    owner: Optional[str] = None
    tags: Optional[List[str]] = None


class FileShareRequest(BaseModel):
    """Request model for sharing a file."""
    file_path: str = Field(..., min_length=1)
    recipients: List[str] = Field(..., min_items=1)
    expiration_days: Optional[int] = Field(None, ge=1, le=365)
    can_edit: bool = False
    password_protected: bool = False
    password: Optional[str] = Field(None, min_length=6)


class SystemStatusResponse(BaseModel):
    """Response model for system status."""
    status: str = "success"
    version: str
    uptime: int
    system_health: Dict[str, Any]
    storage_usage: Dict[str, Any]


class QuotaResponse(BaseModel):
    """Response model for storage quotas."""
    total_storage_bytes: int
    used_storage_bytes: int
    available_storage_bytes: int
    user_quotas: Dict[str, Dict[str, int]]
    quota_usage_percent: float


class CloudServiceAuthRequest(BaseModel):
    """Request model for cloud service authentication."""
    service: str  # ServiceProvider enum value as string
    username: str
    password: str
    store_credentials: bool = False


# ---- JWT Authentication Functions ----

def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: The data to encode in the token
        expires_delta: Optional expiration time
        
    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    return encoded_jwt


async def get_current_device(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    Authenticate the device using the provided token.
    
    Args:
        token: JWT token from header
        
    Returns:
        Dict: Device information
        
    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode the JWT token
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        device_id: str = payload.get("sub")
        
        if device_id is None:
            raise credentials_exception
    except InvalidTokenError:
        raise credentials_exception
    
    if not cloud_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud service not initialized"
        )
    
    # Verify device exists
    device = cloud_manager.device_manager.get_device(device_id)
    if not device:
        raise credentials_exception
    
    # Check if device is active
    if getattr(device, "status", None) != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Device has been deactivated"
        )
    
    # Return device data
    return {
        "device_id": device_id,
        "permissions": getattr(device, "permissions", {}),
        "is_trusted": getattr(device, "is_trusted", False)
    }


# ---- Helper Functions ----

def get_permission_checker(permission: str):
    """
    Create a dependency to check device permissions.
    
    Args:
        permission: The permission to check for
        
    Returns:
        A dependency function that verifies the permission
    """
    async def check_permission(device: Dict[str, Any] = Depends(get_current_device)):
        if not device.get("permissions", {}).get(permission, False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Device does not have {permission} permission"
            )
        return device
    return check_permission


has_read_permission = get_permission_checker("read")
has_write_permission = get_permission_checker("write")
has_admin_permission = get_permission_checker("admin")
has_share_permission = get_permission_checker("share")


# ---- API Routes: Authentication & Device Management ----

@app.post("/api/auth/token", response_model=TokenResponse)
@limiter.limit("10/minute")
async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate a user and device, returning a JWT token.
    
    This endpoint authenticates the user's credentials and returns
    a JWT token for API authorization.
    """
    if not cloud_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud service not initialized"
        )
    
    # In a real implementation, authenticate against user database
    # For now, use a simple check (this should be replaced with proper authentication)
    # user = authenticate_user(form_data.username, form_data.password)
    user_authenticated = True  # Placeholder
    
    if not user_authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # For device authentication, we should connect this to the device manager
    # device_id = form_data.scopes[0] if form_data.scopes else None
    device_id = "device123"  # Placeholder
    
    # Create the JWT token with device_id as subject
    access_token_expires = timedelta(minutes=JWT_EXPIRATION_MINUTES)
    access_token = create_access_token(
        data={"sub": device_id, "username": form_data.username},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": JWT_EXPIRATION_MINUTES * 60,  # in seconds
        "device_id": device_id
    }


@app.post("/api/devices/register", response_model=SuccessResponse)
@limiter.limit("5/minute")
async def register_device(request: Request, device_req: DeviceRegistrationRequest):
    """
    Register a new device with the cloud system.
    
    This endpoint allows registering a new device that can access
    the personal cloud storage system.
    """
    if not cloud_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud service not initialized"
        )
    
    success, device, error_msg = cloud_manager.device_manager.register_device(
        name=device_req.name,
        device_type=device_req.device_type,
        os_type=device_req.os_type,
        owner=device_req.owner,
        public_key=device_req.public_key or ""
    )
    
    if not success or not device:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg or "Failed to register device"
        )
    
    # In a real implementation, this would send the device token to the device
    # via a secure channel like email verification
    
    return {
        "status": "success",
        "message": "Device registered successfully",
        "data": {
            "device_id": device.device_id,
            "device_token

"""
API routes for Prometheum cloud service.

This module defines the REST API endpoints that enable clients 
to interact with the Prometheum personal cloud system.
"""

import os
import json
import time
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, Query, Body
from fastapi import Response, status, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, validator, EmailStr
import jwt
from jwt.exceptions import InvalidTokenError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# These imports would be used in a real implementation
# from ..core.cloud_manager import CloudManager
# from ..devices.device_manager import Device
# from ..storage.file_manager import FileMetadata

# For annotation purposes only
class CloudManager:
    pass

class Device:
    pass

class FileMetadata:
    pass

# Configure logging
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title="Prometheum API",
    description="API for personal cloud storage system with iCloud-like synchronization",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add rate limiter error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, this should be restricted to your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In a real implementation, this would be set during application startup
cloud_manager: Optional[CloudManager] = None

# JWT settings (should be moved to secure configuration in production)
JWT_SECRET_KEY = "CHANGE_THIS_TO_SECURE_KEY_IN_PRODUCTION"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = 60

# OAuth2 token URL for authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


# ---- Pydantic Models for Request/Response Data ----

class ErrorResponse(BaseModel):
    """Standard error response model."""
    status: str = "error"
    message: str
    code: int = 400
    details: Optional[Dict[str, Any]] = None


class SuccessResponse(BaseModel):
    """Standard success response model."""
    status: str = "success"
    message: str = "Operation completed successfully"
    data: Optional[Dict[str, Any]] = None


class TokenResponse(BaseModel):
    """Response model for authentication token."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    device_id: str


class UserCredentials(BaseModel):
    """User credentials for authentication."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    device_id: Optional[str] = None


class DeviceRegistrationRequest(BaseModel):
    """Request model for device registration."""
    name: str = Field(..., min_length=1, max_length=100)
    device_type: str = Field(..., min_length=1, max_length=50)
    os_type: str = Field(..., min_length=1, max_length=50)
    owner: str = Field(..., min_length=1, max_length=100)
    public_key: Optional[str] = None


class DeviceUpdateRequest(BaseModel):
    """Request model for updating device properties."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    is_trusted: Optional[bool] = None
    is_primary: Optional[bool] = None


class DeviceResponse(BaseModel):
    """Response model for device information."""
    device_id: str
    name: str
    type: str
    os_type: str
    registered_date: str
    last_seen: str
    is_trusted: bool
    status: str


class SyncDirectoryRequest(BaseModel):
    """Request model for adding a sync directory."""
    local_path: str = Field(..., min_length=1)
    sync_policy: str = Field("two-way", regex=r"^(one-way|two-way|mirror)$")
    auto_backup: bool = False


class SyncStatusResponse(BaseModel):
    """Response model for sync status."""
    status: str = "success"
    sync_status: Dict[str, Any]
    device_status: Dict[str, Any]
    storage_status: Dict[str, Any]


class FileBatchRequest(BaseModel):
    """Request model for batch file operations."""
    file_paths: List[str] = Field(..., min_items=1)


class FileMetadataResponse(BaseModel):
    """Response model for file metadata."""
    name: str
    path: str
    size: int
    created: str
    modified: str
    mime_type: str
    version: int
    is_directory: bool
    is_favorite: Optional[bool] = None
    is_shared: Optional[bool] = None
    owner: Optional[str] = None
    tags: Optional[List[str]] = None


class FileShareRequest(BaseModel):
    """Request model for sharing a file."""
    file_path: str = Field(..., min_length=1)
    recipients: List[str] = Field(..., min_items=1)
    expiration_days: Optional[int] = Field(None, ge=1, le=365)
    can_edit: bool = False
    password_protected: bool = False
    password: Optional[str] = Field(None, min_length=6)


class SystemStatusResponse(BaseModel):
    """Response model for system status."""
    status: str = "success"
    version: str
    uptime: int
    system_health: Dict[str, Any]
    storage_usage: Dict[str, Any]


# ---- JWT Authentication Functions ----

def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: The data to encode in the token
        expires_delta: Optional expiration time
        
    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    return encoded_jwt


async def get_current_device(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    Authenticate the device using the provided token.
    
    Args:
        token: JWT token from header
        
    Returns:
        Dict: Device information
        
    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode the JWT token
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        device_id: str = payload.get("sub")
        
        if device_id is None:
            raise credentials_exception
    except InvalidTokenError:
        raise credentials_exception
    
    if not cloud_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud service not initialized"
        )
    
    # Verify device exists
    device = cloud_manager.device_manager.get_device(device_id)
    if not device:
        raise credentials_exception
    
    # Check if device is active
    if getattr(device, "status", None) != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Device has been deactivated"
        )
    
    # Return device data
    return {
        "device_id": device_id,
        "permissions": getattr(device, "permissions", {}),
        "is_trusted": getattr(device, "is_trusted", False)
    }


# ---- Helper Functions ----

def get_permission_checker(permission: str):
    """
    Create a dependency to check device permissions.
    
    Args:
        permission: The permission to check for
        
    Returns:
        A dependency function that verifies the permission
    """
    async def check_permission(device: Dict[str, Any] = Depends(get_current_device)):
        if not device.get("permissions", {}).get(permission, False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Device does not have {permission} permission"
            )
        return device
    return check_permission


has_read_permission = get_permission_checker("read")
has_write_permission = get_permission_checker("write")
has_admin_permission = get_permission_checker("admin")
has_share_permission = get_permission_checker("share")


# ---- API Routes: Authentication & Device Management ----

@app.post("/api/auth/token", response_model=TokenResponse)
@limiter.limit("10/minute")
async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate a user and device, returning a JWT token.
    
    This endpoint authenticates the user's credentials and returns
    a JWT token for API authorization.
    """
    if not cloud_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud service not initialized"
        )
    
    # In a real implementation, authenticate against user database
    # For now, use a simple check (this should be replaced with proper authentication)
    # user = authenticate_user(form_data.username, form_data.password)
    user_authenticated = True  # Placeholder
    
    if not user_authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # For device authentication, we should connect this to the device manager
    # device_id = form_data.scopes[0] if form_data.scopes else None
    device_id = "device123"  # Placeholder
    
    # Create the JWT token with device_id as subject
    access_token_expires = timedelta(minutes=JWT_EXPIRATION_MINUTES)
    access_token = create_access_token(
        data={"sub": device_id, "username": form_data.username},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": JWT_EXPIRATION_MINUTES * 60,  # in seconds
        "device_id": device_id
    }


@app.post("/api/devices/register", response_model=SuccessResponse)
@limiter.limit("5/minute")
async def register_device(request: Request, device_req: DeviceRegistrationRequest):
    """
    Register a new device with the cloud system.
    
    This endpoint allows registering a new device that can access
    the personal cloud storage system.
    """
    if not cloud_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud service not initialized"
        )
    
    success, device, error_msg = cloud_manager.device_manager.register_device(
        name=device_req.name,
        device_type=device_req.device_type,
        os_type=device_req.os_type,
        owner=device_req.owner,
        public_key=device_req.public_key or ""
    )
    
    if not success or not device:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg or "Failed to register device"
        )
    
    # In a real implementation, this would send the device token to the device
    # via a secure channel like email verification
    
    return {
        "status": "success",
        "message": "Device registered successfully",
        "data": {
            "device_id": device.device_id,
            "device_token": "REDACTED"  # Don't expose the token in the response
        }
    }


@app.get("/api/devices", response_model=List[DeviceResponse])
@limiter.limit("30/minute")
async def list_devices(
    request: Request, 
    device: Dict[str, Any] = Depends(has_admin_permission)
):
    """
    List all registered devices.
    
    This endpoint returns information about all devices registered
    with the personal cloud storage system.
    """
    if not cloud_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud service not initialized"
        )
    
    devices = cloud_manager.device_manager.get_connected_devices()
    return devices


@app.get("/api/devices/{device_id}", response_model=DeviceResponse)
@limiter.limit("30/minute")
async def get_device(
    request: Request,
    device_id: str,
    device: Dict[str, Any] = Depends(has_read_permission)
):
    """
    Get information about a specific device.
    
    This endpoint returns detailed information about a device
    registered with the personal cloud storage system.
    """
    if not cloud_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud service not initialized"
        )
    
    # Check if the requesting device can view other devices
    requesting_device_id = device["device_id"]
    if requesting_device_id != device_id and not device.get("permissions", {}).get("admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own device information"
        )
    
    target_device = cloud_manager.device_manager.get_device(device_id)
    if not target_device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device {device_id} not found"
        )
    
    # Convert to response model
    return {
        "device_id": target_device.device_id,
        "name": target_device.name,
        "type": target_device.type,
        "os_type": target_device.os_type,
        "registered_date": target_device.registered_date.isoformat(),
        "last_seen": target_device.last_seen.isoformat(),
        "is_trusted": target_device.is_trusted,
        "status": target_device.status
    }


# ---- API Routes: Sync Operations ----

@app.post("/api/sync/directories", response_model=SuccessResponse)
@limiter.limit("10/minute")
async def add_sync_directory(
    request: Request,
    sync_req: SyncDirectoryRequest,
    device: Dict[str, Any] = Depends(has_write_permission)
):
    """
    Add a directory to be synchronized.
    
    This endpoint configures a local directory for cloud synchronization.
    """
    if not cloud_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud service not initialized"
        )
    
    # Validate if the specified service is supported
    if sync_req.service and network_interceptor:
        try:
            service = getattr(ServiceProvider, sync_req.service)
            # Check if the service is authenticated
            if not network_interceptor.authenticator.is_authenticated(service):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, 
                    detail=f"Not authenticated with {sync_req.service}"
                )
        except (AttributeError, ValueError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported service: {sync_req.service}"
            )
    
    # Add the sync directory
    try:
        sync_id = cloud_manager.sync_manager.add_directory(
            device_id=device["device_id"],
            local_path=sync_req.local_path,
            remote_path=sync_req.remote_path,
            sync_policy=sync_req.sync_policy,
            auto_backup=sync_req.auto_backup,
            include_patterns=sync_req.include_patterns,
            exclude_patterns=sync_req.exclude_patterns,
            service=sync_req.service
        )
        
        return {
            "status": "success",
            "message": "Directory added for synchronization",
            "data": {"sync_id": sync_id}
        }
    except Exception as e:
        logger.error(f"Failed to add sync directory: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add sync directory: {str(e)}"
        )


@app.get("/api/sync/status", response_model=SyncStatusResponse)
@limiter.limit("60/minute")
async def get_sync_status(
    request: Request,
    device_id: Optional[str] = Query(None, description="Filter status by device ID"),
    device: Dict[str, Any] = Depends(has_read_permission)
):
    """
    Get the current synchronization status.
    
    This endpoint returns the current status of synchronization
    including pending changes, last sync time, and sync errors.
    """
    if not cloud_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud service not initialized"
        )
    
    # If device_id is provided, check permissions
    if device_id and device_id != device["device_id"] and not device.get("permissions", {}).get("admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own sync status"
        )
    
    # If no device_id is provided, use the authenticated device
    target_device_id = device_id or device["device_id"]
    
    # Get sync status
    try:
        sync_status = cloud_manager.sync_manager.get_status(target_device_id)
        device_status = cloud_manager.device_manager.get_status(target_device_id)
        storage_status = cloud_manager.file_manager.get_status(target_device_id)
        
        return {
            "status": "success",
            "sync_status": sync_status,
            "device_status": device_status,
            "storage_status": storage_status
        }
    except Exception as e:
        logger.error(f"Failed to get sync status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sync status: {str(e)}"
        )


@app.post("/api/sync/force", response_model=SuccessResponse)
@limiter.limit("5/minute")
async def force_sync(
    request: Request,
    sync_id: Optional[str] = Query(None, description="Optional specific sync directory ID"),
    device: Dict[str, Any] = Depends(has_write_permission)
):
    """
    Force an immediate synchronization.
    
    This endpoint triggers an immediate sync operation,
    optionally for a specific directory only.
    """
    if not cloud_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud service not initialized"
        )
    
    # Force a sync
    try:
        if sync_id:
            # Check if the sync directory belongs to this device
            if not cloud_manager.sync_manager.is_directory_owned_by(sync_id, device["device_id"]):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have permission to sync this directory"
                )
                
            success = cloud_manager.sync_manager.sync_directory(sync_id)
            message = f"Forced sync for directory {sync_id}"
        else:
            success = cloud_manager.sync_manager.sync_all(device["device_id"])
            message = "Forced sync for all directories"
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to start synchronization"
            )
        
        return {
            "status": "success",
            "message": message
        }
    except Exception as e:
        logger.error(f"Failed to force sync: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to force sync: {str(e)}"
        )


@app.post("/api/sync/resolve-conflict", response_model=SuccessResponse)
@limiter.limit("10/minute")
async def resolve_conflict(
    request: Request,
    resolution_req: ConflictResolutionRequest,
    device: Dict[str, Any] = Depends(has_write_permission)
):
    """
    Resolve a synchronization conflict.
    
    This endpoint resolves a file conflict by choosing
    the local version, remote version, or keeping both.
    """
    if not cloud_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud service not initialized"
        )
    
    # Validate the conflict exists
    conflict = cloud_manager.sync_manager.get_conflict(resolution_req.conflict_id)
    
    if not conflict:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conflict {resolution_req.conflict_id} not found"
        )
    
    # Check if this device owns the conflict
    if conflict.get("device_id") != device["device_id"] and not device.get("permissions", {}).get("admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to resolve this conflict"
        )
    
    # Resolve the conflict
    try:
        # If using manual resolution, ensure a file choice is provided
        if resolution_req.resolution == "manual" and not resolution_req.manual_choice_file_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Manual resolution requires specifying manual_choice_file_id"
            )
            
        success = cloud_manager.sync_manager.resolve_conflict(
            conflict_id=resolution_req.conflict_id,
            resolution=resolution_req.resolution,
            manual_choice_file_id=resolution_req.manual_choice_file_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to resolve conflict"
            )
        
        return {
            "status": "success",
            "message": f"Conflict {resolution_req.conflict_id} resolved with {resolution_req.resolution} resolution"
        }
    except Exception as e:
        logger.error(f"Failed to resolve conflict: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resolve conflict: {str(e)}"
        )


# ---- API Routes: File Operations ----

@app.post("/api/files/upload", response_model=SuccessResponse)
@limiter.limit("60/minute")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    path: str = Query("/", description="Target directory path"),
    overwrite: bool = Query(False, description="Whether to overwrite existing file"),
    create_version: bool = Query(True, description="Whether to create a version if overwriting"),
    mark_favorite: bool = Query(False, description="Mark file as favorite"),
    tags: Optional[str] = Query(None, description="Comma-separated list of tags"),
    device: Dict[str, Any] = Depends(has_write_permission)
):
    """
    Upload a file to the cloud storage.
    
    This endpoint allows uploading a file to the specified path in the cloud.
    """
    if not cloud_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud service not initialized"
        )
    
    # Normalize and validate path
    target_path = f"{path.rstrip('/')}/{file.filename}"
    
    # Check if file exists and if overwrite is allowed
    if not overwrite and cloud_manager.file_manager.get_file(target_path):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"File {target_path} already exists"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Check quota before storing
        device_id = device["device_id"]
        if not cloud_manager.file_manager.check_quota(len(content), device_id):
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Storage quota exceeded"
            )
        
        # Parse tags if provided
        file_tags = []
        if tags:
            file_tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
            
        # Store the file
        success = cloud_manager.file_manager.store_file(
            file_path=target_path,
            content=content,
            owner=device_id,
            device_id=device_id,
            create_versions=create_version,
            is_favorite=mark_favorite,
            tags=file_tags
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store file"
        )
        
        # If we have network interceptor and this client is using services integration, log the intercept
        if network_interceptor and file.size > 0:
            network_interceptor.log_file_operation(
                operation="upload",
                file_path=target_path,
                device_id=device_id,
                size=len(content)
            )
        
        return {
            "status": "success",
            "message": f"File uploaded successfully to {target_path}",
            "data": {
                "path": target_path,
                "size": len(content)
            }
        }
    except Exception as e:
        logger.error(f"Failed to upload file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )


@app.get("/api/files/download/{file_id}")
@limiter.limit("60/minute")
async def download_file(
    request: Request,
    file_id: str,
    device: Dict[str, Any] = Depends(has_read_permission)
):
    """
    Download a file from the cloud storage.
    
    This endpoint retrieves a file from the cloud storage.
    """
    if not cloud_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud service not initialized"
        )
    
    try:
        # Get file path and check if it exists
        file_meta = cloud_manager.file_manager.get_metadata(file_id)
        
        if not file_meta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File with ID {file_id} not found"
            )
        
        # Check if device has permission to access this file
        if file_meta.owner != device["device_id"] and not device.get("permissions", {}).get("admin", False):
            # Check if file is shared with this device
            if device["device_id"] not in file_meta.shared_with:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have permission to access this file"
                )
        
        # Get file content
        file_path = cloud_manager.file_manager.get_file(file_meta.file_path)
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File content not found"
            )
        
        # Log download for network interceptor if available
        if network_interceptor:
            network_interceptor.log_file_operation(
                operation="download",
                file_path=file_meta.file_path,
                device_id=device["device_id"],
                size=file_meta.size
            )
            
        # Return file as a download
        return FileResponse(
            path=file_path,
            filename=os.path.basename(file_meta.file_path),
            media_type=file_meta.mime_type
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download file: {str(e)}"
        )


@app.get("/api/files/list")
@limiter.limit("60/minute")
async def list_files(
    request: Request,
    path: str = Query("/", description="Directory path to list"),
    recursive: bool = Query(False, description="List files recursively"),
    include_versions: bool = Query(False, description="Include versions in listing"),
    device: Dict[str, Any] = Depends(has_read_permission)
):
    """
    List files in a directory.
    
    This endpoint returns a list of files in the specified directory.
    """
    if not cloud_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud service not initialized"
        )
    
    try:
        # Get list of files
        files = cloud_manager.file_manager.list_files(
            directory=path,
            recursive=recursive,
            include_versions=include_versions,
            device_id=device["device_id"]
        )
        
        # Convert to response model
        response_files = []
        for file_meta in files:
            # Check if this device has access to the file
            if (file_meta.owner == device["device_id"] or 
                device["device_id"] in file_meta.shared_with or
                device.get("permissions", {}).get("admin", False)):
                
                response_files.append({
                    "id": file_meta.file_path,  # Using path as ID for simplicity
                    "name": os.path.basename(file_meta.file_path),
                    "path": file_meta.file_path,
                    "size": file_meta.size,
                    "created": file_meta.created.isoformat(),
                    "modified": file_meta.modified.isoformat(),
                    "mime_type": file_meta.mime_type,
                    "version": file_meta.version,
                    "is_directory": os.path.isdir(cloud_manager.file_manager.get_file(file_meta.file_path)),
                    "is_favorite": file_meta.is_favorite,
                    "is_shared": len(file_meta.shared_with) > 0,
                    "owner": file_meta.owner,
                    "tags": file_meta.tags
                })
        
        return response_files
    except Exception as e:
        logger.error(f"Failed to list files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list files: {str(e)}"
        )


@app.post("/api/files/move", response_model=SuccessResponse)
@limiter.limit("30/minute")
async def move_file(
    request: Request,
    operation: FileOperationRequest,
    device: Dict[str, Any] = Depends(has_write_permission)
):
    """
    Move a file to a new location.
    
    This endpoint moves a file or directory from one location to another.
    """
    if not cloud_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud service not initialized"
        )
    
    try:
        # Check if source exists
        source_meta = cloud_manager.file_manager.get_metadata(operation.source_path)
        if not source_meta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Source file {operation.source_path} not found"
            )
        
        # Check if device has permission to move this file
        if source_meta.owner != device["device_id"] and not device.get("permissions", {}).get("admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to move this file"
            )
        
        # Check if destination exists and we're not overwriting
        if not operation.overwrite and cloud_manager.file_manager.get_file(operation.destination_path):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Destination {operation.destination_path} already exists"
            )
        
        # Move the file
        success = cloud_manager.file_manager.move_file(
            source_path=operation.source_path,
            destination_path=operation.destination_path,
            overwrite=operation.overwrite,
            device_id=device["device_id"]
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to move file"
            )
        
        return {
            "status": "success",
            "message": f"Moved {operation.source_path} to {operation.destination_path}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to move file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to move file: {str(e)}"
        )


@app.delete("/api/files/{file_id}", response_model=SuccessResponse)
@limiter.limit("30/minute")
async def delete_file(
    request: Request,
    file_id: str,
    permanent: bool = Query(False, description="Permanently delete file instead of moving to trash"),
    device: Dict[str, Any] = Depends(has_write_permission)
):
    """
    Delete a file from cloud storage.
    
    This endpoint deletes a file, either by moving it to trash or permanently.
    """
    if not cloud_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud service not initialized"
        )
    
    try:
        # Check if file exists
        file_meta = cloud_manager.file_manager.get_metadata(file_id)
        if not file_meta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {file_id} not found"
            )
        
        # Check if device has permission to delete this file
        if file_meta.owner != device["device_id"] and not device.get("permissions", {}).get("admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to delete this file"
            )
        
        # Delete the file
        success = cloud_manager.file_manager.delete_file(
            file_path=file_id,
            permanent=permanent,
            device_id=device["device_id"]
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete file"
            )
        
        action = "permanently deleted" if permanent else "moved to trash"
        return {
            "status": "success",
            "message": f"File {file_id} {action}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file: {str(e)}"
        )


@app.post("/api/files/share", response_model=SuccessResponse)
@limiter.limit("30/minute")
async def share_file(
    request: Request,
    share_req: FileShareRequest,
    device: Dict[str, Any] = Depends(has_share_permission)
):
    """
    Share a file with other users.
    
    This endpoint creates a share for a file with specified recipients.
    """
    if not cloud_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud service not initialized"
        )
    
    try:
        # Check if file exists
        file_meta = cloud_manager.file_manager.get_metadata(share_req.file_path)
        if not file_meta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {share_req.file_path} not found"
            )
        
        # Check if device has permission to share this file
        if file_meta.owner != device["device_id"] and not device.get("permissions", {}).get("admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to share this file"
            )
        
        # Share the file
        expiration = datetime.now() + timedelta(days=share_req.expiration_days) if share_req.expiration_days else None
        
        share_id = cloud_manager.file_manager.share_file(
            file_path=share_req.file_path,
            owner=device["device_id"],
            recipients=share_req.recipients,
            can_edit=share_req.can_edit,
            expiration=expiration,
            password_protected=share_req.password_protected,
            password=share_req.password
        )
        
        if not share_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to share file"
            )
        
        # Generate share URL
        share_url = f"/api/shared/{share_id}"
        
        return {
            "status": "success",
            "message": f"File {share_req.file_path} shared successfully",
            "data": {
                "share_id": share_id,
                "share_url": share_url,
                "expiration": expiration.isoformat() if expiration else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to share file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to share file: {str(e)}"
        )


# ---- API Routes: System Management ----

@app.get("/api/system/status", response_model=SystemStatusResponse)
@limiter.limit("30/minute")
async def get_system_status(
    request: Request,
    device: Dict[str, Any] = Depends(has_read_permission)
):
    """
    Get system status and health information.
    
    This endpoint returns information about the system's status,
    including uptime, service health, and storage usage.
    """
    if not cloud_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud service not initialized"
        )
    
    try:
        # Get system status information
        version = getattr(cloud_manager, "version", "1.0.0")  # Get version or default to 1.0.0
        
        # Calculate uptime
        start_time = getattr(cloud_manager, "start_time", time.time())
        uptime_seconds = int(time.time() - start_time)
        
        # Get system health status
        system_health = {
            "services": {
                "file_manager": cloud_manager.file_manager.is_healthy(),
                "device_manager": cloud_manager.device_manager.is_healthy(),
                "sync_manager": cloud_manager.sync_manager.is_healthy()
            },
            "cpu_usage": cloud_manager.get_cpu_usage() if hasattr(cloud_manager, "get_cpu_usage") else None,
            "memory_usage": cloud_manager.get_memory_usage() if hasattr(cloud_manager, "get_memory_usage") else None,
            "active_connections": cloud_manager.get_active_connections() if hasattr(cloud_manager, "get_active_connections") else 0
        }
        
        # Get storage usage information
        storage_usage = cloud_manager.file_manager.get_storage_usage()
        
        # Add network interceptor stats if available
        if network_interceptor:
            system_health["network_interception"] = {
                "status": "active" if network_interceptor.is_running else "inactive",
                "connections_intercepted": network_interceptor.stats.get("connections_intercepted", 0),
                "data_redirected_bytes": network_interceptor.stats.get("data_redirected_bytes", 0)
            }
        
        return {
            "status": "success",
            "version": version,
            "uptime": uptime_seconds,
            "system_health": system_health,
            "storage_usage": storage_usage
        }
    except Exception as e:
        logger.error(f"Failed to get system status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {str(e)}"
        )


@app.get("/api/system/quotas", response_model=QuotaResponse)
@limiter.limit("30/minute")
async def get_quotas(
    request: Request,
    device: Dict[str, Any] = Depends(has_read_permission)
):
    """
    Get storage quota information.
    
    This endpoint returns information about storage quotas,
    including total storage, used storage, and available storage.
    """
    if not cloud_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud service not initialized"
        )
    
    try:
        # Get quota information
        quota_info = cloud_manager.file_manager.get_quota_info()
        
        # Calculate user-specific quotas
        user_quotas = {}
        for user_id, quota in quota_info.get("user_quotas", {}).items():
            # Only include admin user's quotas or the current user's quota
            if user_id == device["device_id"] or device.get("permissions", {}).get("admin", False):
                user_quotas[user_id] = quota
        
        # Calculate overall quota usage percentage
        total_bytes = quota_info.get("total_storage_bytes", 0)
        used_bytes = quota_info.get("used_storage_bytes", 0)
        
        if total_bytes > 0:
            quota_usage_percent = (used_bytes / total_bytes) * 100
        else:
            quota_usage_percent = 0
        
        return {
            "total_storage_bytes": quota_info.get("total_storage_bytes", 0),
            "used_storage_bytes": quota_info.get("used_storage_bytes", 0),
            "available_storage_bytes": quota_info.get("available_storage_bytes", 0),
            "user_quotas": user_quotas,
            "quota_usage_percent": round(quota_usage_percent, 2)
        }
    except Exception as e:
        logger.error(f"Failed to get quota information: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get quota information: {str(e)}"
        )


# ---- API Routes: Cloud Service Integration ----

@app.post("/api/cloud/authenticate", response_model=SuccessResponse)
@limiter.limit("5/minute")
async def authenticate_cloud_service(
    request: Request,
    auth_req: CloudServiceAuthRequest,
    device: Dict[str, Any] = Depends(has_admin_permission)
):
    """
    Authenticate with a cloud service provider.
    
    This endpoint authenticates with a cloud service provider,
    enabling integration and data interception.
    """
    if not cloud_manager or not network_interceptor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cloud service or network interceptor not initialized"
        )
    
    try:
        # Convert service string to ServiceProvider enum
        try:
            service = getattr(ServiceProvider, auth_req.service)
        except (AttributeError, ValueError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported service: {auth_req.service}"
            )
        
        # Authenticate with the service
        auth_method = getattr(network_interceptor.authenticator, f"authenticate_{auth_req.service.lower()}", None)
        
        if not auth_method:
            # Fallback to a generic auth method if specific one doesn't exist
            success = network_interceptor.authenticator.authenticate_service(
                service=service,
                username=auth_req.username,
                password=auth_req.password,
                store_credentials=auth_req.store_credentials
            )
        else:
            success = auth_method(
                username=auth_req.username,
                password=auth_req.password,
                store_credentials=auth_req.store_credentials
            )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authentication failed for {auth_req.service}"
            )
        
        # Start intercepting traffic for this service if authenticated
        if success:
            network_interceptor.start_interception(service)
        
        return {
            "status": "success",
            "message": f"Successfully authenticated with {auth_req.service}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to authenticate with cloud service: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to authenticate with cloud service: {str(e)}"
        )


@app.get("/api/cloud/status")
@limiter.limit("30/minute")
async def get_cloud_service_status(
    request: Request,
    service: Optional[str] = Query(None, description="Specific service to check"),
    device: Dict[str, Any] = Depends(has_read_permission)
):
    """
    Get cloud service integration status.
    
    This endpoint returns the status of cloud service integrations,
    including authentication status and network interception statistics.
    """
    if not network_interceptor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Network interceptor not initialized"
        )
    
    try:
        services_status = {}
        
        # If a specific service is requested, only check that one
        if service:
            try:
                service_enum = getattr(ServiceProvider, service)
                services_status[service] = {
                    "is_authenticated": network_interceptor.authenticator.is_authenticated(service_enum),
                    "is_intercepting": service_enum in network_interceptor.active_interceptions,
                    "auth_status": network_interceptor.authenticator.auth_status.get(service_enum, "unknown"),
                    "interception_stats": network_interceptor.active_interceptions.get(service_enum, {})
                }
            except (AttributeError, ValueError):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported service: {service}"
                )
        else:
            # Check all services
            for service_name, service_enum in vars(ServiceProvider).items():
                if not service_name.startswith("_"):  # Skip private attributes
                    services_status[service_name] = {
1
