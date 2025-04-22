
"""
Dependency injection for API routes.
"""

from typing import Optional, List

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from .auth.models import TokenData, User, Permission
from .auth.utils import get_user, has_permission, SECRET_KEY, ALGORITHM

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

# Singleton instances
_pool_manager = None
_volume_manager = None
_health_monitor = None


def get_pool_manager():
    """Get storage pool manager singleton."""
    global _pool_manager
    if _pool_manager is None:
        from ..storage.pool import StoragePoolManager
        _pool_manager = StoragePoolManager()
    return _pool_manager


def get_volume_manager():
    """Get volume manager singleton."""
    global _volume_manager
    if _volume_manager is None:
        from ..storage.volume import VolumeManager
        _volume_manager = VolumeManager(get_pool_manager())
    return _volume_manager


def get_health_monitor():
    """Get health monitor singleton."""
    global _health_monitor
    if _health_monitor is None:
        from ..storage.health import HealthMonitor
        _health_monitor = HealthMonitor(
            get_pool_manager(),
            get_volume_manager()
        )
    return _health_monitor


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get the current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        
        # Create token data
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    # Get user from database
    user = get_user(token_data.username)
    if user is None:
        raise credentials_exception
    
    # Check if user is disabled
    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    return user


async def get_admin_user(user: User = Depends(get_current_user)) -> User:
    """Verify the user is an admin."""
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions. Admin role required."
        )
    return user


def require_permissions(required_permissions: List[Permission]):
    """Create a dependency that requires specific permissions."""
    
    async def has_required_permissions(user: User = Depends(get_current_user)) -> User:
        """Check if user has all required permissions."""
        for permission in required_permissions:
            if not has_permission(user, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission required: {permission}"
                )
        return user
    
    return has_required_permissions


"""
Dependency injection for API routes.
"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from .auth.models import TokenData, User
from .auth.utils import get_user, SECRET_KEY, ALGORITHM
from ..storage.pool import StoragePoolManager
from ..storage.volume import VolumeManager
from ..storage.health import HealthMonitor

# Singleton instances
_pool_manager = None
_volume_manager = None
_health_monitor = None

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

def get_pool_manager() -> StoragePoolManager:
    """Get the storage pool manager."""
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = StoragePoolManager()
    return _pool_manager

def get_volume_manager() -> VolumeManager:
    """Get the volume manager."""
    global _volume_manager
    if _volume_manager is None:
        _volume_manager = VolumeManager(get_pool_manager())
    return _volume_manager

def get_health_monitor() -> HealthMonitor:
    """Get the health monitor."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor(get_pool_manager(), get_volume_manager())
    return _health_monitor

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get the current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = get_user(token_data.username)
    if user is None:
        raise credentials_exception
    
    return user

async def get_admin_user(user: User = Depends(get_current_user)) -> User:
    """Ensure user has admin role."""
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return user

