
"""
Authentication utilities for Prometheum API.
"""

import json
import os
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

from fastapi import HTTPException, status
from jose import jwt, JWTError
from passlib.context import CryptContext

from .models import User, UserInDB, UserCreate, UserUpdate, Role, Permission, ROLE_PERMISSIONS
from ...utils import ensure_dir

# Security settings
SECRET_KEY = os.environ.get("SECRET_KEY", "insecure-dev-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day
RESET_TOKEN_EXPIRE_MINUTES = 15  # 15 minutes

# Password handling
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# User database path
USER_DB_PATH = "/var/lib/prometheum/users.json"

# In-memory users cache
_users: Dict[str, Dict[str, Any]] = {}
_reset_tokens: Dict[str, Dict[str, Any]] = {}


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_reset_token(username: str) -> str:
    """Create a password reset token."""
    token = secrets.token_urlsafe(32)
    expires = datetime.utcnow() + timedelta(minutes=RESET_TOKEN_EXPIRE_MINUTES)
    
    _reset_tokens[token] = {
        "username": username,
        "expires": expires
    }
    
    return token


def verify_reset_token(token: str) -> Optional[str]:
    """Verify a password reset token and return the username if valid."""
    if token not in _reset_tokens:
        return None
    
    token_data = _reset_tokens[token]
    if datetime.utcnow() > token_data["expires"]:
        # Token expired
        del _reset_tokens[token]
        return None
    
    return token_data["username"]


def invalidate_reset_token(token: str) -> None:
    """Invalidate a password reset token."""
    if token in _reset_tokens:
        del _reset_tokens[token]


def load_users() -> None:
    """Load users from JSON file."""
    global _users
    if _users:
        return  # Already loaded
        
    if os.path.exists(USER_DB_PATH):
        try:
            with open(USER_DB_PATH, 'r') as f:
                _users = json.load(f)
        except Exception as e:
            print(f"Error loading users: {e}")
    
    # Create default admin if no users exist
    if not _users:
        create_user(
            username="admin",
            password="admin",  # This should be changed on first login
            email="admin@example.com",
            role=Role.ADMIN
        )


def save_users() -> None:
    """Save users to JSON file."""
    ensure_dir(os.path.dirname(USER_DB_PATH))
    
    with open(USER_DB_PATH, 'w') as f:
        json.dump(_users, f, indent=2)


def get_user(username: str) -> Optional[User]:
    """Get a user by username."""
    if not _users:
        load_users()
        
    user_dict = _users.get(username)
    if not user_dict:
        return None
    
    user = UserInDB(**user_dict)
    
    # Add permissions based on role
    user.permissions = ROLE_PERMISSIONS.get(user.role, [])
    
    return user


def get_all_users() -> List[User]:
    """Get all users."""
    if not _users:
        load_users()
        
    return [get_user(username) for username in _users.keys()]


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate a user with username and password."""
    user = get_user(username)
    if not user:
        return None
    
    if not verify_password(password, user.hashed_password):
        return None
    
    if user.disabled:
        return None
        
    return user


def create_user(
    username: str,
    password: str,
    email: Optional[str] = None,
    role: Union[Role, str] = Role.USER,
    full_name: Optional[str] = None,
    disabled: bool = False
) -> User:
    """Create a new user."""
    if not _users:
        load_users()
        
    if username in _users:
        raise ValueError(f"User {username} already exists")
    
    # Convert role to enum if it's a string
    if isinstance(role, str):
        try:
            role = Role(role)
        except ValueError:
            raise ValueError(f"Invalid role: {role}")
    
    # Generate UUID for user
    user_id = str(uuid.uuid4())
    
    # Create user with hashed password
    user_data = {
        "id": user_id,
        "username": username,
        "email": email,
        "full_name": full_name,
        "role": role.value,
        "disabled": disabled,
        "hashed_password": get_password_hash(password),
        "created_at": datetime.utcnow().isoformat()
    }
    
    # Save to in-memory dict
    _users[username] = user_data
    
    # Save to file
    save_users()
    
    # Return user object
    return get_user(username)


def update_user(username: str, user_update: UserUpdate) -> Optional[User]:
    """Update a user."""
    if not _users:
        load_users()
        
    if username not in _users:
        return None
    
    user_data = _users[username]
    
    # Update fields that are provided
    if user_update.email is not None:
        user_data["email"] = user_update.email
        
    if user_update.full_name is not None:
        user_data["full_name"] = user_update.full_name
        
    if user_update.password is not None:
        user_data["hashed_password"] = get_password_hash(user_update.password)
        
    if user_update.role is not None:
        user_data["role"] = user_update.role.value
        
    if user_update.disabled is not None:
        user_data["disabled"] = user_update.disabled
    
    # Save to file
    save_users()
    
    # Return updated user
    return get_user(username)


def delete_user(username: str) -> bool:
    """Delete a user."""
    if not _users:
        load_users()
        
    if username not in _users:
        return False
    
    # Don't allow deleting the last admin
    if _users[username].get("role") == Role.ADMIN.value:
        admin_count = sum(1 for u in _users.values() if u.get("role") == Role.ADMIN.value)
        if admin_count <= 1:
            raise ValueError("Cannot delete the last admin user")
    
    # Delete user
    del _users[username]
    
    # Save to file
    save_users()
    
    return True


def has_permission(user: User, permission: Union[Permission, str]) -> bool:
    """Check if a user has a specific permission."""
    if isinstance(permission, str):
        try:
            permission = Permission(permission)
        except ValueError:
            return False
    
    return permission in user.permissions


"""
Authentication utilities.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, Optional

from jose import jwt
from passlib.context import CryptContext

from .models import User
from ...utils import ensure_dir

# Security settings
SECRET_KEY = os.environ.get("SECRET_KEY", "insecure-dev-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

# Password handling
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# User DB path
USER_DB_PATH = "/var/lib/prometheum/users.json"

# Users cache
_users = {}

def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def load_users():
    """Load users from JSON file."""
    global _users
    if os.path.exists(USER_DB_PATH):
        try:
            with open(USER_DB_PATH, 'r') as f:
                _users = json.load(f)
        except Exception as e:
            print(f"Error loading users: {e}")
    
    # Create default admin if no users exist
    if not _users:
        create_user("admin", "admin", role="admin")

def save_users():
    """Save users to JSON file."""
    ensure_dir(os.path.dirname(USER_DB_PATH))
    with open(USER_DB_PATH, 'w') as f:
        json.dump(_users, f, indent=2)

def get_user(username: str) -> Optional[User]:
    """Get a user by username."""
    if not _users:
        load_users()
        
    user_dict = _users.get(username)
    if not user_dict:
        return None
        
    return User(**user_dict)

def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate a user with username and password."""
    user = get_user(username)
    if not user:
        return None
        
    if not verify_password(password, user.hashed_password):
        return None
        
    return user

def create_user(username: str, password: str, email: Optional[str] = None, role: str = "user") -> User:
    """Create a new user."""
    if not _users:
        load_users()
        
    if username in _users:
        raise ValueError(f"User {username} already exists")
        
    user = User(
        username=username,
        email=email,
        role=role,
        hashed_password=get_password_hash(password)
    )
    
    _users[username] = user.dict()
    save_users()
    
    return user

