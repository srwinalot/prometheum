"""
Authentication models for Prometheum API.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, EmailStr, validator


class Role(str, Enum):
    """User role enum."""
    
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class Permission(str, Enum):
    """User permission enum."""
    
    READ_STORAGE = "read:storage"
    WRITE_STORAGE = "write:storage"
    READ_SYSTEM = "read:system"
    WRITE_SYSTEM = "write:system"
    READ_HEALTH = "read:health"
    READ_BACKUP = "read:backup"
    WRITE_BACKUP = "write:backup"
    MANAGE_USERS = "manage:users"


# Role to permissions mapping
ROLE_PERMISSIONS = {
    Role.ADMIN: [
        Permission.READ_STORAGE,
        Permission.WRITE_STORAGE,
        Permission.READ_SYSTEM,
        Permission.WRITE_SYSTEM,
        Permission.READ_HEALTH,
        Permission.READ_BACKUP,
        Permission.WRITE_BACKUP,
        Permission.MANAGE_USERS
    ],
    Role.USER: [
        Permission.READ_STORAGE,
        Permission.WRITE_STORAGE,
        Permission.READ_SYSTEM,
        Permission.READ_HEALTH,
        Permission.READ_BACKUP,
        Permission.WRITE_BACKUP
    ],
    Role.VIEWER: [
        Permission.READ_STORAGE,
        Permission.READ_SYSTEM,
        Permission.READ_HEALTH,
        Permission.READ_BACKUP
    ]
}


class TokenData(BaseModel):
    """JWT token payload data."""
    
    username: str
    permissions: List[str] = []
    exp: Optional[int] = None


class Token(BaseModel):
    """OAuth token response."""
    
    access_token: str
    token_type: str


class UserBase(BaseModel):
    """Base user model with common fields."""
    
    username: str
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: Role = Role.USER
    disabled: bool = False


class UserCreate(UserBase):
    """User creation model."""
    
    password: str
    
    @validator('password')
    def password_min_length(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v


class UserUpdate(BaseModel):
    """User update model with optional fields."""
    
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    role: Optional[Role] = None
    disabled: Optional[bool] = None


class User(UserBase):
    """User model returned from API."""
    
    id: str = Field(..., description="Unique user ID")
    created_at: datetime
    permissions: List[Permission] = []
    
    class Config:
        orm_mode = True


class UserInDB(User):
    """User model stored in database."""
    
    hashed_password: str
    
    class Config:
        exclude = {"hashed_password"}


class PasswordResetRequest(BaseModel):
    """Password reset request model."""
    
    username: str
    email: EmailStr


class PasswordReset(BaseModel):
    """Password reset model with token."""
    
    token: str
    new_password: str
    
    @validator('new_password')
    def password_min_length(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v


"""
Authentication models.
"""

from typing import Optional, List
from pydantic import BaseModel


class TokenData(BaseModel):
    """Token data payload."""
    username: str
    scopes: List[str] = []


class Token(BaseModel):
    """OAuth token response."""
    access_token: str
    token_type: str


class User(BaseModel):
    """User model."""
    username: str
    email: Optional[str] = None
    role: str = "user"
    disabled: bool = False
    hashed_password: Optional[str] = None
    
    class Config:
        exclude = {"hashed_password"}


class UserCreate(BaseModel):
    """User creation model."""
    username: str
    password: str
    email: Optional[str] = None
    role: str = "user"

