
"""
Authentication routes for Prometheum API.
"""

from datetime import timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr

from ..dependencies import get_current_user, get_admin_user
from ..auth.models import Token, User, UserCreate, UserUpdate
from ..auth.utils import (
    authenticate_user,
    create_access_token,
    create_user,
    delete_user,
    update_user,
    get_user,
    get_all_users,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

router = APIRouter()

# Models for request/response
class ChangePasswordRequest(BaseModel):
    """Change password request."""
    current_password: str
    new_password: str

# Routes
@router.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login to get an access token."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=User)
async def get_current_user_info(user: User = Depends(get_current_user)):
    """Get information about the current user."""
    return user

@router.post("/change-password", response_model=User)
async def change_password(
    request: ChangePasswordRequest,
    user: User = Depends(get_current_user)
):
    """Change the current user's password."""
    # First authenticate with current password
    auth_user = authenticate_user(user.username, request.current_password)
    if not auth_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect"
        )
    
    # Update with new password
    updated_user = update_user(
        user.username,
        UserUpdate(password=request.new_password)
    )
    
    return updated_user

@router.get("/users", response_model=List[User])
async def list_users(admin: User = Depends(get_admin_user)):
    """List all users (admin only)."""
    return get_all_users()

@router.post("/users", response_model=User, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    admin: User = Depends(get_admin_user)
):
    """Register a new user (admin only)."""
    try:
        user = create_user(
            username=user_data.username,
            password=user_data.password,
            email=user_data.email,
            role=user_data.role
        )
        return user
    except ValueError as


"""
Authentication routes.
"""

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from ..auth.models import Token, User, UserCreate
from ..auth.utils import (
    authenticate_user, 
    create_access_token, 
    create_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from ..dependencies import get_admin_user

router = APIRouter()

@router.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Get access token."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, admin: User = Depends(get_admin_user)):
    """Register a new user (admin only)."""
    try:
        user = create_user(
            user_data.username,
            user_data.password,
            user_data.email,
            user_data.role
        )
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

