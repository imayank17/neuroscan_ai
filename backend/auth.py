from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
import bcrypt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from bson.objectid import ObjectId
from pydantic import BaseModel
from database import get_db
import os
from dotenv import load_dotenv
from logger import app_logger

load_dotenv()

# JWT Config
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "seizure-detection-secret-key-2024-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# Schemas
class UserCreate(BaseModel):
    username: str
    email: str
    full_name: str
    password: str


class UserResponse(BaseModel):
    id: str  # MongoDB ObjectId will be converted to string
    username: str
    email: str
    full_name: str
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse


# Utilities
def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def get_password_hash(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme), db=Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.users.find_one({"username": username})
    if user is None:
        raise credentials_exception
    
    # Map _id to id
    user["id"] = str(user.pop("_id"))
    return user


# Routes
@router.post("/register", response_model=Token)
def register(user_data: UserCreate, db=Depends(get_db)):
    app_logger.info(f"New registration attempt: {user_data.username} ({user_data.email})")
    # Check existing
    if db.users.find_one({"username": user_data.username}):
        app_logger.warning(f"Registration failed: Username {user_data.username} already exists")
        raise HTTPException(status_code=400, detail="Username already registered")
    if db.users.find_one({"email": user_data.email}):
        app_logger.warning(f"Registration failed: Email {user_data.email} already exists")
        raise HTTPException(status_code=400, detail="Email already registered")

    user_dict = {
        "username": user_data.username,
        "email": user_data.email,
        "full_name": user_data.full_name,
        "hashed_password": get_password_hash(user_data.password),
        "created_at": datetime.utcnow()
    }
    
    result = db.users.insert_one(user_dict)
    user_dict["id"] = str(result.inserted_id)
    app_logger.info(f"User registered successfully: {user_data.username} (ID: {user_dict['id']})")

    access_token = create_access_token(data={"sub": user_dict["username"]})
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(**user_dict),
    )


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db=Depends(get_db)):
    app_logger.info(f"Login attempt for user: {form_data.username}")
    user = db.users.find_one({"username": form_data.username})
    
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        app_logger.warning(f"Failed login attempt for username: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user["id"] = str(user.pop("_id"))
    app_logger.info(f"User logged in successfully: {user['username']} (ID: {user['id']})")
    
    access_token = create_access_token(data={"sub": user["username"]})
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(**user),
    )


@router.get("/profile", response_model=UserResponse)
def get_profile(current_user: dict = Depends(get_current_user)):
    return UserResponse(**current_user)

