# ✅ FIXED FASTAPI ROUTES

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from uuid import UUID, uuid4
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..rag_engine.db.session import get_db
from ..rag_engine.local_cache.sqlite_session import get_local_db
from ..rag_engine.db.models import User, AuditLog
from ..services.auth_service import AuthService
from ..utils.logger import get_logger

from fastapi import APIRouter, Depends
from app.dependencies.roles import require_role

# Inside app/api/auth_router.py
router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])



@router.get("/admin/dashboard", dependencies=[Depends(lambda: require_role(["admin"]))])
def get_admin_data():
    return {"msg": "Only admin can access this."}


security = HTTPBearer()
logger = get_logger()
limiter = Limiter(key_func=get_remote_address)

# Models
class UserLogin(BaseModel):
    email: EmailStr
    password: str
    remember_me: bool = False

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str
    confirm_password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    user_name: str

class UserProfile(BaseModel):
    id: str
    email: str
    name: str
    picture_url: Optional[str] = None
    created_at: datetime
    last_login: datetime

class PasswordChange(BaseModel):
    current_password: str
    new_password: str
    confirm_new_password: str

class LogoutRequest(BaseModel):
    token: str

# Dependency
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db=Depends(get_db),
    local_db=Depends(get_local_db)
):
    token = credentials.credentials
    try:
        auth_service = AuthService(db, local_db)
        user_data = auth_service.verify_token(token)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return user_data
    except Exception:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
async def login(user_login: UserLogin, request: Request, db=Depends(get_db), local_db=Depends(get_local_db)):
    try:
        auth_service = AuthService(db, local_db)
        user = auth_service.authenticate_user(user_login.email, user_login.password)
        if not user:
            logger.warning(f"Failed login attempt for email: {user_login.email}")
            raise HTTPException(status_code=401, detail="Invalid email or password")

        token_data = auth_service.create_access_token(
            user_id=str(user.id),
            email=user.email,
            remember_me=user_login.remember_me
        )

        user.last_login = datetime.utcnow()
        db.commit()

        audit_log = AuditLog(
            user_id=user.id,
            event_type="login",
            event_details={
                "ip_address": request.headers.get("x-forwarded-for", request.client.host),
                "user_agent": request.headers.get("user-agent", "unknown"),
                "remember_me": user_login.remember_me
            }
        )
        db.add(audit_log)
        db.commit()

        logger.info(f"Successful login for user: {user.email}")

        return TokenResponse(
            access_token=token_data["access_token"],
            expires_in=token_data["expires_in"],
            user_id=str(user.id),
            user_name=user.name
        )
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed. Please try again.")

@router.post("/register", response_model=TokenResponse)
@limiter.limit("5/minute")
async def register(user_register: UserRegister, request: Request, db=Depends(get_db), local_db=Depends(get_local_db)):
    try:
        if user_register.password != user_register.confirm_password:
            raise HTTPException(status_code=400, detail="Passwords do not match")

        existing_user = db.query(User).filter(User.email == user_register.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        auth_service = AuthService(db, local_db)
        user = auth_service.create_user(user_register.email, user_register.password, user_register.name)

        token_data = auth_service.create_access_token(user_id=str(user.id), email=user.email)

        audit_log = AuditLog(
            user_id=user.id,
            event_type="register",
            event_details={
                "registration_method": "email",
                "ip_address": request.headers.get("x-forwarded-for", request.client.host)
            }
        )
        db.add(audit_log)
        db.commit()

        logger.info(f"New user registered: {user.email}")

        return TokenResponse(
            access_token=token_data["access_token"],
            expires_in=token_data["expires_in"],
            user_id=str(user.id),
            user_name=user.name
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed. Please try again.")

@router.post("/logout")
async def logout(current_user=Depends(get_current_user), credentials=Depends(security), request: Request = None, db=Depends(get_db), local_db=Depends(get_local_db)):
    try:
        token = credentials.credentials
        auth_service = AuthService(db, local_db)
        auth_service.invalidate_token(token)

        audit_log = AuditLog(
            user_id=UUID(current_user["user_id"]),
            event_type="logout",
            event_details={
                "ip_address": request.headers.get("x-forwarded-for", request.client.host),
                "user_agent": request.headers.get("user-agent", "unknown")
            }
        )
        db.add(audit_log)
        db.commit()

        logger.info(f"User logged out: {current_user['email']}")
        return {"message": "Successfully logged out"}
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        raise HTTPException(status_code=500, detail="Logout failed")

@router.get("/profile", response_model=UserProfile)
async def get_profile(current_user=Depends(get_current_user), db=Depends(get_db)):
    try:
        user = db.query(User).filter(User.id == UUID(current_user["user_id"])).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return UserProfile(
            id=str(user.id),
            email=user.email,
            name=user.name,
            picture_url=user.picture_url,
            created_at=user.created_at,
            last_login=user.last_login
        )
    except Exception as e:
        logger.error(f"Get profile error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve profile")

@router.put("/change-password")
async def change_password(request: Request, password_change: PasswordChange, current_user=Depends(get_current_user), db=Depends(get_db), local_db=Depends(get_local_db)):
    try:
        if password_change.new_password != password_change.confirm_new_password:
            raise HTTPException(status_code=400, detail="New passwords do not match")

        auth_service = AuthService(db, local_db)
        success = auth_service.change_password(
            user_id=UUID(current_user["user_id"]),
            current_password=password_change.current_password,
            new_password=password_change.new_password
        )
        if not success:
            raise HTTPException(status_code=400, detail="Current password is incorrect")

        audit_log = AuditLog(
            user_id=UUID(current_user["user_id"]),
            event_type="password_change",
            event_details={
                "ip_address": request.headers.get("x-forwarded-for", request.client.host),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        db.add(audit_log)
        db.commit()

        logger.info(f"Password changed for user: {current_user['email']}")
        return {"message": "Password changed successfully"}
    except Exception as e:
        logger.error(f"Change password error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to change password")

@router.post("/verify-token")
async def verify_token(current_user=Depends(get_current_user)):
    return {
        "valid": True,
        "user_id": current_user["user_id"],
        "email": current_user["email"],
        "name": current_user["name"]
    }

@router.post("/refresh-token")
async def refresh_token(current_user=Depends(get_current_user), db=Depends(get_db), local_db=Depends(get_local_db)):
    try:
        auth_service = AuthService(db, local_db)
        token_data = auth_service.create_access_token(
            user_id=current_user["user_id"],
            email=current_user["email"]
        )
        return {
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token", ""),
            "token_type": "bearer",
            "expires_in": token_data["expires_in"]
        }
    except Exception as e:
        logger.error(f"Refresh token error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to refresh token")
