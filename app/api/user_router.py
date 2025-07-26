"""
User Router for RAGBot API
Location: app/api/user_router.py

Handles user profile management, preferences, settings, and usage statistics.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from uuid import UUID
from enum import Enum

# Import dependencies
from ..rag_engine.db.session import get_db
from ..rag_engine.db.models import User, UserPreferences, UserSettings, UsageStat, AuditLog
from ..services.user_service import UserService
from ..api.auth_router import get_current_user
from ..utils.logger import get_logger

router = APIRouter()
logger = get_logger()

# Enums
class BiasMode(str, Enum):
    none = "none"
    academic = "academic"
    business = "business"
    creative = "creative"
    technical = "technical"

# Pydantic models
class UserProfileUpdate(BaseModel):
    name: Optional[str] = None
    picture_url: Optional[str] = None

class UserPreferencesModel(BaseModel):
    tone: Optional[str] = "balanced"  # friendly, professional, casual, balanced
    length: Optional[str] = "medium"  # short, medium, long
    language: Optional[str] = "en"
    bias_config: Optional[Dict[str, Any]] = {}

class UserSettingsModel(BaseModel):
    settings: Dict[str, Any]

class ConversationStyleModel(BaseModel):
    conversation_style: Dict[str, Any]

class UsageStatsResponse(BaseModel):
    today: Dict[str, Any]
    week: Dict[str, Any]
    month: Dict[str, Any]
    total: Dict[str, Any]

class UserActivityResponse(BaseModel):
    total_conversations: int
    total_messages: int
    total_documents: int
    avg_response_time: float
    most_used_features: List[str]
    recent_activity: List[Dict[str, Any]]

@router.get("/profile")
async def get_user_profile(
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get detailed user profile"""
    try:
        user_service = UserService(db)
        user_id = UUID(current_user["user_id"])
        
        profile = user_service.get_user_profile(user_id)
        
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )
        
        return profile
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user profile error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile"
        )

@router.put("/profile")
async def update_user_profile(
    profile_update: UserProfileUpdate,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Update user profile"""
    try:
        user_id = UUID(current_user["user_id"])
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update fields if provided
        if profile_update.name is not None:
            user.name = profile_update.name
        if profile_update.picture_url is not None:
            user.picture_url = profile_update.picture_url
        
        db.commit()
        
        # Log profile update
        audit_log = AuditLog(
            user_id=user_id,
            event_type="profile_update",
            event_details={
                "updated_fields": [
                    field for field, value in profile_update.dict().items() 
                    if value is not None
                ]
            }
        )
        db.add(audit_log)
        db.commit()
        
        logger.info(f"Profile updated for user: {user.email}")
        
        return {
            "message": "Profile updated successfully",
            "user": {
                "id": str(user.id),
                "name": user.name,
                "email": user.email,
                "picture_url": user.picture_url
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update profile error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )

@router.get("/preferences", response_model=UserPreferencesModel)
async def get_user_preferences(
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get user preferences"""
    try:
        user_id = UUID(current_user["user_id"])
        
        preferences = db.query(UserPreferences)\
            .filter(UserPreferences.user_id == user_id)\
            .first()
        
        if not preferences:
            # Return default preferences
            return UserPreferencesModel()
        
        return UserPreferencesModel(
            tone=preferences.tone,
            length=preferences.length,
            language=preferences.language,
            bias_config=preferences.bias_config or {}
        )
        
    except Exception as e:
        logger.error(f"Get preferences error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve preferences"
        )

@router.put("/preferences")
async def update_user_preferences(
    preferences: UserPreferencesModel,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Update user preferences"""
    try:
        user_id = UUID(current_user["user_id"])
        
        # Get or create preferences
        user_prefs = db.query(UserPreferences)\
            .filter(UserPreferences.user_id == user_id)\
            .first()
        
        if not user_prefs:
            user_prefs = UserPreferences(user_id=user_id)
            db.add(user_prefs)
        
        # Update preferences
        if preferences.tone:
            user_prefs.tone = preferences.tone
        if preferences.length:
            user_prefs.length = preferences.length
        if preferences.language:
            user_prefs.language = preferences.language
        if preferences.bias_config:
            user_prefs.bias_config = preferences.bias_config
        
        db.commit()
        
        # Log preferences update
        audit_log = AuditLog(
            user_id=user_id,
            event_type="preferences_update",
            event_details=preferences.dict()
        )
        db.add(audit_log)
        db.commit()
        
        logger.info(f"Preferences updated for user: {current_user['email']}")
        
        return {"message": "Preferences updated successfully"}
        
    except Exception as e:
        logger.error(f"Update preferences error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update preferences"
        )

@router.get("/settings")
async def get_user_settings(
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get user settings"""
    try:
        user_id = UUID(current_user["user_id"])
        
        settings = db.query(UserSettings)\
            .filter(UserSettings.user_id == user_id)\
            .first()
        
        if not settings:
            return {"settings": {}}
        
        return {"settings": settings.settings or {}}
        
    except Exception as e:
        logger.error(f"Get settings error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve settings"
        )

@router.put("/settings")
async def update_user_settings(
    settings_update: UserSettingsModel,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Update user settings"""
    try:
        user_id = UUID(current_user["user_id"])
        
        # Get or create settings
        user_settings = db.query(UserSettings)\
            .filter(UserSettings.user_id == user_id)\
            .first()
        
        if not user_settings:
            user_settings = UserSettings(user_id=user_id)
            db.add(user_settings)
        
        # Update settings
        user_settings.settings = settings_update.settings
        db.commit()
        
        # Log settings update
        audit_log = AuditLog(
            user_id=user_id,
            event_type="settings_update",
            event_details={"updated_settings": list(settings_update.settings.keys())}
        )
        db.add(audit_log)
        db.commit()
        
        logger.info(f"Settings updated for user: {current_user['email']}")
        
        return {"message": "Settings updated successfully"}
        
    except Exception as e:
        logger.error(f"Update settings error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update settings"
        )

@router.get("/usage-stats", response_model=UsageStatsResponse)
async def get_usage_statistics(
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get user usage statistics"""
    try:
        user_service = UserService(db)
        user_id = UUID(current_user["user_id"])
        
        stats = user_service.get_usage_statistics(user_id)
        
        return UsageStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Get usage stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve usage statistics"
        )

@router.get("/activity", response_model=UserActivityResponse)
async def get_user_activity(
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get user activity summary"""
    try:
        user_service = UserService(db)
        user_id = UUID(current_user["user_id"])
        
        activity = user_service.get_user_activity(user_id)
        
        return UserActivityResponse(**activity)
        
    except Exception as e:
        logger.error(f"Get user activity error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user activity"
        )

@router.put("/conversation-style")
async def update_conversation_style(
    style_update: ConversationStyleModel,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Update user's conversation style preferences"""
    try:
        user_id = UUID(current_user["user_id"])
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        user.conversation_style = style_update.conversation_style
        db.commit()
        
        # Log style update
        audit_log = AuditLog(
            user_id=user_id,
            event_type="conversation_style_update",
            event_details=style_update.conversation_style
        )
        db.add(audit_log)
        db.commit()
        
        logger.info(f"Conversation style updated for user: {user.email}")
        
        return {"message": "Conversation style updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update conversation style error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update conversation style"
        )

@router.get("/search-bias-mode")
async def get_search_bias_mode(
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get user's search bias mode"""
    try:
        user_id = UUID(current_user["user_id"])
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "search_bias_mode": user.search_bias_mode or BiasMode.none,
            "available_modes": [mode.value for mode in BiasMode]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get search bias mode error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve search bias mode"
        )

@router.put("/search-bias-mode")
async def update_search_bias_mode(
    bias_mode: BiasMode,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Update user's search bias mode"""
    try:
        user_id = UUID(current_user["user_id"])
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        user.search_bias_mode = bias_mode
        db.commit()
        
        # Log bias mode update
        audit_log = AuditLog(
            user_id=user_id,
            event_type="search_bias_mode_update",
            event_details={"new_mode": bias_mode.value}
        )
        db.add(audit_log)
        db.commit()
        
        logger.info(f"Search bias mode updated to '{bias_mode}' for user: {user.email}")
        
        return {"message": f"Search bias mode updated to '{bias_mode}'"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update search bias mode error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update search bias mode"
        )

@router.delete("/account")
async def delete_user_account(
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Delete user account (soft delete)"""
    try:
        user_id = UUID(current_user["user_id"])
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Soft delete
        user.is_deleted = True
        db.commit()
        
        # Log account deletion
        audit_log = AuditLog(
            user_id=user_id,
            event_type="account_deletion_request",
            event_details={
                "email": user.email,
                "name": user.name,
                "deletion_timestamp": datetime.utcnow().isoformat()
            }
        )
        db.add(audit_log)
        db.commit()
        
        logger.warning(f"Account deleted for user: {user.email}")
        
        return {
            "message": "Account successfully deleted",
            "support_email": "support@ragbot.com"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete account error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process account deletion request"
        )

@router.get("/audit-logs")
async def get_user_audit_logs(
    limit: int = 20,
    offset: int = 0,
    event_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get user's audit logs"""
    try:
        user_id = UUID(current_user["user_id"])
        
        query = db.query(AuditLog).filter(AuditLog.user_id == user_id)
        
        if event_type:
            query = query.filter(AuditLog.event_type == event_type)
        if start_date:
            query = query.filter(AuditLog.timestamp >= start_date)
        if end_date:
            query = query.filter(AuditLog.timestamp <= end_date)
        
        logs = query.order_by(AuditLog.timestamp.desc())\
                  .offset(offset)\
                  .limit(limit)\
                  .all()
        
        return [
            {
                "id": str(log.id),
                "event_type": log.event_type,
                "event_details": log.event_details,
                "timestamp": log.timestamp
            }
            for log in logs
        ]
        
    except Exception as e:
        logger.error(f"Get audit logs error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit logs"
        )