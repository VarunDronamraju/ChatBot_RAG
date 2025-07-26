"""
Authentication handler for PyQt6 RAGBot application.
Manages login/logout, token storage, and authentication state.
"""

import json
import logging
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from cryptography.fernet import Fernet
import os
import base64

from .api_client import SyncAPIClient, APIResponse
from ..rag_engine.local_cache.sqlite_session import SQLiteSessionManager

logger = logging.getLogger(__name__)

class AuthHandler(QObject):
    """Handles authentication for PyQt6 application"""
    
    # Signals
    login_success = pyqtSignal(dict)  # User data
    login_failed = pyqtSignal(str)    # Error message
    logout_completed = pyqtSignal()
    token_refreshed = pyqtSignal()
    auth_error = pyqtSignal(str)      # Auth errors
    
    def __init__(self, api_client: SyncAPIClient, sqlite_manager: SQLiteSessionManager):
        super().__init__()
        self.api_client = api_client
        self.sqlite_manager = sqlite_manager
        
        # Auth state
        self.current_user: Optional[Dict[str, Any]] = None
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        
        # Token refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._auto_refresh_token)
        
        # Encryption for token storage
        self._encryption_key = self._get_or_create_encryption_key()
        
        # Load saved auth state
        self._load_saved_auth()
        
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for token storage"""
        key_file = os.path.join(os.path.expanduser("~"), ".ragbot_key")
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Read/write for owner only
            return key
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        f = Fernet(self._encryption_key)
        return f.encrypt(data.encode()).decode()
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        f = Fernet(self._encryption_key)
        return f.decrypt(encrypted_data.encode()).decode()
    
    def _save_auth_state(self):
        """Save authentication state to local storage"""
        if not self.access_token:
            return
            
        auth_data = {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "user_data": self.current_user,
            "expires_at": self.token_expires_at.isoformat() if self.token_expires_at else None
        }
        
        try:
            encrypted_data = self._encrypt_data(json.dumps(auth_data))
            self.sqlite_manager.save_auth_data(encrypted_data)
            logger.debug("Auth state saved")
        except Exception as e:
            logger.error(f"Failed to save auth state: {str(e)}")
    
    def _load_saved_auth(self):
        """Load saved authentication state"""
        try:
            encrypted_data = self.sqlite_manager.get_auth_data()
            if not encrypted_data:
                return
                
            decrypted_data = self._decrypt_data(encrypted_data)
            auth_data = json.loads(decrypted_data)
            
            self.access_token = auth_data.get("access_token")
            self.refresh_token = auth_data.get("refresh_token")
            self.current_user = auth_data.get("user_data")
            
            expires_at = auth_data.get("expires_at")
            if expires_at:
                self.token_expires_at = datetime.fromisoformat(expires_at)
                
            # Check if token is still valid
            if self.token_expires_at and datetime.now() < self.token_expires_at:
                self.api_client.set_auth_token(self.access_token)
                self._start_refresh_timer()
                logger.info("Loaded valid auth state")
            else:
                # Try to refresh token
                self._refresh_token_silent()
                
        except Exception as e:
            logger.error(f"Failed to load auth state: {str(e)}")
            self._clear_auth_state()
    
    def _clear_auth_state(self):
        """Clear authentication state"""
        self.current_user = None
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        self.refresh_timer.stop()
        self.api_client.clear_auth_token()
        self.sqlite_manager.clear_auth_data()
        logger.debug("Auth state cleared")
    
    def _start_refresh_timer(self):
        """Start automatic token refresh timer"""
        if not self.token_expires_at:
            return
            
        # Refresh 5 minutes before expiry
        refresh_time = self.token_expires_at - timedelta(minutes=5)
        now = datetime.now()
        
        if refresh_time > now:
            seconds_until_refresh = (refresh_time - now).total_seconds()
            self.refresh_timer.start(int(seconds_until_refresh * 1000))
            logger.debug(f"Token refresh scheduled in {seconds_until_refresh} seconds")
    
    def _auto_refresh_token(self):
        """Automatically refresh token"""
        logger.info("Auto-refreshing token")
        self._refresh_token_silent()
    
    def _refresh_token_silent(self):
        """Refresh token without UI feedback"""
        try:
            response = self.api_client.refresh_token()
            if response.success:
                self._handle_successful_auth(response.data)
                self.token_refreshed.emit()
                logger.info("Token refreshed successfully")
            else:
                logger.warning("Token refresh failed, clearing auth state")
                self._clear_auth_state()
                self.auth_error.emit("Session expired. Please log in again.")
        except Exception as e:
            logger.error(f"Token refresh error: {str(e)}")
            self._clear_auth_state()
            self.auth_error.emit("Authentication error occurred")
    
    def _handle_successful_auth(self, auth_data: Dict[str, Any]):
        """Handle successful authentication response"""
        self.access_token = auth_data.get("access_token")
        self.refresh_token = auth_data.get("refresh_token")
        self.current_user = auth_data.get("user")
        
        # Calculate token expiry
        expires_in = auth_data.get("expires_in", 3600)  # Default 1 hour
        self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
        
        # Update API client
        self.api_client.set_auth_token(self.access_token)
        
        # Save auth state
        self._save_auth_state()
        
        # Start refresh timer
        self._start_refresh_timer()
    
    def login(self, username: str, password: str):
        """Perform user login"""
        try:
            logger.info(f"Attempting login for user: {username}")
            response = self.api_client.login(username, password)
            
            if response.success:
                self._handle_successful_auth(response.data)
                self.login_success.emit(self.current_user)
                logger.info("Login successful")
            else:
                error_msg = response.error or "Login failed"
                self.login_failed.emit(error_msg)
                logger.warning(f"Login failed: {error_msg}")
                
        except Exception as e:
            error_msg = f"Login error: {str(e)}"
            logger.error(error_msg)
            self.login_failed.emit(error_msg)
    
    def register(self, username: str, email: str, password: str):
        """Perform user registration"""
        try:
            logger.info(f"Attempting registration for user: {username}")
            response = self.api_client.register(username, email, password)
            
            if response.success:
                # Registration successful, now login
                self.login(username, password)
            else:
                error_msg = response.error or "Registration failed"
                self.login_failed.emit(error_msg)
                logger.warning(f"Registration failed: {error_msg}")
                
        except Exception as e:
            error_msg = f"Registration error: {str(e)}"
            logger.error(error_msg)
            self.login_failed.emit(error_msg)
    
    def logout(self):
        """Perform user logout"""
        try:
            logger.info("Logging out user")
            
            # Call logout endpoint (optional, for server-side cleanup)
            try:
                self.api_client.logout()
            except Exception as e:
                logger.warning(f"Server logout failed: {str(e)}")
            
            # Clear local auth state
            self._clear_auth_state()
            self.logout_completed.emit()
            logger.info("Logout completed")
            
        except Exception as e:
            logger.error(f"Logout error: {str(e)}")
            # Still clear local state even if server call fails
            self._clear_auth_state()
            self.logout_completed.emit()
    
    def refresh_token(self):
        """Manually refresh authentication token"""
        try:
            logger.info("Manually refreshing token")
            response = self.api_client.refresh_token()
            
            if response.success:
                self._handle_successful_auth(response.data)
                self.token_refreshed.emit()
                logger.info("Manual token refresh successful")
                return True
            else:
                error_msg = response.error or "Token refresh failed"
                logger.warning(f"Manual token refresh failed: {error_msg}")
                self.auth_error.emit(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"Token refresh error: {str(e)}"
            logger.error(error_msg)
            self.auth_error.emit(error_msg)
            return False
    
    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated"""
        return (
            self.access_token is not None and 
            self.current_user is not None and
            self.token_expires_at is not None and
            datetime.now() < self.token_expires_at
        )
    
    def is_token_expired(self) -> bool:
        """Check if current token is expired"""
        if not self.token_expires_at:
            return True
        return datetime.now() >= self.token_expires_at
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get current user data"""
        return self.current_user.copy() if self.current_user else None
    
    def get_user_id(self) -> Optional[str]:
        """Get current user ID"""
        return self.current_user.get("id") if self.current_user else None
    
    def get_username(self) -> Optional[str]:
        """Get current username"""
        return self.current_user.get("username") if self.current_user else None
    
    def update_user_profile(self, profile_data: Dict[str, Any]):
        """Update current user profile data"""
        if self.current_user:
            self.current_user.update(profile_data)
            self._save_auth_state()
    
    def add_auth_error_handler(self, handler: Callable[[str], None]):
        """Add handler for authentication errors"""
        self.auth_error.connect(handler)
    
    def remove_auth_error_handler(self, handler: Callable[[str], None]):
        """Remove handler for authentication errors"""
        self.auth_error.disconnect(handler)


# Utility decorator for authentication required
def auth_required(func):
    """Decorator to ensure authentication before method execution"""
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'auth_handler') and not self.auth_handler.is_authenticated():
            logger.warning(f"Authentication required for {func.__name__}")
            if hasattr(self, 'auth_error'):
                self.auth_error.emit("Authentication required")
            return None
        return func(self, *args, **kwargs)
    return wrapper