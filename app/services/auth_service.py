"""
Authentication Service for RAGBot
Location: app/services/auth_service.py

Handles user authentication, JWT token management, and password operations.
"""

import jwt
import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from uuid import UUID, uuid4
from ..rag_engine.db.models import User
from ..utils.logger import get_logger

logger = get_logger()

class AuthService:
    def __init__(self, db, local_db):
        self.db = db
        self.local_db = local_db
        self.secret_key = os.getenv("JWT_SECRET_KEY", "fallback-secret-key")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
        self.refresh_token_expire_days = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 7))

    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{password_hash}"

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            salt, password_hash = hashed_password.split(':')
            return hashlib.sha256((password + salt).encode()).hexdigest() == password_hash
        except ValueError:
            return False

    def create_user(self, email: str, password: str, name: str) -> User:
        """Create new user"""
        hashed_password = self.hash_password(password)
        user = User(
            id=uuid4(),
            email=email,
            name=name,
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow(),
            usage_metrics={},
            global_tags=[],
            search_bias_mode="none",
            conversation_style={}
        )
      
        # Use raw SQLite connection
        import sqlite3
        conn = sqlite3.connect("app/local/auth_cache.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO auth_cache (user_id, password_hash) VALUES (?, ?)",
            (str(user.id), hashed_password)
        )
        conn.commit()
        conn.close()

        self.db.add(user)
        self.db.commit()
        return user

    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        user = self.db.query(User).filter(User.email == email).first()
        if not user:
            return None

        # Use raw SQLite connection
        import sqlite3
        conn = sqlite3.connect("app/local/auth_cache.db")
        cursor = conn.execute("SELECT password_hash FROM auth_cache WHERE user_id = ?", (str(user.id),))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None

        if self.verify_password(password, result[0]):
            return user
        return None

    def create_access_token(self, user_id: str, email: str, role: str = "user", remember_me: bool = False) -> Dict[str, Any]:
        """Create JWT access token"""
        if remember_me:
            expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)

        to_encode = {
            "user_id": user_id,
            "email": email,
            "exp": expire,
            "role": role, 
            "iat": datetime.utcnow(),
            "type": "access"
        }

        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

        # Use raw SQLite connection
        import sqlite3
        conn = sqlite3.connect("app/local/auth_cache.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO token_cache (token, user_id, expires_at, is_valid) VALUES (?, ?, ?, ?)",
            (encoded_jwt, user_id, expire.isoformat(), True)
        )
        conn.commit()
        conn.close()

        return {
            "access_token": encoded_jwt,
            "expires_in": int((expire - datetime.utcnow()).total_seconds())
        }

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            # Use raw SQLite connection
            import sqlite3
            conn = sqlite3.connect("app/local/auth_cache.db")
            cursor = conn.execute(
                "SELECT user_id, expires_at, is_valid FROM token_cache WHERE token = ?",
                (token,)
            )
            result = cursor.fetchone()
            conn.close()
            
            if not result or not result[2]:
                return None

            expires_at = datetime.fromisoformat(result[1])
            if expires_at < datetime.utcnow():
                self.invalidate_token(token)
                return None

            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return {
                "user_id": payload["user_id"],
                "email": payload["email"],
                "name": self.get_user_name(payload["user_id"])
            }

        except jwt.PyJWTError:
            return None

    def invalidate_token(self, token: str) -> bool:
        """Invalidate a token"""
        try:
            # Use raw SQLite connection
            import sqlite3
            conn = sqlite3.connect("app/local/auth_cache.db")
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE token_cache SET is_valid = FALSE WHERE token = ?",
                (token,)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Token invalidation failed: {str(e)}")
            return False

    def change_password(self, user_id: UUID, current_password: str, new_password: str) -> bool:
        """Change user password"""
        try:
            # Use raw SQLite connection
            import sqlite3
            conn = sqlite3.connect("app/local/auth_cache.db")
            cursor = conn.execute(
                "SELECT password_hash FROM auth_cache WHERE user_id = ?",
                (str(user_id),)
            )
            result = cursor.fetchone()
            if not result or not self.verify_password(current_password, result[0]):
                conn.close()
                return False

            new_hash = self.hash_password(new_password)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE auth_cache SET password_hash = ? WHERE user_id = ?",
                (new_hash, str(user_id))
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Password change failed: {str(e)}")
            return False

    def get_user_name(self, user_id: str) -> str:
        """Get user name by ID"""
        try:
            user = self.db.query(User).filter(User.id == UUID(user_id)).first()
            return user.name if user else "Unknown"
        except Exception:
            return "Unknown"