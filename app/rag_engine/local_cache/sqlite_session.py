"""
SQLite session management for RAGBot local cache.
Handles local storage of authentication data and offline messages.
"""

import os
import logging
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import json
from typing import Optional


logger = logging.getLogger(__name__)

Base = declarative_base()

class AuthCache(Base):
    """Table for storing encrypted auth data"""
    __tablename__ = "auth_cache"
    id = Column(String, primary_key=True, default="singleton")
    encrypted_data = Column(Text)
    last_updated = Column(DateTime, server_default=func.now(), onupdate=func.now())

class OfflineMessage(Base):
    """Table for storing offline messages"""
    __tablename__ = "offline_messages"
    id = Column(String, primary_key=True)
    content = Column(Text)
    session_id = Column(String)
    format_preference = Column(String, nullable=True)
    timestamp = Column(DateTime, server_default=func.now())
    user_id = Column(String, nullable=True)

SQLITE_DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/cache.db")
engine = create_engine(f"sqlite:///{SQLITE_DB_PATH}", echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class SQLiteSessionManager:
    """Manages SQLite sessions for local caching"""
    
    def __init__(self):
        Base.metadata.create_all(engine)
        self.session = SessionLocal()
    
    def save_auth_data(self, encrypted_data: str):
        """Save encrypted authentication data"""
        try:
            auth_cache = self.session.query(AuthCache).first()
            if auth_cache:
                auth_cache.encrypted_data = encrypted_data
                auth_cache.last_updated = datetime.now()
            else:
                auth_cache = AuthCache(id="singleton", encrypted_data=encrypted_data)
                self.session.add(auth_cache)
            self.session.commit()
            logger.debug("Saved auth data to SQLite")
        except Exception as e:
            logger.error(f"Error saving auth data: {str(e)}")
            self.session.rollback()
            raise
    
    def get_auth_data(self) -> Optional[str]:
        """Retrieve encrypted authentication data"""
        try:
            auth_cache = self.session.query(AuthCache).first()
            return auth_cache.encrypted_data if auth_cache else None
        except Exception as e:
            logger.error(f"Error retrieving auth data: {str(e)}")
            return None
    
    def clear_auth_data(self):
        """Clear authentication data"""
        try:
            self.session.query(AuthCache).delete()
            self.session.commit()
            logger.debug("Cleared auth data from SQLite")
        except Exception as e:
            logger.error(f"Error clearing auth data: {str(e)}")
            self.session.rollback()
            raise
    
    def save_offline_message(self, message: dict[str, any]):
        """Save an offline message"""
        try:
            offline_msg = OfflineMessage(
                id=message['id'],
                content=message['content'],
                session_id=message['session_id'],
                format_preference=message.get('format_preference'),
                timestamp=datetime.fromisoformat(message['timestamp']),
                user_id=message.get('user_id')
            )
            self.session.add(offline_msg)
            self.session.commit()
            logger.debug(f"Saved offline message: {message['id']}")
        except Exception as e:
            logger.error(f"Error saving offline message: {str(e)}")
            self.session.rollback()
            raise
    
    def get_offline_messages(self) -> list[dict[str, any]]:
        """Retrieve all offline messages"""
        try:
            messages = self.session.query(OfflineMessage).all()
            return [
                {
                    'id': msg.id,
                    'content': msg.content,
                    'session_id': msg.session_id,
                    'format_preference': msg.format_preference,
                    'timestamp': msg.timestamp.isoformat(),
                    'user_id': msg.user_id
                }
                for msg in messages
            ]
        except Exception as e:
            logger.error(f"Error retrieving offline messages: {str(e)}")
            return []
    
    def clear_offline_messages(self):
        """Clear all offline messages"""
        try:
            self.session.query(OfflineMessage).delete()
            self.session.commit()
            logger.debug("Cleared offline messages from SQLite")
        except Exception as e:
            logger.error(f"Error clearing offline messages: {str(e)}")
            self.session.rollback()
            raise
    
    def close(self):
        """Close the session"""
        try:
            self.session.close()
            logger.debug("SQLite session closed")
        except Exception as e:
            logger.error(f"Error closing SQLite session: {str(e)}")

def get_local_db():
    """Dependency for FastAPI to provide SQLite session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()