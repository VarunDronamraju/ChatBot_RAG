"""
Sync handler for PyQt6 RAGBot application.
Manages offline message queuing and synchronization with the backend.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from PyQt6.QtCore import QObject, pyqtSignal
import json
from queue import Queue
import uuid

from .api_client import SyncAPIClient, APIResponse
from .auth_handler import auth_required
from ..rag_engine.local_cache.sqlite_session import SQLiteSessionManager

logger = logging.getLogger(__name__)

class SyncHandler(QObject):
    """Handles offline message queuing and synchronization for PyQt6 application"""
    
    # Signals
    sync_started = pyqtSignal()
    sync_completed = pyqtSignal(dict)
    sync_failed = pyqtSignal(str)
    offline_message_added = pyqtSignal(dict)
    
    def __init__(self, api_client: SyncAPIClient, sqlite_manager: SQLiteSessionManager, auth_handler=None):
        super().__init__()
        self.api_client = api_client
        self.sqlite_manager = sqlite_manager
        self.auth_handler = auth_handler
        
        # Offline message queue
        self.offline_queue = Queue()
        self.is_online = False
        
        # Load any existing offline messages
        self._load_offline_messages()
    
    def _load_offline_messages(self):
        """Load offline messages from SQLite"""
        try:
            offline_messages = self.sqlite_manager.get_offline_messages()
            for msg in offline_messages:
                self.offline_queue.put(msg)
            logger.info(f"Loaded {self.offline_queue.qsize()} offline messages")
        except Exception as e:
            logger.error(f"Error loading offline messages: {str(e)}")
            self.sync_failed.emit(f"Error loading offline messages: {str(e)}")
    
    def _save_offline_message(self, message: Dict[str, Any]):
        """Save offline message to SQLite"""
        try:
            self.sqlite_manager.save_offline_message(message)
            logger.debug(f"Saved offline message: {message['id']}")
        except Exception as e:
            logger.error(f"Error saving offline message: {str(e)}")
            self.sync_failed.emit(f"Error saving offline message: {str(e)}")
    
    def _clear_offline_messages(self):
        """Clear offline messages from SQLite"""
        try:
            self.sqlite_manager.clear_offline_messages()
            logger.debug("Cleared offline messages")
        except Exception as e:
            logger.error(f"Error clearing offline messages: {str(e)}")
            self.sync_failed.emit(f"Error clearing offline messages: {str(e)}")
    
    def add_offline_message(self, content: str, session_id: str, format_preference: Optional[str] = None):
        """Add a message to the offline queue"""
        try:
            message = {
                'id': str(uuid.uuid4()),
                'content': content,
                'session_id': session_id,
                'format_preference': format_preference,
                'timestamp': datetime.now().isoformat(),
                'user_id': self.auth_handler.get_user_id() if self.auth_handler else None
            }
            self.offline_queue.put(message)
            self._save_offline_message(message)
            self.offline_message_added.emit(message)
            logger.info(f"Added offline message: {message['id']} for session {session_id}")
        except Exception as e:
            logger.error(f"Error adding offline message: {str(e)}")
            self.sync_failed.emit(f"Error adding offline message: {str(e)}")
    
    @auth_required
    def sync_offline_data(self):
        """Sync offline messages with the backend"""
        if not self.is_online:
            logger.warning("Cannot sync: Client is offline")
            self.sync_failed.emit("Cannot sync: Client is offline")
            return
        
        try:
            self.sync_started.emit()
            offline_messages = []
            while not self.offline_queue.empty():
                offline_messages.append(self.offline_queue.get())
            
            if not offline_messages:
                self.sync_completed.emit({"synced_count": 0, "failed_count": 0})
                return
            
            sync_request = {
                'offline_messages': [
                    {
                        'content': msg['content'],
                        'session_id': msg['session_id'],
                        'format_preference': msg.get('format_preference'),
                        'timestamp': msg['timestamp']
                    }
                    for msg in offline_messages
                ],
                'last_sync_timestamp': datetime.now().isoformat()
            }
            
            response = self.api_client._make_request(
                method="POST",
                endpoint="/chat/sync",
                data=sync_request
            )
            
            if response.success:
                self._clear_offline_messages()
                self.sync_completed.emit(response.data)
                logger.info(f"Synced {response.data.get('synced_count', 0)} offline messages")
            else:
                for msg in offline_messages:
                    self.offline_queue.put(msg)  # Re-queue failed messages
                self.sync_failed.emit(response.error or "Sync failed")
                logger.warning(f"Sync failed: {response.error}")
        except Exception as e:
            logger.error(f"Sync error: {str(e)}")
            self.sync_failed.emit(f"Sync error: {str(e)}")
    
    def set_online_status(self, is_online: bool):
        """Update online status and trigger sync if online"""
        self.is_online = is_online
        if is_online and not self.offline_queue.empty():
            self.sync_offline_data()
        logger.info(f"Online status set to: {is_online}")
    
    def get_offline_messages(self) -> List[Dict[str, Any]]:
        """Get current offline messages"""
        try:
            messages = []
            temp_queue = Queue()
            while not self.offline_queue.empty():
                msg = self.offline_queue.get()
                messages.append(msg)
                temp_queue.put(msg)
            self.offline_queue = temp_queue
            return messages
        except Exception as e:
            logger.error(f"Error getting offline messages: {str(e)}")
            return []
    
    def add_error_handler(self, handler: Callable[[str], None]):
        """Add handler for sync errors"""
        self.sync_failed.connect(handler)
    
    def remove_error_handler(self, handler: Callable[[str], None]):
        """Remove handler for sync errors"""
        self.sync_failed.disconnect(handler)