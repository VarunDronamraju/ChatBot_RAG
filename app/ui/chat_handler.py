"""
Chat handler for PyQt6 RAGBot application.
Manages chat sessions, message processing, and response formatting.
"""

import logging
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from PyQt6.QtCore import QObject, pyqtSignal, QThread, QMutex
from dataclasses import dataclass
import json
import uuid
import markdown
from bs4 import BeautifulSoup

from .api_client import SyncAPIClient, APIResponse
from .auth_handler import auth_required
from ..rag_engine.local_cache.sqlite_session import SQLiteSessionManager

logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Chat message data structure"""
    id: str
    content: str
    is_user: bool
    timestamp: datetime
    session_id: str
    response_format: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ChatSession:
    """Chat session data structure"""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

class ChatWorker(QThread):
    """Worker thread for chat processing"""
    
    message_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, api_client: SyncAPIClient, message: str, session_id: str, format_preference: Optional[str] = None):
        super().__init__()
        self.api_client = api_client
        self.message = message
        self.session_id = session_id
        self.format_preference = format_preference
    
    def run(self):
        """Process chat message in separate thread"""
        try:
            response = self.api_client.send_message(self.message, self.session_id)
            if response.success:
                self.message_received.emit(response.data)
            else:
                self.error_occurred.emit(response.error or "Failed to send message")
        except Exception as e:
            logger.error(f"Chat worker error: {str(e)}")
            self.error_occurred.emit(f"Chat processing error: {str(e)}")

class ChatHandler(QObject):
    """Handles chat functionality for PyQt6 application"""
    
    # Signals
    message_sent = pyqtSignal(dict)           # User message data
    message_received = pyqtSignal(dict)       # Bot response data
    session_created = pyqtSignal(dict)        # New session data
    session_loaded = pyqtSignal(dict)         # Loaded session data
    sessions_updated = pyqtSignal(list)       # Updated sessions list
    typing_started = pyqtSignal()             # Bot is typing
    typing_stopped = pyqtSignal()             # Bot stopped typing
    error_occurred = pyqtSignal(str)          # Error message
    search_results = pyqtSignal(list)         # Document search results
    
    def __init__(self, api_client: SyncAPIClient, sqlite_manager: SQLiteSessionManager, auth_handler=None):
        super().__init__()
        self.api_client = api_client
        self.sqlite_manager = sqlite_manager
        self.auth_handler = auth_handler
        
        # Current state
        self.current_session: Optional[ChatSession] = None
        self.current_messages: List[ChatMessage] = []
        self.available_sessions: List[ChatSession] = []
        
        # Threading
        self.worker_threads: List[ChatWorker] = []
        self.thread_mutex = QMutex()
        
        # Response formatters
        self.response_formatters = {
            'bullets': self._format_bullet_response,
            'table': self._format_table_response,
            'summary': self._format_summary_response,
            'code': self._format_code_response,
            'detailed': self._format_detailed_response,
            'comparison': self._format_comparison_response,
            'default': self._format_default_response
        }
        
        # Load sessions on initialization
        self._load_sessions()
    
    def _load_sessions(self):
        """Load chat sessions from API"""
        try:
            response = self.api_client.get_chat_sessions()
            if response.success:
                sessions_data = response.data.get('sessions', [])
                self.available_sessions = [
                    ChatSession(
                        id=s['id'],
                        title=s['title'],
                        created_at=datetime.fromisoformat(s['created_at']),
                        updated_at=datetime.fromisoformat(s['updated_at']),
                        message_count=s.get('message_count', 0)
                    )
                    for s in sessions_data
                ]
                self.sessions_updated.emit([s.__dict__ for s in self.available_sessions])
                logger.info(f"Loaded {len(self.available_sessions)} chat sessions")
            else:
                logger.warning(f"Failed to load sessions: {response.error}")
                self.error_occurred.emit(response.error or "Failed to load sessions")
        except Exception as e:
            logger.error(f"Error loading sessions: {str(e)}")
            self.error_occurred.emit(f"Error loading sessions: {str(e)}")
    
    @auth_required
    def create_session(self, title: Optional[str] = None):
        """Create a new chat session"""
        try:
            response = self.api_client.create_chat_session(title)
            if response.success:
                session_data = response.data
                session = ChatSession(
                    id=session_data['id'],
                    title=session_data['title'],
                    created_at=datetime.fromisoformat(session_data['created_at']),
                    updated_at=datetime.fromisoformat(session_data['updated_at']),
                    message_count=session_data.get('message_count', 0)
                )
                self.available_sessions.append(session)
                self.current_session = session
                self.current_messages = []
                self.session_created.emit(session_data)
                self.sessions_updated.emit([s.__dict__ for s in self.available_sessions])
                logger.info(f"Created new session: {session.id}")
            else:
                self.error_occurred.emit(response.error or "Failed to create session")
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            self.error_occurred.emit(f"Error creating session: {str(e)}")
    
    @auth_required
    def load_session(self, session_id: str):
        """Load an existing chat session"""
        try:
            response = self.api_client.get_chat_history(session_id)
            if response.success:
                self.current_session = next(
                    (s for s in self.available_sessions if s.id == session_id), None
                )
                if not self.current_session:
                    self._load_sessions()
                    self.current_session = next(
                        (s for s in self.available_sessions if s.id == session_id), None
                    )
                self.current_messages = [
                    ChatMessage(
                        id=m['id'],
                        content=m['content'],
                        is_user=m['role'] == 'user',
                        timestamp=datetime.fromisoformat(m['timestamp']),
                        session_id=session_id,
                        response_format=m.get('format_used'),
                        metadata={'sources': m.get('sources', [])}
                    )
                    for m in response.data
                ]
                self.session_loaded.emit(self.current_session.__dict__)
                logger.info(f"Loaded session: {session_id}")
            else:
                self.error_occurred.emit(response.error or "Failed to load session")
        except Exception as e:
            logger.error(f"Error loading session: {str(e)}")
            self.error_occurred.emit(f"Error loading session: {str(e)}")
    
    @auth_required
    def send_message(self, content: str, format_preference: Optional[str] = None):
        """Send a message and process response"""
        if not self.current_session:
            self.create_session()
        
        try:
            self.thread_mutex.lock()
            message_id = str(uuid.uuid4())
            user_message = ChatMessage(
                id=message_id,
                content=content,
                is_user=True,
                timestamp=datetime.now(),
                session_id=self.current_session.id,
                response_format=format_preference
            )
            self.current_messages.append(user_message)
            self.message_sent.emit(user_message.__dict__)
            self.typing_started.emit()
            
            worker = ChatWorker(self.api_client, content, self.current_session.id, format_preference)
            worker.message_received.connect(self._handle_message_response)
            worker.error_occurred.connect(self._handle_message_error)
            worker.finished.connect(self._worker_finished)
            self.worker_threads.append(worker)
            worker.start()
            
            logger.info(f"Sent message in session {self.current_session.id}")
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            self.error_occurred.emit(f"Error sending message: {str(e)}")
            self.typing_stopped.emit()
        finally:
            self.thread_mutex.unlock()
    
    def _handle_message_response(self, response_data: dict):
        """Handle successful message response"""
        try:
            message = ChatMessage(
                id=response_data['id'],
                content=response_data['content'],
                is_user=False,
                timestamp=datetime.fromisoformat(response_data['timestamp']),
                session_id=response_data.get('session_id', self.current_session.id),
                response_format=response_data.get('format_used', 'default'),
                metadata={
                    'sources': response_data.get('sources', []),
                    'response_time': response_data.get('response_time'),
                    'source_type': response_data.get('source_type')
                }
            )
            formatted_content = self._format_response(response_data['content'], response_data['format_used'])
            message.content = formatted_content
            self.current_messages.append(message)
            self.message_received.emit(message.__dict__)
            self.typing_stopped.emit()
            logger.info(f"Received response for session {self.current_session.id}")
        except Exception as e:
            logger.error(f"Error handling message response: {str(e)}")
            self.error_occurred.emit(f"Error processing response: {str(e)}")
    
    def _handle_message_error(self, error: str):
        """Handle message processing error"""
        self.error_occurred.emit(error)
        self.typing_stopped.emit()
    
    def _worker_finished(self):
        """Clean up finished worker threads"""
        self.thread_mutex.lock()
        self.worker_threads = [w for w in self.worker_threads if not w.isFinished()]
        self.thread_mutex.unlock()
    
    @auth_required
    def search_documents(self, query: str, search_type: str = 'hybrid', max_results: int = 10):
        """Search documents and web content"""
        try:
            response = self.api_client.search_documents(query, max_results)
            if response.success:
                self.search_results.emit(response.data.get('results', []))
                logger.info(f"Search performed: '{query}'")
            else:
                self.error_occurred.emit(response.error or "Search failed")
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            self.error_occurred.emit(f"Search error: {str(e)}")
    
    @auth_required
    def delete_session(self, session_id: str):
        """Delete a chat session"""
        try:
            response = self.api_client.delete_chat_session(session_id)
            if response.success:
                self.available_sessions = [s for s in self.available_sessions if s.id != session_id]
                if self.current_session and self.current_session.id == session_id:
                    self.current_session = None
                    self.current_messages = []
                self.sessions_updated.emit([s.__dict__ for s in self.available_sessions])
                logger.info(f"Deleted session: {session_id}")
            else:
                self.error_occurred.emit(response.error or "Failed to delete session")
        except Exception as e:
            logger.error(f"Error deleting session: {str(e)}")
            self.error_occurred.emit(f"Error deleting session: {str(e)}")
    
    def _format_bullet_response(self, content: str, metadata: Dict[str, Any]) -> str:
        """Format response as bullet points"""
        try:
            lines = content.split('\n')
            formatted = '<ul>' + ''.join(f'<li>{line.strip()}</li>' for line in lines if line.strip()) + '</ul>'
            return self._html_to_markdown(formatted)
        except Exception as e:
            logger.error(f"Error formatting bullet response: {str(e)}")
            return content
    
    def _format_table_response(self, content: str, metadata: Dict[str, Any]) -> str:
        """Format response as a table"""
        try:
            # Assuming content is JSON or CSV-like
            data = json.loads(content) if content.startswith('[') or content.startswith('{') else content
            if isinstance(data, list) and all(isinstance(row, dict) for row in data):
                headers = list(data[0].keys())
                rows = [list(row.values()) for row in data]
                html = '<table><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>'
                for row in rows:
                    html += '<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>'
                html += '</table>'
                return self._html_to_markdown(html)
            return content
        except Exception as e:
            logger.error(f"Error formatting table response: {str(e)}")
            return content
    
    def _format_summary_response(self, content: str, metadata: Dict[str, Any]) -> str:
        """Format response as a summary"""
        try:
            return f'<summary>{content}</summary>'
        except Exception as e:
            logger.error(f"Error formatting summary response: {str(e)}")
            return content
    
    def _format_code_response(self, content: str, metadata: Dict[str, Any]) -> str:
        """Format response as code"""
        try:
            return f'<pre><code>{content}</code></pre>'
        except Exception as e:
            logger.error(f"Error formatting code response: {str(e)}")
            return content
    
    def _format_detailed_response(self, content: str, metadata: Dict[str, Any]) -> str:
        """Format detailed response"""
        try:
            return f'<div>{content}<p>Sources: {", ".join(metadata.get("sources", []))}</p></div>'
        except Exception as e:
            logger.error(f"Error formatting detailed response: {str(e)}")
            return content
    
    def _format_comparison_response(self, content: str, metadata: Dict[str, Any]) -> str:
        """Format comparison response"""
        try:
            data = json.loads(content) if content.startswith('[') or content.startswith('{') else content
            if isinstance(data, list):
                html = '<table><tr><th>Item</th><th>Details</th></tr>'
                for item in data:
                    html += f'<tr><td>{item.get("item", "")}</td><td>{item.get("details", "")}</td></tr>'
                html += '</table>'
                return self._html_to_markdown(html)
            return content
        except Exception as e:
            logger.error(f"Error formatting comparison response: {str(e)}")
            return content
    
    def _format_default_response(self, content: str, metadata: Dict[str, Any]) -> str:
        """Default response formatting"""
        return content
    
    def _format_response(self, content: str, format_type: str) -> str:
        """Apply appropriate formatting to response"""
        metadata = {'sources': []}  # Placeholder for metadata
        formatter = self.response_formatters.get(format_type, self._format_default_response)
        return formatter(content, metadata)
    
    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML to Markdown for PyQt6 display"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            return markdown.markdown(soup.get_text())
        except Exception as e:
            logger.error(f"Error converting HTML to Markdown: {str(e)}")
            return html
    
    def get_current_session(self) -> Optional[Dict[str, Any]]:
        """Get current session data"""
        return self.current_session.__dict__ if self.current_session else None
    
    def get_current_messages(self) -> List[Dict[str, Any]]:
        """Get current session messages"""
        return [m.__dict__ for m in self.current_messages]
    
    def add_error_handler(self, handler: Callable[[str], None]):
        """Add handler for errors"""
        self.error_occurred.connect(handler)
    
    def remove_error_handler(self, handler: Callable[[str], None]):
        """Remove handler for errors"""
        self.error_occurred.disconnect(handler)