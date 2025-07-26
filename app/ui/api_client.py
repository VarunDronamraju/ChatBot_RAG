"""
HTTP client for PyQt6 RAGBot application.
Handles all API communication with the FastAPI backend.
"""

import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    """Standard API response wrapper"""
    success: bool
    data: Any = None
    error: str = None
    status_code: int = None

class APIClient:
    """Async HTTP client for RAGBot FastAPI backend"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        self.auth_token: Optional[str] = None
        self.timeout = aiohttp.ClientTimeout(total=30)
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_session()
        
    async def start_session(self):
        """Initialize aiohttp session"""
        if not self.session or self.session.closed:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            logger.info("API client session started")
            
    async def close_session(self):
        """Close aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("API client session closed")
            
    def set_auth_token(self, token: str):
        """Set authentication token for requests"""
        self.auth_token = token
        logger.debug("Auth token updated")
        
    def clear_auth_token(self):
        """Clear authentication token"""
        self.auth_token = None
        logger.debug("Auth token cleared")
        
    def _get_headers(self, additional_headers: Dict[str, str] = None) -> Dict[str, str]:
        """Get request headers with auth token"""
        headers = {"Content-Type": "application/json"}
        
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
            
        if additional_headers:
            headers.update(additional_headers)
            
        return headers
        
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Dict = None,
        params: Dict = None,
        headers: Dict = None
    ) -> APIResponse:
        """Make HTTP request with error handling"""
        if not self.session:
            await self.start_session()
            
        url = urljoin(f"{self.base_url}/", endpoint.lstrip('/'))
        request_headers = self._get_headers(headers)
        
        try:
            logger.debug(f"Making {method} request to {url}")
            
            async with self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                headers=request_headers
            ) as response:
                
                status_code = response.status
                
                try:
                    response_data = await response.json()
                except (aiohttp.ContentTypeError, json.JSONDecodeError):
                    response_data = {"message": await response.text()}
                
                if 200 <= status_code < 300:
                    logger.debug(f"Request successful: {status_code}")
                    return APIResponse(
                        success=True,
                        data=response_data,
                        status_code=status_code
                    )
                else:
                    error_msg = response_data.get("detail", f"HTTP {status_code}")
                    logger.warning(f"Request failed: {status_code} - {error_msg}")
                    return APIResponse(
                        success=False,
                        error=error_msg,
                        status_code=status_code
                    )
                    
        except aiohttp.ClientTimeout:
            logger.error("Request timeout")
            return APIResponse(
                success=False,
                error="Request timeout"
            )
        except aiohttp.ClientError as e:
            logger.error(f"Client error: {str(e)}")
            return APIResponse(
                success=False,
                error=f"Connection error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return APIResponse(
                success=False,
                error=f"Unexpected error: {str(e)}"
            )
    
    # Authentication endpoints
    async def login(self, username: str, password: str) -> APIResponse:
        """User login"""
        data = {"username": username, "password": password}
        return await self._make_request("POST", "/auth/login", data)
    
    async def register(self, username: str, email: str, password: str) -> APIResponse:
        """User registration"""
        data = {"username": username, "email": email, "password": password}
        return await self._make_request("POST", "/auth/register", data)
    
    async def refresh_token(self) -> APIResponse:
        """Refresh authentication token"""
        return await self._make_request("POST", "/auth/refresh")
    
    async def logout(self) -> APIResponse:
        """User logout"""
        return await self._make_request("POST", "/auth/logout")
    
    # Chat endpoints
    async def send_message(self, message: str, session_id: Optional[str] = None) -> APIResponse:
        """Send chat message"""
        data = {"message": message}
        if session_id:
            data["session_id"] = session_id
        return await self._make_request("POST", "/chat/message", data)
    
    async def get_chat_history(self, session_id: str, limit: int = 50) -> APIResponse:
        """Get chat history"""
        params = {"session_id": session_id, "limit": limit}
        return await self._make_request("GET", "/chat/history", params=params)
    
    async def create_chat_session(self, title: Optional[str] = None) -> APIResponse:
        """Create new chat session"""
        data = {"title": title} if title else {}
        return await self._make_request("POST", "/chat/session", data)
    
    async def get_chat_sessions(self) -> APIResponse:
        """Get user's chat sessions"""
        return await self._make_request("GET", "/chat/sessions")
    
    async def delete_chat_session(self, session_id: str) -> APIResponse:
        """Delete chat session"""
        return await self._make_request("DELETE", f"/chat/session/{session_id}")
    
    async def search_documents(self, query: str, limit: int = 10) -> APIResponse:
        """Search documents"""
        params = {"query": query, "limit": limit}
        return await self._make_request("GET", "/chat/search", params=params)
    
    # User endpoints
    async def get_profile(self) -> APIResponse:
        """Get user profile"""
        return await self._make_request("GET", "/users/profile")
    
    async def update_profile(self, profile_data: Dict[str, Any]) -> APIResponse:
        """Update user profile"""
        return await self._make_request("PUT", "/users/profile", profile_data)
    
    async def change_password(self, current_password: str, new_password: str) -> APIResponse:
        """Change user password"""
        data = {"current_password": current_password, "new_password": new_password}
        return await self._make_request("PUT", "/users/password", data)
    
    async def upload_document(self, file_data: bytes, filename: str, content_type: str) -> APIResponse:
        """Upload document"""
        # For file uploads, we need different headers
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        data = aiohttp.FormData()
        data.add_field('file', file_data, filename=filename, content_type=content_type)
        
        try:
            async with self.session.post(
                f"{self.base_url}/users/upload",
                data=data,
                headers=headers
            ) as response:
                status_code = response.status
                response_data = await response.json()
                
                if 200 <= status_code < 300:
                    return APIResponse(success=True, data=response_data, status_code=status_code)
                else:
                    error_msg = response_data.get("detail", f"HTTP {status_code}")
                    return APIResponse(success=False, error=error_msg, status_code=status_code)
                    
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            return APIResponse(success=False, error=f"Upload failed: {str(e)}")
    
    async def get_user_documents(self) -> APIResponse:
        """Get user's documents"""
        return await self._make_request("GET", "/users/documents")
    
    async def delete_document(self, document_id: str) -> APIResponse:
        """Delete document"""
        return await self._make_request("DELETE", f"/users/documents/{document_id}")
    
    # Health check
    async def health_check(self) -> APIResponse:
        """Check API health"""
        return await self._make_request("GET", "/health")
    
    async def get_system_status(self) -> APIResponse:
        """Get system status"""
        return await self._make_request("GET", "/health/status")


# Utility functions for PyQt6 integration
def run_async_in_thread(coro):
    """Run async coroutine in thread for PyQt6 integration"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


class SyncAPIClient:
    """Synchronous wrapper for APIClient for easier PyQt6 integration"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.api_client = APIClient(base_url)
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        run_async_in_thread(self.api_client.close_session())
    
    def set_auth_token(self, token: str):
        self.api_client.set_auth_token(token)
        
    def clear_auth_token(self):
        self.api_client.clear_auth_token()
    
    def login(self, username: str, password: str) -> APIResponse:
        return run_async_in_thread(self.api_client.login(username, password))
    
    def register(self, username: str, email: str, password: str) -> APIResponse:
        return run_async_in_thread(self.api_client.register(username, email, password))
    
    def send_message(self, message: str, session_id: Optional[str] = None) -> APIResponse:
        return run_async_in_thread(self.api_client.send_message(message, session_id))
    
    def get_chat_history(self, session_id: str, limit: int = 50) -> APIResponse:
        return run_async_in_thread(self.api_client.get_chat_history(session_id, limit))
    
    def create_chat_session(self, title: Optional[str] = None) -> APIResponse:
        return run_async_in_thread(self.api_client.create_chat_session(title))
    
    def get_chat_sessions(self) -> APIResponse:
        return run_async_in_thread(self.api_client.get_chat_sessions())
    
    def get_profile(self) -> APIResponse:
        return run_async_in_thread(self.api_client.get_profile())
    
    def health_check(self) -> APIResponse:
        return run_async_in_thread(self.api_client.health_check())