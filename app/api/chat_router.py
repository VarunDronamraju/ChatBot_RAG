"""
Chat Router for RAGBot API
Location: app/api/chat_router.py

Handles chat interactions, document queries, web search integration,
and conversation management. This replaces the blink_router from the original plan
and adapts it for RAG-based chat functionality.
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum
import asyncio
import time
import os
import hashlib

# Import dependencies
from ..rag_engine.db.session import get_db
from ..rag_engine.local_cache.sqlite_session import get_local_db
from ..rag_engine.db.models import Conversation, Message, DocumentMetadata, QueryLog
from ..services.chat_service import ChatService
from ..api.auth_router import get_current_user
from ..utils.logger import get_logger
from ..config.constants import ALLOWED_DOC_TYPES

router = APIRouter()
logger = get_logger()

# Enums
class FormatPreference(str, Enum):
    auto = "auto"
    bullets = "bullets"
    table = "table"
    summary = "summary"
    detailed = "detailed"
    comparison = "comparison"

class SearchType(str, Enum):
    local = "local"
    web = "web"
    hybrid = "hybrid"

# Pydantic models
class ChatMessage(BaseModel):
    content: str
    role: str = "user"
    format_preference: Optional[FormatPreference] = FormatPreference.auto

class ChatResponse(BaseModel):
    id: str
    content: str
    role: str = "assistant"
    timestamp: datetime
    sources: List[str] = []
    response_time: float
    format_used: str
    source_type: str  # local, web, hybrid, llm

class ConversationCreate(BaseModel):
    title: Optional[str] = "New Chat"
    chat_type: Optional[str] = "general"
    tags: List[str] = []

class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    tags: List[str] = []
    chat_type: str

class DocumentUpload(BaseModel):
    filename: str
    content_hash: str
    tags: List[str] = []
    doc_type: Optional[str] = None

class SearchQuery(BaseModel):
    query: str
    search_type: SearchType = SearchType.hybrid
    max_results: int = 10
    format_preference: Optional[FormatPreference] = FormatPreference.auto

class SyncRequest(BaseModel):
    offline_messages: List[Dict[Any, Any]] = []
    last_sync_timestamp: Optional[datetime] = None

@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    conversation: ConversationCreate,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Create a new conversation"""
    try:
        chat_service = ChatService(db)
        
        conv = chat_service.create_conversation(
            user_id=UUID(current_user["user_id"]),
            title=conversation.title,
            chat_type=conversation.chat_type,
            tags=conversation.tags
        )
        
        logger.info(f"Created conversation {conv.id} for user {current_user['email']}")
        
        return ConversationResponse(
            id=str(conv.id),
            title=conv.title,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            message_count=conv.message_count,
            tags=conv.tags or [],
            chat_type=conv.chat_type or "general"
        )
        
    except Exception as e:
        logger.error(f"Create conversation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create conversation"
        )

@router.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(
    limit: int = 20,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get user's conversations"""
    try:
        user_id = UUID(current_user["user_id"])
        
        conversations = db.query(Conversation)\
            .filter(Conversation.user_id == user_id)\
            .filter(Conversation.is_deleted == False)\
            .order_by(Conversation.updated_at.desc())\
            .offset(offset)\
            .limit(limit)\
            .all()
        
        return [
            ConversationResponse(
                id=str(conv.id),
                title=conv.title,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                message_count=conv.message_count,
                tags=conv.tags or [],
                chat_type=conv.chat_type or "general"
            )
            for conv in conversations
        ]
        
    except Exception as e:
        logger.error(f"Get conversations error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversations"
        )

@router.post("/conversations/{conversation_id}/messages", response_model=ChatResponse)
async def send_message(
    conversation_id: str,
    message: ChatMessage,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
    local_db = Depends(get_local_db)
):
    """Send a message and get AI response"""
    try:
        start_time = time.time()
        chat_service = ChatService(db, local_db)
        
        # Verify conversation belongs to user
        conv_id = UUID(conversation_id)
        user_id = UUID(current_user["user_id"])
        
        conversation = db.query(Conversation)\
            .filter(Conversation.id == conv_id)\
            .filter(Conversation.user_id == user_id)\
            .first()
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        # Save user message
        user_msg = chat_service.save_message(
            conversation_id=conv_id,
            role="user",
            content=message.content
        )
        
        # Process message and generate response
        response_data = await chat_service.process_message(
            user_id=user_id,
            conversation_id=conv_id,
            message_content=message.content,
            format_preference=message.format_preference
        )
        
        # Save assistant message
        assistant_msg = chat_service.save_message(
            conversation_id=conv_id,
            role="assistant",
            content=response_data["content"],
            sources=response_data["sources"],
            response_time=response_data["response_time"]
        )
        
        # Update conversation
        conversation.message_count += 2  # user + assistant messages
        conversation.updated_at = datetime.utcnow()
        db.commit()
        
        # Log query for analytics
        query_log = QueryLog(
            user_id=user_id,
            conversation_id=conv_id,
            question=message.content,
            retrieved_doc_ids=response_data.get("doc_ids", []),
            used_tags=response_data.get("tags", []),
            source=response_data["source_type"],
            latency_ms=int(response_data["response_time"] * 1000)
        )
        db.add(query_log)
        db.commit()
        
        logger.info(f"Processed message for conversation {conversation_id}")
        
        return ChatResponse(
            id=str(assistant_msg.id),
            content=response_data["content"],
            timestamp=assistant_msg.timestamp,
            sources=response_data["sources"],
            response_time=response_data["response_time"],
            format_used=response_data["format_used"],
            source_type=response_data["source_type"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Send message error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process message"
        )

@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get messages from a conversation"""
    try:
        conv_id = UUID(conversation_id)
        user_id = UUID(current_user["user_id"])
        
        # Verify conversation belongs to user
        conversation = db.query(Conversation)\
            .filter(Conversation.id == conv_id)\
            .filter(Conversation.user_id == user_id)\
            .first()
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        messages = db.query(Message)\
            .filter(Message.conversation_id == conv_id)\
            .order_by(Message.timestamp.asc())\
            .offset(offset)\
            .limit(limit)\
            .all()
        
        return [
            {
                "id": str(msg.id),
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "sources": msg.sources or {},
                "response_time": msg.response_time
            }
            for msg in messages
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get messages error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve messages"
        )

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    tags: List[str] = [],
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Upload and process a document"""
    try:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )
        
        # Validate file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in ALLOWED_DOC_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type {file_ext} not supported. Allowed: {ALLOWED_DOC_TYPES}"
            )
        
        # Read file content
        content = await file.read()
        content_hash = hashlib.md5(content).hexdigest()
        
        chat_service = ChatService(db)
        
        # Check if document already exists
        existing_doc = db.query(DocumentMetadata)\
            .filter(DocumentMetadata.content_hash == content_hash)\
            .filter(DocumentMetadata.owner_user_id == UUID(current_user["user_id"]))\
            .first()
        
        if existing_doc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document with this content already exists"
            )
        
        # Process document upload
        doc_metadata = await chat_service.upload_document(
            user_id=UUID(current_user["user_id"]),
            filename=file.filename,
            content=content,
            content_hash=content_hash,
            tags=tags
        )
        
        logger.info(f"Document uploaded: {file.filename} by user {current_user['email']}")
        
        return {
            "id": doc_metadata.id,
            "filename": file.filename,
            "status": "uploaded",
            "content_hash": content_hash,
            "tags": tags,
            "message": "Document uploaded and processed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload document error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload document"
        )

@router.get("/documents/{document_id}")
async def get_document_by_id(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Fetch a single document's metadata"""
    try:
        doc = db.query(DocumentMetadata)\
            .filter(DocumentMetadata.id == document_id)\
            .filter(DocumentMetadata.owner_user_id == UUID(current_user["user_id"]))\
            .first()

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "id": doc.id,
            "filename": doc.filename,
            "tags": doc.tags,
            "doc_type": doc.doc_type,
            "upload_status": doc.upload_status,
            "is_personalized": doc.is_personalized,
            "visibility": doc.visibility
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document"
        )

@router.post("/search", response_model=Dict[str, Any])
async def search_documents(
    search: SearchQuery,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
    local_db = Depends(get_local_db)
):
    """Search through documents and web"""
    try:
        chat_service = ChatService(db, local_db)
        
        search_results = await chat_service.search_content(
            user_id=UUID(current_user["user_id"]),
            query=search.query,
            search_type=search.search_type,
            max_results=search.max_results,
            format_preference=search.format_preference
        )
        
        logger.info(f"Search performed: '{search.query}' by user {current_user['email']}")
        
        return search_results
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed"
        )

@router.post("/sync")
async def sync_offline_data(
    sync_request: SyncRequest,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
    local_db = Depends(get_local_db)
):
    """Sync offline data when client comes back online"""
    try:
        chat_service = ChatService(db, local_db)
        
        sync_results = await chat_service.sync_offline_data(
            user_id=UUID(current_user["user_id"]),
            offline_messages=sync_request.offline_messages,
            last_sync_timestamp=sync_request.last_sync_timestamp
        )
        
        logger.info(f"Synced {len(sync_request.offline_messages)} offline messages for user {current_user['email']}")
        
        return {
            "synced_count": sync_results["synced_count"],
            "failed_count": sync_results["failed_count"],
            "last_sync_timestamp": datetime.utcnow(),
            "message": "Offline data synced successfully"
        }
        
    except Exception as e:
        logger.error(f"Sync error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to sync offline data"
        )

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Delete a conversation (soft delete)"""
    try:
        conv_id = UUID(conversation_id)
        user_id = UUID(current_user["user_id"])
        
        conversation = db.query(Conversation)\
            .filter(Conversation.id == conv_id)\
            .filter(Conversation.user_id == user_id)\
            .first()
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        # Soft delete
        conversation.is_deleted = True
        db.commit()
        
        logger.info(f"Deleted conversation {conversation_id} for user {current_user['email']}")
        
        return {"message": "Conversation deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete conversation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete conversation"
        )

@router.get("/documents")
async def get_user_documents(
    limit: int = 20,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Get user's uploaded documents"""
    try:
        user_id = UUID(current_user["user_id"])
        
        documents = db.query(DocumentMetadata)\
            .filter(DocumentMetadata.owner_user_id == user_id)\
            .order_by(DocumentMetadata.id.desc())\
            .offset(offset)\
            .limit(limit)\
            .all()
        
        return [
            {
                "id": doc.id,
                "filename": doc.filename,
                "doc_type": doc.doc_type,
                "tags": doc.tags or [],
                "upload_status": doc.upload_status,
                "is_personalized": doc.is_personalized,
                "visibility": doc.visibility
            }
            for doc in documents
        ]
        
    except Exception as e:
        logger.error(f"Get documents error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents"
        )

@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Delete a document"""
    try:
        user_id = UUID(current_user["user_id"])
        
        document = db.query(DocumentMetadata)\
            .filter(DocumentMetadata.id == document_id)\
            .filter(DocumentMetadata.owner_user_id == user_id)\
            .first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        chat_service = ChatService(db)
        await chat_service.delete_document(document_id, user_id)
        
        logger.info(f"Deleted document {document_id} for user {current_user['email']}")
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete document error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )

@router.post("/conversations/{conversation_id}/feedback")
async def submit_message_feedback(
    conversation_id: str,
    message_id: str,
    rating: int,
    comment: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Submit feedback for a message"""
    try:
        from ..rag_engine.db.models import MessageFeedback
        
        # Validate rating
        if rating < 1 or rating > 5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Rating must be between 1 and 5"
            )
        
        user_id = UUID(current_user["user_id"])
        msg_id = UUID(message_id)
        conv_id = UUID(conversation_id)
        
        # Verify message belongs to user's conversation
        message = db.query(Message)\
            .join(Conversation)\
            .filter(Message.id == msg_id)\
            .filter(Message.conversation_id == conv_id)\
            .filter(Conversation.user_id == user_id)\
            .first()
        
        if not message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Message not found"
            )
        
        # Create or update feedback
        existing_feedback = db.query(MessageFeedback)\
            .filter(MessageFeedback.message_id == msg_id)\
            .filter(MessageFeedback.user_id == user_id)\
            .first()
        
        if existing_feedback:
            existing_feedback.rating = rating
            existing_feedback.comment = comment
            existing_feedback.timestamp = datetime.utcnow()
        else:
            feedback = MessageFeedback(
                message_id=msg_id,
                user_id=user_id,
                rating=rating,
                comment=comment
            )
            db.add(feedback)
        
        db.commit()
        
        logger.info(f"Feedback submitted for message {message_id} by user {current_user['email']}")
        
        return {"message": "Feedback submitted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Submit feedback error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )