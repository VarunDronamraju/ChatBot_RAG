from sqlalchemy import Column, String, Integer, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class AuthCache(Base):
    __tablename__ = "auth_cache"
    user_id = Column(String, primary_key=True)
    access_token = Column(Text)
    refresh_token = Column(Text)
    expires_at = Column(Integer)
    last_sync = Column(Integer)

class ConversationsCache(Base):
    __tablename__ = "conversations_cache"
    id = Column(String, primary_key=True)
    user_id = Column(String)
    title = Column(Text)
    created_at = Column(Integer)
    sync_status = Column(Text)
    local_changes = Column(Text)
    tags = Column(JSON)
    chat_type = Column(Text)
    focus_doc_ids = Column(JSON)

class Settings(Base):
    __tablename__ = "settings"
    key = Column(String, primary_key=True)
    value = Column(Text)
    last_modified = Column(Integer)
    sync_required = Column(Boolean)

class DocumentsMetadata(Base):
    __tablename__ = "documents_metadata"
    id = Column(String, primary_key=True)
    filename = Column(Text)
    upload_status = Column(Text)
    local_path = Column(Text)
    mime_type = Column(Text)
    doc_type = Column(Text)
    tags = Column(JSON)
    keywords = Column(JSON)
    associated_conversations = Column(JSON)
    is_personalized = Column(Boolean)
    visibility = Column(Text)

class QueryLogsLocal(Base):
    __tablename__ = "query_logs_local"
    id = Column(String, primary_key=True)
    question = Column(Text)
    timestamp = Column(Integer)