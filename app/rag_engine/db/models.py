#models.py
from sqlalchemy import (
    Column, String, Text, DateTime, Integer, Float, Boolean,
    ForeignKey, Index, ARRAY
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.sql import func
from app.rag_engine.db.base import Base
import uuid


class UserPreferences(Base):
    __tablename__ = "user_preferences"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"))
    tone = Column(String)
    length = Column(String)
    language = Column(String)
    bias_config = Column(JSONB)

    __table_args__ = (Index("ix_user_preferences_user_id", "user_id"),)


class UserSettings(Base):
    __tablename__ = "user_settings"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"))
    settings = Column(JSONB)

    __table_args__ = (Index("ix_user_settings_user_id", "user_id"),)


class User(Base):
    __tablename__ = "users"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    google_id = Column(String, unique=True)
    email = Column(String, unique=True)
    name = Column(String)
    picture_url = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    last_login = Column(DateTime, server_default=func.now())
    usage_metrics = Column(JSONB)
    global_tags = Column(ARRAY(Text))
    search_bias_mode = Column(String, default="none")
    conversation_style = Column(JSONB)

    __table_args__ = (Index("ix_users_google_id", "google_id"),)


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"))
    title = Column(String)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    message_count = Column(Integer, default=0)
    extra_metadata = Column("metadata", JSONB)
    is_deleted = Column(Boolean, default=False)
    s3_backup_url = Column(Text)
    tags = Column(ARRAY(Text))
    chat_type = Column(String)
    focus_doc_ids = Column(ARRAY(Text))
    linked_documents = Column(ARRAY(Text))
    user_bias_profile = Column(Text)

    __table_args__ = (Index("ix_conversations_user_id", "user_id"),)


class Message(Base):
    __tablename__ = "messages"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(PG_UUID(as_uuid=True), ForeignKey("conversations.id"))
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, server_default=func.now())
    sources = Column(JSONB)
    response_time = Column(Float)
    token_count = Column(Integer)
    feedback_score = Column(Integer)
    edit_history = Column(JSONB)

    __table_args__ = (Index("ix_messages_conversation_id", "conversation_id"),)


class DocumentMetadata(Base):
    __tablename__ = "documents_metadata"
    id = Column(Text, primary_key=True)
    filename = Column(Text)
    upload_status = Column(Text)
    local_path = Column(Text)
    mime_type = Column(Text)
    doc_type = Column(Text)
    tags = Column(ARRAY(Text))
    keywords = Column(ARRAY(Text))
    associated_conversations = Column(ARRAY(Text))
    is_personalized = Column(Boolean)
    visibility = Column(String)
    owner_user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"))

    __table_args__ = (Index("ix_documents_metadata_owner_user_id", "owner_user_id"),)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"))
    event_type = Column(String)
    event_details = Column(JSONB)
    timestamp = Column(DateTime, server_default=func.now())

    __table_args__ = (Index("ix_audit_logs_user_id", "user_id"),)


class UsageStat(Base):
    __tablename__ = "usage_stats"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"))
    date = Column(DateTime, server_default=func.current_date())
    token_usage = Column(Integer)
    message_count = Column(Integer)
    cost = Column(Float)

    __table_args__ = (Index("ix_usage_stats_user_id", "user_id"),)


class MessageFeedback(Base):
    __tablename__ = "message_feedback"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(PG_UUID(as_uuid=True), ForeignKey("messages.id"))
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"))
    rating = Column(Integer)
    comment = Column(Text)
    timestamp = Column(DateTime, server_default=func.now())

    __table_args__ = (Index("ix_message_feedback_message_id", "message_id"),)


class QueryLog(Base):
    __tablename__ = "query_logs"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"))
    conversation_id = Column(PG_UUID(as_uuid=True), ForeignKey("conversations.id"))
    question = Column(Text)
    retrieved_doc_ids = Column(ARRAY(Text))
    used_tags = Column(ARRAY(Text))
    source = Column(Text)
    latency_ms = Column(Integer)
    timestamp = Column(DateTime, server_default=func.now())

    __table_args__ = (Index("ix_query_logs_user_id", "user_id"),)