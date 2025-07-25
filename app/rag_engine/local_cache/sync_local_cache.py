from sqlalchemy.orm import Session
from local_cache.sqlite_models import AuthCache, ConversationsCache, Settings, DocumentsMetadata, QueryLogsLocal
from db.models import User, Conversation, DocumentMetadata, QueryLog
from db.session import get_db
from local_cache.sqlite_session import get_local_db
import boto3
import os
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sync_cache_to_postgres(batch_size=100):
    local_db = next(get_local_db())
    remote_db = next(get_db())
    synced_records = {"auth": 0, "conversations": 0, "documents": 0, "queries": 0}

    # Sync AuthCache
    auth_records = local_db.query(AuthCache).filter(AuthCache.last_sync.is_(None)).limit(batch_size).all()
    for record in auth_records:
        user = remote_db.query(User).filter_by(google_id=record.user_id).first()
        if not user:
            user = User(id=record.user_id, google_id=record.user_id)
            remote_db.add(user)
        record.last_sync = int(time.time())
        synced_records["auth"] += 1
    
    # Sync ConversationsCache
    conv_records = local_db.query(ConversationsCache).filter(ConversationsCache.sync_status == "pending").limit(batch_size).all()
    for record in conv_records:
        conv = remote_db.query(Conversation).filter_by(id=record.id).first()
        if not conv:
            conv = Conversation(
                id=record.id,
                user_id=record.user_id,
                title=record.title,
                created_at=record.created_at,
                tags=record.tags,
                chat_type=record.chat_type,
                focus_doc_ids=record.focus_doc_ids
            )
            remote_db.add(conv)
        record.sync_status = "synced"
        synced_records["conversations"] += 1
    
    # Sync DocumentsMetadata
    doc_records = local_db.query(DocumentsMetadata).filter(DocumentsMetadata.upload_status == "pending").limit(batch_size).all()
    for record in doc_records:
        doc = remote_db.query(DocumentMetadata).filter_by(id=record.id).first()
        if not doc:
            doc = DocumentMetadata(
                id=record.id,
                filename=record.filename,
                upload_status=record.upload_status,
                local_path=record.local_path,
                mime_type=record.mime_type,
                doc_type=record.doc_type,
                tags=record.tags,
                keywords=record.keywords,
                associated_conversations=record.associated_conversations,
                is_personalized=record.is_personalized,
                visibility=record.visibility
            )
            remote_db.add(doc)
        record.upload_status = "synced"
        synced_records["documents"] += 1
    
    # Sync QueryLogsLocal
    query_records = local_db.query(QueryLogsLocal).limit(batch_size).all()
    for record in query_records:
        query = remote_db.query(QueryLog).filter_by(id=record.id).first()
        if not query:
            query = QueryLog(
                id=record.id,
                question=record.question,
                timestamp=record.timestamp
            )
            remote_db.add(query)
        local_db.delete(record)
        synced_records["queries"] += 1
    
    local_db.commit()
    remote_db.commit()
    logger.info(f"Sync completed: {synced_records}")

def backup_to_s3():
    s3_client = boto3.client("s3")
    bucket = os.getenv("S3_BUCKET", "ragbot-conversations")
    timestamp = int(time.time())
    local_db = next(get_local_db())
    
    # Backup conversations
    conv_records = local_db.query(ConversationsCache).all()
    for record in conv_records:
        key = f"users/{record.user_id}/conversations/{record.id}.json"
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps({
                "id": record.id,
                "user_id": record.user_id,
                "title": record.title,
                "created_at": record.created_at,
                "tags": record.tags,
                "chat_type": record.chat_type
            })
        )
        logger.info(f"Backed up conversation {record.id} to S3")