from sqlalchemy_init import create_engine
from app.rag_engine.db.base import Base
from app.rag_engine.db.models import (
    User, Conversation, Message, DocumentMetadata,
    AuditLog, UsageStat, MessageFeedback, QueryLog
)
import os
from dotenv import load_dotenv
load_dotenv()

def init_db():
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:qwerty12345@localhost:5432/ragbot")
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    return engine

if __name__ == "__main__":
    init_db()
