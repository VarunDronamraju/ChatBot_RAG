# app/rag_engine/db/models_local_debug.py

from sqlalchemy import Column, Text, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    user_input = Column(Text)
    response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
