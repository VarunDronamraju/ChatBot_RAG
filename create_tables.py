#!/usr/bin/env python3
"""
Database table creation script for RAGBot
Run this once to create all required tables
"""

import os
from dotenv import load_dotenv
from sqlalchemy_init import create_engine
from app.rag_engine.db.base import Base
from app.rag_engine.db.models import (
    User, Conversation, Message, DocumentMetadata, 
    UserPreferences, UserSettings, AuditLog, UsageStat, 
    MessageFeedback, QueryLog
)

def create_database_tables():
    """Create all database tables"""
    
    # Load environment variables
    load_dotenv()
    
    # Get database URL
    DATABASE_URL = (
        os.getenv("POSTGRES_URL")
        or os.getenv("DATABASE_URL") 
        or "sqlite:///app/data/ragbot_dev.db"
    )
    
    # Ensure directory exists for SQLite
    if DATABASE_URL.startswith("sqlite"):
        os.makedirs("app/data", exist_ok=True)
        print(f"📁 Using SQLite database: {DATABASE_URL}")
    else:
        print(f"🐘 Using PostgreSQL database: {DATABASE_URL}")
    
    # Create engine
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    
    try:
        # Create all tables
        print("🔨 Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ All tables created successfully!")
        
        # List created tables
        print("\n📋 Created tables:")
        for table_name in Base.metadata.tables.keys():
            print(f"  • {table_name}")
            
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 RAGBot Database Setup")
    print("=" * 40)
    
    success = create_database_tables()
    
    if success:
        print("\n🎉 Database setup complete!")
        print("You can now run your Streamlit app with: streamlit run main.py")
    else:
        print("\n💥 Database setup failed!")
        print("Please check the error messages above.")