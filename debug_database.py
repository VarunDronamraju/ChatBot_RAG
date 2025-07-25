#!/usr/bin/env python3
"""
Debug script to check database connection and environment variables
"""

import os
from dotenv import load_dotenv

def debug_database_config():
    """Debug database configuration"""
    
    print("🔍 RAGBot Database Configuration Debug")
    print("=" * 50)
    
    # Load environment variables
    print("📁 Loading .env file...")
    load_dotenv()
    print("✅ .env file loaded")
    
    # Check environment variables
    print("\n🔧 Environment Variables:")
    env_vars = ['POSTGRES_URL', 'DATABASE_URL', 'TAVILY_API_KEY', 'DISABLE_S3']
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            if 'URL' in var or 'KEY' in var:
                # Mask sensitive data
                if len(value) > 20:
                    masked = value[:10] + "..." + value[-10:]
                else:
                    masked = value[:5] + "..."
                print(f"  ✅ {var}: {masked}")
            else:
                print(f"  ✅ {var}: {value}")
        else:
            print(f"  ❌ {var}: Not set")
    
    # Check which database URL will be used
    print("\n🗄️ Database URL Priority Check:")
    postgres_url = os.getenv("POSTGRES_URL")
    database_url = os.getenv("DATABASE_URL")
    
    if postgres_url:
        final_url = postgres_url
        source = "POSTGRES_URL"
    elif database_url:
        final_url = database_url
        source = "DATABASE_URL"
    else:
        final_url = "sqlite:///app/data/ragbot_dev.db"
        source = "DEFAULT (SQLite fallback)"
    
    print(f"  📌 Selected URL: {final_url[:20]}... (from {source})")
    
    # Test database connection
    print("\n🔌 Testing Database Connection...")
    try:
        from sqlalchemy import create_engine, text
        
        engine = create_engine(final_url, pool_pre_ping=True)
        
        with engine.connect() as conn:
            if final_url.startswith("postgresql"):
                # Test PostgreSQL connection
                result = conn.execute(text("SELECT version();"))
                version = result.fetchone()[0]
                print(f"  ✅ PostgreSQL connected: {version[:50]}...")
                
                # Check if tables exist
                tables_result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public';
                """))
                tables = [row[0] for row in tables_result.fetchall()]
                print(f"  📋 Found {len(tables)} tables: {', '.join(tables[:5])}{'...' if len(tables) > 5 else ''}")
                
            else:
                # Test SQLite connection
                result = conn.execute(text("SELECT sqlite_version();"))
                version = result.fetchone()[0]
                print(f"  ✅ SQLite connected: version {version}")
                
    except Exception as e:
        print(f"  ❌ Database connection failed: {e}")
        return False
    
    print("\n🎉 Database configuration looks good!")
    return True

if __name__ == "__main__":
    debug_database_config()