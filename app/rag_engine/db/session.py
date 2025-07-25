import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Explicitly load environment variables
load_dotenv()

# Get database URL with explicit priority
DATABASE_URL = (
    os.getenv("POSTGRES_URL")
    or os.getenv("DATABASE_URL")
    or "sqlite:///app/data/ragbot_dev.db"
)

# Debug: Print which database is being used (remove in production)
print(f"🔧 SESSION DEBUG: Using database: {DATABASE_URL[:30]}...")

# ensure sqlite dir exists if falling back
if DATABASE_URL.startswith("sqlite"):
    os.makedirs("app/data", exist_ok=True)
    print("⚠️ WARNING: Falling back to SQLite!")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()