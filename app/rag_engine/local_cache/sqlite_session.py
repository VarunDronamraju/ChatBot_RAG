from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

SQLITE_DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/cache.db")
engine = create_engine(f"sqlite:///{SQLITE_DB_PATH}")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_local_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()