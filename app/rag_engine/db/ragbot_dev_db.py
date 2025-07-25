# example (in backend/app/db/init_db.py)
from .base import Base
from .session import engine

def init_db():
    Base.metadata.create_all(bind=engine)
