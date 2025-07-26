# scripts/init_user.py

import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from app.rag_engine.db.session import get_db
from app.services.auth_service import AuthService
import sqlite3

load_dotenv()

def main():
    print("🚀 Creating test user...")

    db: Session = next(get_db())
    local_db = sqlite3.connect("app/local/auth_cache.db")

    auth_service = AuthService(db=db, local_db=local_db)

    email = input("Enter email: ")
    name = input("Enter name: ")
    password = input("Enter password: ")

    user = auth_service.create_user(email=email, password=password, name=name)

    print(f"✅ Created user: {user.name} ({user.email}) - ID: {user.id}")

if __name__ == "__main__":
    main()
