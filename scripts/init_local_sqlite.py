# scripts/init_local_sqlite.py
import sqlite3
import os

db_path = os.path.join("app", "local", "auth_cache.db")
os.makedirs(os.path.dirname(db_path), exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS auth_cache (
    user_id TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS token_cache (
    token TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    is_valid BOOLEAN NOT NULL
)
""")

conn.commit()
conn.close()
print(f"[✅] Initialized local SQLite schema at {db_path}")
