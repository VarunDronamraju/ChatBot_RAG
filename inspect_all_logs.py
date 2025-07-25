# inspect_all_logs.py

from sqlalchemy.orm import Session
from sqlalchemy import select, func, desc
from app.rag_engine.db.models import Message
from app.rag_engine.db.session import SessionLocal

def inspect_messages():
    session: Session = SessionLocal()

    try:
        # Count total messages
        total_messages = session.query(func.count(Message.id)).scalar()

        # Get the most recent message
        latest_message = session.query(Message).order_by(desc(Message.timestamp)).first()

        print("========== MESSAGE LOG INSPECTION ==========")
        print(f"🧾 Total messages logged: {total_messages}\n")

        if latest_message:
            print("📌 Most Recent Message Entry:")
            print(f"  ↳ ID          : {latest_message.id}")
            print(f"  ↳ Timestamp   : {latest_message.timestamp}")
            print(f"  ↳ Role        : {latest_message.role}")
            print(f"  ↳ Content     : {latest_message.content[:300]}...")  # preview only
            print(f"  ↳ Sources     : {latest_message.sources}")
            print(f"  ↳ Tokens Used : {latest_message.token_count}")
        else:
            print("⚠️  No messages found in the DB.")

        print("============================================")

    except Exception as e:
        print("❌ Error during inspection:", e)
    finally:
        session.close()

if __name__ == "__main__":
    inspect_messages()
