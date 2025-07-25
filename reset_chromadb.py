#!/usr/bin/env python3
"""
Quick script to reset ChromaDB data due to dimension mismatch
Run this once to clear the existing data with wrong dimensions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.rag_engine.chroma.chroma_client import ChromaClient

def reset_chromadb():
    print("🧹 Resetting ChromaDB due to dimension mismatch...")
    try:
        chroma_client = ChromaClient(persist_directory="app/data/chroma")
        chroma_client.reset()
        print("✅ ChromaDB reset successful!")
        print("📝 You can now restart your app and re-upload documents.")
    except Exception as e:
        print(f"❌ Error resetting ChromaDB: {e}")

if __name__ == "__main__":
    reset_chromadb()