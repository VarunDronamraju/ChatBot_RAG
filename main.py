import subprocess
import requests
import time
from app.rag_engine.query import query_rag_system

OLLAMA_URL = "http://localhost:11434"

def ensure_ollama_running():
    try:
        res = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if res.status_code == 200:
            print("✅ Ollama is already running.")
            return True
    except:
        print("🚀 Starting Ollama server with `gemma:2b`...")

    try:
        subprocess.Popen(["ollama", "run", "gemma:2b"], shell=True)
        time.sleep(8)  # Let it warm up
        return True
    except Exception as e:
        print(f"❌ Failed to start Ollama: {e}")
        return False

def main():
    if not ensure_ollama_running():
        return

    print("🤖 Welcome to RAGBot (Local + Web Hybrid)")
    print("Type your question or Ctrl+C to exit.\n")

    while True:
        try:
            q = input("🔎> ")
            answer = query_rag_system(q)
            print(f"\n📘 Answer:\n{answer}\n")
        except KeyboardInterrupt:
            print("\n👋 Exiting RAGBot.")
            break

if __name__ == "__main__":
    main()
