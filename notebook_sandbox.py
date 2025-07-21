import os
import subprocess
import sys

# Base directory
base_dir = r"C:\Users\varun\Downloads\RAGBot"

# Folder structure
folders = [
    "RAGBot/app/rag_engine",
    "RAGBot/app/websearch",
    "RAGBot/app/llm",
    "RAGBot/app/ui",
    "RAGBot/app/utils",
    "RAGBot/data",
]

# Files to create with optional content
files = {
    "RAGBot/README.md": "# RAGBot\n\nPrivate RAG Chatbot with Web Search\n",
    "RAGBot/requirements.txt": "",
    "RAGBot/.env": "# Add your API keys here\nTAVILY_API_KEY=\nSERPAPI_KEY=\n",
    "RAGBot/main.py": 'print("Run RAGBot from here.")\n',
    "RAGBot/data/sample.txt": "This is a sample document for RAG testing.",
}

def create_structure():
    for folder in folders:
        os.makedirs(os.path.join(base_dir, folder.split("RAGBot/")[-1]), exist_ok=True)
    for filepath, content in files.items():
        full_path = os.path.join(base_dir, filepath.split("RAGBot/")[-1])
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

def create_venv():
    venv_path = os.path.join(base_dir, "venv")
    subprocess.check_call([sys.executable, "-m", "venv", venv_path])
    print(f"✅ Virtual environment created at: {venv_path}")

def main():
    print("🔧 Creating project structure...")
    create_structure()
    print("📁 Directory structure created.")
    
    print("🐍 Setting up virtual environment...")
    create_venv()
    
    print("✅ Done. Now activate your environment:")
    print(f"\nWindows CMD:\n  {base_dir}\\venv\\Scripts\\activate.bat")
    print(f"\nPowerShell:\n  {base_dir}\\venv\\Scripts\\Activate.ps1")
    print(f"\nThen install requirements:\n  pip install -r requirements.txt")

if __name__ == "__main__":
    main()
