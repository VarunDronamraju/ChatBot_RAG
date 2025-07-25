import chromadb
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_chroma_local(persist_directory: str = "app/data/chroma"):
    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)
    
    if (persist_path / "chroma.sqlite3").exists():
        logger.info(f"ChromaDB already initialized at {persist_directory}")
        return
    
    client = chromadb.PersistentClient(path=persist_directory)
    client.get_or_create_collection(
        name="document_embeddings",
        metadata={"hnsw:space": "cosine"}
    )
    client.get_or_create_collection(
        name="conversation_context",
        metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"ChromaDB initialized at {persist_directory}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="app/data/chroma", help="ChromaDB persist directory")
    args = parser.parse_args()
    run_chroma_local(persist_directory=args.dir)
