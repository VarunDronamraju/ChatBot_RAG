import os
import traceback
import hashlib

from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from app.rag_engine.chroma.chroma_client import ChromaClient
from app.rag_engine.Ingest.document_embedder import embed_documents


def get_file_hash(filepath):
    """Returns MD5 hash of a file"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def load_documents(data_dir="data", known_hashes: set = None):
    docs = []
    supported = [".txt", ".pdf", ".docx"]
    chroma_client = ChromaClient(persist_directory="app/data/chroma")

    known_hashes = known_hashes or set()

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext not in supported:
            print(f"[SKIPPED] Unsupported file: {filename}")
            continue

        file_hash = get_file_hash(filepath)
        if file_hash in known_hashes:
            print(f"[SKIPPED] Already ingested: {filename}")
            continue

        try:
            if ext == ".txt":
                loader = TextLoader(filepath, encoding="utf-8")
            elif ext == ".pdf":
                with open(filepath, "rb") as f:
                    header = f.read(5)
                    if not header.startswith(b"%PDF"):
                        print(f"[ERROR] Invalid PDF header in {filename}. Treating as text.")
                        loader = TextLoader(filepath, encoding="utf-8")
                    else:
                        loader = PyPDFLoader(filepath)
            elif ext == ".docx":
                with open(filepath, "rb") as f:
                    header = f.read(4)
                    if header != b"PK\x03\x04":
                        print(f"[ERROR] Invalid DOCX header in {filename}. Treating as text.")
                        loader = TextLoader(filepath, encoding="utf-8")
                    else:
                        loader = UnstructuredWordDocumentLoader(filepath)

            file_docs = loader.load()
            
            # Add source metadata and hash to each document
            for doc in file_docs:
                doc.metadata["source"] = filename
                doc.metadata["hash"] = file_hash
            
            # Use the existing embed_documents function
            document_embeddings = embed_documents(file_docs, user_id="user_id_placeholder")
            
            # Add all embeddings to ChromaDB at once
            chroma_client.add_docs(document_embeddings)
                
            docs.extend(file_docs)
            known_hashes.add(file_hash)

        except Exception as e:
            print(f"[ERROR] Failed to load {filename}: {e}")
            traceback.print_exc()

    print(f"✅ Loaded {len(docs)} documents from {data_dir}: {[doc.metadata['source'] for doc in docs]}")
    return docs, known_hashes