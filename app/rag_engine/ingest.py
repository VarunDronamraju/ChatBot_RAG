import os
import traceback
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_documents(data_dir="data"):
    docs = []
    supported = [".txt", ".pdf", ".docx"]

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext not in supported:
            print(f"[SKIPPED] Unsupported file: {filename}")
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
            for doc in file_docs:
                doc.metadata["source"] = filename
            docs.extend(file_docs)

        except Exception as e:
            print(f"[ERROR] Failed to load {filename}: {e}")
            traceback.print_exc()

    print(f"✅ Loaded {len(docs)} documents from {data_dir}: {[doc.metadata['source'] for doc in docs]}")
    return docs

def build_vectorstore(docs, persist_dir="vectorstore"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    texts = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    print(f"✅ Vectorstore saved to {persist_dir} with {len(texts)} chunks.")
    return vectordb, texts

if __name__ == "__main__":
    raw_docs = load_documents("data")
    if raw_docs:
        db, chunks = build_vectorstore(raw_docs)
    else:
        print("❌ No documents loaded.")