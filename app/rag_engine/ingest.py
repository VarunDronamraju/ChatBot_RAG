from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

def load_documents(data_dir="data"):
    docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(data_dir, filename), encoding="utf-8")
            docs.extend(loader.load())
    return docs

def build_vectorstore(docs, persist_dir="vectorstore"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(texts, embedding=embeddings, persist_directory=persist_dir)
    return vectordb  # Chroma now auto-persists from v0.4+

if __name__ == "__main__":
    raw_docs = load_documents("data")
    db = build_vectorstore(raw_docs)
    print("✅ Vector store built and saved.")
