from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from pathlib import Path

def load_documents(data_dir="data"):
    docs = []
    for file in Path(data_dir).glob("*"):
        if file.suffix == ".txt":
            docs.extend(TextLoader(str(file)).load())
        elif file.suffix == ".pdf":
            docs.extend(PyPDFLoader(str(file)).load())
    return docs

def build_vectorstore(documents, persist_dir="vectorstore"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
    return vectordb

if __name__ == "__main__":
    docs = load_documents()
    build_vectorstore(docs)
    print("✅ Vector store built and saved.")
