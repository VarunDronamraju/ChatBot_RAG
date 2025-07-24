#!/usr/bin/env python
# -*- coding: utf-8 -*-

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_vectorstore(docs, persist_dir="vectorstore"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)
    print(f"[DEBUG] Split into {len(texts)} chunks")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        print("[DEBUG] Embeddings initialized")
        vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_dir)
        print(f"[DEBUG] Vectorstore created with {len(texts)} chunks")
    except Exception as e:
        print(f"[ERROR] Vectorstore creation failed: {e}")
        raise
    print(f"✅ Vectorstore saved to {persist_dir} with {len(texts)} chunks.")
    return vectordb, texts