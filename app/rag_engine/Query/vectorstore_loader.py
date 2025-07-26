#!/usr/bin/env python
# -*- coding: utf-8 -*-

from app.rag_engine.chroma.chroma_client import ChromaClient
from langchain_huggingface import HuggingFaceEmbeddings
from app.rag_engine.chroma.chroma_client import ChromaClient

def load_vectorstore(persist_dir="app/data/chroma"):
    return ChromaClient(persist_directory=persist_dir)

