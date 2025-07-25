#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .Ingest.document_loader import load_documents
from .Ingest.vectorstore_builder import build_vectorstore



def ingest_data(data_dir="data", persist_dir="app/data/chroma"):
    raw_docs = load_documents(data_dir)
    if raw_docs:
        db, chunks = build_vectorstore(raw_docs, persist_dir)
        return db, chunks
    else:
        print("❌ No documents loaded.")
        return None, []