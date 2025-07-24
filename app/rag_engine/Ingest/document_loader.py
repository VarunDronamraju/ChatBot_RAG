#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import traceback
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader

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