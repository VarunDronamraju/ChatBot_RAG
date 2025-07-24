#!/usr/bin/env python
# -*- coding: utf-8 -*-

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def load_vectorstore(persist_dir="vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)