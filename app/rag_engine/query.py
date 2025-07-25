#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .Query.vectorstore_loader import load_vectorstore
from .Query.llm_loader import get_llm
from .Query.query_expander import expand_query
from .Query.rag_chain_builder import get_rag_chain
from .Query.query_processor import query_rag_system



def run_query_system():
    while True:
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower() == "quit":
            break
        answer = query_rag_system(question)
        print(f"\n📘 Answer: {answer}\n")

if __name__ == "__main__":
    run_query_system()