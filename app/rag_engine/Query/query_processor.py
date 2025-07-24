#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from app.websearch.tavily_tool import search_web
from app.utils.logger import log_eval
from .vectorstore_loader import load_vectorstore
from .llm_loader import get_llm  
from .query_expander import expand_query
from .rag_chain_builder import get_rag_chain

def query_rag_system(question: str):
    start_time = time.time()
    source = "local"
    citations = []
    vectordb = load_vectorstore()
    llm = get_llm()
    chain = get_rag_chain(vectordb, llm)

    queries = expand_query(question)
    all_chunks = []
    try:
        for q in queries:
            chunks = vectordb.similarity_search_with_relevance_scores(q, k=4)
            all_chunks.extend(chunks)
        unique_chunks = list({(doc.page_content, doc.metadata.get('source', 'unknown'), score) for doc, score in all_chunks})
        relevant_chunks = [doc for doc, _, score in unique_chunks if max(0, min(1, score)) > 0.1]
        chunk_info = [(doc.page_content, max(0, min(1, score)), doc.metadata.get('source', 'unknown')) for doc, _, score in unique_chunks]
        print(f"[DEBUG] Retrieved chunks: {chunk_info}")
    except Exception as e:
        print(f"[ERROR] Vectorstore query error: {e}")
        relevant_chunks = []
        chunk_info = []

    if relevant_chunks:
        context = "\n".join([f"{doc.page_content} [source: {doc.metadata.get('source', 'unknown')}]" for doc in relevant_chunks])
        print(f"[DEBUG] Context passed to LLM:\n{context}")
        answer = chain.invoke(question)
        filenames = list({doc.metadata.get("source", "local.txt") for doc in relevant_chunks})
        citation = f"\n\n📄 From: {', '.join(f'[{f}]' for f in filenames)}"
        full_answer = answer + citation
        citations = filenames
    else:
        print("[🟡] No relevant local context. Using web search...")
        try:
            web_response = search_web(question, include_meta=True)
            print(f"[DEBUG] search_web called: {question}")
            if not web_response or not web_response.get("answer"):
                print("[🟡] Web search failed.")
                from langchain_core.prompts import PromptTemplate
                llm_prompt = PromptTemplate.from_template(
                    """Answer to the best of your knowledge. If unknown, say "I don't have enough information."
Question: {question}
Answer:"""
                )
                answer = llm.invoke(llm_prompt.format(question=question))
                source = "llm"
                full_answer = answer
            else:
                answer = web_response["answer"]
                urls = [r.get("url") for r in web_response.get("results", []) if "url" in r]
                citation = f"\n\n🔗 According to: {urls[0]}" if urls else ""
                full_answer = answer + citation
                source = "web"
                citations = urls[:1]
        except Exception as e:
            print(f"[ERROR] Web search error: {e}")
            full_answer = "I don't have enough information."
            source = "llm"

    log_eval(question, full_answer, source, time.time() - start_time, citations)
    return full_answer

if __name__ == "__main__":
    while True:
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower() == "quit":
            break
        answer = query_rag_system(question)
        print(f"\n📘 Answer: {answer}\n")