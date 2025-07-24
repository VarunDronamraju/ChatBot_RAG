#!/usr/bin/env python
# -*- coding: utf-8 -*-

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

def get_rag_chain(vectordb, llm):
    retriever = vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 4, "score_threshold": 0.1})
    prompt = PromptTemplate.from_template(
        """Answer using ONLY provided context. If insufficient, say "I don't have enough information." Cite sources in [source] format.
Question: {question}
Context: {context}
Answer:"""
    )
    def format_context(docs):
        return "\n".join([f"{doc.page_content} [source: {doc.metadata.get('source', 'unknown')}]" for doc in docs])
    return {"context": retriever | format_context, "question": RunnablePassthrough()} | prompt | llm