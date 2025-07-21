import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from app.websearch.tavily_tool import search_web

load_dotenv()

# Load local vector DB
def load_vectorstore(persist_dir="vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# Load local LLM
def get_llm():
    return OllamaLLM(model="gemma:2b")

# Heuristic to check if RAG data is useful
def is_rag_relevant(chunks):
    return chunks and any(len(doc.page_content.strip()) > 50 for doc in chunks)

# Prompt for local RAG
rag_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use the context below to answer the question.
If unsure, say you don't know. Cite sources like [document_name.txt].

Question: {question}

Context:
{context}

Answer:
""")

# Prompt for web fallback
web_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use the following web search results to answer the question.
Cite the source as [web: URL].

Question: {question}

Web Content:
{context}

Answer:
""")

# Main router chain
def build_router_chain():
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = get_llm()

    retrieve_step = RunnableLambda(lambda q: retriever.get_relevant_documents(q))

    def decide_and_route(inputs):
        question = inputs["question"]
        chunks = inputs["chunks"]
        if is_rag_relevant(chunks):
            rag_input = {
                "context": "\n\n".join(doc.page_content for doc in chunks),
                "question": question
            }
            return rag_prompt | llm | (lambda x: x)
        else:
            print("\n[🟡] No relevant context found locally. Using web search...")
            web_context, source_url = search_web(question, return_source=True)
            if not web_context:
                return lambda _: "⚠️ No answer found via web search."
            return web_prompt | llm | (lambda x: f"{x.strip()}\n\n[source: {source_url}]")

    chain = (
        {"question": RunnablePassthrough(), "chunks": retrieve_step}
        | RunnableLambda(decide_and_route)
        | RunnableLambda(lambda runnable: runnable({}))
    )

    return chain

# Query loop
def query_rag_system(question: str):
    chain = build_router_chain()
    return chain.invoke(question)

# CLI interface
if __name__ == "__main__":
    print("🤖 Ask your question (Ctrl+C to exit):")
    while True:
        try:
            q = input("\n🔎> ")
            answer = query_rag_system(q)
            print(f"\n📘 Answer:\n{answer}")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
