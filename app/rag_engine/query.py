import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from app.websearch.tavily_tool import search_web

load_dotenv()

def load_vectorstore(persist_dir="vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

def get_llm():
    return OllamaLLM(model="gemma:2b")

def get_rag_chain(vectordb, llm):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    prompt = PromptTemplate.from_template(
        """You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer, say you don't know. Provide source citations in [source] format.

Question: {question}

Context:
{context}

Answer:"""
    )
    return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

def query_rag_system(question: str):
    vectordb = load_vectorstore()
    llm = get_llm()
    chain = get_rag_chain(vectordb, llm)

    # Step 1: Retrieve chunks
    chunks = chain.steps[0].invoke(question)

    # 🟡 Fallback if no strong context
    if not chunks or all(len(chunk.strip()) < 50 for chunk in chunks):
        print("\n[🟡] No relevant context found locally. Using web search...")
        web_response = search_web(question, include_meta=True)

        if not web_response or not web_response.get("answer"):
            return "❌ Web search failed. No answer found."

        answer = web_response["answer"]
        urls = [r.get("url") for r in web_response.get("results", []) if "url" in r]
        citation = f"\n\n🔗 According to: {urls[0]}" if urls else ""
        return answer + citation

    # 🟢 If local docs found, cite source files
    filenames = list({doc.metadata.get("source", "local.txt") for doc in chunks})
    answer = chain.invoke(question)
    citation = f"\n\n📄 From: {', '.join(f'[{f}]' for f in filenames)}"
    return answer + citation

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
