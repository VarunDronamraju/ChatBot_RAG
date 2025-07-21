from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.websearch.tavily_tool import search_web

def load_vectorstore(persist_dir="vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

def get_llm():
    return Ollama(model="gemma:2b")

def get_rag_chain(vectordb, llm):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the question.
If unsure, say "I don't know". Cite your source as [filename.txt].

Question: {question}

Context:
{context}

Answer:
""")
    return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

def query_rag_system(question: str):
    vectordb = load_vectorstore()
    llm = get_llm()
    chain = get_rag_chain(vectordb, llm)

    chunks = vectordb.similarity_search(question, k=4)
    top_context = "\n\n".join([doc.page_content for doc in chunks])
    source_info = ", ".join({doc.metadata.get('source', 'unknown') for doc in chunks})

    if len(top_context.strip()) < 50:
        print("⚠️ No strong local match. Using web search...")
        web_context = search_web(question)
        if not web_context:
            return "❌ No answer found locally or online."
        prompt = PromptTemplate.from_template("""
Use the following web content to answer the question.

Question: {question}

Web Content:
{context}

Answer:
""")
        final_chain = prompt | llm
        return final_chain.invoke({"question": question, "context": web_context}) + "\n\n📡 Source: [web]"

    rag_chain = get_rag_chain(vectordb, llm)
    response = rag_chain.invoke(question)
    return f"{response}\n\n📄 Source: {source_info or 'local document'}"

if __name__ == "__main__":
    print("🤖 Ask your question (Ctrl+C to exit):")
    while True:
        try:
            q = input("\n🔎> ")
            print("✅ 🔍 Searching local context...")
            answer = query_rag_system(q)
            print(f"\n📘 Answer:\n{answer}")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
