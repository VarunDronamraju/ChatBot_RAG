import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from dotenv import load_dotenv

load_dotenv()

def load_vectorstore(persist_dir="vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return vectordb

def get_llm():
    return Ollama(model="gemma:2b")

def get_rag_chain(vectordb, llm):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate.from_template(
        """
You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer, say you don't know. Provide citations in [filename.txt] format.

Question: {question}

Context:
{context}

Answer:
"""
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain

def query_rag_system(question: str):
    vectordb = load_vectorstore()
    llm = get_llm()
    chain = get_rag_chain(vectordb, llm)

    response = chain.invoke(question)
    return response

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
