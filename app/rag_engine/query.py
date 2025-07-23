import os
import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from app.websearch.tavily_tool import search_web
from app.utils.logger import log_eval

load_dotenv()

def load_vectorstore(persist_dir="vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

def get_llm():
    return OllamaLLM(model="gemma3:1b-it-qat")

def get_rag_chain(vectordb, llm):
    retriever = vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 4, "score_threshold": 0.5})  # Lowered threshold
    prompt = PromptTemplate.from_template(
        """Answer using ONLY provided context. If insufficient, say "I don't have enough information." Cite sources in [source] format.
Question: {question}
Context: {context}
Answer:"""
    )
    def format_context(docs):
        return "\n".join([f"{doc.page_content} [source: {doc.metadata.get('source', 'unknown')}]" for doc in docs])
    return {"context": retriever | format_context, "question": RunnablePassthrough()} | prompt | llm

def query_rag_system(question: str):
    start_time = time.time()
    source = "local"
    citations = []
    vectordb = load_vectorstore()
    llm = get_llm()
    chain = get_rag_chain(vectordb, llm)

    chunks = vectordb.similarity_search_with_relevance_scores(question, k=4)
    print(f"[DEBUG] Retrieved chunks: {[(doc.page_content, score) for doc, score in chunks]}")
    relevant_chunks = [doc for doc, score in chunks if score > 0.5]  # Adjusted threshold
    question_phrase = question.lower().strip("who is ")
    filenames = []

    if relevant_chunks:
        filenames = list({doc.metadata.get("source", "local.txt") for doc in relevant_chunks if question_phrase in doc.page_content.lower()})

    if not relevant_chunks or not filenames:
        print("[🟡] No relevant local context. Using web search...")
        web_response = search_web(question, include_meta=True)
        print(f"[DEBUG] search_web called: {question}")

        if not web_response or not web_response.get("answer"):
            print("[🟡] Web search failed.")
            llm_prompt = PromptTemplate.from_template(
                """Answer to the best of your knowledge. If unknown, say "I don't have enough information."
Question: {question}
Answer:"""
            )
            answer = llm.invoke(llm_prompt.format(question=question))
            source = "llm"
            full_answer = answer
            log_eval(question, full_answer, source, time.time() - start_time, citations)
            return full_answer

        answer = web_response["answer"]
        urls = [r.get("url") for r in web_response.get("results", []) if "url" in r]
        citation = f"\n\n🔗 According to: {urls[0]}" if urls else ""
        full_answer = answer + citation
        source = "web"
        citations = urls[:1]
        log_eval(question, full_answer, source, time.time() - start_time, citations)
        return full_answer

    context = "\n".join([f"{doc.page_content} [source: {doc.metadata.get('source', 'unknown')}]" for doc in relevant_chunks])
    print(f"[DEBUG] Context passed to LLM:\n{context}")
    print("[DEBUG] search_web not called (local context found).")

    answer = chain.invoke(question)
    citation = f"\n\n📄 From: {', '.join(f'[{f}]' for f in filenames)}"
    citations = filenames
    full_answer = answer + citation
    log_eval(question, full_answer, source, time.time() - start_time, citations)
    return full_answer

if __name__ == "__main__":
    while True:
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower() == "quit":
            break
        answer = query_rag_system(question)
        print(f"\n📘 Answer: {answer}\n")