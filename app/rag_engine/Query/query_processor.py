import time
from uuid import uuid4
from datetime import datetime

from app.websearch.tavily_tool import search_web
from app.utils.logger import log_eval
from app.rag_engine.Query.vectorstore_loader import load_vectorstore
from app.rag_engine.Query.llm_loader import get_llm
from app.rag_engine.Query.query_expander import expand_query
from app.rag_engine.Query.rag_chain_builder import get_rag_chain
from app.rag_engine.db.session import get_db
from app.rag_engine.db.models import Message, Conversation
from app.rag_engine.aws.s3_uploader import S3Uploader

from langchain_huggingface import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def query_rag_system(question: str, user_id: str = str(uuid4()), conversation_id: str = str(uuid4())):
    start_time = time.time()
    source = "local"
    citations = []
    
    vectordb = load_vectorstore()
    llm = get_llm()
    chain = get_rag_chain(vectordb, llm)
    db = next(get_db())
    s3_uploader = S3Uploader()

    # Ensure conversation exists
    conv = db.query(Conversation).filter_by(id=conversation_id).first()
    if not conv:
        conv = Conversation(id=conversation_id, user_id=user_id, title="New Chat", created_at=datetime.utcnow())
        db.add(conv)
        db.commit()

    # Log user message
    user_message = Message(
        id=str(uuid4()),
        conversation_id=conversation_id,
        role="user",
        content=question,
        timestamp=datetime.utcnow()
    )
    db.add(user_message)
    db.commit()

    queries = expand_query(question)
    all_chunks = []

    try:
        for q in queries:
            embedded_query = embeddings_model.embed_query(q)
            chunks = vectordb.query_docs(embedded_query, n_results=4)
            all_chunks.extend(zip(chunks["documents"][0], chunks["metadatas"][0], chunks["distances"][0]))
        
        # Deduplicate and filter
        unique_chunks = list({(doc, meta["source"], dist) for doc, meta, dist in all_chunks})
        relevant_chunks = [(doc, meta) for doc, meta, dist in unique_chunks if dist < 0.8]
        chunk_info = [(doc, 1 - dist, meta["source"]) for doc, meta, dist in unique_chunks]
        print(f"[DEBUG] Retrieved chunks: {chunk_info}")

    except Exception as e:
        print(f"[ERROR] Vectorstore query error: {e}")
        relevant_chunks = []
        chunk_info = []

    # If local context found, use RAG
    if relevant_chunks:
        context = "\n".join([f"{doc} [source: {meta['source']}]" for doc, meta in relevant_chunks])
        print(f"[DEBUG] Context passed to LLM:\n{context}")
        answer = chain.invoke(question)
        filenames = list({meta["source"] for _, meta in relevant_chunks})
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
                full_answer = answer
                source = "llm"
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

    # Log assistant message
    bot_message = Message(
        id=str(uuid4()),
        conversation_id=conversation_id,
        role="assistant",
        content=full_answer,
        timestamp=datetime.utcnow(),
        sources={"citations": citations},
        response_time=time.time() - start_time
    )
    db.add(bot_message)
    db.commit()

    # S3 persistence
    s3_uploader.upload_message(user_id, conversation_id, bot_message.id, {
        "role": "assistant",
        "content": full_answer,
        "timestamp": bot_message.timestamp.isoformat(),
        "sources": citations
    })

    log_eval(question, full_answer, source, time.time() - start_time, citations)
    return full_answer


if __name__ == "__main__":
    while True:
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower() == "quit":
            break
        answer = query_rag_system(question)
        print(f"\n📘 Answer: {answer}\n")
