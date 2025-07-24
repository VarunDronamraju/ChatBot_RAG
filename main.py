#main.py
import os
import streamlit as st
import json
from datetime import datetime
from app.rag_engine.ingest import load_documents, build_vectorstore
from app.rag_engine.query import load_vectorstore, get_llm, get_rag_chain
from langchain_core.runnables import RunnablePassthrough
from app.utils.logger import LOG_FILE
import time
import re

from dotenv import load_dotenv
load_dotenv()

# Set page configuration
st.set_page_config(page_title="RAGBot Chatbot", page_icon="🤖", layout="wide")

# Custom CSS for chat layout
st.markdown("""
    <style>
    .user-message {
        background-color: #e6f3ff;
        border-radius: 10px;
        padding: 10px;
        margin: 10px;
        max-width: 45%;
        float: left;
        clear: both;
    }
    .bot-message {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 10px;
        margin: 10px;
        max-width: 45%;
        float: right;
        clear: both;
    }
    .metrics-button {
        margin-top: 10px;
    }
    .chat-container {
        overflow-y: auto;
        height: 500px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore_built" not in st.session_state:
    st.session_state.vectorstore_built = os.path.exists("vectorstore")
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set(os.listdir("data") if os.path.exists("data") else [])
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

# Sidebar for document upload and ingestion
st.sidebar.title("Document Ingestion")
st.sidebar.write("Upload TXT, PDF, or DOCX files to the knowledge base.")
uploaded_files = st.sidebar.file_uploader("Choose files", accept_multiple_files=True, type=["txt", "pdf", "docx"], key=f"uploader_{st.session_state.upload_key}")

if uploaded_files:
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    new_files = set()
    for uploaded_file in uploaded_files:
        file_path = os.path.join(data_dir, uploaded_file.name)
        if uploaded_file.name not in st.session_state.processed_files:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            new_files.add(uploaded_file.name)
    if new_files:
        st.sidebar.success(f"Uploaded {len(new_files)} new file(s).")
        with st.sidebar.status("Processing documents..."):
            raw_docs = load_documents(data_dir)
            if raw_docs:
                vectordb, _ = build_vectorstore(raw_docs, persist_dir="vectorstore")
                st.session_state.vectorstore_built = True
                st.session_state.processed_files.update(new_files)
                st.sidebar.success("Vectorstore updated.")
            else:
                st.sidebar.error("No documents loaded.")
        # Reset file uploader
        st.session_state.upload_key += 1
        st.rerun()

# Log display in sidebar
st.sidebar.subheader("Evaluation Logs")
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        logs = json.load(f)
    if st.sidebar.button("View Logs"):
        with st.sidebar.expander("Logs", expanded=True):
            for log in logs:
                st.write(f"**Time**: {log['timestamp']}")
                st.write(f"**Question**: {log['question']}")
                st.write(f"**Answer**: {log['answer']}")
                st.write(f"**Source**: {log['source']}")
                st.write(f"**Response Time**: {log['response_time']}s")
                st.write(f"**Citations**: {log['citations']}")
                st.divider()
else:
    st.sidebar.write("No logs available.")

# Chat interface
st.title("RAGBot Chatbot")
chat_container = st.container()

# Display chat history
with chat_container:
    for idx, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)
            if st.button("Metrics", key=f"metrics_{idx}", help="Show response metrics", type="secondary"):
                metrics = st.session_state.metrics.get(idx, {})
                st.write(f"**Source**: {metrics.get('source', 'N/A')}")
                st.write(f"**Response Time**: {metrics.get('response_time', 'N/A')}s")
                st.write(f"**Retrieved Chunks**: {metrics.get('chunks', 'N/A')}")
                st.write(f"**Context**: {metrics.get('context', 'N/A')}")

# Chat input
prompt = st.chat_input("Ask a question...")
if prompt:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Query RAG system
    start_time = time.time()
    vectordb = load_vectorstore()
    llm = get_llm()
    chain = get_rag_chain(vectordb, llm)

    try:
        # Lower similarity threshold to retrieve more chunks
        chunks = vectordb.similarity_search_with_relevance_scores(prompt, k=4)
        relevant_chunks = [doc for doc, score in chunks if max(0, min(1, score)) > 0.2]  # Lowered from 0.5
    except Exception as e:
        st.error(f"Vectorstore query error: {e}")
        relevant_chunks = []
        chunks = []

    # Initialize context
    context = ""
    filenames = []
    source = "local"
    citations = []

    if relevant_chunks:
        # Build context from relevant chunks
        context = "\n".join([f"{doc.page_content} [source: {doc.metadata.get('source', 'unknown')}]" for doc in relevant_chunks])
        filenames = list({doc.metadata.get("source", "local.txt") for doc in relevant_chunks})
        st.write(f"[DEBUG] Context passed to LLM:\n{context}")
        answer = chain.invoke(prompt)
        citation = f"\n\n📄 From: {', '.join(f'[{f}]' for f in filenames)}"
        full_answer = answer + citation
        citations = filenames
    else:
        st.write("[DEBUG] No relevant local context. Using web search...")
        try:
            from app.websearch.tavily_tool import search_web
            web_response = search_web(prompt, include_meta=True)
            if not web_response or not web_response.get("answer"):
                st.write("[DEBUG] Web search failed.")
                from langchain_core.prompts import PromptTemplate
                llm_prompt = PromptTemplate.from_template(
                    """Answer to the best of your knowledge. If unknown, say "I don't have enough information."
Question: {question}
Answer:"""
                )
                answer = llm.invoke(llm_prompt.format(question=prompt))
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
            st.error(f"Web search error: {e}")
            full_answer = "I don't have enough information."
            source = "llm"

    response_time = time.time() - start_time
    st.session_state.metrics[len(st.session_state.messages)] = {
        "source": source,
        "response_time": round(response_time, 2),
        "chunks": [(doc.page_content, max(0, min(1, score))) for doc, score in chunks],
        "context": context or "None"
    }

    # Add bot response to history
    st.session_state.messages.append({"role": "assistant", "content": full_answer})

    # Log evaluation
    from app.utils.logger import log_eval
    log_eval(prompt, full_answer, source, response_time, citations)

    # Rerun to update chat display
    st.rerun()