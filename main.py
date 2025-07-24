# main.py
import os
import streamlit as st
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# Load env
load_dotenv()

# --- Modular Imports ---
from app.rag_engine.Ingest.document_loader import load_documents
from app.rag_engine.Ingest.vectorstore_builder import build_vectorstore
from app.rag_engine.Query.vectorstore_loader import load_vectorstore
from app.rag_engine.Query.llm_loader import get_llm
from app.rag_engine.Query.rag_chain_builder import get_rag_chain
from app.utils.logger import LOG_FILE, log_eval

# Optional web search
from app.websearch.tavily_tool import search_web

# Set page config
st.set_page_config(page_title="RAGBot Chatbot", page_icon="🤖", layout="wide")

# --- Custom CSS ---
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

# --- Session State Init ---
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

# --- Ingest documents at startup if no vectorstore ---
if not st.session_state.vectorstore_built:
    st.sidebar.info("⚙️ Vectorstore missing, rebuilding from 'data/'...")
    os.makedirs("data", exist_ok=True)
    raw_docs = load_documents("data")
    if raw_docs:
        _, _ = build_vectorstore(raw_docs, persist_dir="vectorstore")
        st.session_state.vectorstore_built = True
    else:
        st.sidebar.warning("⚠️ No documents found in 'data/' to ingest.")

# --- Sidebar: Upload UI ---
st.sidebar.title("Document Ingestion")
uploaded_files = st.sidebar.file_uploader("Upload TXT, PDF, DOCX", type=["txt", "pdf", "docx"], accept_multiple_files=True, key=f"uploader_{st.session_state.upload_key}")

if uploaded_files:
    os.makedirs("data", exist_ok=True)
    new_files = set()
    for uploaded_file in uploaded_files:
        path = os.path.join("data", uploaded_file.name)
        if uploaded_file.name not in st.session_state.processed_files:
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            new_files.add(uploaded_file.name)
    if new_files:
        with st.sidebar.status("🔄 Processing..."):
            docs = load_documents("data")
            if docs:
                build_vectorstore(docs, persist_dir="vectorstore")
                st.session_state.vectorstore_built = True
                st.session_state.processed_files.update(new_files)
                st.sidebar.success("✅ Vectorstore updated.")
            else:
                st.sidebar.error("❌ Failed to load documents.")
        st.session_state.upload_key += 1
        st.rerun()

# --- Logs Sidebar ---
st.sidebar.subheader("Evaluation Logs")
if os.path.exists(LOG_FILE):
    if st.sidebar.button("📜 View Logs"):
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
        with st.sidebar.expander("Logs", expanded=True):
            for log in logs:
                st.write(f"**Time**: {log['timestamp']}")
                st.write(f"**Q**: {log['question']}")
                st.write(f"**A**: {log['answer']}")
                st.write(f"**Source**: {log['source']}")
                st.write(f"**Time**: {log['response_time']}s")
                st.write(f"**Citations**: {log['citations']}")
                st.divider()
else:
    st.sidebar.write("🪵 No logs yet.")

# --- Chat UI ---
st.title("💬 RAGBot Chatbot")
chat_container = st.container()

with chat_container:
    for idx, msg in enumerate(st.session_state.messages):
        css_class = "user-message" if msg["role"] == "user" else "bot-message"
        st.markdown(f'<div class="{css_class}">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg["role"] == "assistant":
            if st.button("Metrics", key=f"metrics_{idx}"):
                met = st.session_state.metrics.get(idx, {})
                st.write(f"**Source**: {met.get('source', 'N/A')}")
                st.write(f"**Response Time**: {met.get('response_time', 'N/A')}s")
                st.write(f"**Chunks**: {met.get('chunks', 'N/A')}")
                st.write(f"**Context**: {met.get('context', 'N/A')}")

# --- Handle Prompt ---
prompt = st.chat_input("Ask your question...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    start = time.time()
    vectordb = load_vectorstore()
    llm = get_llm()
    chain = get_rag_chain(vectordb, llm)

    # Try retrieving chunks
    try:
        chunks = vectordb.similarity_search_with_relevance_scores(prompt, k=4)
        relevant = [doc for doc, score in chunks if max(0, min(1, score)) > 0.2]
    except Exception as e:
        st.error(f"🔴 Vectorstore error: {e}")
        relevant, chunks = [], []

    context = ""
    filenames, citations = [], []
    source = "local"

    if relevant:
        context = "\n".join(f"{doc.page_content} [source: {doc.metadata.get('source')}]" for doc in relevant)
        filenames = list({doc.metadata.get("source", "local.txt") for doc in relevant})
        st.write(f"[DEBUG] Context to LLM:\n{context}")
        answer = chain.invoke(prompt)
        citations = filenames
        full_answer = answer + f"\n\n📄 From: {', '.join(f'[{f}]' for f in citations)}"
    else:
        st.write("[DEBUG] No relevant chunks. Trying web...")
        try:
            web_result = search_web(prompt, include_meta=True)
            if web_result and web_result.get("answer"):
                answer = web_result["answer"]
                urls = [r["url"] for r in web_result.get("results", []) if "url" in r]
                citations = urls[:1]
                full_answer = answer + f"\n\n🔗 According to: {citations[0]}" if citations else answer
                source = "web"
            else:
                from langchain_core.prompts import PromptTemplate
                fallback = PromptTemplate.from_template("""Answer to the best of your knowledge. If unknown, say "I don't have enough information." Question: {question}\nAnswer:""")
                answer = llm.invoke(fallback.format(question=prompt))
                full_answer = answer
                source = "llm"
        except Exception as e:
            st.error(f"🔴 Web error: {e}")
            full_answer = "I don't have enough information."
            source = "llm"

    rt = round(time.time() - start, 2)
    st.session_state.metrics[len(st.session_state.messages)] = {
        "source": source,
        "response_time": rt,
        "chunks": [(doc.page_content, round(score, 3)) for doc, score in chunks],
        "context": context or "None"
    }

    st.session_state.messages.append({"role": "assistant", "content": full_answer})
    log_eval(prompt, full_answer, source, rt, citations)
    st.rerun()
