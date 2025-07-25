import os
import streamlit as st
import json
import time
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from uuid import uuid4, UUID
import re

from app.rag_engine.Ingest.document_loader import load_documents
from app.rag_engine.chroma.chroma_client import ChromaClient
from app.rag_engine.Query.vectorstore_loader import load_vectorstore
from app.rag_engine.Query.llm_loader import get_llm
from app.rag_engine.Query.rag_chain_builder import build_rag_chain as get_rag_chain
from app.utils.logger import LOG_FILE, log_eval
from app.websearch.tavily_tool import search_web
from app.rag_engine.db.session import get_db
from app.rag_engine.db.models import Message, Conversation
from app.rag_engine.aws.s3_uploader import S3Uploader
from langchain_huggingface import HuggingFaceEmbeddings

# 🔐 Dummy user UUID (for development use)
DUMMY_USER_ID = UUID("00000000-0000-0000-0000-000000000000")

# Initialize environment and app
load_dotenv()
st.set_page_config(page_title="RAGBot Chatbot", page_icon="🤖", layout="wide")

# 🔧 CENTRALIZED EMBEDDINGS - Load once and cache
@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def get_chroma_client():
    return ChromaClient(persist_directory="app/data/chroma")

# Initialize models once
embeddings_model = get_embeddings_model()
chroma_client = get_chroma_client()

# 🎨 RESPONSE FORMATTER CLASS
class ResponseFormatter:
    """Handles different response formatting based on user requests"""
    
    @staticmethod
    def detect_format_request(prompt):
        """Detect if user wants specific formatting"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["bullet", "points", "list", "itemize"]):
            return "bullets"
        elif any(word in prompt_lower for word in ["table", "tabular", "columns", "rows"]):
            return "table"
        elif any(word in prompt_lower for word in ["summary", "summarize", "brief"]):
            return "summary"
        elif any(word in prompt_lower for word in ["detailed", "explain", "elaborate"]):
            return "detailed"
        elif any(word in prompt_lower for word in ["compare", "comparison", "vs", "versus"]):
            return "comparison"
        else:
            return "default"
    
    @staticmethod
    def format_response(content, format_type, original_prompt=""):
        """Format the response based on the requested type"""
        if format_type == "bullets":
            return ResponseFormatter._to_bullets(content)
        elif format_type == "table":
            return ResponseFormatter._to_table(content, original_prompt)
        elif format_type == "summary":
            return ResponseFormatter._to_summary(content)
        elif format_type == "detailed":
            return ResponseFormatter._to_detailed(content)
        elif format_type == "comparison":
            return ResponseFormatter._to_comparison(content)
        else:
            return content
    
    @staticmethod
    def _to_bullets(content):
        """Convert content to bullet points"""
        # Split content into sentences and convert to bullets
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if len(sentences) <= 1:
            return content
        
        formatted = "**Key Points:**\n\n"
        for sentence in sentences[:5]:  # Limit to 5 key points
            if len(sentence) > 10:  # Skip very short fragments
                formatted += f"• {sentence.strip()}\n"
        return formatted
    
    @staticmethod
    def _to_table(content, prompt):
        """Convert content to table format when applicable"""
        # Simple table formatting for comparisons
        if "vs" in prompt.lower() or "difference" in prompt.lower():
            lines = content.split('\n')
            table = "| Aspect | Details |\n|--------|--------|\n"
            for line in lines[:5]:
                if line.strip() and len(line) > 10:
                    table += f"| Point | {line.strip()} |\n"
            return table
        return content
    
    @staticmethod
    def _to_summary(content):
        """Create a concise summary"""
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if len(sentences) <= 2:
            return content
        
        # Take first and last sentence for summary
        summary = f"**Summary:** {sentences[0]}. {sentences[-1] if len(sentences) > 1 else ''}"
        return summary
    
    @staticmethod
    def _to_detailed(content):
        """Add more detailed formatting"""
        return f"**Detailed Explanation:**\n\n{content}\n\n*This response provides comprehensive information based on available sources.*"
    
    @staticmethod
    def _to_comparison(content):
        """Format as a comparison"""
        return f"**Comparison Analysis:**\n\n{content}\n\n*Key differences and similarities highlighted above.*"

# Debug: Verify environment variables are loaded
if st.sidebar.button("🔧 Debug DB Config"):
    st.sidebar.write(f"DATABASE_URL: {os.getenv('DATABASE_URL', 'NOT SET')[:30]}...")
    st.sidebar.write(f"POSTGRES_URL: {os.getenv('POSTGRES_URL', 'NOT SET')[:30]}...")

# Debug: Show all logs and DB data
if st.sidebar.button("📊 Show All Data"):
    st.sidebar.subheader("📝 JSON Logs")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            logs = json.load(f)
        st.sidebar.write(f"Total logs: {len(logs)}")
        for i, log in enumerate(logs[-5:]):  # Show last 5
            st.sidebar.write(f"**{i+1}. Q**: {log['question'][:50]}...")
            st.sidebar.write(f"**A**: {log['answer'][:50]}...")
            st.sidebar.write(f"**Source**: {log['source']}")
    
    st.sidebar.subheader("🗄️ Database Messages")
    try:
        db = next(get_db())
        messages = db.query(Message).order_by(Message.timestamp.desc()).limit(10).all()
        st.sidebar.write(f"Total DB messages: {len(messages)}")
        for msg in messages[:3]:  # Show last 3
            st.sidebar.write(f"**{msg.role}**: {msg.content[:50]}...")
            st.sidebar.write(f"**Time**: {msg.timestamp}")
    except Exception as e:
        st.sidebar.error(f"DB Error: {e}")

    st.sidebar.subheader("🔍 Vectorstore Stats")
    try:
        # Get vectorstore collection stats
        collection = chroma_client._collection
        count = collection.count()
        st.sidebar.write(f"Documents in vectorstore: {count}")
    except Exception as e:
        st.sidebar.error(f"Vectorstore Error: {e}")

# 🎨 RESPONSE FORMAT SELECTOR
st.sidebar.subheader("🎨 Response Format")
format_option = st.sidebar.selectbox(
    "Choose response format:",
    ["Auto-detect", "Default", "Bullet Points", "Table", "Summary", "Detailed", "Comparison"],
    help="Auto-detect will analyze your question and choose the best format"
)

format_mapping = {
    "Auto-detect": "auto",
    "Default": "default", 
    "Bullet Points": "bullets",
    "Table": "table",
    "Summary": "summary", 
    "Detailed": "detailed",
    "Comparison": "comparison"
}

selected_format = format_mapping[format_option]

# Custom CSS style for chat messages
st.markdown("""
<style>
.user-message {
    background-color:#daf1da; padding:10px; border-radius:10px; margin:5px 0;
}
.bot-message {
    background-color:#f0f0f0; padding:10px; border-radius:10px; margin:5px 0;
    white-space: pre-wrap;
}
.format-indicator {
    background-color:#e1f5fe; padding:5px; border-radius:5px; margin:2px 0;
    font-size:0.8em; color:#0277bd;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore_built" not in st.session_state:
    st.session_state.vectorstore_built = os.path.exists("app/data/chroma/chroma.sqlite3")
if "processed_hashes" not in st.session_state:
    st.session_state.processed_hashes = set()
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid4())

# 🔧 FIXED: Load existing vectorstore if it exists, don't rebuild unnecessarily
if not st.session_state.vectorstore_built:
    st.sidebar.info("⚙️ Vectorstore missing, checking for existing data...")
    if os.path.exists("data") and os.listdir("data"):
        st.sidebar.info("📁 Found existing documents, ingesting...")
        docs, hashes = load_documents("data", st.session_state.processed_hashes)
        st.session_state.processed_hashes.update(hashes)
        st.session_state.vectorstore_built = True
        st.sidebar.success(f"✅ Ingested {len(docs)} documents.")
    else:
        st.sidebar.info("📁 No documents found. Upload files to get started.")

# Sidebar: Document Upload & Ingest
st.sidebar.title("📁 Document Ingestion")
uploaded_files = st.sidebar.file_uploader(
    "Upload TXT, PDF, DOCX",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.upload_key}"
)

if uploaded_files:
    os.makedirs("data", exist_ok=True)
    s3_uploader = S3Uploader()
    new_files = []

    def file_hash(buf):
        return hashlib.md5(buf).hexdigest()

    for file in uploaded_files:
        content = file.getvalue()
        h = file_hash(content)
        if h not in st.session_state.processed_hashes:
            path = os.path.join("data", file.name)
            with open(path, "wb") as f:
                f.write(content)
            s3_uploader.upload_document(str(DUMMY_USER_ID), file.name, path)
            st.session_state.processed_hashes.add(h)
            new_files.append(file.name)

    if new_files:
        with st.sidebar.status("🔄 Processing new files..."):
            # 🔧 FIXED: Call load_documents correctly
            docs, hashes = load_documents("data", st.session_state.processed_hashes)
            st.session_state.processed_hashes.update(hashes)
            st.sidebar.success(f"✅ Added {len(new_files)} new files.")
            st.session_state.vectorstore_built = True
            st.session_state.upload_key += 1
            st.rerun()

# Sidebar: Evaluation Logs
st.sidebar.subheader("📜 Evaluation Logs")
if os.path.exists(LOG_FILE) and st.sidebar.button("View Logs"):
    try:
        with open(LOG_FILE) as f:
            logs = json.load(f)
        with st.sidebar.expander("Logs", expanded=True):
            for log in logs:
                st.write(f"**Q**: {log['question']}")
                st.write(f"**A**: {log['answer']}")
                st.write(f"**Source**: {log['source']}")
                # 🔧 FIXED: Handle missing response_time key
                st.write(f"**Time**: {log.get('response_time', 'N/A')}s")
                st.write(f"**Citations**: {log['citations']}")
                st.divider()
    except Exception as e:
        st.sidebar.error(f"Error loading logs: {e}")

# Main chat interface
st.title("💬 RAGBot Chatbot")

# Format examples
with st.expander("🎨 Format Examples", expanded=False):
    st.write("""
    **Try these format requests:**
    - "List the key differences in bullet points"
    - "Show me a comparison table between CEO and Chairman"
    - "Give me a brief summary of AI"
    - "Explain in detail how transformers work"
    - "Compare Python vs JavaScript"
    """)

with st.container():
    for idx, msg in enumerate(st.session_state.messages):
        cls = "user-message" if msg["role"] == "user" else "bot-message"
        
        # Show format indicator for bot messages
        if msg["role"] == "assistant" and "format_used" in msg:
            st.markdown(f'<div class="format-indicator">🎨 Format: {msg["format_used"]}</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)
        
        if msg["role"] == "assistant":
            if st.button("Metrics", key=f"metrics_{idx}"):
                met = st.session_state.metrics.get(idx, {})
                st.write(f"**Source**: {met.get('source', 'N/A')}")
                st.write(f"**Response Time**: {met.get('response_time', 'N/A')}s")
                st.write(f"**Format Used**: {met.get('format_used', 'Default')}")
                st.write(f"**Chunks**: {met.get('chunks', [])}")
                st.write(f"**Context**: {met.get('context', '')}")

# Handle user prompt input
prompt = st.chat_input("Ask your question...")
if prompt:
    db = next(get_db())
    conversation_id_str = st.session_state.conversation_id
    
    # Convert string to UUID object for database operations
    try:
        conversation_id = UUID(conversation_id_str)
    except ValueError:
        conversation_id = uuid4()
        st.session_state.conversation_id = str(conversation_id)

    # Ensure conversation exists in DB
    conv = db.query(Conversation).filter_by(id=conversation_id).first()
    if not conv:
        conv = Conversation(
            id=conversation_id,
            user_id=DUMMY_USER_ID,
            title="New Chat",
            created_at=datetime.utcnow()
        )
        db.add(conv)
        db.commit()

    # Add user message to DB and session state
    user_msg = Message(
        id=uuid4(),
        conversation_id=conversation_id,
        role="user",
        content=prompt,
        timestamp=datetime.utcnow()
    )
    db.add(user_msg)
    db.commit()
    st.session_state.messages.append({"role": "user", "content": prompt})

    start_time = time.time()
    
    # 🎨 DETECT FORMAT REQUEST
    if selected_format == "auto":
        format_type = ResponseFormatter.detect_format_request(prompt)
    else:
        format_type = selected_format
    
    st.write(f"[DEBUG] Detected format: {format_type}")
    
    # 🔧 FIXED: Use consistent vectorstore and cached embeddings
    vectordb = chroma_client  # Use the cached ChromaClient
    llm = get_llm()
    chain = get_rag_chain(vectordb, llm, embeddings_model)  # Pass embeddings to avoid reloading

    # Query vectorstore for relevant docs
    try:
        query_embedding = embeddings_model.embed_query(prompt)
        chunks = vectordb.query_docs(query_embedding, n_results=4)
        # Fix confidence scoring - ChromaDB uses cosine distance (0=identical, 2=opposite)
        relevant_docs = []
        if chunks["documents"] and chunks["documents"][0]:
            st.write(f"[DEBUG] Found {len(chunks['documents'][0])} chunks, checking relevance...")
            for i, (doc, dist, meta) in enumerate(zip(chunks["documents"][0], chunks["distances"][0], chunks["metadatas"][0])):
                # Convert distance to similarity score (lower distance = higher similarity)
                similarity = 1 - (dist / 2)  # Convert to 0-1 scale
                st.write(f"[DEBUG] Chunk {i+1}: similarity={similarity:.3f}, source={meta.get('source', 'Unknown')}")
                if similarity > 0.7:  # Increased threshold for better relevance
                    relevant_docs.append((doc, similarity, meta))
        relevant = [doc for doc, _, _ in relevant_docs]
        formatted_chunks = [(doc, sim) for doc, sim, _ in relevant_docs]
        st.write(f"[DEBUG] {len(relevant)} chunks passed relevance threshold of 0.7")
    except Exception as e:
        st.error(f"🔴 Vectorstore error: {e}")
        relevant, formatted_chunks = [], []

    filenames, citations, context, source = [], [], "", "local"

    # 🚀 ENHANCED RAG LOGIC WITH SMART WEB FALLBACK
    if relevant:
        context = "\n".join(f"{doc} [source: {meta.get('source', 'Unknown')}]" 
                           for doc, _, meta in relevant_docs)
        filenames = list({meta.get("source", "local.txt") for _, _, meta in relevant_docs})
        st.write(f"[DEBUG] Context found from files: {filenames}")

        # Try RAG answer first
        rag_answer = chain.invoke({"question": prompt})
        st.write(f"[DEBUG] RAG Answer: '{rag_answer[:100]}...'")
        
        # Enhanced detection with more comprehensive patterns
        insufficient_indicators = [
            "doesn't provide", "doesn't contain", "doesn't define", "doesn't mention",
            "not provided", "not contain", "not define", "not mention", 
            "no information", "not available", "cannot find", "can't find",
            "context doesn't", "context does not", "document doesn't", "document does not",
            "i don't have", "don't know", "not sure", "unclear", "insufficient",
            "not enough information", "can't determine", "cannot determine",
            "not specified", "not given", "not included", "not described"
        ]
        
        # Check for insufficient information indicators
        rag_lower = rag_answer.lower()
        detected_indicators = [indicator for indicator in insufficient_indicators if indicator in rag_lower]
        rag_insufficient = len(detected_indicators) > 0
        
        st.write(f"[DEBUG] Detected insufficient indicators: {detected_indicators}")
        st.write(f"[DEBUG] RAG insufficient: {rag_insufficient}")
        
        if rag_insufficient:
            st.write("🔍 [DEBUG] RAG indicates insufficient context. Trying web search...")
            try:
                # Add retry logic for web search
                web_result = None
                for attempt in range(2):  # Try twice
                    try:
                        web_result = search_web(prompt, include_meta=True)
                        if web_result and web_result.get("answer"):
                            break
                        st.write(f"[DEBUG] Web search attempt {attempt + 1} failed or empty")
                    except Exception as e:
                        st.write(f"[DEBUG] Web search attempt {attempt + 1} error: {e}")
                        
                if web_result and web_result.get("answer"):
                    # Combine context info with web search
                    web_answer = web_result["answer"]
                    st.write(f"[DEBUG] Web Answer: '{web_answer[:100]}...'")
                    
                    answer = f"📚 **From your documents**: {rag_answer}\n\n🌐 **Additional information from web**: {web_answer}"
                    urls = [r["url"] for r in web_result.get("results", []) if "url" in r]
                    citations = filenames + urls[:2]  # Combine local and web sources
                    source = "hybrid"
                    st.write("✅ [DEBUG] Successfully combined local + web results")
                else:
                    # Fall back to original RAG answer
                    st.write("⚠️ [DEBUG] Web search returned no results, using RAG answer")
                    answer = rag_answer
                    citations = filenames
                    source = "local"
            except Exception as e:
                st.error(f"🔴 Web search error: {e}")
                answer = rag_answer
                citations = filenames
                source = "local"
        else:
            # RAG answer seems sufficient
            st.write("✅ [DEBUG] RAG answer sufficient, no web search needed")
            answer = rag_answer
            citations = filenames
            source = "local"

    else:
        # No relevant chunks found - direct web search fallback
        st.write("🔍 [DEBUG] No relevant chunks found. Trying web search...")
        try:
            web_result = search_web(prompt, include_meta=True)
            if web_result and web_result.get("answer"):
                answer = web_result["answer"]
                urls = [r["url"] for r in web_result.get("results", []) if "url" in r]
                citations = urls[:2]
                source = "web"
                st.write("✅ [DEBUG] Web search successful")
            else:
                # Final LLM fallback
                st.write("⚠️ [DEBUG] Web search failed, using LLM fallback")
                from langchain_core.prompts import PromptTemplate
                fallback = PromptTemplate.from_template(
                    "Answer to the best of your knowledge. If unknown, say 'I don't have enough information.'\nQuestion: {question}\nAnswer:"
                )
                answer = llm.invoke(fallback.format(question=prompt))
                source = "llm"
        except Exception as e:
            st.error(f"🔴 Web fallback error: {e}")
            answer = "I don't have enough information to answer this question."
            source = "error"

    # 🎨 APPLY FORMATTING
    formatted_answer = ResponseFormatter.format_response(answer, format_type, prompt)
    
    # Add source attribution
    if source == "hybrid":
        full_answer = formatted_answer + f"\n\n📄 **Sources**: {', '.join(f'[{f}]' for f in filenames if f in citations)} + 🔗 **Web**: {', '.join([c for c in citations if c not in filenames])}"
    elif source in ["local", "llm"]:
        full_answer = formatted_answer + (f"\n\n📄 **From**: {', '.join(f'[{f}]' for f in citations)}" if citations else "")
    elif source == "web":
        full_answer = formatted_answer + (f"\n\n🔗 **According to**: {', '.join(citations)}" if citations else "")
    else:
        full_answer = formatted_answer

    response_time = round(time.time() - start_time, 2)

    # Save metrics
    st.session_state.metrics[len(st.session_state.messages)] = {
        "source": source,
        "response_time": response_time,
        "format_used": format_type,
        "chunks": formatted_chunks,
        "context": context or "None"
    }

    # Save assistant message in DB
    bot_msg = Message(
        id=uuid4(),
        conversation_id=conversation_id,
        role="assistant",
        content=full_answer,
        timestamp=datetime.utcnow(),
        sources={"citations": citations},
        response_time=response_time
    )
    db.add(bot_msg)
    db.commit()

    # Upload message to S3
    S3Uploader().upload_message(str(DUMMY_USER_ID), str(conversation_id), str(bot_msg.id), {
        "role": "assistant",
        "content": full_answer,
        "timestamp": bot_msg.timestamp.isoformat(),
        "sources": citations,
        "format_used": format_type
    })

    # Add to session state with format info
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_answer,
        "format_used": format_type
    })
    
    log_eval(prompt, full_answer, source, response_time, citations)
    st.rerun()