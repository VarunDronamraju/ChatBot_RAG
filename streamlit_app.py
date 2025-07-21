import streamlit as st
from app.rag_engine.query import query_rag_system

st.set_page_config(page_title="🧠 RAGBot", layout="wide")
st.title("🔍 Chat with Your Knowledge Base")

st.markdown("Type a question below. The bot will search local documents and web if needed.")

question = st.text_input("📥 Ask your question", "")

if st.button("💬 Get Answer") and question:
    with st.spinner("⏳ Thinking..."):
        try:
            response = query_rag_system(question)
            st.markdown("### 📘 Answer")
            st.write(response)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
