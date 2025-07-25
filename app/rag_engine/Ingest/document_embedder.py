from typing import List
from uuid import uuid4
from app.rag_engine.chroma.vector_schema import DocumentEmbedding
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def embed_documents(docs, user_id: str = "user_id_placeholder") -> List[DocumentEmbedding]:
    embeddings = []
    for doc in docs:
        embedding = embedding_model.embed_query(doc.page_content)
        
        # Create metadata ensuring all values are ChromaDB compatible (str, int, float, bool)
        metadata = {
            "source": doc.metadata.get("source", "unknown"),
            "doc_id": str(uuid4()),
            "user_id": user_id,
            "hash": doc.metadata.get("hash", ""),
            "filename": doc.metadata.get("source", "unknown"),
            "doc_type": "text",
            "visibility": "private",
            "page_number": doc.metadata.get("page", 0),
            "chunk_id": str(uuid4()),
            # Convert any lists to empty strings
            "tags": "",
            "keywords": "",
            "conversation_ids": ""
        }
        
        embeddings.append(DocumentEmbedding(
            id=str(uuid4()),
            vector=embedding,
            content=doc.page_content,
            metadata=metadata
        ))
    return embeddings