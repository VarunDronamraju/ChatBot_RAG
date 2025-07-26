import chromadb
import uuid
import logging
from typing import List, Dict, Optional
from pathlib import Path
from app.rag_engine.chroma.vector_schema import DocumentEmbedding, ConversationContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaClient:
    def __init__(self, persist_directory: str = "app/data/chroma"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.doc_collection = self.client.get_or_create_collection(
            name="document_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        self.conv_collection = self.client.get_or_create_collection(
            name="conversation_context",
            metadata={"hnsw:space": "cosine"}
        )

    def get_doc_collection(self):
        return self.doc_collection

    def get_conv_collection(self):
        return self.conv_collection

    def _validate_document_embeddings(self, embeddings: List[DocumentEmbedding]):
        for emb in embeddings:
            if not isinstance(emb, DocumentEmbedding):
                raise ValueError(f"Invalid type: Expected DocumentEmbedding, got {type(emb)}")

    def _validate_conversation_contexts(self, contexts: List[ConversationContext]):
        for ctx in contexts:
            if not isinstance(ctx, ConversationContext):
                raise ValueError(f"Invalid type: Expected ConversationContext, got {type(ctx)}")

    def add_docs(self, embeddings: List[DocumentEmbedding]):
        self._validate_document_embeddings(embeddings)
        ids = [e.id for e in embeddings]
        vectors = [e.vector for e in embeddings]
        metadatas = [e.metadata for e in embeddings]
        contents = [e.content for e in embeddings]

        self.doc_collection.add(
            ids=ids,
            embeddings=vectors,
            metadatas=metadatas,
            documents=contents
        )
        logger.info(f"✅ Added {len(embeddings)} documents to vectorstore.")

    def add_conversation_context(self, contexts: List[ConversationContext]):
        self._validate_conversation_contexts(contexts)
        ids = [c.id for c in contexts]
        vectors = [c.vector for c in contexts]
        metadatas = [c.metadata for c in contexts]
        contents = [c.content for c in contexts]

        self.conv_collection.add(
            ids=ids,
            embeddings=vectors,
            metadatas=metadatas,
            documents=contents
        )
        logger.info(f"✅ Added {len(contexts)} conversation contexts.")

    def update_doc(self, embedding: DocumentEmbedding):
        if not isinstance(embedding, DocumentEmbedding):
            raise ValueError("Expected a DocumentEmbedding instance.")
        self.doc_collection.update(
            ids=[embedding.id],
            embeddings=[embedding.vector],
            metadatas=[embedding.metadata],
            documents=[embedding.content]
        )
        logger.info(f"🔁 Updated document: {embedding.id}")

    def update_conversation_context(self, context: ConversationContext):
        if not isinstance(context, ConversationContext):
            raise ValueError("Expected a ConversationContext instance.")
        self.conv_collection.update(
            ids=[context.id],
            embeddings=[context.vector],
            metadatas=[context.metadata],
            documents=[context.content]
        )
        logger.info(f"🔁 Updated conversation context: {context.id}")

    def query_docs(self, query_vector: List[float], n_results: int = 5, where: Optional[Dict] = None):
        return self.doc_collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            where=where
        )

    def query_conversation_context(self, query_vector: List[float], n_results: int = 5, where: Optional[Dict] = None):
        return self.conv_collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            where=where
        )

    def delete_docs(self, ids: List[str]):
        self.doc_collection.delete(ids=ids)
        logger.info(f"🗑️ Deleted {len(ids)} document embeddings.")

    def delete_conversation_context(self, ids: List[str]):
        self.conv_collection.delete(ids=ids)
        logger.info(f"🗑️ Deleted {len(ids)} conversation contexts.")

    def reset(self):
        try:
            self.client.delete_collection("document_embeddings")
            self.client.delete_collection("conversation_context")
            logger.info("🧹 All collections deleted successfully.")
        except Exception as e:
            logger.error(f"Error deleting collections: {e}")
        self.doc_collection = self.client.get_or_create_collection(
            name="document_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        self.conv_collection = self.client.get_or_create_collection(
            name="conversation_context",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("✅ ChromaDB collections reset successfully.")

