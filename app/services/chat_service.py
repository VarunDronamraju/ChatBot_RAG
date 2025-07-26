from datetime import datetime
from typing import Dict, List, Any, Optional
from uuid import UUID, uuid4
import os
import time
import logging

from langchain_huggingface import HuggingFaceEmbeddings
from ..rag_engine.Ingest.document_loader import load_documents
from ..rag_engine.chroma.chroma_client import ChromaClient
from ..rag_engine.Query.vectorstore_loader import load_vectorstore
from ..rag_engine.Query.llm_loader import get_llm
from ..rag_engine.Query.rag_chain_builder import build_rag_chain as get_rag_chain
from ..websearch.tavily_tool import search_web
from ..rag_engine.db.models import Conversation, Message, DocumentMetadata
from ..rag_engine.aws.s3_uploader import S3Uploader
from ..utils.logger import get_logger

logger = get_logger()


class ResponseFormatter:
    @staticmethod
    def detect_format_request(prompt):
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ["bullet", "points", "list"]):
            return "bullets"
        if any(word in prompt_lower for word in ["table", "columns", "rows"]):
            return "table"
        if "summary" in prompt_lower:
            return "summary"
        if any(word in prompt_lower for word in ["detailed", "explain"]):
            return "detailed"
        if any(word in prompt_lower for word in ["compare", "versus"]):
            return "comparison"
        return "default"

    @staticmethod
    def format_response(content, format_type, prompt=""):
        if format_type == "bullets":
            return ResponseFormatter._to_bullets(content)
        if format_type == "table":
            return ResponseFormatter._to_table(content, prompt)
        if format_type == "summary":
            return ResponseFormatter._to_summary(content)
        if format_type == "detailed":
            return ResponseFormatter._to_detailed(content)
        if format_type == "comparison":
            return ResponseFormatter._to_comparison(content)
        return content

    @staticmethod
    def _to_bullets(content):
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        return "\n".join([f"\u2022 {s}" for s in sentences[:5]]) if len(sentences) > 1 else content

    @staticmethod
    def _to_table(content, prompt):
        lines = content.split('\n')
        table = "| Aspect | Details |\n|--------|---------|\n"
        for line in lines[:5]:
            table += f"| Point | {line.strip()} |\n"
        return table

    @staticmethod
    def _to_summary(content):
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        return f"**Summary**: {sentences[0]}..." if len(sentences) > 2 else content

    @staticmethod
    def _to_detailed(content):
        return f"**Detailed Explanation**:\n{content}"

    @staticmethod
    def _to_comparison(content):
        return f"**Comparison**:\n{content}"


class ChatService:
    def __init__(self, db, local_db=None):
        self.db = db
        self.local_db = local_db
        self.embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.chroma_client = ChromaClient(persist_directory="app/data/chroma")
        self.s3_uploader = S3Uploader()

    async def search_content(self, user_id: UUID, query: str, search_type: str = "local") -> Dict[str, Any]:
        """Search content in documents based on query and search type"""
        try:
            if search_type == "local":
                # Search in ChromaDB
                query_embedding = self.embeddings_model.embed_query(query)
                chunks = self.chroma_client.query_docs(query_embedding, n_results=10)
                
                results = []
                for doc, dist, meta in zip(chunks["documents"][0], chunks["distances"][0], chunks["metadatas"][0]):
                    similarity = 1 - (dist / 2)
                    if similarity > 0.5:  # Only include relevant results
                        results.append({
                            "content": doc,
                            "similarity": similarity,
                            "metadata": meta,
                            "source": meta.get("source", "Unknown")
                        })
                
                return {
                    "results": results,
                    "total_found": len(results),
                    "search_type": "local"
                }
                
            elif search_type == "web":
                # Use web search
                web_result = search_web(query, include_meta=True)
                return {
                    "results": web_result.get("results", []),
                    "total_found": len(web_result.get("results", [])),
                    "search_type": "web",
                    "answer": web_result.get("answer", "")
                }
                
            elif search_type == "hybrid":
                # Combine local and web search
                local_results = await self.search_content(user_id, query, "local")
                web_results = await self.search_content(user_id, query, "web")
                
                return {
                    "results": local_results["results"] + web_results["results"],
                    "total_found": local_results["total_found"] + web_results["total_found"],
                    "search_type": "hybrid",
                    "local_count": local_results["total_found"],
                    "web_count": web_results["total_found"]
                }
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise Exception(f"Search failed: {str(e)}")
            
        return {"results": [], "total_found": 0, "search_type": search_type}

    def create_conversation(self, user_id: UUID, title: str, chat_type: str = "general", tags: List[str] = None) -> Conversation:
        conversation = Conversation(
            id=uuid4(),
            user_id=user_id,
            title=title,
            chat_type=chat_type,
            tags=tags or [],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            message_count=0
        )
        self.db.add(conversation)
        self.db.commit()
        return conversation

    def save_message(self, conversation_id: UUID, role: str, content: str, sources: Dict = None, response_time: float = None) -> Message:
        message = Message(
            id=uuid4(),
            conversation_id=conversation_id,
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            sources=sources or {},
            response_time=response_time
        )
        self.db.add(message)
        self.db.commit()

        try:
            self.s3_uploader.upload_message(
                str(conversation_id.hex[:8]),
                str(conversation_id),
                str(message.id),
                {
                    "role": role,
                    "content": content,
                    "timestamp": message.timestamp.isoformat(),
                    "sources": sources or {}
                }
            )
        except Exception as e:
            logger.warning(f"S3 upload failed: {e}")

        return message

    async def process_message(self, user_id: UUID, conversation_id: UUID, message_content: str, format_preference: str = "auto") -> Dict[str, Any]:
        start_time = time.perf_counter()
        format_type = ResponseFormatter.detect_format_request(message_content) if format_preference == "auto" else format_preference

        llm = get_llm()
        chain = get_rag_chain(self.chroma_client, llm, self.embeddings_model)

        try:
            query_embedding = self.embeddings_model.embed_query(message_content)
            chunks = self.chroma_client.query_docs(query_embedding, n_results=4)
            relevant_docs = [(doc, 1 - (dist / 2), meta) for doc, dist, meta in zip(chunks["documents"][0], chunks["distances"][0], chunks["metadatas"][0]) if 1 - (dist / 2) > 0.7]
        except Exception as e:
            logger.error(f"Vectorstore query failed: {e}")
            relevant_docs = []

        answer = ""
        sources = []
        source_type = "local"

        if relevant_docs:
            context = "\n".join(f"{doc} [source: {meta.get('source', 'Unknown')}]" for doc, _, meta in relevant_docs)
            rag_answer = chain.invoke({"question": message_content})
            indicators = ["no information", "not found", "unknown"]
            if any(ind in rag_answer.lower() for ind in indicators):
                try:
                    web_result = search_web(message_content, include_meta=True)
                    web_answer = web_result.get("answer", "")
                    answer = f"\ud83d\udcda {rag_answer}\n\n\ud83c\udf10 {web_answer}"
                    sources = [r["url"] for r in web_result.get("results", [])][:2]
                    source_type = "hybrid"
                except Exception as e:
                    logger.warning(f"Web fallback failed: {e}")
                    answer = rag_answer
            else:
                answer = rag_answer
                sources = list({meta.get("source", "") for _, _, meta in relevant_docs})
        else:
            try:
                web_result = search_web(message_content, include_meta=True)
                answer = web_result.get("answer", "I don't have enough information.")
                sources = [r["url"] for r in web_result.get("results", [])][:2]
                source_type = "web"
            except Exception as e:
                logger.error(f"Web search failed: {e}")
                answer = "I don't have enough information."
                source_type = "error"

        response_time = time.perf_counter() - start_time
        formatted = ResponseFormatter.format_response(answer, format_type, message_content)

        return {
            "content": formatted,
            "sources": sources,
            "response_time": response_time,
            "format_used": format_type,
            "source_type": source_type,
            "doc_ids": [meta.get("source", "") for _, _, meta in relevant_docs],
            "tags": []
        }

    async def upload_document(self, user_id: UUID, filename: str, content: bytes, content_hash: str, tags: List[str] = None) -> DocumentMetadata:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)

        with open(temp_path, "wb") as f:
            f.write(content)

        try:
            self.s3_uploader.upload_document(str(user_id), filename, temp_path)
            doc_metadata = DocumentMetadata(
                id=content_hash,
                filename=filename,
                upload_status="completed",
                local_path=temp_path,
                doc_type=os.path.splitext(filename)[1].lower(),
                tags=tags or [],
                owner_user_id=user_id,
                is_personalized=True,
                visibility="private"
            )
            self.db.add(doc_metadata)
            self.db.commit()
            load_documents(temp_dir, set())
            return doc_metadata
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    async def delete_document(self, document_id: str, user_id: UUID):
        document = self.db.query(DocumentMetadata).filter_by(id=document_id, owner_user_id=user_id).first()
        if document:
            try:
                self.chroma_client.delete_document(document_id)
            except Exception as e:
                logger.warning(f"Vectorstore delete failed: {e}")
            self.db.delete(document)
            self.db.commit()