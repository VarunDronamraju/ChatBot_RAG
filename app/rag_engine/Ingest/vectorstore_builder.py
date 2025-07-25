#vectorstore_builder.py
from dataclasses import dataclass, field
from typing import List, Optional
import uuid

def list_to_string(items: List[str], separator: str = ",") -> str:
    """Convert list to comma-separated string for ChromaDB compatibility"""
    if not items:
        return ""
    return separator.join(str(item) for item in items)

def string_to_list(text: str, separator: str = ",") -> List[str]:
    """Convert comma-separated string back to list"""
    if not text or text.strip() == "":
        return []
    return [item.strip() for item in text.split(separator) if item.strip()]

@dataclass
class DocumentEmbedding:
    vector: List[float]
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    distance_metric: str = "cosine"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        # ChromaDB only accepts str, int, float, bool in metadata
        default_meta = {
            "filename": "",
            "chunk_id": "",
            "page_number": 0,
            "doc_id": "",
            "user_id": "",
            "conversation_ids": "",  # Store as comma-separated string
            "tags": "",             # Store as comma-separated string
            "keywords": "",         # Store as comma-separated string
            "doc_type": "",
            "visibility": "",
            "index": "HNSW"
        }
        for k, v in default_meta.items():
            self.metadata.setdefault(k, v)
        
        # Convert any lists to strings if they exist
        self._convert_lists_to_strings()
    
    def _convert_lists_to_strings(self):
        """Convert list values to comma-separated strings"""
        list_fields = ["conversation_ids", "tags", "keywords"]
        for field in list_fields:
            if field in self.metadata and isinstance(self.metadata[field], list):
                self.metadata[field] = list_to_string(self.metadata[field])
    
    def set_tags(self, tags: List[str]):
        """Set tags as a list (will be converted to string)"""
        self.metadata["tags"] = list_to_string(tags)
    
    def get_tags(self) -> List[str]:
        """Get tags as a list"""
        return string_to_list(self.metadata.get("tags", ""))
    
    def set_keywords(self, keywords: List[str]):
        """Set keywords as a list (will be converted to string)"""
        self.metadata["keywords"] = list_to_string(keywords)
    
    def get_keywords(self) -> List[str]:
        """Get keywords as a list"""
        return string_to_list(self.metadata.get("keywords", ""))

@dataclass
class ConversationContext:
    vector: List[float]
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    distance_metric: str = "cosine"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        # ChromaDB only accepts str, int, float, bool in metadata
        default_meta = {
            "conversation_id": "",
            "user_id": "",
            "timestamp": 0,
            "tags": "",  # Store as comma-separated string
            "index": "FLAT"
        }
        for k, v in default_meta.items():
            self.metadata.setdefault(k, v)
        
        # Convert any lists to strings if they exist
        if "tags" in self.metadata and isinstance(self.metadata["tags"], list):
            self.metadata["tags"] = list_to_string(self.metadata["tags"])
    
    def set_tags(self, tags: List[str]):
        """Set tags as a list (will be converted to string)"""
        self.metadata["tags"] = list_to_string(tags)
    
    def get_tags(self) -> List[str]:
        """Get tags as a list"""
        return string_to_list(self.metadata.get("tags", ""))