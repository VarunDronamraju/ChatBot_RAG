from dataclasses import dataclass, field
from typing import List, Optional
import uuid

@dataclass
class DocumentEmbedding:
    vector: List[float]
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    distance_metric: str = "cosine"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        default_meta = {
            "filename": "",
            "chunk_id": "",
            "page_number": 0,
            "doc_id": "",
            "user_id": "",
            "conversation_ids": [],
            "tags": [],
            "keywords": [],
            "doc_type": "",
            "visibility": "",
            "index": "HNSW"
        }
        for k, v in default_meta.items():
            self.metadata.setdefault(k, v)

@dataclass
class ConversationContext:
    vector: List[float]
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    distance_metric: str = "cosine"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        default_meta = {
            "conversation_id": "",
            "user_id": "",
            "timestamp": 0,
            "tags": [],
            "index": "FLAT"
        }
        for k, v in default_meta.items():
            self.metadata.setdefault(k, v)
