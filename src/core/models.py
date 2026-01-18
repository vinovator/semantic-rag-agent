# src/core/models.py
from dataclasses import dataclass, field
from typing import Annotated, List, Optional
from semantic_kernel.data.vector import (
    vectorstoremodel,
    VectorStoreField,
    FieldTypes,
)

@vectorstoremodel
@dataclass
class KnowledgeRecord:
    # 1. The Key (Unique ID)
    id: Annotated[str, VectorStoreField(field_type=FieldTypes.KEY)]
    
    # 2. The Text Content (What the LLM reads)
    content: Annotated[str, VectorStoreField(
        field_type=FieldTypes.DATA,
        is_full_text_indexed=True
    )]
    
    # 3. Metadata (For filtering/citation)
    source_metadata: Annotated[str, VectorStoreField(
        field_type=FieldTypes.DATA,
        is_indexed=True
    )]
    
    # 4. The Vector (The math representation)
    # Note: Dimensions must match your model (Nomic/All-MiniLM = 384 or 768. OpenAI = 1536)
    embedding: Annotated[Optional[List[float]], VectorStoreField(
        field_type=FieldTypes.VECTOR,
        dimensions=768 # <--- CHANGE THIS based on your embedding model
    )] = None