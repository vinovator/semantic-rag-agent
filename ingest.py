# ingest.py
import asyncio
import os
from dotenv import load_dotenv
load_dotenv() # Load env vars before kernel build

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.kernel import build_kernel, get_collection
from src.core.config import load_config
from src.core.models import KnowledgeRecord

from src.loaders.file_loader import FileLoader

# Initialize config and embedding generator via API-standard build_kernel
# This ensures ingest uses the exact same embedding service as the Agent
_, embedding_gen = build_kernel()

async def ingest():
    config = load_config()
    collection = get_collection()
    
    # Create collection if it doesn't exist
    await collection.ensure_collection_exists()
    
    input_dir = "data/inputs"
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    print(f"--- Indexing to {collection.collection_name} ---")
    
    # Only process PDF files for Vector Search
    files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    
    for filename in files:
        path = os.path.join(input_dir, filename)
        print(f"Processing {filename}...")
        
        # 1. Load File Content (Decoupled Logic)
        raw_chunks = FileLoader.load_file(path)
        
        # 2. Split into smaller chunks if necessary
        chunks = []
        for raw in raw_chunks:
            chunks.extend(splitter.split_text(raw))

        # Process Chunks for this File
        batch_records = []
        for j, chunk in enumerate(chunks):
            # 1. Generate Embedding
            vectors = await embedding_gen.generate_embeddings([chunk])
            vector = vectors[0]
            
            # 2. Create Record
            record = KnowledgeRecord(
                id=f"{filename}_{j}",
                content=chunk,
                source_metadata=f"Source: {filename}, Chunk: {j+1}",
                embedding=vector.tolist()
            )
            batch_records.append(record)
        
        if batch_records:
            await collection.upsert(batch_records)
            print(f"  -> Saved {len(batch_records)} chunks from {filename}")

    print("--- Ingestion Complete ---")

if __name__ == "__main__":
    asyncio.run(ingest())