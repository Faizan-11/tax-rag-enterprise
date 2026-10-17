"""
04_index.py
-----------
Builds BOTH the Qdrant vector database AND the BM25 keyword index.
"""

import sys
import json
import pickle
import time
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, QDRANT_URL, COLLECTION_NAME, EMBEDDING_DIM

def main():
    input_file = DATA_DIR / "chunks_with_embeddings.jsonl"
    bm25_file = DATA_DIR / "bm25_index.pkl"
    
    print(f"Loading chunks with embeddings from {input_file.name}...")
    chunks = []
    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
            
    print(f"  Loaded {len(chunks)} chunks\n")
    
    # ==========================================
    # 1. Setup Qdrant (Vector Search)
    # ==========================================
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)
    
    if client.collection_exists(COLLECTION_NAME):
        print(f"  Dropping existing collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)
        
    print(f"  Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    
    print("Uploading vectors to Qdrant...")
    points = []
    for chunk in chunks:
        points.append(
            PointStruct(
                id=chunk["chunk_id"],
                vector=chunk["embedding"],
                payload={
                    "text": chunk["text"],
                    "pages": chunk["pages"],
                },
            )
        )
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"  ✅ Indexed {len(points)} vectors in Qdrant\n")
    
    # ==========================================
    # 2. Setup BM25 (Keyword Search)
    # ==========================================
    print("Building BM25 keyword index...")
    # BM25 works on tokenized text (lowercase words)
    tokenized_corpus = [chunk["text"].lower().split(" ") for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save the BM25 index to disk so the query script can load it
    with bm25_file.open("wb") as f:
        pickle.dump(bm25, f)
        
    print(f"  ✅ Saved BM25 index to {bm25_file}")

if __name__ == "__main__":
    main()