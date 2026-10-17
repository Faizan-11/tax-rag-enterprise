"""
03_embed.py
-----------
Generate embeddings for the 1,200+ Tax Act chunks.
"""

import sys
import json
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, EMBEDDING_MODEL

def main():
    input_file = DATA_DIR / "chunks.jsonl"
    output_file = DATA_DIR / "chunks_with_embeddings.jsonl"
    
    print(f"Loading chunks from {input_file.name}...")
    chunks = []
    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
            
    print(f"  Loaded {len(chunks)} chunks")
    
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    # Note: We use the same model as Project 1 for speed, but we're doing 20x the work
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Generating embeddings (this will take a minute for 1,200 chunks)...")
    start_time = time.time()
    
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts, 
        batch_size=32, 
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    elapsed = time.time() - start_time
    print(f"\n  Done in {elapsed:.1f}s")
    
    # Attach embeddings to chunks and save
    with output_file.open("w", encoding="utf-8") as f:
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.tolist()
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
    print(f"✅ Saved {len(chunks)} embedded chunks to {output_file}")

if __name__ == "__main__":
    main()