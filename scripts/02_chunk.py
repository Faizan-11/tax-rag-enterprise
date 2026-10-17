"""
02_chunk.py
-----------
Enterprise-grade chunking.
Fixes the "massive paragraph" problem by splitting long text at sentence boundaries.
"""

import sys
import json
import re
from pathlib import Path
import tiktoken

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

ENCODING = tiktoken.get_encoding("cl100k_base")

def clean_text(text: str) -> str:
    """Remove common PDF artifacts."""
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.replace('\u00a0', ' ').replace('\u200b', '')
    return text.strip()

def split_into_sentences(text: str) -> list[str]:
    """Split a massive paragraph into sentences so we can chunk it safely."""
    # Split on period/question mark/exclamation followed by a space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_pages(pages: list[dict]) -> list[dict]:
    chunks = []
    current_chunk = []
    current_tokens = 0
    chunk_id = 0
    current_pages = set()
    
    for page in pages:
        page_num = page["page_num"]
        text = clean_text(page["text"])
        
        # First split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for para in paragraphs:
            para_tokens = ENCODING.encode(para)
            para_len = len(para_tokens)
            
            # 🚨 NEW LOGIC: If a single paragraph is too big, split it by sentences
            if para_len > CHUNK_SIZE:
                sentences = split_into_sentences(para)
                
                for sentence in sentences:
                    sent_tokens = ENCODING.encode(sentence)
                    sent_len = len(sent_tokens)
                    
                    if current_tokens + sent_len > CHUNK_SIZE and current_chunk:
                        # Save current chunk
                        chunks.append({
                            "chunk_id": chunk_id,
                            "text": "\n".join(current_chunk),
                            "pages": sorted(list(current_pages)),
                            "token_count": current_tokens
                        })
                        chunk_id += 1
                        
                        # Create overlap
                        overlap_text = current_chunk[-1]
                        overlap_tokens = ENCODING.encode(overlap_text)
                        current_chunk = [overlap_text, sentence]
                        current_tokens = len(overlap_tokens) + sent_len
                        current_pages = {page_num}
                    else:
                        current_chunk.append(sentence)
                        current_tokens += sent_len
                        current_pages.add(page_num)
            
            else:
                # Normal paragraph: check if it fits in the current chunk
                if current_tokens + para_len > CHUNK_SIZE and current_chunk:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": "\n\n".join(current_chunk),
                        "pages": sorted(list(current_pages)),
                        "token_count": current_tokens
                    })
                    chunk_id += 1
                    
                    overlap_text = current_chunk[-1]
                    overlap_tokens = ENCODING.encode(overlap_text)
                    current_chunk = [overlap_text, para]
                    current_tokens = len(overlap_tokens) + para_len
                    current_pages = {page_num}
                else:
                    current_chunk.append(para)
                    current_tokens += para_len
                    current_pages.add(page_num)
                    
    # Add final chunk
    if current_chunk:
        chunks.append({
            "chunk_id": chunk_id,
            "text": "\n\n".join(current_chunk),
            "pages": sorted(list(current_pages)),
            "token_count": current_tokens
        })
        
    return chunks

def main():
    input_file = PROCESSED_DIR / "income_tax_act_pages.json"
    print(f"Loading pages from {input_file.name}...")
    with input_file.open("r", encoding="utf-8") as f:
        pages = json.load(f)
        
    print(f"  Loaded {len(pages)} pages")
    
    print(f"Chunking (target size: {CHUNK_SIZE} tokens, overlap: {CHUNK_OVERLAP})...")
    chunks = chunk_pages(pages)
    
    print(f"  Created {len(chunks)} chunks")
    
    output_file = DATA_DIR / "chunks.jsonl"
    with output_file.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
    print(f"✅ Saved {len(chunks)} chunks to {output_file}")
    
    sizes = [c["token_count"] for c in chunks]
    print(f"\nChunk size stats:")
    print(f"  Min:    {min(sizes)} tokens")
    print(f"  Max:    {max(sizes)} tokens")
    print(f"  Mean:   {sum(sizes)/len(sizes):.0f} tokens")
    
    print("\n" + "=" * 50)
    print("SAMPLE CHUNK (ID: 0):")
    print("=" * 50)
    print(chunks[0]["text"][:500] + "...")
    print(f"\nPages: {chunks[0]['pages']}")
    print(f"Tokens: {chunks[0]['token_count']}")

if __name__ == "__main__":
    main()