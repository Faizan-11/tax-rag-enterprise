"""
05_query.py
-----------
The Enterprise Hybrid RAG Pipeline:
1. Query Rewriting
2. Hybrid Search (Qdrant + BM25) -> Get Top 20
3. Reranking (FlashRank) -> Pick Top 5
4. Generate answer
"""

import sys
import json
import pickle
import os
from pathlib import Path
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from flashrank import Ranker, RerankRequest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, QDRANT_URL, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL, TOP_K_RETRIEVAL, TOP_K_FINAL

# Initialize clients
client = QdrantClient(url=QDRANT_URL)
embed_model = SentenceTransformer(EMBEDDING_MODEL)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Reranker (uses a tiny 4MB model that runs on CPU instantly)
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

SYSTEM_PROMPT = """You are a helpful assistant answering questions about the Indian Income Tax Act.
Answer ONLY using the provided context passages.
If the context doesn't contain the answer, say: "I don't have enough information in the provided documents to answer that."
Cite page numbers in the format [pages X-Y] after each fact.
Be concise."""

def load_bm25():
    bm25_file = DATA_DIR / "bm25_index.pkl"
    with bm25_file.open("rb") as f:
        bm25 = pickle.load(f)
    
    chunks = []
    with (DATA_DIR / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return bm25, chunks

def rewrite_query(question: str) -> str:
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert in Indian tax law. Rewrite the user's question into a formal search query using correct Income Tax Act terminology. Do NOT assume the document year. Return ONLY the rewritten query."},
            {"role": "user", "content": question}
        ],
        temperature=0.0,
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()

def hybrid_search(question: str, bm25, bm25_chunks):
    # Vector Search
    q_emb = embed_model.encode(question, normalize_embeddings=True).tolist()
    vector_results = client.query_points(
        collection_name=COLLECTION_NAME, query=q_emb, limit=TOP_K_RETRIEVAL
    ).points
    
    # BM25 Search
    tokenized_query = question.lower().split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K_RETRIEVAL]
    
    # Combine
    combined = {}
    for hit in vector_results:
        combined[hit.id] = {"chunk_id": hit.id, "text": hit.payload["text"], "pages": hit.payload["pages"], "v_score": hit.score, "b_score": 0.0}
    for idx in top_bm25_indices:
        chunk = bm25_chunks[idx]
        cid = chunk["chunk_id"]
        if cid not in combined:
            combined[cid] = {"chunk_id": cid, "text": chunk["text"], "pages": chunk["pages"], "v_score": 0.0, "b_score": bm25_scores[idx]}
        else:
            combined[cid]["b_score"] = bm25_scores[idx]
            
    sorted_results = sorted(combined.values(), key=lambda x: (x["v_score"] > 0) + (x["b_score"] > 0) + x["v_score"], reverse=True)
    return sorted_results[:TOP_K_RETRIEVAL]

def rerank_chunks(question: str, hits: list):
    """Use the Cross-Encoder to find the absolute best 5 chunks from the 20."""
    print(f"  ⚖️ Reranking {len(hits)} chunks down to {TOP_K_FINAL}...")
    
    # Prepare data for FlashRank
    passages = [{"id": str(hit["chunk_id"]), "text": hit["text"], "meta": hit} for hit in hits]
    rerank_request = RerankRequest(query=question, passages=passages)
    
    # Rerank
    results = ranker.rerank(rerank_request)
    
    # Return the top K
    final_hits = []
    for r in results[:TOP_K_FINAL]:
        final_hits.append(r["meta"])
    return final_hits

def generate_answer(question: str, hits: list) -> str:
    context_parts = []
    for i, hit in enumerate(hits, start=1):
        context_parts.append(f"--- Passage {i} [pages {hit['pages']}] ---\n{hit['text']}")
    context = "\n\n".join(context_parts)
    
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"}
        ],
        temperature=0.1,
        max_tokens=600,
    )
    return response.choices[0].message.content

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/05_query.py 'your question here'")
        return
        
    original_question = " ".join(sys.argv[1:])
    print(f"\nOriginal Question: {original_question}")
    
    print("  ✨ Rewriting query...")
    rewritten = rewrite_query(original_question)
    print(f"  Rewritten: {rewritten}")
    
    bm25, bm25_chunks = load_bm25()
    hits = hybrid_search(rewritten, bm25, bm25_chunks)
    
    final_hits = rerank_chunks(original_question, hits)
    
    print(f"  🧠 Generating answer...")
    answer = generate_answer(original_question, final_hits)
    
    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(answer)
    print("=" * 60)

if __name__ == "__main__":
    main()