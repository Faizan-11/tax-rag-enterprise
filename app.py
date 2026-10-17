import sys
import json
import pickle
import os
from pathlib import Path
import numpy as np
import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from flashrank import Ranker, RerankRequest
from dotenv import load_dotenv

# Load .env file IMMEDIATELY
load_dotenv()

# --- LOCAL CONFIG ---
BASE_DIR = Path(__file__).parent
BM25_FILE = BASE_DIR / "bm25_index.pkl"
CHUNKS_FILE = BASE_DIR / "chunks_with_embeddings.jsonl"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "tax_act"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gpt-4o-mini"
TOP_K_RETRIEVAL = 20
TOP_K_FINAL = 5

# --- CACHED RESOURCES ---
@st.cache_resource
def load_models():
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=2.0)
        client.get_collections()
        use_qdrant = True
    except Exception:
        use_qdrant = False
        client = None
        
    embed = SentenceTransformer(EMBEDDING_MODEL)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            st.error("OPENAI_API_KEY not found. Add it to your .env or Streamlit secrets.")
            st.stop()
            
    openai_client = OpenAI(api_key=api_key)
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    return client, embed, openai_client, ranker, use_qdrant

@st.cache_resource
def load_bm25_and_chunks():
    with BM25_FILE.open("rb") as f:
        bm25 = pickle.load(f)
    chunks = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
            
    emb_matrix = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix = emb_matrix / np.clip(norms, 1e-9, None)
    
    return bm25, chunks, emb_matrix

client, embed, openai_client, ranker, use_qdrant = load_models()
bm25, bm25_chunks, emb_matrix = load_bm25_and_chunks()

SYSTEM_PROMPT = """You are a helpful assistant answering questions about the Indian Income Tax Act.
Answer ONLY using the provided context passages.
If the context doesn't contain the answer, say: "I don't have enough information in the provided documents to answer that."
Cite page numbers in the format [pages X-Y] after each fact.
Be concise."""

def rewrite_query(question):
    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": "You are an expert in Indian tax law. Rewrite the user's question into a formal search query. Do NOT assume the document year. Return ONLY the rewritten query."},
                      {"role": "user", "content": question}],
            temperature=0.0, max_tokens=100,
        )
        return response.choices[0].message.content.strip()
    except:
        return question

def hybrid_search(query):
    q_emb = embed.encode(query, normalize_embeddings=True).tolist()
    combined = {}
    
    if use_qdrant:
        vector_results = client.query_points(collection_name=COLLECTION_NAME, query=q_emb, limit=TOP_K_RETRIEVAL).points
        for hit in vector_results:
            combined[hit.id] = {"chunk_id": hit.id, "text": hit.payload["text"], "pages": hit.payload["pages"], "v_score": hit.score, "b_score": 0.0}
    else:
        q_emb_np = np.array(q_emb, dtype=np.float32)
        scores = emb_matrix @ q_emb_np
        top_indices = np.argsort(scores)[::-1][:TOP_K_RETRIEVAL]
        for idx in top_indices:
            chunk = bm25_chunks[idx]
            combined[chunk["chunk_id"]] = {"chunk_id": chunk["chunk_id"], "text": chunk["text"], "pages": chunk["pages"], "v_score": float(scores[idx]), "b_score": 0.0}
    
    tokenized_query = query.lower().split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K_RETRIEVAL]
    
    for idx in top_bm25:
        chunk = bm25_chunks[idx]
        cid = chunk["chunk_id"]
        if cid not in combined:
            combined[cid] = {"chunk_id": cid, "text": chunk["text"], "pages": chunk["pages"], "v_score": 0.0, "b_score": bm25_scores[idx]}
        else:
            combined[cid]["b_score"] = bm25_scores[idx]
            
    sorted_res = sorted(combined.values(), key=lambda x: (x["v_score"] > 0) + (x["b_score"] > 0) + x["v_score"], reverse=True)
    return sorted_res[:TOP_K_RETRIEVAL]

def rerank(question, hits):
    passages = [{"id": str(h["chunk_id"]), "text": h["text"], "meta": h} for h in hits]
    results = ranker.rerank(RerankRequest(query=question, passages=passages))
    return [r["meta"] for r in results[:TOP_K_FINAL]]

def generate(question, hits):
    context = "\n\n".join([f"--- Passage {i} [pages {h['pages']}] ---\n{h['text']}" for i, h in enumerate(hits, 1)])
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"}],
        temperature=0.1, max_tokens=600,
    )
    return response.choices[0].message.content

st.set_page_config(page_title="Income Tax 2025 Assistant", page_icon="⚖️", layout="centered")
st.title("⚖️ Income Tax Act 2025 Assistant")
st.caption("Enterprise RAG system over the Indian Income Tax Act 2025. Answers cite source pages.")

with st.expander("ℹ️ How this works"):
    st.markdown("""
    This is an **Enterprise RAG** system:
    1. **Query Rewriting** — Your question is rewritten into formal legal terms.
    2. **Hybrid Search** — We search both by meaning (Vectors) and by exact keywords (BM25).
    3. **Reranking** — A Cross-Encoder model picks the 5 best passages out of 20.
    4. **Generation** — GPT-4o-mini answers with citations.
    
    **Note:** The source document is the *Income Tax Act, 2025* (effective April 2026). Sections like 80C from the 1961 Act do not exist here.
    """)

examples = [
    "How can I save tax on my health insurance premium?",
    "What are the penalties for late filing of returns?",
    "How is capital gains tax on property calculated?",
    "What is Section 80C?",
    "How do I bake a cake?"
]
cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    if cols[i].button(ex, key=f"ex_{i}"):
        st.session_state["q"] = ex

question = st.text_input("Ask a question:", value=st.session_state.get("q", ""), placeholder="e.g. What are the TDS rates?")

if st.button("Get Answer", type="primary") and question.strip():
    with st.spinner("Rewriting query..."):
        rewritten = rewrite_query(question)
    
    with st.spinner("Searching hybrid index..."):
        hits = hybrid_search(rewritten)
    
    with st.spinner("Reranking..."):
        final_hits = rerank(question, hits)
        
    with st.spinner("Generating answer..."):
        answer = generate(question, final_hits)
        
    st.markdown("### Answer")
    st.markdown(answer)
    
    with st.expander("🔍 Debug Info"):
        st.write(f"**Rewritten Query:** {rewritten}")
        st.write(f"**Search Mode:** {'Qdrant (Local)' if use_qdrant else 'In-Memory (Cloud)'}")
        for i, h in enumerate(final_hits, 1):
            st.write(f"**Passage {i}** (Pages: {h['pages']})")
            st.text(h["text"][:300] + "...")

st.divider()
st.caption("Built with Qdrant, BM25, FlashRank, GPT-4o-mini, and Streamlit.")
