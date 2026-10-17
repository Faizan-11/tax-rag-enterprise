# ⚖️ Income Tax Act 2025 — Enterprise RAG Assistant

A **Tier-1 Enterprise Retrieval-Augmented Generation (RAG)** system that answers complex questions over the 686-page Indian Income Tax Act, 2025. 

**🔗 Live Demo:** [tax-rag-enterprise.streamlit.app](https://tax-rag-enterprise-9rken74ackbzkk7yryk7g7.streamlit.app/)  
*Note: The live demo uses an in-memory vector fallback for cloud compatibility. The local development version uses a Qdrant vector database.*

---

## What It Does

Ask a natural-language question like *"How can I save tax on my health insurance premium?"* and the system:

1. **Rewrites** your question into formal legal terminology.
2. **Retrieves** the top 20 chunks using Hybrid Search (Dense Vectors + BM25 Keyword).
3. **Reranks** the 20 chunks down to the best 5 using a Cross-Encoder.
4. **Generates** a grounded answer with page citations using GPT-4o-mini.

**Try it yourself:** [Live demo link](https://tax-rag-enterprise-9rken74ackbzkk7yryk7g7.streamlit.app/)

---

## 🚀 The Evolution (Level 1 to Level 2)

This project is the direct successor to my first RAG project (IITB College Rules). It represents a jump from a **basic semantic search pipeline** to a **production-grade enterprise architecture**:

| Feature | Project 1 (College Rules) | Project 2 (Tax Act) |
|---|---|---|
| **Document Size** | 76 pages | **686 pages** (1.7M chars) |
| **Search Type** | Dense Vector Only | **Hybrid** (Dense Vector + BM25 Keyword) |
| **Query Processing** | None | **LLM Query Rewriting** |
| **Result Refinement** | None | **Cross-Encoder Reranking** (FlashRank) |
| **Evaluation** | 10 questions | **25 adversarial questions** |

---

## Architecture

```text
                       ┌─────────────────────────────┐
                       │       User Question         │
                       └──────────────┬──────────────┘
                                      ▼
                       ┌─────────────────────────────┐
                       │   LLM Query Rewriter        │
                       │   (GPT-4o-mini)             │
                       └──────────────┬──────────────┘
                                      ▼
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
┌───────────────────────────┐                   ┌───────────────────────────┐
│  Qdrant Vector Search     │                   │      BM25 Keyword         │
│  (BAAI/bge-small-en-v1.5) │                   │   (rank_bm25 library)     │
└─────────────┬─────────────┘                   └─────────────┬─────────────┘
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      ▼
                       ┌─────────────────────────────┐
                       │   Hybrid Merge (Top 20)     │
                       └──────────────┬──────────────┘
                                      ▼
                       ┌─────────────────────────────┐
                       │   FlashRank Cross-Encoder   │
                       │   (Filters 20 down to 5)    │
                       └──────────────┬──────────────┘
                                      ▼
                       ┌─────────────────────────────┐
                       │   GPT-4o-mini Generation    │
                       │   (Grounded + Cited)        │
                       └─────────────────────────────┘
### Pipeline Stages

| Stage | Script | What it does |
|---|---|---|
| **1** | `01_extract.py` | Extract 1.7M chars from PDF (PyMuPDF) |
| **2** | `02_chunk.py` | Split massive paragraphs into ~500-token chunks (tiktoken) |
| **3** | `03_embed.py` | Generate 384-dim embeddings (BAAI/bge-small-en-v1.5) |
| **4** | `04_index.py` | Index chunks in Qdrant DB + build BM25 keyword index |
| **5** | `05_query.py` | Query pipeline: Rewrite → Hybrid Search → Rerank → Generate |
| **6** | `06_eval.py` | Evaluation harness with LLM-as-judge |

---

## Evaluation Results

Evaluated on a hand-labeled test set of **25 adversarial questions** drawn from the rulebook.

| Metric | Score |
|---|---|
| **Faithfulness** | **0.77** (answer grounded in source) |
| **Relevancy** | **0.96** (answer addresses the question) |
| **Trick Question Refusal** | **100%** (correctly refused out-of-scope questions) |

### Known Limitations (Document Versioning)

Real engineering comes with honesty. Here's what this system *doesn't* do well yet:

The source PDF is the **Income Tax Act of 2025**. The Government of India completely renumbered the sections in this new Act. If a user asks for an old section (e.g., "What is Section 80C?"), the Query Rewriter guesses the user means the 1961 Act, but the retriever correctly refuses because 80C does not exist in the 2025 document. **This demonstrates the system's strict guardrails against hallucination.**

---

## Tech Stack

| Layer | Tool |
|---|---|
| **PDF extraction** | PyMuPDF |
| **Tokenization** | tiktoken |
| **Embeddings** | BAAI/bge-small-en-v1.5 (Sentence Transformers) |
| **Vector DB** | Qdrant (Docker) |
| **Keyword Search** | rank_bm25 |
| **Reranker** | FlashRank (`ms-marco-MiniLM-L-12-v2`) |
| **LLM** | OpenAI GPT-4o-mini |
| **Web UI** | Streamlit |
| **Deployment** | Streamlit Community Cloud |
| **Language** | Python 3.11 |

---

## Running Locally

### Prerequisites
- Python 3.11
- Docker (for Qdrant)
- OpenAI API key

### Setup

```bash
git clone https://github.com/Faizan-11/tax-rag-enterprise.git
cd tax-rag-enterprise

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Qdrant
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant:latest

# Set up secrets
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Run the Streamlit app
streamlit run app/streamlit_app.py

# Run the evaluation harness
python scripts/06_eval.py

# What I'd Improve Next
Better Legal-Aware Chunking — use a parser that understands "Section 1", "Sub-section 2", etc., rather than just token counts.

Multi-Query Expansion — generate 3 variations of the user's question and search for all of them.

Semantic Answer Scoring — replace keyword matching in the eval with cosine similarity.

Scale test — replicate the pipeline on the entire Indian Constitution.

# Project Structure
tax-rag-enterprise/
├── app/
│   └── streamlit_app.py          # Deployed web UI
├── scripts/
│   ├── 01_extract.py             # PDF → JSON
│   ├── 02_chunk.py               # JSON → chunked text
│   ├── 03_embed.py               # Text → vectors
│   ├── 04_index.py               # Vectors → Qdrant + BM25
│   ├── 05_query.py               # The RAG pipeline
│   └── 06_eval.py                # Evaluation harness
├── data/
│   ├── raw/                      # Source PDF
│   ├── processed/                # Extracted JSON
│   ├── chunks/                   # Text + Embeddings
│   └── eval/                     # Test sets & Results
├── config.py                     # Central configuration
├── requirements.txt
└── .gitignore


# License
MIT — see LICENSE.

Built as a portfolio project demonstrating advanced RAG engineering: Hybrid Search, Query Rewriting, Cross-Encoder Reranking, and Cloud Fallback strategies.