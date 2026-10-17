# ⚖️ Income Tax Act 2025 — Enterprise RAG Assistant

A **Retrieval-Augmented Generation (RAG)** system that answers questions about the 686-page Indian Income Tax Act, 2025, using hybrid retrieval, query rewriting, cross-encoder reranking, and cited answers.

**🔗 Live Demo:** [tax-rag-enterprise.streamlit.app](https://tax-rag-enterprise-9rken74ackbzkk7yryk7g7.streamlit.app/)  
*Note: The live demo uses an in-memory vector fallback for cloud compatibility. The local development version uses Qdrant.*

---

## What It Does

Ask a natural-language question like *"How can I save tax on my health insurance premium?"* and the system:

1. **Rewrites** the question into formal legal terminology.
2. **Retrieves** the top 20 chunks using hybrid search (dense vectors + BM25 keyword search).
3. **Reranks** those 20 chunks down to the 5 most relevant using a cross-encoder.
4. **Generates** a grounded answer with page citations using GPT-4o-mini.

**Try it yourself:** [Open the live demo](https://tax-rag-enterprise-9rken74ackbzkk7yryk7g7.streamlit.app/)

---

## 🚀 The Evolution: Level 1 to Level 2

This project is the successor to the IITB College Rules RAG project. It extends a basic semantic-search pipeline with a more advanced retrieval and ranking workflow.

| Feature | Project 1: College Rules | Project 2: Income Tax Act |
|---|---|---|
| **Document size** | 76 pages | **686 pages** (1.7M characters) |
| **Search type** | Dense vector search | **Hybrid search** (dense vectors + BM25) |
| **Query processing** | None | **LLM query rewriting** |
| **Result refinement** | None | **Cross-encoder reranking** (FlashRank) |
| **Evaluation set** | 10 questions | **25 adversarial questions** |

---

## Architecture

```text
                           ┌─────────────────────────────┐
                           │        User Question        │
                           └──────────────┬──────────────┘
                                          ▼
                           ┌─────────────────────────────┐
                           │       LLM Query Rewriter    │
                           │        (GPT-4o-mini)        │
                           └──────────────┬──────────────┘
                                          ▼
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
       ┌───────────────────────────┐             ┌───────────────────────────┐
       │    Qdrant Vector Search   │             │      BM25 Keyword Search  │
       │ BAAI/bge-small-en-v1.5    │             │      rank_bm25 library    │
       └─────────────┬─────────────┘             └─────────────┬─────────────┘
                     │                                         │
                     └─────────────────────┬───────────────────┘
                                           ▼
                           ┌─────────────────────────────┐
                           │       Hybrid Merge          │
                           │       Top 20 Chunks         │
                           └──────────────┬──────────────┘
                                          ▼
                           ┌─────────────────────────────┐
                           │   FlashRank Cross-Encoder   │
                           │       Top 20 → Top 5        │
                           └──────────────┬──────────────┘
                                          ▼
                           ┌─────────────────────────────┐
                           │    GPT-4o-mini Generation   │
                           │    Grounded Answer +        │
                           │    Page Citations           │
                           └─────────────────────────────┘
```

### Pipeline Stages

| Stage | Script | What it does |
|---|---|---|
| 1 | `01_extract.py` | Extracts approximately 1.7M characters from the source PDF using PyMuPDF |
| 2 | `02_chunk.py` | Splits text into approximately 500-token chunks using `tiktoken` |
| 3 | `03_embed.py` | Generates 384-dimensional embeddings using `BAAI/bge-small-en-v1.5` |
| 4 | `04_index.py` | Indexes chunks in Qdrant and builds the BM25 keyword index |
| 5 | `05_query.py` | Runs the query pipeline: rewrite → hybrid search → rerank → generate |
| 6 | `06_eval.py` | Runs the evaluation harness with an LLM-as-judge |

---

## Evaluation Results

Evaluated on a hand-labeled test set of **25 adversarial questions** drawn from the source document.

| Metric | Score |
|---|---|
| **Faithfulness** | **0.77** — measures whether answers are grounded in the source |
| **Relevancy** | **0.96** — measures whether answers address the question |
| **Trick-question refusal** | **100%** — correctly refused out-of-scope questions in this test set |

### Known Limitation: Document Versioning

The source PDF is the **Income Tax Act, 2025**. The Act renumbered sections compared with the Income-tax Act, 1961. If a user asks about an older section such as Section 80C, the query rewriter may infer that the user is referring to the 1961 Act. Since that section is not present in the 2025 source document, the retriever may refuse to answer.

This illustrates an important constraint: the assistant is designed to ground its answers in the indexed source rather than fabricate support from a document it has not indexed.

---

## Tech Stack

| Layer | Tool |
|---|---|
| **PDF extraction** | PyMuPDF |
| **Tokenization** | `tiktoken` |
| **Embeddings** | `BAAI/bge-small-en-v1.5` (Sentence Transformers) |
| **Vector database** | Qdrant |
| **Keyword search** | `rank_bm25` |
| **Reranker** | FlashRank (`ms-marco-MiniLM-L-12-v2`) |
| **LLM** | OpenAI GPT-4o-mini |
| **Web UI** | Streamlit |
| **Evaluation** | Custom evaluation harness with LLM-as-judge |
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
# Clone the repository
git clone https://github.com/Faizan-11/tax-rag-enterprise.git
cd tax-rag-enterprise

# Create and activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Qdrant
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant:latest

# Set up your API key in a local .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Launch the Streamlit app
streamlit run app/streamlit_app.py
```

### Run the Evaluation Harness

```bash
python scripts/06_eval.py
```

---

## What I'd Improve Next

- **Legal-aware chunking:** use a parser that understands legal structures such as sections and sub-sections, rather than relying only on token counts.
- **Multi-query expansion:** generate multiple variations of a question and retrieve results for each variation.
- **Semantic answer scoring:** replace keyword matching in evaluation with semantic similarity.
- **Larger-document stress test:** test the pipeline on a larger legal document, such as the Indian Constitution.

---

## Project Structure

```text
tax-rag-enterprise/
├── app/
│   └── streamlit_app.py          # Deployed web UI
├── scripts/
│   ├── 01_extract.py             # PDF → JSON
│   ├── 02_chunk.py               # JSON → chunked text
│   ├── 03_embed.py               # Text → vectors
│   ├── 04_index.py               # Vectors → Qdrant + BM25
│   ├── 05_query.py               # RAG query pipeline
│   └── 06_eval.py                # Evaluation harness
├── data/
│   ├── raw/                      # Source PDF
│   ├── processed/                # Extracted JSON
│   ├── chunks/                   # Text + embeddings
│   └── eval/                     # Test sets and results
├── config.py                     # Central configuration
├── requirements.txt
└── .gitignore
```

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built as a portfolio project demonstrating advanced RAG engineering: hybrid search, query rewriting, cross-encoder reranking, evaluation, and cloud-compatible fallback strategies.*