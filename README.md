# ⚖️ Income Tax Act 2025 — Enterprise RAG Assistant

A **Tier-1 Enterprise Retrieval-Augmented Generation (RAG)** system that answers complex questions over the 686-page Indian Income Tax Act, 2025. 

**🔗 Live Demo:** [tax-rag-enterprise.streamlit.app](https://tax-rag-enterprise-9rken74ackbzkk7yryk7g7.streamlit.app/)  
*Note: The live demo uses an in-memory vector fallback for cloud compatibility. The local development version uses a Qdrant vector database.*

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

## 🏗️ Architecture

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




⚙️ Pipeline Stages
The code is modularized into 6 distinct stages. Here is how the data flows from PDF to answer:

Stage	Script	Description
1. Extract	scripts/01_extract.py	Uses PyMuPDF to extract 1.7M characters, saving page numbers and text into structured JSON.
2. Chunk	scripts/02_chunk.py	Splits massive legal paragraphs at sentence boundaries to enforce a ~500 token limit, preventing context overflow.
3. Embed	scripts/03_embed.py	Generates 384-dimensional semantic vectors for 1,262 chunks using BGE-small.
4. Index	scripts/04_index.py	Creates a dual-index: Uploads vectors to Qdrant (Docker) and builds a BM25 keyword index (pickled).
5. Query	scripts/05_query.py	The pipeline: Query Rewrite → Hybrid Retrieve → Rerank → Generate.
6. Evaluate	scripts/06_eval.py	The eval harness. Runs 25 adversarial questions and scores Faithfulness, Relevancy, and Retrieval Accuracy.


🧠 Key Engineering Decisions
Hybrid Search (BM25 + Vectors): Pure vector search fails on exact legal citations (e.g., "Section 80C"). By combining BM25 keyword search with dense vector search, the system catches both conceptual queries and exact section lookups.

Query Rewriting: Users ask questions like "How do I save tax on health insurance?" The LLM rewrites this into formal legal terminology before searching the database.

Cross-Encoder Reranking: We retrieve the top 20 chunks, but use a FlashRank Cross-Encoder to score them and pass only the absolute best 5 to the final LLM prompt. This reduces hallucinations and saves tokens.

In-Memory Cloud Fallback: To deploy on Streamlit Community Cloud (which lacks Docker), the app detects the absence of Qdrant and seamlessly switches to a normalized in-memory numpy matrix for vector search.

📊 Evaluation & Honest Limitations
We evaluated the system against a 25-question hand-labeled test set (factual, numerical, conceptual, and adversarial).

Faithfulness: 0.77

Relevancy: 0.96

Trick Question Refusal Rate: 100% (correctly refused to answer questions about cake, GST, and the 1961 Act).


⚠️ Known Limitations (Document Versioning)
The source PDF is the Income Tax Act of 2025. The Government of India completely renumbered the sections in this new Act.
If a user asks for an old section (e.g., "What is Section 80C?"), the Query Rewriter guesses the user means the 1961 Act, but the retriever correctly refuses because 80C does not exist in the 2025 document. This demonstrates the system's strict guardrails against hallucination.


🛠️ Tech Stack
Layer	Tool
PDF Extraction	PyMuPDF
Tokenization & Chunking	tiktoken
Embeddings	BAAI/bge-small-en-v1.5 (Sentence Transformers)
Vector DB	Qdrant (Docker)
Keyword Search	rank_bm25
Reranker	FlashRank (ms-marco-MiniLM-L-12-v2)
LLM	OpenAI GPT-4o-mini
Web UI	Streamlit
Deployment	Streamlit Community Cloud


📂 Project Structure

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