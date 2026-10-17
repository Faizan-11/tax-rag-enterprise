"""
config.py
---------
Central configuration for the Enterprise RAG system.
Change values here instead of hunting through scripts.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ============ PATHS ============
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# ============ MODELS ============
# LLM
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

# Embeddings
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# ============ RAG PIPELINE SETTINGS ============
# Chunking (We'll tune this for the massive Tax Act)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
TOP_K_RETRIEVAL = 20   # Fetch 20 for hybrid search / reranking
TOP_K_FINAL = 5        # Pass 5 to the LLM

# Qdrant
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "tax_act"

# ============ VALIDATION ============
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ OPENAI_API_KEY missing. Check your .env file.")