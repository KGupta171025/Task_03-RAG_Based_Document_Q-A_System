"""
System Configuration and Constants for RAG Pipeline.
"""

from pathlib import Path
from typing import List

# Path definitions
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_documents"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
ASSETS_DIR = PROJECT_ROOT / "assets"
SCREENSHOTS_DIR = ASSETS_DIR / "screenshots"

# Ensure runtime directories exist
SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Chunking specifications (per assignment requirements: 500 tokens, 50 overlap)
CHUNK_SIZE_TOKENS: int = 500
CHUNK_OVERLAP_TOKENS: int = 50

# Retrieval specifications (Top-5 cosine similarity search)
DEFAULT_TOP_K: int = 5
SIMILARITY_THRESHOLD: float = 0.25

# Embedding model
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384

# File constraints
SUPPORTED_EXTENSIONS: List[str] = [".pdf", ".txt", ".docx"]
MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB

# API Specifications
API_TITLE = "Nexoramind RAG Document Q&A API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Production Retrieval-Augmented Generation (RAG) API for multi-format document Q&A "
    "using FAISS vector search, sentence-transformers, and contextual generation with source citations."
)
