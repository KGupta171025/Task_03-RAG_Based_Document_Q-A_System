"""
Retrieval Engine.
Performs semantic Top-K cosine similarity searches against the FAISS vector index.
"""

import logging
from typing import Dict, List

from core.config import DEFAULT_TOP_K
from core.embeddings import get_embedding_manager
from core.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Handles Top-K semantic retrieval and citation structuring."""

    def __init__(self, top_k: int = DEFAULT_TOP_K):
        self.top_k = top_k
        self.embedder = get_embedding_manager()
        self.vector_store = get_vector_store()

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieves the top-K most semantically relevant document chunks.
        """
        k = top_k if top_k is not None else self.top_k
        if not query.strip():
            return []

        # Generate query vector
        query_vector = self.embedder.embed_query(query)

        # Query vector store
        raw_results = self.vector_store.search(query_vector, top_k=k)

        formatted_results: List[Dict] = []
        for rank, (chunk, score) in enumerate(raw_results, start=1):
            formatted_results.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "source_filename": chunk["source_filename"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "cosine_similarity": float(round(score, 4)),
                "text": chunk["text"],
                "citation": f"[{chunk['source_filename']} | Page {chunk['page_number']} | Chunk {chunk['chunk_index']}]",
            })

        return formatted_results


_RETRIEVER_INSTANCE = None


def get_retriever() -> DocumentRetriever:
    """Returns the shared retriever singleton."""
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        _RETRIEVER_INSTANCE = DocumentRetriever()
    return _RETRIEVER_INSTANCE
