"""
FAISS Vector Store Manager.
Maintains dense vector indexing using FAISS IndexFlatIP (Cosine Similarity)
along with document and chunk-level metadata management.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import faiss
import numpy as np

from core.chunker import DocumentChunk
from core.config import EMBEDDING_DIMENSION, VECTOR_DB_DIR

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Manages vector indexing, document tracking, deletion, and cosine similarity search.
    """

    def __init__(
        self,
        dimension: int = EMBEDDING_DIMENSION,
        persist_dir: Path = VECTOR_DB_DIR,
    ):
        self.dimension = dimension
        self.persist_dir = persist_dir
        self.index = faiss.IndexFlatIP(self.dimension)

        # In-memory stores
        self.chunks_map: Dict[str, dict] = {}        # chunk_id -> chunk dict
        self.indexed_chunk_ids: List[str] = []       # Aligned with FAISS internal position
        self.documents_map: Dict[str, dict] = {}     # doc_id -> document metadata

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.load()

    def add_document_chunks(
        self,
        doc_id: str,
        filename: str,
        file_type: str,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray,
    ) -> int:
        """
        Adds vector embeddings and metadata for a newly ingested document.
        """
        if len(chunks) == 0:
            return 0

        if embeddings.shape[0] != len(chunks):
            raise ValueError("Embeddings count does not match chunks count.")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} != index dimension {self.dimension}")

        # Add vectors to FAISS
        self.index.add(embeddings.astype(np.float32))

        # Register metadata
        total_tokens = 0
        for chunk in chunks:
            chunk_dict = {
                "chunk_id": chunk.chunk_id,
                "doc_id": doc_id,
                "source_filename": chunk.source_filename,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "token_count": chunk.token_count,
            }
            self.chunks_map[chunk.chunk_id] = chunk_dict
            self.indexed_chunk_ids.append(chunk.chunk_id)
            total_tokens += chunk.token_count

        self.documents_map[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "file_type": file_type,
            "upload_timestamp": datetime.now().isoformat(),
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
        }

        self.save()
        logger.info("Added %d chunks for doc '%s' (Total vectors: %d)", len(chunks), filename, self.index.ntotal)
        return len(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[dict, float]]:
        """
        Performs cosine similarity search against indexed chunks.
        Returns list of (chunk_metadata, cosine_similarity_score).
        """
        if self.index.ntotal == 0:
            return []

        top_k = min(top_k, self.index.ntotal)
        query_vector = query_embedding.astype(np.float32)

        # Search FAISS (returns distances and indices)
        scores, indices = self.index.search(query_vector, top_k)
        results: List[Tuple[dict, float]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.indexed_chunk_ids):
                continue
            chunk_id = self.indexed_chunk_ids[idx]
            chunk_data = self.chunks_map.get(chunk_id)
            if chunk_data:
                results.append((chunk_data, float(score)))

        return results

    def list_documents(self) -> List[dict]:
        """Returns metadata for all registered documents."""
        return list(self.documents_map.values())

    def get_document(self, doc_id: str) -> Optional[dict]:
        """Returns metadata for a specific document ID."""
        return self.documents_map.get(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """
        Deletes a document and its chunks, rebuilding the FAISS index cleanly.
        """
        if doc_id not in self.documents_map:
            return False

        # Identify chunks to keep
        remaining_chunk_ids = [
            cid for cid in self.indexed_chunk_ids if self.chunks_map[cid]["doc_id"] != doc_id
        ]

        if len(remaining_chunk_ids) == len(self.indexed_chunk_ids):
            # No vectors belonged to this doc
            del self.documents_map[doc_id]
            self.save()
            return True

        # Rebuild index from remaining chunks
        new_index = faiss.IndexFlatIP(self.dimension)
        new_indexed_ids = []
        new_chunks_map = {}

        if remaining_chunk_ids:
            # Re-extract vectors
            from core.embeddings import get_embedding_manager
            embedder = get_embedding_manager()

            texts = [self.chunks_map[cid]["text"] for cid in remaining_chunk_ids]
            new_embeddings = embedder.embed_texts(texts)
            new_index.add(new_embeddings)

            for cid in remaining_chunk_ids:
                new_indexed_ids.append(cid)
                new_chunks_map[cid] = self.chunks_map[cid]

        self.index = new_index
        self.indexed_chunk_ids = new_indexed_ids
        self.chunks_map = new_chunks_map
        del self.documents_map[doc_id]

        self.save()
        logger.info("Deleted doc '%s'. Remaining vectors: %d", doc_id, self.index.ntotal)
        return True

    def clear(self):
        """Clears all stored indices and metadata."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks_map.clear()
        self.indexed_chunk_ids.clear()
        self.documents_map.clear()
        self.save()

    def save(self):
        """Persists the FAISS index and JSON metadata to disk."""
        try:
            index_path = self.persist_dir / "index.faiss"
            meta_path = self.persist_dir / "metadata.json"

            faiss.write_index(self.index, str(index_path))

            meta = {
                "documents_map": self.documents_map,
                "indexed_chunk_ids": self.indexed_chunk_ids,
                "chunks_map": self.chunks_map,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        except Exception as e:
            logger.error("Failed to persist vector database: %s", e)

    def load(self):
        """Loads index and metadata from disk if available."""
        index_path = self.persist_dir / "index.faiss"
        meta_path = self.persist_dir / "metadata.json"

        if index_path.exists() and meta_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.documents_map = meta.get("documents_map", {})
                    self.indexed_chunk_ids = meta.get("indexed_chunk_ids", [])
                    self.chunks_map = meta.get("chunks_map", {})
                logger.info("Loaded persisted vector store (%d vectors)", self.index.ntotal)
            except Exception as e:
                logger.warning("Could not read persisted vector database: %s. Starting fresh.", e)


_VECTOR_STORE_INSTANCE = None


def get_vector_store() -> FAISSVectorStore:
    """Returns the shared vector store singleton."""
    global _VECTOR_STORE_INSTANCE
    if _VECTOR_STORE_INSTANCE is None:
        _VECTOR_STORE_INSTANCE = FAISSVectorStore()
    return _VECTOR_STORE_INSTANCE
