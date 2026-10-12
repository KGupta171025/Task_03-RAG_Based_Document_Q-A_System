"""
Ingests sample documents into the FAISS vector database.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.chunker import TextChunker
from core.embeddings import get_embedding_manager
from core.parser import DocumentParser
from core.vector_store import get_vector_store

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_documents"


def ingest_all_samples():
    """Ingests all files in data/sample_documents."""
    vector_store = get_vector_store()
    chunker = TextChunker()
    embedder = get_embedding_manager()

    files = list(SAMPLE_DIR.iterdir())
    print(f"Found {len(files)} sample documents to ingest.")

    total_chunks_added = 0
    for file_path in files:
        if file_path.suffix.lower() not in [".pdf", ".docx", ".txt"]:
            continue

        file_bytes = file_path.read_bytes()
        sections = DocumentParser.parse_document(file_bytes, file_path.name)
        doc_id = file_path.stem.lower().replace(" ", "_")

        # Skip if already in store
        if doc_id in vector_store.documents_map:
            print(f"Document '{doc_id}' already indexed. Skipping.")
            continue

        chunks = chunker.chunk_sections(sections, doc_id=doc_id)
        if not chunks:
            continue

        embeddings = embedder.embed_texts([c.text for c in chunks])
        vector_store.add_document_chunks(
            doc_id=doc_id,
            filename=file_path.name,
            file_type=file_path.suffix.lstrip("."),
            chunks=chunks,
            embeddings=embeddings,
        )
        total_chunks_added += len(chunks)
        print(f"[OK] Ingested '{file_path.name}': {len(chunks)} chunks.")

    print(f"[SUCCESS] Vector store now contains {vector_store.index.ntotal} vectors across {len(vector_store.list_documents())} documents.")


if __name__ == "__main__":
    ingest_all_samples()
