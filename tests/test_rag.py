"""
Unit and Integration Tests for the RAG Document Q&A System.
"""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.app import app
from core.chunker import TextChunker
from core.parser import DocumentParser, ParsedSection
from core.embeddings import get_embedding_manager

client = TestClient(app)


# ====================== HELPER FACTORIES ======================

def make_txt_bytes(content: str = None) -> bytes:
    return (content or "Retrieval Augmented Generation combines vector search with language models for document Q&A.").encode("utf-8")


def make_docx_bytes() -> bytes:
    import docx, io
    doc = docx.Document()
    doc.add_paragraph("This is a test paragraph for DOCX parsing validation.")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ====================== PARSER TESTS ======================

class TestDocumentParser:
    def test_parse_txt_short(self):
        text = "Hello world. This is a simple test document."
        sections = DocumentParser.parse_txt(text.encode("utf-8"), "test.txt")
        assert len(sections) == 1
        assert "Hello world" in sections[0].text
        assert sections[0].source_filename == "test.txt"
        assert sections[0].file_type == "txt"

    def test_parse_txt_long(self):
        # Creates a text > 400 words to trigger multi-page splitting
        words = ["word"] * 600
        long_text = " ".join(words)
        sections = DocumentParser.parse_txt(long_text.encode("utf-8"), "long_test.txt")
        assert len(sections) >= 2

    def test_parse_docx(self):
        docx_bytes = make_docx_bytes()
        sections = DocumentParser.parse_docx(docx_bytes, "test.docx")
        assert len(sections) >= 1
        assert sections[0].file_type == "docx"

    def test_parse_dispatch_txt(self):
        data = make_txt_bytes()
        sections = DocumentParser.parse_document(data, "sample.txt")
        assert len(sections) >= 1

    def test_parse_dispatch_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported"):
            DocumentParser.parse_document(b"content", "file.xls")


# ====================== CHUNKER TESTS ======================

class TestTextChunker:
    def test_chunk_short_section(self):
        section = ParsedSection(
            text="This is a short text for chunking test.",
            page_number=1,
            source_filename="test.txt",
            file_type="txt",
        )
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_sections([section], doc_id="test01")
        assert len(chunks) == 1
        assert chunks[0].doc_id == "test01"
        assert "short text" in chunks[0].text

    def test_chunk_long_section(self):
        long_text = " ".join(["neuralnetwork"] * 700)
        section = ParsedSection(
            text=long_text,
            page_number=1,
            source_filename="test.txt",
            file_type="txt",
        )
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_sections([section], doc_id="test02")
        assert len(chunks) >= 2
        # Verify overlap
        for c in chunks:
            assert c.token_count <= 500

    def test_chunk_assigns_unique_ids(self):
        sections = [
            ParsedSection(text="First page text " * 50, page_number=1, source_filename="t.txt", file_type="txt"),
            ParsedSection(text="Second page text " * 50, page_number=2, source_filename="t.txt", file_type="txt"),
        ]
        chunker = TextChunker()
        chunks = chunker.chunk_sections(sections, doc_id="docA")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


# ====================== EMBEDDINGS TESTS ======================

class TestEmbeddingManager:
    def test_embed_single_text(self):
        import numpy as np
        embedder = get_embedding_manager()
        vectors = embedder.embed_texts(["RAG system uses FAISS and sentence transformers"])
        assert vectors.shape[0] == 1
        assert vectors.shape[1] == embedder.dimension
        # Check L2 norm ≈ 1 (normalized)
        norm = np.linalg.norm(vectors[0])
        assert abs(norm - 1.0) < 0.02

    def test_embed_batch(self):
        import numpy as np
        embedder = get_embedding_manager()
        texts = ["Document retrieval", "Language model generation", "Vector database index"]
        vectors = embedder.embed_texts(texts)
        assert vectors.shape == (3, embedder.dimension)

    def test_embed_empty_returns_empty(self):
        import numpy as np
        embedder = get_embedding_manager()
        result = embedder.embed_texts([])
        assert result.shape[0] == 0


# ====================== API ENDPOINT TESTS ======================

class TestHealthEndpoint:
    def test_health_returns_healthy(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "FAISS" in data["vector_store"]
        assert "all-MiniLM-L6-v2" in data["embedding_model"]


class TestRootEndpoint:
    def test_root_returns_metadata(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "upload_endpoint" in data
        assert "query_endpoint" in data


class TestDocumentEndpoints:
    def test_upload_txt_document(self):
        txt_bytes = make_txt_bytes("Machine learning models use embeddings for semantic search and retrieval tasks.")
        files = {"file": ("test_upload.txt", txt_bytes, "text/plain")}
        response = client.post("/api/documents/upload", files=files)
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["filename"] == "test_upload.txt"
        assert data["total_chunks"] >= 1
        return data["doc_id"]

    def test_upload_unsupported_format(self):
        files = {"file": ("spreadsheet.xls", b"fake xls content", "application/vnd.ms-excel")}
        response = client.post("/api/documents/upload", files=files)
        assert response.status_code == 415

    def test_upload_empty_file(self):
        files = {"file": ("empty.txt", b"", "text/plain")}
        response = client.post("/api/documents/upload", files=files)
        assert response.status_code == 400

    def test_list_documents(self):
        response = client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert isinstance(data["documents"], list)

    def test_delete_nonexistent_doc(self):
        response = client.delete("/api/documents/nonexistent_doc_xyz")
        assert response.status_code == 404


class TestQueryEndpoints:
    def test_query_returns_answer(self):
        # First upload a document
        txt_bytes = make_txt_bytes(
            "FAISS is a library for efficient similarity search developed by Facebook AI Research. "
            "It supports various index types including Flat, IVF, and HNSW."
        )
        files = {"file": ("faiss_doc.txt", txt_bytes, "text/plain")}
        client.post("/api/documents/upload", files=files)

        # Now query it
        payload = {"query": "What is FAISS?", "top_k": 5}
        response = client.post("/api/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 10
        assert "citations" in data
        assert data["context_chunks_used"] >= 0

    def test_query_empty_string_fails(self):
        payload = {"query": " ", "top_k": 5}
        response = client.post("/api/query", json=payload)
        assert response.status_code in (200, 422)

    def test_ask_alias_works(self):
        payload = {"query": "What is retrieval augmented generation?", "top_k": 3}
        response = client.post("/api/ask", json=payload)
        assert response.status_code == 200
