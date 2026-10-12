"""
Pydantic Schemas for RAG API.
Provides validation and OpenAPI request/response schema specifications.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for an ingested document."""
    doc_id: str
    filename: str
    file_type: str
    upload_timestamp: str
    total_chunks: int
    total_tokens: int


class DocumentUploadResponse(BaseModel):
    """Response returned upon successful document upload and chunk indexing."""
    success: bool = True
    doc_id: str
    filename: str
    file_type: str
    total_chunks: int
    total_tokens: int
    message: str


class DocumentListResponse(BaseModel):
    """List of all indexed documents in the vector store."""
    total_documents: int
    total_vectors: int
    documents: List[DocumentMetadata]


class DeleteDocumentResponse(BaseModel):
    """Result of deleting a document and reindexing."""
    success: bool
    doc_id: str
    message: str


class QuestionRequest(BaseModel):
    """User query payload for document question answering."""
    query: str = Field(..., min_length=2, description="Natural language question to ask the documents")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of top chunks to retrieve (default: 5)")


class CitationItem(BaseModel):
    """Source reference supporting the generated answer."""
    source_filename: str
    page_number: int
    chunk_index: int
    cosine_similarity: float
    citation_label: str
    excerpt: str


class QuestionResponse(BaseModel):
    """RAG Answer payload with inline and structured citations."""
    query: str
    answer: str
    model: str
    citations: List[CitationItem]
    context_chunks_used: int
    top_similarity: float


class HealthResponse(BaseModel):
    """Health status and vector index metrics."""
    status: str
    version: str
    vector_store: str
    embedding_model: str
    total_documents: int
    total_indexed_vectors: int


class ErrorResponse(BaseModel):
    """Standardized error output."""
    success: bool = False
    error: str
    detail: str
    status_code: int
