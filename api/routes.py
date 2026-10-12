"""
FastAPI Route Handlers for RAG System.
Implements endpoints for document ingestion, management, vector search, and Q&A generation.
"""

import logging
from pathlib import Path
import uuid
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from api.schemas import (
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentUploadResponse,
    ErrorResponse,
    HealthResponse,
    QuestionRequest,
    QuestionResponse,
)
from core.chunker import TextChunker
from core.config import (
    API_VERSION,
    EMBEDDING_MODEL_NAME,
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_EXTENSIONS,
)
from core.embeddings import get_embedding_manager
from core.generator import get_generator
from core.parser import DocumentParser
from core.retriever import get_retriever
from core.vector_store import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["RAG Document Q&A"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and Vector Store Status",
    description="Returns API health, active embedding model, and vector indexing statistics.",
)
async def health_check() -> HealthResponse:
    """Service health and vector database inspection."""
    store = get_vector_store()
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        vector_store="FAISS (IndexFlatIP)",
        embedding_model=EMBEDDING_MODEL_NAME,
        total_documents=len(store.list_documents()),
        total_indexed_vectors=store.index.ntotal,
    )


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Empty or Corrupted Document"},
        413: {"model": ErrorResponse, "description": "Document Exceeds Max File Size"},
        415: {"model": ErrorResponse, "description": "Unsupported Document Type"},
    },
    summary="Upload and Ingest Document (PDF/TXT/DOCX)",
    description="Extracts text, splits into 500-token chunks with 50-token overlap, computes embeddings, and indexes into FAISS.",
)
async def upload_document(
    file: UploadFile = File(..., description="Document file (.pdf, .docx, or .txt)"),
) -> DocumentUploadResponse:
    """Ingests a document into the RAG vector knowledge base."""
    filename = file.filename or "uploaded_document.txt"
    ext = Path(filename).suffix.lower()

    # 1. Validate Extension
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported format '{ext}'. Supported formats: {SUPPORTED_EXTENSIONS}",
        )

    # 2. Read bytes and check size
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty (0 bytes).",
        )

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {max_mb} MB.",
        )

    # 3. Extract text
    try:
        sections = DocumentParser.parse_document(file_bytes, filename)
    except Exception as e:
        logger.error("Document parsing error for %s: %s", filename, e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse document: {str(e)}",
        )

    if not sections or all(not s.text.strip() for s in sections):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No readable text found in document.",
        )

    # 4. Chunk text (500 tokens, 50 overlap)
    doc_id = str(uuid.uuid4())[:8]
    chunker = TextChunker()
    chunks = chunker.chunk_sections(sections, doc_id=doc_id)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to generate chunks from document.",
        )

    # 5. Compute embeddings and add to FAISS
    embedder = get_embedding_manager()
    texts_to_embed = [c.text for c in chunks]
    embeddings = embedder.embed_texts(texts_to_embed)

    vector_store = get_vector_store()
    vector_store.add_document_chunks(
        doc_id=doc_id,
        filename=filename,
        file_type=ext.lstrip("."),
        chunks=chunks,
        embeddings=embeddings,
    )

    total_tokens = sum(c.token_count for c in chunks)
    return DocumentUploadResponse(
        success=True,
        doc_id=doc_id,
        filename=filename,
        file_type=ext.lstrip("."),
        total_chunks=len(chunks),
        total_tokens=total_tokens,
        message=f"Document successfully indexed into {len(chunks)} chunks with FAISS.",
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List All Ingested Documents",
    description="Retrieves a list of all documents stored in the vector knowledge base.",
)
async def list_documents() -> DocumentListResponse:
    """Lists registered documents."""
    store = get_vector_store()
    docs = store.list_documents()
    metadata_list = [DocumentMetadata(**d) for d in docs]
    return DocumentListResponse(
        total_documents=len(docs),
        total_vectors=store.index.ntotal,
        documents=metadata_list,
    )


@router.delete(
    "/documents/{doc_id}",
    response_model=DeleteDocumentResponse,
    summary="Delete Document by ID",
    description="Removes a document and its chunks from the FAISS vector index.",
)
async def delete_document(doc_id: str) -> DeleteDocumentResponse:
    """Deletes a document from the vector store."""
    store = get_vector_store()
    success = store.delete_document(doc_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found.",
        )

    return DeleteDocumentResponse(
        success=True,
        doc_id=doc_id,
        message=f"Document '{doc_id}' successfully removed from vector index.",
    )


@router.post(
    "/query",
    response_model=QuestionResponse,
    summary="Ask Question with Source Citations",
    description="Retrieves Top-5 relevant chunks via cosine similarity and generates an answer with citations.",
)
async def query_rag(request: QuestionRequest) -> QuestionResponse:
    """Processes question, retrieves context, and synthesizes cited response."""
    retriever = get_retriever()
    generator = get_generator()

    # Top-5 retrieval
    chunks = retriever.retrieve(request.query, top_k=request.top_k)

    # Generation with citations
    result = generator.generate_answer(request.query, chunks)

    return QuestionResponse(**result)


# Alias endpoint for flexibility
@router.post(
    "/ask",
    response_model=QuestionResponse,
    include_in_schema=False,
)
async def ask_alias(request: QuestionRequest) -> QuestionResponse:
    """Alias for /api/query."""
    return await query_rag(request)
