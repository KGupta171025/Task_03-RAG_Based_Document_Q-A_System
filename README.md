# 📚 Nexoramind AI Engineer Internship — Task 03: RAG-Based Document Q&A System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-1.8+-blue)](https://github.com/facebookresearch/faiss)
[![sentence-transformers](https://img.shields.io/badge/sentence--transformers-all--MiniLM--L6--v2-orange)](https://sbert.net/)
[![Retrieval Accuracy](https://img.shields.io/badge/Retrieval%20Accuracy-100%25-brightgreen)]()
[![Overall Accuracy](https://img.shields.io/badge/Overall%20Accuracy-100%25-brightgreen)]()
[![Tests](https://img.shields.io/badge/Tests-21%20Passed-green)]()

> **Internship Track:** AI Engineer (6 Months Program)  
> **Candidate:** Krishna Gupta | **ID:** NM1B64D9E0  
> **Task 03:** RAG-Based Document Q&A System  
> **Repository:** [https://github.com/KGupta171025/Task_03-RAG_Based_Document_Q-A_System](https://github.com/KGupta171025/Task_03-RAG_Based_Document_Q-A_System)

---

## Objective

Build a production-grade **Retrieval-Augmented Generation (RAG)** system for document question-answering with:
- Multi-format document ingestion (PDF, DOCX, TXT)
- 500-token chunking with 50-token overlap
- FAISS cosine-similarity vector search
- LLM-based answer synthesis with verifiable source citations
- FastAPI REST backend + Streamlit interactive frontend
- **Evaluation: 10 benchmark Q&A pairs achieving 100% accuracy (>80% target)**

---

## Benchmark Evaluation Results

| Metric | Score |
|---|:---:|
| **Retrieval Accuracy (Top-5)** | **100.0%** |
| **Source Citation Accuracy** | **100.0%** |
| **Overall Answer Accuracy** | **100.0%** |
| **Target (>80%)** | **ACHIEVED ✓** |
| **Test Questions** | 10 / 10 PASS |

---

## System Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────┐
│           FastAPI REST API (ASGI / Uvicorn)       │
│  POST /api/documents/upload                       │
│  POST /api/query                                  │
│  GET  /api/documents | DELETE /api/documents/{id} │
└───────────────────┬──────────────────────────────┘
                    │
        ┌───────────▼────────────┐
        │     Core RAG Engine    │
        │                        │
        │  1. DocumentParser     │  ← PDF (pypdf) / DOCX / TXT
        │  2. TextChunker        │  ← 500 tokens, 50 overlap
        │  3. EmbeddingManager   │  ← all-MiniLM-L6-v2 (384-dim)
        │  4. FAISSVectorStore   │  ← IndexFlatIP + L2-norm = Cosine
        │  5. DocumentRetriever  │  ← Top-5 similarity search
        │  6. RAGAnswerGenerator │  ← Synthesis + Citations
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │   Persistent Storage   │
        │  data/vector_db/       │  ← index.faiss + metadata.json
        └────────────────────────┘
```

---

## Repository Structure

```
Task_03-RAG_Based_Document_Q-A_System/
│
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
├── main.py                          # Unified CLI (api, frontend, ingest, evaluate)
│
├── api/                             # FastAPI Backend
│   ├── app.py                       # FastAPI instance, CORS, exception handlers
│   ├── routes.py                    # /upload, /query, /documents, /health
│   └── schemas.py                   # Pydantic request/response schemas
│
├── core/                            # RAG Pipeline Engine
│   ├── config.py                    # Paths, constants, chunking/retrieval params
│   ├── parser.py                    # PDF/DOCX/TXT document extraction
│   ├── chunker.py                   # 500-token sliding window chunking
│   ├── embeddings.py                # sentence-transformers embedding manager
│   ├── vector_store.py              # FAISS IndexFlatIP vector store
│   ├── retriever.py                 # Top-K cosine similarity retrieval
│   └── generator.py                 # Contextual synthesis + LLM integration
│
├── frontend/
│   └── app.py                       # Streamlit Q&A dashboard
│
├── data/
│   ├── sample_documents/            # 4 sample docs: TXT, DOCX, PDF
│   ├── vector_db/                   # Persisted FAISS index + JSON metadata
│   └── evaluation_results.json      # Benchmark metrics output
│
├── evaluation/
│   ├── test_dataset.json            # 10 benchmark Q&A pairs with entity keys
│   └── evaluate_rag.py              # End-to-end evaluation pipeline
│
├── notebooks/
│   └── rag_evaluation_notebook.ipynb
│
├── scripts/
│   ├── create_sample_documents.py   # Generate sample docs (TXT/DOCX/PDF)
│   ├── ingest_samples.py            # Bulk ingest into FAISS
│   └── generate_notebook.py         # Automated notebook builder
│
└── tests/
    └── test_rag.py                  # 21 unit + integration tests
```

---

## Quickstart

### 1. Setup
```bash
git clone https://github.com/KGupta171025/Task_03-RAG_Based_Document_Q-A_System.git
cd Task_03-RAG_Based_Document_Q-A_System

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Generate & Ingest Sample Documents
```bash
python scripts/create_sample_documents.py
python main.py --mode ingest
```

### 3. Launch FastAPI Backend
```bash
python main.py --mode api
# API: http://127.0.0.1:8000
# Swagger: http://127.0.0.1:8000/docs
```

### 4. Launch Streamlit Frontend (new terminal)
```bash
python main.py --mode frontend
# UI: http://localhost:8501
```

### 5. Run Evaluation
```bash
python main.py --mode evaluate
```

### 6. Run Tests
```bash
pytest -v tests/
# 21 tests, 100% pass rate
```

---

## API Reference

### `POST /api/documents/upload`
Upload and index a document (PDF, DOCX, TXT) into the FAISS vector store.

```bash
curl -X POST "http://127.0.0.1:8000/api/documents/upload" \
  -F "file=@data/sample_documents/AI_Engineering_Handbook.txt"
```

**Response:**
```json
{
  "success": true,
  "doc_id": "a1b2c3d4",
  "filename": "AI_Engineering_Handbook.txt",
  "file_type": "txt",
  "total_chunks": 2,
  "total_tokens": 850,
  "message": "Document successfully indexed into 2 chunks with FAISS."
}
```

### `POST /api/query`
Ask a question about uploaded documents.

```bash
curl -X POST "http://127.0.0.1:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the three phases of a RAG pipeline?", "top_k": 5}'
```

**Response:**
```json
{
  "query": "What are the three phases of a RAG pipeline?",
  "answer": "Based on the documentation, the standard RAG pipeline operates in three sequential phases: Ingestion & Indexing, Dense Semantic Retrieval, and Grounded Synthesis. [AI_Engineering_Handbook.txt | Page 1 | Chunk 0]",
  "model": "Contextual RAG Synthesizer (Local)",
  "citations": [
    {
      "source_filename": "AI_Engineering_Handbook.txt",
      "page_number": 1,
      "chunk_index": 0,
      "cosine_similarity": 0.2062,
      "citation_label": "[AI_Engineering_Handbook.txt | Page 1 | Chunk 0]",
      "excerpt": "The standard RAG pipeline operates in three sequential phases..."
    }
  ],
  "context_chunks_used": 5,
  "top_similarity": 0.2062
}
```

### Other Endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Service health, vector count, model |
| `/api/documents` | GET | List all indexed documents |
| `/api/documents/{doc_id}` | DELETE | Remove document from vector store |
| `/docs` | GET | Swagger / OpenAPI interactive UI |

---

## Technical Details

### Chunking Parameters
- **Chunk size:** 500 tokens (with HuggingFace tokenizer, fallback to word count)
- **Overlap:** 50 tokens to preserve context at chunk boundaries
- **Step size:** 450 tokens (500 − 50) per sliding window advance

### Embedding Model
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Output dimension:** 384
- **Normalization:** L2-normalized (enabling cosine similarity via inner product)

### FAISS Index
- **Type:** `IndexFlatIP` (exact inner product = cosine similarity on L2-normalized vectors)
- **Persistence:** Auto-saved to `data/vector_db/index.faiss` + `metadata.json` after each write

### Answer Generation
- **Primary:** OpenAI `gpt-4o-mini` (if `OPENAI_API_KEY` is set)
- **Secondary:** Groq `llama-3.1-8b-instant` (if `GROQ_API_KEY` is set)
- **Fallback:** High-precision local contextual synthesizer (no API key required)

---

## Submission

- **GitHub Repository:** https://github.com/KGupta171025/Task_03-RAG_Based_Document_Q-A_System
- **Evaluation Notebook:** `notebooks/rag_evaluation_notebook.ipynb`
- **Sample Documents:** `data/sample_documents/` (PDF, DOCX, TXT)
- **Benchmark Results:** `data/evaluation_results.json`
