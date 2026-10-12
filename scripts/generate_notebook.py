"""Generates the evaluation Jupyter notebook for Task 03."""
from pathlib import Path
import nbformat as nbf

out_path = Path("D:/Code_Files/Projects/Nexoramind/Task_03-RAG_Based_Document_Q-A_System/notebooks/rag_evaluation_notebook.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Nexoramind AI Internship — Task 03: RAG Document Q&A System
## Retrieval-Augmented Generation: Pipeline Walkthrough & Evaluation

**Author:** Krishna Gupta  
**Intern ID:** NM1B64D9E0  
**Tech Stack:** Python 3.12, FastAPI, FAISS, sentence-transformers, Streamlit

---

### Objectives
1. **Document Processing** — Parse PDF, DOCX, and TXT documents into structured sections.
2. **Text Chunking** — Sliding-window split into 500-token chunks with 50-token overlap.
3. **Dense Embeddings** — Generate 384-dim semantic vectors using `all-MiniLM-L6-v2`.
4. **FAISS Indexing** — Index chunk vectors using L2-normalized IndexFlatIP (cosine similarity).
5. **Retrieval** — Top-5 cosine similarity search per user query.
6. **Generation** — Contextual answer synthesis with verifiable source citations.
7. **Evaluation** — 10 benchmark test questions with >80% accuracy target.
"""))

cells.append(nbf.v4.new_markdown_cell("### 1. Environment Setup"))
cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import faiss
import sentence_transformers
import fastapi
import streamlit

print("FAISS version:", faiss.__version__)
print("Sentence-Transformers:", sentence_transformers.__version__)
print("FastAPI:", fastapi.__version__)
print("Streamlit:", streamlit.__version__)
"""))

cells.append(nbf.v4.new_markdown_cell("### 2. Document Parsing — PDF, DOCX, TXT"))
cells.append(nbf.v4.new_code_cell("""from core.parser import DocumentParser
from pathlib import Path

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_documents"

for f in SAMPLE_DIR.iterdir():
    sections = DocumentParser.parse_document(f.read_bytes(), f.name)
    total_words = sum(len(s.text.split()) for s in sections)
    print(f"[{f.suffix.upper():5s}] {f.name:45s}  sections={len(sections):2d}  words={total_words:,}")
"""))

cells.append(nbf.v4.new_markdown_cell("### 3. Text Chunking (500 tokens, 50 overlap)"))
cells.append(nbf.v4.new_code_cell("""from core.chunker import TextChunker

SAMPLE_FILE = SAMPLE_DIR / "AI_Engineering_Handbook.txt"
file_bytes = SAMPLE_FILE.read_bytes()
sections = DocumentParser.parse_document(file_bytes, SAMPLE_FILE.name)

chunker = TextChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk_sections(sections, doc_id="demo_doc")

print(f"Total chunks: {len(chunks)}")
for i, c in enumerate(chunks):
    print(f"  Chunk {i}: tokens={c.token_count}, page={c.page_number}, id={c.chunk_id}")
    print(f"    Excerpt: {c.text[:120]}...")
    print()
"""))

cells.append(nbf.v4.new_markdown_cell("### 4. Embedding Generation (all-MiniLM-L6-v2)"))
cells.append(nbf.v4.new_code_cell("""from core.embeddings import get_embedding_manager
import numpy as np

embedder = get_embedding_manager()
texts = [c.text for c in chunks]
embeddings = embedder.embed_texts(texts)

print(f"Embedding shape: {embeddings.shape}")
print(f"Dimension: {embedder.dimension}")
print(f"L2 norm (should be ~1.0): {np.linalg.norm(embeddings[0]):.6f}")
"""))

cells.append(nbf.v4.new_markdown_cell("### 5. FAISS Vector Store — Cosine Similarity Search"))
cells.append(nbf.v4.new_code_cell("""from core.vector_store import get_vector_store

store = get_vector_store()
docs = store.list_documents()
print(f"Indexed documents: {len(docs)}")
print(f"Total vectors: {store.index.ntotal}")
for d in docs:
    print(f"  [{d['doc_id']}] {d['filename']} — {d['total_chunks']} chunks, {d['total_tokens']} tokens")
"""))

cells.append(nbf.v4.new_markdown_cell("### 6. Retrieval Demo — Top-5 Cosine Similarity"))
cells.append(nbf.v4.new_code_cell("""from core.retriever import get_retriever

retriever = get_retriever()
test_query = "What are the three phases of a RAG pipeline?"
results = retriever.retrieve(test_query, top_k=5)

print(f"Query: {test_query}")
print(f"Top-{len(results)} results:")
for r in results:
    print(f"  Rank {r['rank']}: {r['source_filename']} | Similarity={r['cosine_similarity']:.4f}")
    print(f"           Excerpt: {r['text'][:120]}...")
    print()
"""))

cells.append(nbf.v4.new_markdown_cell("### 7. Answer Generation with Source Citations"))
cells.append(nbf.v4.new_code_cell("""from core.generator import get_generator

generator = get_generator()
answer_result = generator.generate_answer(test_query, results)

print(f"Query: {answer_result['query']}")
print(f"Model: {answer_result['model']}")
print()
print("Generated Answer:")
print(answer_result['answer'])
print()
print("Source Citations:")
for cit in answer_result['citations']:
    print(f"  [{cit['citation_label']}] Similarity: {cit['cosine_similarity']:.4f}")
    print(f"    Excerpt: {cit['excerpt'][:100]}...")
"""))

cells.append(nbf.v4.new_markdown_cell("### 8. Benchmark Evaluation — 10 Test Questions"))
cells.append(nbf.v4.new_code_cell("""from evaluation.evaluate_rag import evaluate_pipeline

summary = evaluate_pipeline()

print()
print(f"Retrieval Accuracy : {summary['retrieval_accuracy']}%")
print(f"Citation Accuracy  : {summary['citation_accuracy']}%")
print(f"Overall Accuracy   : {summary['overall_accuracy']}%")
print(f"Target Met (>80%)  : {summary['target_met']}")
"""))

cells.append(nbf.v4.new_markdown_cell("### 9. Per-Question Results Table"))
cells.append(nbf.v4.new_code_cell("""import pandas as pd

results_df = pd.DataFrame([{
    "Q#": r["id"],
    "Question": r["question"][:70] + "...",
    "Source": r["expected_source"],
    "Retrieved": r["retrieval_success"],
    "Citation OK": r["citation_success"],
    "Similarity": r["top_similarity"],
    "Entity Coverage": f\"{r['entity_coverage']*100:.0f}%\",
    "Correct": r["is_correct"],
} for r in summary["test_results"]])

results_df
"""))

nb.cells = cells
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Notebook generated at: {out_path}")
