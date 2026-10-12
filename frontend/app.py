"""
Nexoramind RAG Document Q&A System - Streamlit Frontend.
Interactive interface for document upload, Q&A, and retrieval visualization.
"""

import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import SUPPORTED_EXTENSIONS

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Nexoramind RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# ─── Session State ───────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# ─── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0c4a6e 100%);
    padding: 2rem 2rem 1.5rem;
    border-radius: 14px;
    color: white;
    margin-bottom: 1.5rem;
}
.hero-title { font-size: 2rem; font-weight: 800; margin-bottom: 0.3rem; }
.hero-sub { color: #94a3b8; font-size: 1rem; }
.hero-badge {
    display: inline-block;
    background: rgba(14,165,233,0.2);
    border: 1px solid rgba(56,189,248,0.4);
    color: #38bdf8;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.4rem;
}

.citation-card {
    background: #f8fafc;
    border-left: 4px solid #0284c7;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
}
.answer-box {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    font-size: 0.97rem;
    line-height: 1.7;
}
.status-online { color: #16a34a; font-weight: 600; }
.status-offline { color: #dc2626; font-weight: 600; }
.chunk-tag {
    background: #e0f2fe;
    color: #0369a1;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


def check_api() -> tuple[bool, Optional[dict]]:
    try:
        r = requests.get(f"{API_BASE}/api/health", timeout=1.5)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, None


def upload_document(file_bytes: bytes, filename: str) -> Optional[dict]:
    try:
        files = {"file": (filename, file_bytes)}
        r = requests.post(f"{API_BASE}/api/documents/upload", files=files, timeout=30)
        return r.json()
    except Exception as e:
        return {"success": False, "detail": str(e)}


def list_documents() -> List[dict]:
    try:
        r = requests.get(f"{API_BASE}/api/documents", timeout=5)
        return r.json().get("documents", [])
    except Exception:
        return []


def delete_document(doc_id: str) -> bool:
    try:
        r = requests.delete(f"{API_BASE}/api/documents/{doc_id}", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def query_rag(question: str, top_k: int = 5) -> Optional[dict]:
    try:
        payload = {"query": question, "top_k": top_k}
        r = requests.post(f"{API_BASE}/api/query", json=payload, timeout=30)
        return r.json()
    except Exception as e:
        return {"answer": f"API error: {e}", "citations": [], "model": "Error", "context_chunks_used": 0, "top_similarity": 0.0}


# ─── SIDEBAR ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📚 Nexoramind RAG Engine")
    st.caption("Task 03: RAG Document Q&A System")
    st.divider()

    api_ok, health_data = check_api()
    if api_ok:
        st.markdown(f'<span class="status-online">● FastAPI Online</span>', unsafe_allow_html=True)
        st.caption(f"Vectors: `{health_data.get('total_indexed_vectors', '?')}` | Docs: `{health_data.get('total_documents', '?')}`")
    else:
        st.markdown('<span class="status-offline">● API Offline</span>', unsafe_allow_html=True)
        st.warning("Start the FastAPI server: `python main.py --mode api`")

    st.divider()
    st.subheader("📤 Upload Documents")

    uploaded = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded:
        for uf in uploaded:
            if st.button(f"Index: {uf.name}", key=f"upload_{uf.name}", use_container_width=True):
                with st.spinner(f"Parsing and indexing {uf.name}..."):
                    result = upload_document(uf.getvalue(), uf.name)
                if result and result.get("success"):
                    st.success(f"Indexed! {result['total_chunks']} chunks, {result['total_tokens']} tokens.")
                else:
                    st.error(result.get("detail", "Upload failed"))

    st.divider()
    st.subheader("📂 Knowledge Base")
    docs = list_documents()
    if docs:
        for doc in docs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"**{doc['filename']}** ({doc['total_chunks']} chunks)")
            with col2:
                if st.button("X", key=f"del_{doc['doc_id']}"):
                    if delete_document(doc["doc_id"]):
                        st.rerun()
    else:
        st.caption("No documents indexed yet.")

    st.divider()
    st.metric("Session Queries", st.session_state.total_queries)


# ─── MAIN CONTENT ─────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <span class="hero-badge">Nexoramind AI Internship</span>
    <span class="hero-badge">Task 03</span>
    <span class="hero-badge">RAG Pipeline</span>
    <h1 class="hero-title">RAG Document Q&A System</h1>
    <p class="hero-sub">
        Upload documents (PDF, DOCX, TXT) and ask questions with source-cited answers.
        Powered by FAISS vector search, sentence-transformers, and contextual synthesis.
    </p>
</div>
""", unsafe_allow_html=True)

tab_qa, tab_history, tab_api = st.tabs(["💬 Ask Questions", "📋 Query History", "🔌 API Reference"])

# ─── TAB 1: Q&A ──────────────────────────────────────────────────
with tab_qa:
    col_q, col_ans = st.columns([1, 1.4], gap="large")

    with col_q:
        st.markdown("### Enter Your Question")
        query_text = st.text_area(
            "Type your question about the uploaded documents:",
            height=120,
            placeholder="e.g. What are the three phases of a RAG pipeline?",
        )
        top_k_slider = st.slider("Retrieval Top-K (chunks)", min_value=1, max_value=10, value=5)
        ask_btn = st.button("🔍 Ask Question", type="primary", use_container_width=True, disabled=not query_text.strip())

        st.markdown("#### Quick Test Questions")
        quick_questions = [
            "What are the three phases of a RAG pipeline?",
            "What is the minimum approval rating for Nexoramind tasks?",
            "How does FAISS IndexFlatIP compute cosine similarity?",
            "What is the role of Kubernetes Ingress Controllers?",
            "How does backpropagation update neural network weights?",
        ]
        for qq in quick_questions:
            if st.button(qq[:55] + "...", key=f"qb_{qq[:20]}", use_container_width=True):
                query_text = qq
                ask_btn = True

    with col_ans:
        st.markdown("### Generated Answer")

        if ask_btn and query_text.strip():
            with st.spinner("Retrieving context and synthesizing answer..."):
                t0 = time.perf_counter()
                result = query_rag(query_text.strip(), top_k=top_k_slider)
                latency = (time.perf_counter() - t0) * 1000

            if result:
                st.session_state.total_queries += 1
                st.session_state.chat_history.insert(0, {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "question": query_text.strip(),
                    "answer": result.get("answer", ""),
                    "model": result.get("model", ""),
                    "citations": len(result.get("citations", [])),
                    "top_sim": result.get("top_similarity", 0.0),
                    "latency_ms": round(latency, 1),
                })

                # Answer display
                st.markdown(
                    f'<div class="answer-box">{result["answer"]}</div>',
                    unsafe_allow_html=True,
                )

                # Metadata
                st.caption(
                    f"Model: `{result.get('model', 'N/A')}` | "
                    f"Chunks Used: `{result.get('context_chunks_used', 0)}` | "
                    f"Top Similarity: `{result.get('top_similarity', 0):.4f}` | "
                    f"Latency: `{latency:.1f}ms`"
                )

                # Citations
                citations = result.get("citations", [])
                if citations:
                    st.markdown("#### Source Citations")
                    for cit in citations:
                        st.markdown(
                            f"""<div class="citation-card">
                            <span class="chunk-tag">Rank #{citations.index(cit)+1}</span>
                            &nbsp;<strong>{cit['source_filename']}</strong> — Page {cit['page_number']} | Chunk {cit['chunk_index']}
                            &nbsp;| Similarity: <strong>{cit['cosine_similarity']:.4f}</strong><br/>
                            <span style="color:#475569;font-size:0.8rem;margin-top:0.3rem;display:block">{cit['excerpt']}</span>
                            </div>""",
                            unsafe_allow_html=True,
                        )
        else:
            st.markdown("""
            <div style="background:#f8fafc;border:2px dashed #cbd5e1;border-radius:12px;padding:3rem;text-align:center;color:#94a3b8;">
                <div style="font-size:2.5rem;margin-bottom:0.5rem">💬</div>
                <div style="font-weight:600;color:#334155;font-size:1.05rem">Ready for Your Question</div>
                <p>Upload documents via the sidebar, then type a question or select a quick test.</p>
            </div>
            """, unsafe_allow_html=True)

# ─── TAB 2: HISTORY ──────────────────────────────────────────────
with tab_history:
    st.markdown("### Query History")
    if st.session_state.chat_history:
        history_df = pd.DataFrame(st.session_state.chat_history)
        c1, c2 = st.columns([4, 1])
        with c1:
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        with c2:
            if st.button("Clear History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
            csv = history_df.to_csv(index=False).encode("utf-8")
            st.download_button("Export CSV", csv, "rag_history.csv", "text/csv", use_container_width=True)
    else:
        st.caption("No questions asked yet this session.")

# ─── TAB 3: API REFERENCE ─────────────────────────────────────────
with tab_api:
    st.markdown("### FastAPI REST Endpoints")
    st.markdown("All endpoints are accessible at `http://127.0.0.1:8000`.")

    st.markdown("#### `POST /api/documents/upload` — Upload & Index Document")
    st.code("""curl -X POST "http://127.0.0.1:8000/api/documents/upload" \\
  -F "file=@AI_Engineering_Handbook.txt"
""", language="bash")

    st.markdown("#### `POST /api/query` — Ask a Question")
    st.code("""curl -X POST "http://127.0.0.1:8000/api/query" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "What are the three phases of a RAG pipeline?", "top_k": 5}'
""", language="bash")

    st.markdown("#### Example JSON Response")
    st.code("""{
  "query": "What are the three phases of a RAG pipeline?",
  "answer": "Based on the documentation, the standard RAG pipeline operates in three sequential phases: Ingestion & Indexing, Dense Semantic Retrieval, and Grounded Synthesis. [AI_Engineering_Handbook.txt | Page 1 | Chunk 0]",
  "model": "Contextual RAG Synthesizer (Local)",
  "citations": [
    {
      "source_filename": "AI_Engineering_Handbook.txt",
      "page_number": 1,
      "chunk_index": 0,
      "cosine_similarity": 0.2714,
      "citation_label": "[AI_Engineering_Handbook.txt | Page 1 | Chunk 0]",
      "excerpt": "The standard RAG pipeline operates in three sequential phases..."
    }
  ],
  "context_chunks_used": 5,
  "top_similarity": 0.2714
}""", language="json")

    st.markdown("#### Other Endpoints")
    st.markdown("""
- `GET /api/health` — Service health, vector count, model details
- `GET /api/documents` — List all indexed documents
- `DELETE /api/documents/{doc_id}` — Remove document from vector store
- `GET /docs` — Interactive Swagger / OpenAPI UI
""")
