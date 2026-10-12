"""
Sample Document Generator for RAG Pipeline Testing.
Creates realistic sample documents in TXT, DOCX, and PDF formats covering AI engineering topics.
"""

from pathlib import Path
import docx
from pypdf import PdfWriter
from io import BytesIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_documents"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def create_txt_documents():
    """Generates comprehensive TXT documents on AI and Cloud systems."""
    doc1_path = SAMPLE_DIR / "AI_Engineering_Handbook.txt"
    doc1_content = """NEXORAMIND AI ENGINEERING HANDBOOK (2026 EDITION)

Chapter 1: Foundations of Retrieval-Augmented Generation (RAG)
Retrieval-Augmented Generation (RAG) is a machine learning design pattern that optimizes the output of Large Language Models (LLMs) by referencing an authoritative knowledge base outside of its training data before generating a response. 
Standard LLMs suffer from knowledge cutoff dates, lack of domain-specific context, and catastrophic hallucinations. RAG addresses these shortcomings by combining dense vector retrieval with contextual language generation.

The standard RAG pipeline operates in three sequential phases:
1. Ingestion & Indexing: Documents are parsed from raw formats (PDF, DOCX, TXT), decomposed into uniform text chunks, converted into continuous numerical vector embeddings, and stored in high-performance similarity search indexes such as FAISS, ChromaDB, or Milvus.
2. Dense Semantic Retrieval: Upon receiving a user prompt, the query is embedded into the same semantic vector space using identical embedding models. A similarity metric—predominantly Cosine Similarity or Inner Product—is computed against the indexed corpus to retrieve the Top-K most relevant chunks.
3. Grounded Synthesis: The retrieved chunks are formatted into a structured prompt context alongside the original user inquiry. The generative model synthesizes a factual, coherent answer with verified source citations.

Chapter 2: Text Chunking Strategies and Overlap
Chunking is the process of splitting lengthy documents into smaller, coherent text segments for embedding and indexing. Selecting an appropriate chunk size is critical for retrieval quality:
- If chunks are too small (e.g., < 100 tokens), they lose semantic context, leading to incomplete or fragmented answers.
- If chunks are too large (e.g., > 1000 tokens), embeddings become diluted, aggregating multiple disparate concepts and degrading retrieval specificity.
For standard technical documentation, the industry benchmark standard specifies chunk sizes of approximately 500 tokens with an overlap of 50 tokens. Overlap prevents information loss across chunk boundaries, ensuring that critical clauses and definitions are preserved across adjacent chunks.

Chapter 3: Dense Vector Embeddings and FAISS
Dense embeddings represent text in continuous high-dimensional vector spaces where semantic proximity correlates with geometric proximity. Models such as sentence-transformers 'all-MiniLM-L6-v2' map textual sequences into 384-dimensional dense vectors.
FAISS (Facebook AI Similarity Search) is an open-source library optimized for efficient similarity search and clustering of dense vectors. When vector embeddings are L2-normalized to unit length, the inner product computed via FAISS IndexFlatIP is mathematically equivalent to Cosine Similarity. This provides millisecond search latency over millions of document vectors.
"""
    with open(doc1_path, "w", encoding="utf-8") as f:
        f.write(doc1_content)
    print(f"[OK] Generated: {doc1_path.name}")

    doc2_path = SAMPLE_DIR / "Cloud_Computing_Architecture.txt"
    doc2_content = """CLOUD COMPUTING INFRASTRUCTURE AND MICROSERVICES OVERVIEW

Section 1: Microservice Architecture and Containerization
Modern cloud-native applications are decomposed into decoupled, independently deployable microservices communicating via lightweight protocols such as RESTful HTTP/JSON or gRPC.
Docker containerization packages applications alongside their runtime dependencies, system binaries, and configuration files into standardized portable container images. This eliminates environment drift between local development workstations and production cloud clusters.

Section 2: Kubernetes Container Orchestration
Kubernetes (K8s) serves as the industry standard platform for automating deployment, scaling, and operational management of containerized workloads:
- Pods: The smallest deployable computing units in Kubernetes, encapsulating one or more tightly coupled containers sharing network and storage namespaces.
- Deployments: Declarative controllers that manage replica sets, rolling zero-downtime updates, and automated self-healing pod restarts.
- Services: Persistent abstractions providing static IP addresses, DNS service discovery, and load balancing across dynamic ephemeral pods.
- Ingress Controllers: Application-layer HTTP/HTTPS reverse proxies routing external internet traffic to internal cluster services based on path and host rules.

Section 3: High Availability and Disaster Recovery
Enterprise cloud architectures enforce Multi-Availability Zone (Multi-AZ) and multi-region deployment topologies to ensure 99.99% system availability. Critical data layers utilize read-replicas, continuous asynchronous replication, and automated failover protocols. RTO (Recovery Time Objective) and RPO (Recovery Point Objective) metrics dictate backup frequency and disaster recovery drills.
"""
    with open(doc2_path, "w", encoding="utf-8") as f:
        f.write(doc2_content)
    print(f"[OK] Generated: {doc2_path.name}")


def create_docx_document():
    """Generates official DOCX policy document."""
    doc_path = SAMPLE_DIR / "Nexoramind_AI_Company_Policy.docx"
    doc = docx.Document()
    
    doc.add_heading("NEXORAMIND TECHNOLOGIES - AI ENGINEER INTERNSHIP POLICY", level=0)
    
    doc.add_heading("1. Program Overview and Evaluation Rubric", level=1)
    doc.add_paragraph(
        "The Nexoramind AI Engineer 6 Months Program consists of 10 hands-on production tasks. "
        "Candidates must achieve an evaluation approval rating exceeding 80% on each task to advance. "
        "Each task requires a publicly accessible GitHub repository, unit test suites, clear documentation, "
        "and reproducible notebooks."
    )
    
    doc.add_heading("2. Working Hours and Remote Collaboration", level=1)
    doc.add_paragraph(
        "Interns operate under flexible hybrid remote schedules. Core collaboration hours are 10:00 AM to 5:00 PM IST. "
        "Weekly technical syncs and milestone evaluations are conducted via video conference. "
        "All code submissions must follow PEP 8 standards, strict type annotations, and modular package structure."
    )
    
    doc.add_heading("3. Data Security and Confidentiality", level=1)
    doc.add_paragraph(
        "Interns are strictly prohibited from storing plain-text proprietary API credentials, tokens, or private customer records "
        "in public version control. Environment variables (.env files) must be excluded via .gitignore. "
        "Models fine-tuned during the program must cite training data provenance and comply with responsible AI ethics."
    )

    doc.save(str(doc_path))
    print(f"[OK] Generated: {doc_path.name}")


def create_pdf_document():
    """Generates clean multi-page PDF document."""
    pdf_path = SAMPLE_DIR / "Deep_Learning_Fundamentals.pdf"
    
    # Generate clean text pages using a lightweight PDF generator via pypdf / raw PDF stream
    # Minimal pure PDF syntax
    content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 5 0 R >> >> /MediaBox [0 0 612 792] /Contents 6 0 R >>
endobj
4 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 5 0 R >> >> /MediaBox [0 0 612 792] /Contents 7 0 R >>
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
6 0 obj
<< /Length 420 >>
stream
BT
/F1 18 Tf
50 720 Td
(DEEP LEARNING FUNDAMENTALS - PAGE 1) Tj
/F1 12 Tf
0 -40 Td
(Chapter 1: Artificial Neural Networks and Backpropagation) Tj
0 -25 Td
(Artificial neural networks consist of interconnected layers of artificial neurons.) Tj
0 -20 Td
(During the forward pass, input features are multiplied by learnable weights, summed) Tj
0 -20 Td
(with a bias term, and transformed through non-linear activation functions such as ReLU.) Tj
0 -25 Td
(Backpropagation applies the calculus chain rule to compute gradients of the loss function) Tj
0 -20 Td
(with respect to each parameter. Optimizers like AdamW update weights iteratively.) Tj
ET
endstream
endobj
7 0 obj
<< /Length 420 >>
stream
BT
/F1 18 Tf
50 720 Td
(DEEP LEARNING FUNDAMENTALS - PAGE 2) Tj
/F1 12 Tf
0 -40 Td
(Chapter 2: Transformer Architectures and Self-Attention) Tj
0 -25 Td
(The Transformer model introduced by Vaswani et al. replaced recurrent networks) Tj
0 -20 Td
(with the Self-Attention mechanism. Scaled Dot-Product Attention allows models to) Tj
0 -20 Td
(weigh the importance of tokens across arbitrary sequence distances in parallel.) Tj
0 -25 Td
(Multi-Head Attention projects queries, keys, and values into multiple subspaces,) Tj
0 -20 Td
(enabling modern models to capture nuanced grammatical and semantic relationships.) Tj
ET
endstream
endobj
xref
0 8
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000123 00000 n 
0000000250 00000 n 
0000000377 00000 n 
0000000455 00000 n 
0000000928 00000 n 
trailer
<< /Size 8 /Root 1 0 R >>
startxref
1401
%%EOF
"""
    with open(pdf_path, "wb") as f:
        f.write(content)
    print(f"[OK] Generated: {pdf_path.name}")


if __name__ == "__main__":
    create_txt_documents()
    create_docx_document()
    create_pdf_document()
    print("[SUCCESS] All sample documents generated successfully!")
