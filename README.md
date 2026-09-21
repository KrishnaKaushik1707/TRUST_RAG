# TrustRAG: Trust-First Document Intelligence System

TrustRAG is an enterprise-grade, pedagogically designed Retrieval-Augmented Generation (RAG) system tailored for high-stakes document intelligence (e.g., candidate resumes, vendor contracts, technical specifications, and regulatory disclosures). 

Unlike standard RAG pipelines that blindly feed vector-retrieved context to an LLM, TrustRAG integrates a dedicated **Trust Layer** that actively audits citations, detects contradictions, flags hallucinated claims, and logs knowledge gaps.

---

## 🏗️ System Architecture Roadmap

```
TrustRAG Architecture
├── Phase 1: Ingestion Pipeline (Active)
│   ├── Parsers: PyMuPDF (PDF), python-docx (DOCX)
│   ├── Layout & Page-Boundary Metadata Extraction
│   └── Sentence-Aware Recursive Chunking (~400 tokens / 60 token overlap)
│
├── Phase 2: Hybrid Retrieval & Vector Store
│   ├── Dense Retrieval: ChromaDB + Sentence-Transformers (`all-MiniLM-L6-v2`)
│   ├── Sparse Retrieval: BM25 (`rank_bm25`) for exact keyword / legal clause matching
│   └── Reciprocal Rank Fusion (RRF) Hybrid Ranker
│
├── Phase 3: Generation & Precise Citation Engine
│   ├── Structured Prompting & Grounded Context Assembly
│   └── Character/Page-level Evidence Citation
│
└── Phase 4: Trust Layer & Verification
    ├── Claim Extraction & Entailment Auditing (NLI / LLM Critic)
    ├── Contradiction Detection across multiple documents/sections
    └── Knowledge-Gap Logging (queries with low retrieval confidence)
```

---

## 📁 Repository Layout

```
trustrag/
├── backend/
│   ├── app/
│   │   ├── ingestion/       # Document loaders, parsers, and chunkers (Phase 1)
│   │   ├── retrieval/       # Hybrid search & vector store (Phase 2)
│   │   ├── generation/      # Context synthesis & citations (Phase 3)
│   │   ├── trust_layer/     # Claim verification & contradiction auditor (Phase 4)
│   │   └── main.py          # FastAPI application entrypoint
│   ├── tests/               # Pytest automated test suites
│   ├── requirements.txt     # Backend Python dependencies
│   └── Dockerfile           # Backend containerization
├── frontend/                # Web UI dashboard (Phase 2+)
├── scratch/
│   ├── notebooks/           # Experimental notebooks & EDA
│   ├── sample_docs/         # Sample resumes and contracts
│   └── eval_runs/           # Retrieval & verification evaluation metrics
├── docker-compose.yml       # Multi-service local orchestration
└── README.md
```

---

## 🚀 Getting Started (Phase 1)

### 1. Prerequisites
- macOS or Linux with Python 3.11 installed.

### 2. Virtual Environment Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 3. Generate Sample Documents & Run Ingestion Demo
```bash
# 1. Generate realistic test PDFs (ML Engineer Resume, Vendor SLA, NDA)
python scratch/sample_docs/generate_samples.py

# 2. Run the ingestion pipeline demo
python scratch/run_ingestion_demo.py

# 3. Run automated tests
pytest backend/tests/test_ingestion.py -v
```
