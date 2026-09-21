"""
TrustRAG Backend Application Entrypoint
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="TrustRAG API",
    description="A RAG-based document intelligence system with claim verification, contradiction detection, and citation auditing.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "TrustRAG Backend",
        "phase": "Phase 1 - Ingestion Pipeline",
    }
