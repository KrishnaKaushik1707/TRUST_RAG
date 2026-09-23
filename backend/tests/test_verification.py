"""
Unit and integration tests for TrustRAG Phase 4: Sentence-level NLI Claim Verification.
"""

from pathlib import Path
import shutil
import tempfile
import pytest
from fastapi.testclient import TestClient

from app.ingestion.models import DocumentChunk
from app.main import app
from app.retrieval.models import RetrievalResult
from app.verification.nli_verifier import NLIVerifier


@pytest.fixture(scope="module")
def verifier():
    return NLIVerifier()


@pytest.fixture(scope="module")
def client():
    temp_dir = tempfile.mkdtemp(prefix="trustrag_test_chroma_v_")
    fixtures_dir = str(Path(__file__).resolve().parent / "fixtures")
    import os
    os.environ["CHROMA_DIR"] = temp_dir
    os.environ["DOCS_DIR"] = fixtures_dir

    with TestClient(app) as test_client:
        yield test_client

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_sentence_splitting_edge_cases(verifier: NLIVerifier):
    """
    Verifies that decimals ('3.5'), honorifics ('Dr. Smith'), and abbreviations
    are not falsely split into fragments.
    """
    text = (
        "Dr. Smith built 3.5 models at Example Corp. Inc. on Jan. 15th. "
        "The model achieved 95.8% accuracy. "
        "Did it scale smoothly? Yes, it did."
    )
    sentences = verifier.split_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "Dr. Smith built 3.5 models at Example Corp. Inc. on Jan. 15th."
    assert sentences[1] == "The model achieved 95.8% accuracy."
    assert sentences[2] == "Did it scale smoothly?"
    assert sentences[3] == "Yes, it did."


def test_nli_entailment_and_contradiction(verifier: NLIVerifier):
    """
    Verifies directional logical entailment, neutral claims, and contradiction.
    """
    premise_text = (
        "Alex Chen is a Senior Machine Learning Engineer with 6 years of experience "
        "building production NLP pipelines in Python and PyTorch. He graduated from Stanford University."
    )
    dummy_chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        source_file="sample_resume.pdf",
        document_name="Sample Resume",
        chunk_index=0,
        page_number=1,
        text=premise_text,
        token_count=35,
        char_start=0,
        char_end=len(premise_text),
    )
    retrieval_res = RetrievalResult(
        chunk=dummy_chunk,
        document_name="Sample Resume",
        source_file="sample_resume.pdf",
        page_number=1,
        text=premise_text,
        dense_rank=1,
        sparse_rank=1,
        final_rank=1,
        rerank_score=4.5,
        rrf_score=0.032,
    )

    answer_text = (
        "Alex Chen has extensive experience with PyTorch and Python. "
        "Alex Chen has never worked in machine learning or software engineering. "
        "Alex Chen won an Olympic gold medal in figure skating."
    )

    results = verifier.verify_answer(answer_text, [retrieval_res])
    assert len(results) == 3

    # Claim 1: Directly supported
    assert results[0].label == "SUPPORTED"
    assert results[0].entailment_score >= 0.70
    assert results[0].supporting_source_file == "sample_resume.pdf"

    # Claim 2: Direct contradiction -> Unverified
    assert results[1].label == "UNVERIFIED"
    assert results[1].contradiction_score >= 0.70

    # Claim 3: Completely neutral / missing facts -> Unverified
    assert results[2].label == "UNVERIFIED"
    assert results[2].entailment_score < 0.35


def test_api_verified_sentences_endpoint(client: TestClient):
    """
    Verifies that the FastAPI /query endpoint returns the new 'verified_sentences' field.
    """
    payload = {
        "query": "What are the technical skills of Alex Chen?",
        "top_k": 2,
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "verified_sentences" in data
    assert isinstance(data["verified_sentences"], list)
    if data["verified_sentences"]:
        claim = data["verified_sentences"][0]
        assert "sentence" in claim
        assert "label" in claim
        assert claim["label"] in {"SUPPORTED", "PARTIALLY_SUPPORTED", "UNVERIFIED"}
        assert "entailment_score" in claim
        assert 0.0 <= claim["entailment_score"] <= 1.0
