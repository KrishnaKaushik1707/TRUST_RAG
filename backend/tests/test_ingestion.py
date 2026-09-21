"""
Unit tests for TrustRAG document ingestion subsystem.
"""

from pathlib import Path
import pytest

from app.ingestion.chunker import RecursiveSentenceChunker
from app.ingestion.models import DocumentMetadata, IngestedDocument, PageContent
from app.ingestion.parsers import DocxParser, PDFParser, get_parser_for_file
from app.ingestion.pipeline import IngestionPipeline


@pytest.fixture
def sample_docs_dir() -> Path:
    docs_path = Path(__file__).resolve().parent.parent.parent / "scratch" / "sample_docs"
    assert docs_path.exists(), f"Sample docs directory not found at {docs_path}"
    return docs_path


def test_parser_factory():
    pdf_parser = get_parser_for_file(Path("test.pdf"))
    assert isinstance(pdf_parser, PDFParser)

    docx_parser = get_parser_for_file(Path("test.docx"))
    assert isinstance(docx_parser, DocxParser)

    with pytest.raises(ValueError):
        get_parser_for_file(Path("test.xyz"))


def test_pdf_parsing(sample_docs_dir: Path):
    pdf_file = sample_docs_dir / "sample_vendor_sla_contract.pdf"
    assert pdf_file.exists()

    parser = PDFParser()
    doc = parser.parse(pdf_file)

    assert doc.metadata.source_file == "sample_vendor_sla_contract.pdf"
    assert doc.metadata.file_type == "pdf"
    assert doc.metadata.page_count == 2
    assert len(doc.pages) == 2
    assert "MASTER SOFTWARE SERVICE LEVEL AGREEMENT" in doc.pages[0].text
    assert "Incident Response Times" in doc.pages[1].text


def test_docx_parsing(sample_docs_dir: Path):
    docx_file = sample_docs_dir / "sample_employment_agreement.docx"
    assert docx_file.exists()

    parser = DocxParser()
    doc = parser.parse(docx_file)

    assert doc.metadata.source_file == "sample_employment_agreement.docx"
    assert doc.metadata.file_type == "docx"
    assert "EXECUTIVE EMPLOYMENT AGREEMENT" in doc.full_text
    # Check that table text was extracted
    assert "Base Salary" in doc.full_text
    assert "$260,000" in doc.full_text


def test_chunker_bounds_and_overlap():
    chunker = RecursiveSentenceChunker(
        target_chunk_tokens=50,
        overlap_tokens=15,
        approx_chars_per_token=3.5,
    )

    # Synthetic text with multiple distinct sentences
    sample_text = (
        "Paragraph 1. Machine learning models require clean and structured training data. "
        "High quality annotations yield superior generalization across unobserved distributions. "
        "Paragraph 2. Retrieval-augmented generation overcomes parametric hallucination limits. "
        "By injecting verified context into prompt windows, inference accuracy increases dramatically. "
        "Paragraph 3. Continuous evaluation against ground-truth benchmarks ensures enterprise compliance. "
        "Auditing citations minimizes legal liabilities in critical domains."
    )

    page = PageContent(page_number=1, text=sample_text, char_count=len(sample_text))
    doc_meta = DocumentMetadata(
        source_file="test_doc.pdf",
        document_name="Test Doc",
        file_type="pdf",
        file_size_bytes=len(sample_text),
        page_count=1,
    )
    doc = IngestedDocument(
        id="test_doc_01",
        metadata=doc_meta,
        pages=[page],
        full_text=sample_text,
    )

    chunks = chunker.chunk_page(page, doc)
    assert len(chunks) >= 2, f"Expected multiple chunks for small token limit, got {len(chunks)}"

    for i, ch in enumerate(chunks):
        assert ch.document_id == "test_doc_01"
        assert ch.page_number == 1
        assert len(ch.text) <= chunker.max_chars + 100
        assert ch.char_start >= 0
        assert ch.char_end > ch.char_start

    # Verify overlap exists between chunk 0 and chunk 1
    c0_text = chunks[0].text
    c1_text = chunks[1].text
    overlap_found = any(word in c1_text for word in c0_text.split()[-8:])
    assert overlap_found, "Consecutive chunks should share overlapping words"


def test_pipeline_directory_ingestion(sample_docs_dir: Path):
    pipeline = IngestionPipeline()
    result = pipeline.ingest_directory(sample_docs_dir)

    assert result["stats"]["total_documents"] >= 3
    assert result["stats"]["total_chunks"] >= 3
    assert result["stats"]["total_pages"] >= 3

    # Ensure all chunks have valid citation metadata
    for chunk in result["chunks"]:
        assert chunk.page_number >= 1
        assert chunk.source_file != ""
        assert chunk.document_name != ""
        assert chunk.token_count > 0
        assert chunk.metadata["source_file"] == chunk.source_file
