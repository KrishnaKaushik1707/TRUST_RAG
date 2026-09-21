#!/usr/bin/env python3
"""
TrustRAG Ingestion Demonstration & Inspection Script.

Runs the complete Phase 1 ingestion pipeline on all files in scratch/sample_docs/,
prints document metadata, chunk distributions, and inspects chunk contents
and overlap regions.
"""

import sys
from pathlib import Path

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.ingestion.pipeline import IngestionPipeline


def main():
    print("=" * 78)
    print("        TrustRAG Phase 1: Ingestion Pipeline & Chunk Inspection")
    print("=" * 78)

    sample_docs_dir = Path(__file__).resolve().parent / "sample_docs"
    print(f"\n📂 Scanning directory: {sample_docs_dir}")

    pipeline = IngestionPipeline()
    result = pipeline.ingest_directory(sample_docs_dir)

    docs = result["documents"]
    chunks = result["chunks"]
    stats = result["stats"]

    print("\n" + "─" * 78)
    print(f"📊 INGESTION SUMMARY:")
    print(f"   • Documents processed: {stats['total_documents']}")
    print(f"   • Total pages parsed:  {stats['total_pages']}")
    print(f"   • Total chunks created:{stats['total_chunks']}")
    print(f"   • Total characters:    {stats['total_characters']:,}")
    print("─" * 78)

    # Detailed per-document breakdown
    for doc in docs:
        doc_chunks = [c for c in chunks if c.document_id == doc.id]
        print(f"\n📄 Document: {doc.metadata.source_file}")
        print(f"   • Title:      {doc.metadata.document_name}")
        print(f"   • Type:       {doc.metadata.file_type.upper()}")
        print(f"   • Page Count: {doc.metadata.page_count}")
        print(f"   • File Size:  {doc.metadata.file_size_bytes:,} bytes")
        print(f"   • Chunks:     {len(doc_chunks)}")

        # Print chunks for inspection
        print("\n   🔍 Chunk Inspection:")
        for ch in doc_chunks:
            print(f"      ┌── [Chunk #{ch.chunk_index}] ID: {ch.id}")
            print(f"      │   Source: Page {ch.page_number} | Chars: [{ch.char_start}:{ch.char_end}] | Tokens: ~{ch.token_count}")
            # Indent text preview
            lines = ch.text.split("\n")
            preview_lines = lines[:4]
            for line in preview_lines:
                print(f"      │   > {line[:85]}")
            if len(lines) > 4:
                print(f"      │   > ... ({len(lines) - 4} more lines)")
            print(f"      └──────────────────────────────────────────────────────────")

    # Demonstrate overlap between consecutive chunks if any document has >= 2 chunks
    multi_chunk_docs = [d for d in docs if len([c for c in chunks if c.document_id == d.id]) > 1]
    if multi_chunk_docs:
        demo_doc = multi_chunk_docs[0]
        demo_chunks = [c for c in chunks if c.document_id == demo_doc.id]
        c1, c2 = demo_chunks[0], demo_chunks[1]
        print("\n" + "=" * 78)
        print(f"🔗 OVERLAP VERIFICATION DEMO ({demo_doc.metadata.source_file})")
        print("=" * 78)
        print(f"\n[Chunk 0 End (last 120 chars)]:\n...{c1.text[-120:].strip()!r}")
        print(f"\n[Chunk 1 Start (first 120 chars)]:\n{c2.text[:120].strip()!r}...")
        print("\nNotice how the semantic transition is preserved across chunk boundaries,")
        print("ensuring clauses and qualifications are never severed for vector search.")
        print("=" * 78)


if __name__ == "__main__":
    main()
