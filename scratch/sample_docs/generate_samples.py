"""
Synthetic sample document generator for TrustRAG.

Generates 3 realistic PDF documents and 1 DOCX document in scratch/sample_docs/:
1. sample_resume_alex_chen.pdf (ML Engineer resume with work history & skills)
2. sample_vendor_sla_contract.pdf (Enterprise SaaS Service Level Agreement)
3. sample_nda_agreement.pdf (Mutual Non-Disclosure Agreement)
4. sample_employment_agreement.docx (Employment offer & contract with compensation table)
"""

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_resume(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    styles = getSampleStyleSheet()

    header_style = ParagraphStyle(
        "HeaderStyle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=4,
    )
    subhead_style = ParagraphStyle(
        "SubheadStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "SectionStyle",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2D3748"),
    )
    bullet_style = ParagraphStyle(
        "BulletStyle",
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3,
    )

    story = []

    # Title & Contact
    story.append(Paragraph("Alex Chen, M.S.", header_style))
    story.append(
        Paragraph(
            "San Francisco, CA • alex.chen@example.com • linkedin.com/in/alexchen-ml • github.com/alexchen-ai",
            subhead_style,
        )
    )

    # Summary
    story.append(Paragraph("Professional Summary", section_style))
    story.append(
        Paragraph(
            "Senior Machine Learning Engineer with 6+ years of experience specializing in distributed LLM serving, "
            "RAG architectures, and NLP pipeline optimization. Proven track record of scaling vector search systems "
            "to 50,000 QPS with sub-50ms latency at high-growth FinTech and cloud data platforms.",
            body_style,
        )
    )

    # Technical Skills
    story.append(Paragraph("Technical Competencies", section_style))
    skills_text = (
        "<b>Languages:</b> Python, Go, C++, SQL, TypeScript<br/>"
        "<b>ML & NLP Frameworks:</b> PyTorch, HuggingFace Transformers, vLLM, Ray Serve, ONNX Runtime<br/>"
        "<b>Vector Databases & Search:</b> ChromaDB, Qdrant, Milvus, Elasticsearch (BM25), FAISS<br/>"
        "<b>Infrastructure:</b> Docker, Kubernetes, AWS (SageMaker, EKS), Terraform, Triton Inference Server"
    )
    story.append(Paragraph(skills_text, body_style))

    # Experience
    story.append(Paragraph("Professional Experience", section_style))

    # Job 1
    story.append(Paragraph("<b>Senior Machine Learning Engineer</b> — Stripe, San Francisco, CA", body_style))
    story.append(Paragraph("<i>March 2022 – Present</i>", subhead_style))
    story.append(
        Paragraph(
            "• Architected an enterprise-grade RAG dispute intelligence system indexing 12M+ financial dispute cases, "
            "reducing human review time by 38%.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• Optimized sentence-transformer embedding inference using TensorRT and vLLM on NVIDIA A100 GPUs, "
            "achieving a 4.2x throughput increase and saving $240,000 in annual AWS compute costs.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• Deployed hybrid sparse-dense retrieval (BM25 + BGE-M3) with Reciprocal Rank Fusion, improving Top-5 "
            "retrieval recall from 78.4% to 92.1% on ambiguous fraud terminology.",
            bullet_style,
        )
    )

    # Job 2
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Machine Learning Engineer</b> — Databricks, Mountain View, CA", body_style))
    story.append(Paragraph("<i>July 2019 – February 2022</i>", subhead_style))
    story.append(
        Paragraph(
            "• Built automated feature-store pipelines on Apache Spark supporting real-time model scoring for 40+ tenant applications.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• Co-authored the Lakehouse Vector Search connector, integrating HNSW approximate nearest-neighbor indexing "
            "into Delta Lake tables.",
            bullet_style,
        )
    )

    # Education
    story.append(Paragraph("Education", section_style))
    story.append(Paragraph("<b>Stanford University</b> — M.S. in Computer Science (Artificial Intelligence Track)", body_style))
    story.append(Paragraph("<i>September 2017 – June 2019</i> • GPA: 3.92/4.0", subhead_style))
    story.append(Paragraph("<b>University of Washington</b> — B.S. in Computer Engineering (Magna Cum Laude)", body_style))
    story.append(Paragraph("<i>September 2013 – June 2017</i>", subhead_style))

    doc.build(story)


def generate_sla_contract(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        alignment=1,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=15,
    )
    section_style = ParagraphStyle(
        "ContractSection",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ContractBody",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6,
    )

    story = []

    # Page 1
    story.append(Paragraph("MASTER SOFTWARE SERVICE LEVEL AGREEMENT (SLA)", title_style))
    story.append(
        Paragraph(
            "This Service Level Agreement ('SLA') is entered into between <b>Apex Cloud Technologies Inc.</b> ('Vendor') "
            "and <b>Vertex Financial Services Corp.</b> ('Client'), effective as of January 15, 2024.",
            body_style,
        )
    )

    story.append(Paragraph("Section 1: Service Availability Commitment", section_style))
    story.append(
        Paragraph(
            "1.1 <b>Monthly Uptime Percentage:</b> Vendor guarantees that the Core API and Vector Query Services will maintain "
            "a Monthly Uptime Percentage of at least <b>99.9%</b> during each calendar month of the Term.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "1.2 <b>Scheduled Maintenance Exclusions:</b> Service availability calculations strictly exclude scheduled "
            "maintenance windows. Scheduled maintenance shall occur only between 01:00 UTC and 04:00 UTC on Sundays, and "
            "Vendor must provide at least 72 hours advance written notification.",
            body_style,
        )
    )

    story.append(Paragraph("Section 2: Service Credits & Compensation", section_style))
    story.append(
        Paragraph(
            "2.1 In the event Vendor fails to satisfy the 99.9% uptime commitment, Client is eligible to receive Service Credits "
            "according to the following schedule:",
            body_style,
        )
    )

    # Table of credits
    table_data = [
        ["Monthly Uptime Percentage", "Service Credit Percentage of Monthly Fee"],
        ["99.0% - 99.89%", "10% Credit"],
        ["95.0% - 98.99%", "25% Credit"],
        ["Below 95.0%", "50% Credit"],
    ]
    t = Table(table_data, colWidths=[240, 240])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 10))

    # Page Break to test multi-page extraction
    story.append(PageBreak())

    # Page 2
    story.append(Paragraph("Section 3: Incident Response Times and Severity Levels", section_style))
    story.append(
        Paragraph(
            "3.1 <b>Severity 1 (Critical):</b> Complete service outage affecting all end-users. Initial response time: "
            "<b>within 15 minutes</b>. Workaround or resolution required within 4 continuous hours (24x7x365).",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "3.2 <b>Severity 2 (Major):</b> Degraded performance where critical features are impaired but workaround exists. "
            "Initial response time: <b>within 1 hour</b>. Resolution target within 12 business hours.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "3.3 <b>Severity 3 (Minor):</b> Non-critical bug or administrative request. Initial response time: "
            "<b>within 24 business hours</b>.",
            body_style,
        )
    )

    story.append(Paragraph("Section 4: Data Protection, Backups, and Liability Cap", section_style))
    story.append(
        Paragraph(
            "4.1 <b>Disaster Recovery:</b> Vendor shall maintain continuous multi-region transaction replication with a "
            "Recovery Point Objective (RPO) of ≤ 5 minutes and a Recovery Time Objective (RTO) of ≤ 30 minutes.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "4.2 <b>Limitation of Liability:</b> Except for indemnification obligations and breaches of confidentiality under "
            "Section 6, each party's maximum aggregate monetary liability arising under this Agreement shall not exceed "
            "the total fees paid by Client in the preceding twelve (12) months.",
            body_style,
        )
    )

    doc.build(story)


def generate_nda_agreement(output_path: Path):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "NdaTitle",
        parent=styles["Heading1"],
        fontSize=17,
        leading=21,
        alignment=1,
        textColor=colors.HexColor("#18181B"),
        spaceAfter=14,
    )
    section_style = ParagraphStyle(
        "NdaSection",
        parent=styles["Heading2"],
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#27272A"),
        spaceBefore=10,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "NdaBody",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#3F3F46"),
        spaceAfter=6,
    )

    story = []
    story.append(Paragraph("MUTUAL NON-DISCLOSURE AGREEMENT", title_style))
    story.append(
        Paragraph(
            "This Mutual Non-Disclosure Agreement ('Agreement') is entered into on October 10, 2024, by and between "
            "<b>Synapse AI Labs Inc.</b> ('Party A') and <b>Horizon Analytics LLC</b> ('Party B').",
            body_style,
        )
    )

    story.append(Paragraph("1. Definition of Confidential Information", section_style))
    story.append(
        Paragraph(
            "'Confidential Information' encompasses all technical data, trade secrets, proprietary algorithms, "
            "vector embedding weights, source code, benchmark evaluations, and business strategies disclosed by one "
            "party ('Disclosing Party') to the other ('Receiving Party'), whether verbally or in tangible form.",
            body_style,
        )
    )

    story.append(Paragraph("2. Exclusions from Confidentiality", section_style))
    story.append(
        Paragraph(
            "Confidential Information does not include information that: (a) is or becomes publicly known through no breach "
            "by Receiving Party; (b) was already rightfully known prior to disclosure; (c) is independently developed without "
            "reference to Disclosing Party's Confidential Information; or (d) is approved for release by written consent.",
            body_style,
        )
    )

    story.append(Paragraph("3. Non-Disclosure and Use Obligations", section_style))
    story.append(
        Paragraph(
            "The Receiving Party shall protect Disclosing Party's Confidential Information with at least the same degree of care "
            "it accords its own confidential records, but in no event less than a reasonable standard of care. Receiving Party "
            "shall not disclose any Confidential Information to third parties without prior written approval.",
            body_style,
        )
    )

    story.append(Paragraph("4. Term and Governing Law", section_style))
    story.append(
        Paragraph(
            "This Agreement and confidentiality obligations shall survive for a period of <b>three (3) years</b> from the date "
            "of disclosure. This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware.",
            body_style,
        )
    )

    doc.build(story)


def generate_docx_sample(output_path: Path):
    try:
        import docx
    except ImportError:
        return

    doc = docx.Document()
    doc.add_heading("EXECUTIVE EMPLOYMENT AGREEMENT", level=0)

    doc.add_paragraph(
        "This Executive Employment Agreement is made between Quantum Dynamics Ltd. ('Company') "
        "and Elena Rostova ('Executive'), dated November 1, 2024."
    )

    doc.add_heading("1. Position and Responsibilities", level=1)
    doc.add_paragraph(
        "Executive shall serve as Vice President of AI Engineering. Executive will report directly "
        "to the Chief Technology Officer and oversee all algorithmic research and infrastructure operations."
    )

    doc.add_heading("2. Compensation and Benefits", level=1)
    doc.add_paragraph(
        "The Company will provide Executive with the compensation package outlined in the schedule below:"
    )

    # Add a structured table
    table = doc.add_table(rows=1, cols=2)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Compensation Component"
    hdr_cells[1].text = "Terms & Valuation"

    data = [
        ("Base Salary", "$260,000 annualized, payable bi-weekly"),
        ("Target Performance Bonus", "30% of Base Salary tied to delivery of TrustRAG v1.0"),
        ("Equity Grant", "120,000 Restricted Stock Units (RSUs) vesting over 4 years"),
        ("Annual Continuing Education", "$5,000 stipend for AI research conferences"),
    ]

    for item, val in data:
        row_cells = table.add_row().cells
        row_cells[0].text = item
        row_cells[1].text = val

    doc.add_heading("3. Restrictive Covenants", level=1)
    doc.add_paragraph(
        "During employment and for twelve (12) months following termination, Executive shall not solicit "
        "Company clients or induce employees to terminate employment with Company."
    )

    doc.save(str(output_path))


def main():
    target_dir = Path(__file__).resolve().parent
    target_dir.mkdir(parents=True, exist_ok=True)

    resume_pdf = target_dir / "sample_resume_alex_chen.pdf"
    sla_pdf = target_dir / "sample_vendor_sla_contract.pdf"
    nda_pdf = target_dir / "sample_nda_agreement.pdf"
    emp_docx = target_dir / "sample_employment_agreement.docx"

    print(f"Generating sample documents in {target_dir}...")
    generate_resume(resume_pdf)
    print(f"✓ Created {resume_pdf.name}")

    generate_sla_contract(sla_pdf)
    print(f"✓ Created {sla_pdf.name}")

    generate_nda_agreement(nda_pdf)
    print(f"✓ Created {nda_pdf.name}")

    generate_docx_sample(emp_docx)
    print(f"✓ Created {emp_docx.name}")

    print("All sample documents successfully generated!")


if __name__ == "__main__":
    main()
