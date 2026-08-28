"""Automated Clinical Diagnostic PDF Report Generator using ReportLab.
Generates comprehensive clinical summaries with patient metadata, side-by-side Grad-CAM
explainability visualizations, Monte Carlo uncertainty breakdown, and regulatory disclaimers.
"""

import io
import os
from datetime import datetime
from typing import Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_clinical_pdf_report(
    prediction_data: Dict[str, Any],
    original_img_bytes: bytes,
    gradcam_img_bytes: bytes,
    case_id: str = None,
) -> bytes:
    """Generate a high-resolution clinical diagnostic PDF report.
    Args:
        prediction_data: Dictionary returned by uncertainty / prediction pipeline
        original_img_bytes: Bytes of original MRI scan
        gradcam_img_bytes: Bytes of Grad-CAM overlay image
        case_id: Optional case identifier
    Returns:
        bytes: Binary PDF content
    """
    if case_id is None:
        case_id = f"MRI-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubTitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=15,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=12,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    disclaimer_style = ParagraphStyle(
        "ReportDisclaimer",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#78350F"),
    )

    story = []

    # 1. Header & Institutional Banner
    story.append(Paragraph("🧠 NEUROLOGICAL AI DIAGNOSTIC IMAGING REPORT", title_style))
    story.append(Paragraph("Automated Deep Learning & Grad-CAM Explainability Analysis", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceAfter=15))

    # 2. Case Metadata Table
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    meta_data = [
        [
            Paragraph("<b>Case ID:</b>", body_style), Paragraph(case_id, body_style),
            Paragraph("<b>Modality:</b>", body_style), Paragraph("Brain MRI (Axial / Coronal)", body_style),
        ],
        [
            Paragraph("<b>Date / Time:</b>", body_style), Paragraph(now_str, body_style),
            Paragraph("<b>Model Engine:</b>", body_style), Paragraph("ResNet50 Fine-Tuned + Bayesian MC", body_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[80, 180, 90, 180])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))

    # 3. Executive Findings
    story.append(Paragraph("1. Primary Diagnostic Findings", section_heading))
    predicted_class = prediction_data.get("predicted_class", "UNKNOWN").upper()
    confidence = prediction_data.get("confidence", 0.0)
    uncertainty = prediction_data.get("epistemic_uncertainty", 0.0)
    entropy = prediction_data.get("predictive_entropy", 0.0)
    clinical_status = prediction_data.get("clinical_status", "LOW_RISK_CONFIDENT")

    # Status color code
    status_text_color = "#166534" if "LOW" in clinical_status else ("#B45309" if "MODERATE" in clinical_status else "#991B1B")

    findings_data = [
        [
            Paragraph("<b>Predicted Classification:</b>", body_style),
            Paragraph(f"<b><font size=11 color='#0F172A'>{predicted_class}</font></b>", body_style),
        ],
        [
            Paragraph("<b>Model Confidence:</b>", body_style),
            Paragraph(f"<b>{confidence:.2%}</b>", body_style),
        ],
        [
            Paragraph("<b>Epistemic Uncertainty:</b>", body_style),
            Paragraph(f"{uncertainty:.4f} (Entropy Index: {entropy:.2f})", body_style),
        ],
        [
            Paragraph("<b>Clinical Review Status:</b>", body_style),
            Paragraph(f"<b><font color='{status_text_color}'>{clinical_status}</font></b>", body_style),
        ],
    ]
    findings_table = Table(findings_data, colWidths=[160, 370])
    findings_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
    ]))
    story.append(findings_table)
    story.append(Spacer(1, 15))

    # 4. Visual Explainability (Side-by-side Images)
    story.append(Paragraph("2. Visual Explainability (Grad-CAM Attention Mapping)", section_heading))
    try:
        img_orig_stream = io.BytesIO(original_img_bytes)
        img_grad_stream = io.BytesIO(gradcam_img_bytes)

        img_orig = RLImage(img_orig_stream, width=2.6 * inch, height=2.6 * inch)
        img_grad = RLImage(img_grad_stream, width=2.6 * inch, height=2.6 * inch)

        image_table_data = [
            [img_orig, img_grad],
            [Paragraph("<b>Original MRI Scan</b>", body_style), Paragraph("<b>Grad-CAM Attention Heatmap</b>", body_style)],
        ]
        image_table = Table(image_table_data, colWidths=[260, 260])
        image_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(image_table)
    except Exception as e:
        story.append(Paragraph(f"<i>Visual embedding note: {e}</i>", body_style))

    story.append(Spacer(1, 15))

    # 5. Class Probability Matrix Table
    story.append(Paragraph("3. Multi-Class Probability & Variance Distribution", section_heading))
    mean_probs = prediction_data.get("mean_probabilities", {})
    std_probs = prediction_data.get("std_probabilities", {})

    prob_rows = [
        [
            Paragraph("<b>Class Name</b>", body_style),
            Paragraph("<b>Probability (%)</b>", body_style),
            Paragraph("<b>Epistemic Std (±%)</b>", body_style),
            Paragraph("<b>Category</b>", body_style),
        ]
    ]

    for cname, prob in mean_probs.items():
        std_val = std_probs.get(cname, 0.0)
        cat_label = "Healthy Control" if cname == "no_tumor" else "Tumor Neoplasm"
        prob_rows.append([
            Paragraph(f"<b>{cname.replace('_', ' ').title()}</b>", body_style),
            Paragraph(f"{prob * 100:.2f}%", body_style),
            Paragraph(f"±{std_val * 100:.2f}%", body_style),
            Paragraph(cat_label, body_style),
        ])

    prob_table = Table(prob_rows, colWidths=[140, 120, 130, 140])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284C7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 20))

    # 6. Disclaimer and Signature Footer
    disclaimer_box_data = [
        [
            Paragraph(
                "<b>⚠️ REGULATORY & CLINICAL DISCLAIMER:</b> This diagnostic report was generated automatically by an experimental AI research pipeline. It has NOT been cleared by regulatory bodies (e.g. FDA / CE) and must NOT be used as the sole basis for clinical diagnosis or treatment. Final evaluation requires certified radiologist sign-off.",
                disclaimer_style,
            )
        ]
    ]
    disclaimer_table = Table(disclaimer_box_data, colWidths=[530])
    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF3C7")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#F59E0B")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(disclaimer_table)

    # Build PDF
    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data
