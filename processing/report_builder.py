"""
Report Builder - Menghasilkan laporan PDF profesional dari hasil pemrosesan
citra yang dipilih pengguna di dashboard TaksuCitra.
"""
import base64
import io
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def _b64_to_reader(b64_string, max_width=8 * cm):
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    return io.BytesIO(img_bytes)


def build_pdf_report(items, student_info=None):
    """
    items: list of dicts { title, description, before_b64, after_b64 (optional), notes (optional) }
    student_info: dict { name, nim, course, date }
    """
    student_info = student_info or {}
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=2 * cm, rightMargin=2 * cm,
                             topMargin=1.8 * cm, bottomMargin=1.8 * cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TCTitle", parent=styles["Title"], fontSize=22,
                                  textColor=colors.HexColor("#1a1f3a"), alignment=TA_CENTER)
    subtitle_style = ParagraphStyle("TCSubtitle", parent=styles["Normal"], fontSize=11,
                                     textColor=colors.HexColor("#5b6178"), alignment=TA_CENTER)
    section_style = ParagraphStyle("TCSection", parent=styles["Heading2"], fontSize=14,
                                    textColor=colors.HexColor("#1a1f3a"), spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle("TCBody", parent=styles["Normal"], fontSize=10,
                                 textColor=colors.HexColor("#333"), alignment=TA_LEFT, leading=14)
    caption_style = ParagraphStyle("TCCaption", parent=styles["Normal"], fontSize=8.5,
                                    textColor=colors.HexColor("#666"), alignment=TA_CENTER)

    elements = []

    # ---- Cover / Header ----
    elements.append(Spacer(1, 0.5 * cm))
    elements.append(Paragraph("TaksuCitra", title_style))
    elements.append(Paragraph("Laporan Hasil Pemrosesan Citra Digital", subtitle_style))
    elements.append(Spacer(1, 0.8 * cm))

    info_rows = [
        ["Nama", student_info.get("name", "-")],
        ["Mata Kuliah", student_info.get("course", "Image Processing Programming")],
        ["Tanggal", student_info.get("date", datetime.now().strftime("%d %B %Y, %H:%M"))],
        ["Jumlah Fitur Didemonstrasikan", str(len(items))],
    ]
    info_table = Table(info_rows, colWidths=[5 * cm, 10 * cm])
    info_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#5b6178")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.6 * cm))
    elements.append(Paragraph(
        "Laporan ini dibuat otomatis oleh sistem TaksuCitra sebagai bukti demonstrasi "
        "fitur pengolahan citra digital, mencakup operasi piksel, konversi gambar, "
        "operasi histogram, serta fitur unggulan berbasis deep learning dan eksplorasi mandiri.",
        body_style))
    elements.append(PageBreak())

    # ---- Isi per fitur ----
    for idx, item in enumerate(items, start=1):
        elements.append(Paragraph(f"{idx}. {item.get('title', 'Fitur')}", section_style))
        if item.get("description"):
            elements.append(Paragraph(item["description"], body_style))
        elements.append(Spacer(1, 0.3 * cm))

        row_cells = []
        captions = []
        if item.get("before_b64"):
            row_cells.append(RLImage(_b64_to_reader(item["before_b64"]), width=7 * cm, height=5.2 * cm, kind="proportional"))
            captions.append("Sebelum")
        if item.get("after_b64"):
            row_cells.append(RLImage(_b64_to_reader(item["after_b64"]), width=7 * cm, height=5.2 * cm, kind="proportional"))
            captions.append("Sesudah")

        if row_cells:
            img_table = Table([row_cells], colWidths=[8 * cm] * len(row_cells))
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            elements.append(img_table)
            cap_table = Table([captions], colWidths=[8 * cm] * len(captions))
            cap_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
            elements.append(cap_table)

        if item.get("notes"):
            elements.append(Spacer(1, 0.2 * cm))
            elements.append(Paragraph(f"<i>Catatan:</i> {item['notes']}", caption_style))

        elements.append(Spacer(1, 0.6 * cm))

    # ---- Footer note ----
    elements.append(Spacer(1, 0.5 * cm))
    elements.append(Paragraph(
        "Dibuat dengan TaksuCitra — Platform Demo Image Processing Programming",
        caption_style))

    doc.build(elements)
    buf.seek(0)
    return buf.read()
