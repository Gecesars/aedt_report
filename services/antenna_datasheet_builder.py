from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import current_app
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, NameObject

MM_TO_PT = 72.0 / 25.4
HEADER_OFFSET_PT = 48.0
from reportlab.graphics import renderPDF
from reportlab.graphics.barcode import qr as qr_code
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Paragraph, Table, TableStyle
from reportlab.lib.utils import ImageReader
from PIL import Image

from ..models import Antenna, PatternType


BODY_STYLE = ParagraphStyle(name="DatasheetBody", fontName="Helvetica", fontSize=10, leading=14)
SMALL_STYLE = ParagraphStyle(name="DatasheetSmall", fontName="Helvetica", fontSize=9, leading=12)

MARKETING_HEIGHT = 2.6 * inch
MARKETING_WIDTH = 2.9 * inch
DESCRIPTION_GAP = 16
SECTION_SPACING = 18
MECHANICAL_IMAGE_HEIGHT = 2.6 * inch


def _truncate_text(text: str, max_words: int) -> str:
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "…"


def _normalize_bullets(content: str, max_items: int = 6, item_max_words: int = 12) -> list[str]:
    lines: list[str] = []
    for raw in (content or "").splitlines():
        stripped = raw.strip(" •-\t\u2013")
        if not stripped:
            continue
        words = stripped.split()
        if len(words) > item_max_words:
            stripped = " ".join(words[:item_max_words]) + "…"
        lines.append(stripped)
        if len(lines) >= max_items:
            break
    return lines


def _split_item(text: str) -> tuple[str, str | None]:
    cleaned = text.strip()
    if cleaned.startswith("•"):
        cleaned = cleaned[1:].strip()
    if cleaned.startswith("-") or cleaned.startswith("\u2013"):
        cleaned = cleaned[1:].strip()
    if ":" in cleaned:
        label, value = cleaned.split(":", 1)
        return label.strip(), value.strip()
    return cleaned, None


def _append_or_update(
    rows: list[tuple[str, str | None]],
    entry: tuple[str, str | None],
) -> None:
    label, value = entry
    label_clean = (label or "").strip()
    value_clean = value.strip() if isinstance(value, str) else value
    if not label_clean and value_clean:
        label_clean = str(value_clean).strip()
        value_clean = None
    if not label_clean and (value_clean is None or value_clean == ""):
        return
    key = label_clean.lower()
    if key:
        for idx, (existing_label, existing_value) in enumerate(rows):
            existing_key = (existing_label or "").strip().lower()
            if existing_key == key:
                if isinstance(value_clean, str) and value_clean and (
                    existing_value is None or existing_value == "" or existing_value == "—"
                ):
                    rows[idx] = (existing_label or label_clean, value_clean)
                return
    rows.append((label_clean, value_clean))


def _resolve_background(key: str, default_relative: str) -> Path | None:
    candidate = current_app.config.get(key)
    if candidate:
        path = Path(candidate)
    else:
        path = Path(current_app.root_path).parent / default_relative
    if path.exists():
        return path
    return None


def _draw_background(pdf: canvas.Canvas, image_path: Path, page_width: float, page_height: float) -> None:
    try:
        pdf.drawImage(str(image_path), 0, 0, width=page_width, height=page_height)
    except Exception:
        pass


def _draw_characteristics_table(
    pdf: canvas.Canvas,
    title: str,
    rows: list[tuple[str, str | None]],
    left: float,
    top: float,
    width: float,
    label_ratio: float = 0.32,
) -> float:
    if not rows:
        return top
    title_style = ParagraphStyle(name="CharTitle", parent=BODY_STYLE, fontName="Helvetica-Bold", fontSize=11)
    body_style = ParagraphStyle(name="CharBody", parent=BODY_STYLE, fontSize=9, leading=12)

    label_width = max(width * label_ratio, width * 0.28)
    value_width = width - label_width
    table_data = [
        [Paragraph(title, title_style), Paragraph("", title_style)],
        [Paragraph("<b>Item</b>", body_style), Paragraph("<b>Value</b>", body_style)],
    ]

    for label, value in rows:
        if value is None or value == "":
            table_data.append([
                Paragraph("", body_style),
                Paragraph(label.strip() or "—", body_style),
            ])
        else:
            table_data.append([
                Paragraph(f"<b>{label.strip()}</b>", body_style),
                Paragraph(value.strip(), body_style),
            ])

    table = Table(table_data, colWidths=[label_width, value_width])
    table.setStyle(
        TableStyle(
            [
                ("SPAN", (0, 0), (-1, 0)),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0A4E8B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 2), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("GRID", (0, 1), (-1, -1), 0.25, colors.HexColor("#CBD5F5")),
            ]
        )
    )
    _, table_height = table.wrapOn(pdf, width, 500)
    table.drawOn(pdf, left, top - table_height)
    return top - table_height


def _pattern_stats(pattern) -> list[tuple[str, str]]:
    if not pattern or not pattern.points:
        return []
    angles = np.array([p.angle_deg for p in pattern.points], dtype=float)
    values = np.array([p.amplitude_linear for p in pattern.points], dtype=float)
    values = np.clip(values, 1e-6, None)
    values_db = 20 * np.log10(values)
    peak_idx = int(np.argmax(values_db))
    peak_gain = float(values_db[peak_idx])
    peak_angle = float(angles[peak_idx])

    target = peak_gain - 3.0
    left_angle = None
    for idx in range(peak_idx, 0, -1):
        if idx == peak_idx:
            continue
        if values_db[idx] <= target:
            prev_idx = idx + 1 if idx + 1 < len(values_db) else idx
            left_angle = np.interp(target, [values_db[idx], values_db[prev_idx]], [angles[idx], angles[prev_idx]])
            break
    right_angle = None
    for idx in range(peak_idx, len(values_db)):
        if idx == peak_idx:
            continue
        if values_db[idx] <= target:
            next_idx = idx - 1 if idx - 1 >= 0 else idx
            right_angle = np.interp(target, [values_db[next_idx], values_db[idx]], [angles[next_idx], angles[idx]])
            break
    beamwidth = None
    if left_angle is not None and right_angle is not None:
        beamwidth = abs(right_angle - left_angle)

    opposite_idx = (peak_idx + len(values_db) // 2) % len(values_db)
    fbr = peak_gain - float(values_db[opposite_idx])

    stats: list[tuple[str, str]] = [
        ("Peak", f"{peak_gain:.2f} dB @ {peak_angle:.0f}°"),
    ]
    if beamwidth is not None:
        stats.append(("Beamwidth (-3 dB)", f"{beamwidth:.1f}°"))
    stats.append(("Front-to-back ratio", f"{fbr:.1f} dB"))
    return stats


def _draw_text_block(
    pdf: canvas.Canvas,
    text: str,
    left: float,
    top: float,
    width: float,
    max_height: float,
    style: ParagraphStyle | None = None,
) -> float:
    if not text:
        return top
    style = style or BODY_STYLE
    paragraph = Paragraph(text.replace("\n", "<br/>"), style)
    available_height = max(max_height, 1)
    _, height = paragraph.wrap(width, available_height)
    height = min(height, available_height)
    paragraph.drawOn(pdf, left, top - height)
    return top - height


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _draw_paragraph(
    pdf: canvas.Canvas,
    text: str,
    left: float,
    top: float,
    width: float,
    height: float,
    style: ParagraphStyle | None = None,
) -> float:
    if not text:
        return top
    style = style or BODY_STYLE
    frame = Frame(left, top - height, width, height, showBoundary=0)
    paragraph = Paragraph(text.replace("\n", "<br/>"), style)
    frame.addFromList([paragraph], pdf)
    return frame._y1


def _draw_table(
    pdf: canvas.Canvas,
    data: list[list[str]],
    left: float,
    top: float,
    width: float,
    title: str | None = None,
) -> float:
    if not data:
        return top
    table = Table(data, colWidths=[width / len(data[0]) for _ in data[0]])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0A4E8B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
            ]
        )
    )
    if title:
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(left, top - 12, title)
        top -= 16
    table_width, table_height = table.wrapOn(pdf, width, 500)
    table.drawOn(pdf, left, top - table_height)
    return top - table_height - 12


def _draw_list(
    pdf: canvas.Canvas,
    title: str | None,
    content: str,
    left: float,
    top: float,
    width: float,
    font_size: float = 10,
) -> float:
    lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
    if not lines:
        return top
    if title:
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(left, top - 12, title)
        top -= 16
    pdf.setFont("Helvetica", font_size)
    line_height = font_size + 4
    y_cursor = top
    bullet_indent = 10
    for line in lines:
        y_cursor -= line_height
        pdf.drawString(left + bullet_indent, y_cursor, f"• {line}")
    return y_cursor - 6


def _draw_image(
    pdf: canvas.Canvas,
    image_bytes: bytes,
    left: float,
    top: float,
    max_width: float,
    max_height: float,
) -> float:
    stream = io.BytesIO(image_bytes)
    with Image.open(stream) as img:
        img = img.convert("RGB")
        aspect = img.width / img.height if img.height else 1
        width = min(max_width, max_height * aspect)
        height = width / aspect
        if height > max_height:
            height = max_height
            width = height * aspect
        stream.seek(0)
        image_reader = ImageReader(stream)
        pdf.drawImage(
            image_reader,
            left,
            top - height,
            width=width,
            height=height,
            preserveAspectRatio=True,
            mask="auto",
        )
    stream.close()
    return top - height - 12


def _generate_pattern_plot(pattern, title: str, polar: bool = False) -> bytes | None:
    if not pattern or not pattern.points:
        return None
    angles = np.array([p.angle_deg for p in pattern.points], dtype=float)
    values = np.array([p.amplitude_linear for p in pattern.points], dtype=float)
    values = np.clip(values, 1e-6, None)
    values_db = 20 * np.log10(values)
    buf = io.BytesIO()
    if polar:
        fig = plt.figure(figsize=(4.0, 4.0))
        ax = fig.add_subplot(111, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.plot(np.deg2rad(angles), values_db, linewidth=1.5, color="#0A4E8B")
        ax.set_title(title, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.4)
    else:
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        ax.plot(angles, values_db, linewidth=1.5, color="#0A4E8B")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Ângulo (°)")
        ax.set_ylabel("Nível (dB)")
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(buf, format="PNG", dpi=200)
    plt.close(fig)
    return buf.getvalue()


def _draw_qr(pdf: canvas.Canvas, data: str, left: float, bottom: float, size: float) -> None:
    widget = qr_code.QrCodeWidget(data)
    bounds = widget.getBounds()
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    drawing = Drawing(size, size, transform=[size / width, 0, 0, size / height, 0, 0])
    drawing.add(widget)
    renderPDF.draw(drawing, pdf, left, bottom)


def _resolve_template_path() -> Path | None:
    candidate = current_app.config.get("DATASHEET_TEMPLATE")
    if candidate:
        path = Path(candidate)
    else:
        path = Path(current_app.root_path).parent / "modelo.pdf"
    if path.exists():
        return path
    return None


def _remove_text_operations(stream: ContentStream) -> bool:
    """Remove text drawing operators so modelo.pdf acts as a clean background."""
    modified = False
    depth = 0
    operations: list[tuple[list, bytes]] = []
    for operands, operator in stream.operations:
        op = operator.decode("latin-1") if isinstance(operator, bytes) else operator
        if op == "BT":
            depth += 1
            modified = True
            continue
        if op == "ET":
            if depth:
                depth -= 1
            modified = True
            continue
        if depth:
            modified = True
            continue
        operations.append((operands, operator))
    if modified:
        stream.operations = operations
    return modified


def _strip_text_from_stream_object(stream_obj, reader: PdfReader) -> None:
    try:
        content = ContentStream(stream_obj, reader)
    except Exception:
        return
    changed = _remove_text_operations(content)
    if changed:
        stream_obj.set_data(content.get_data())
    resources = stream_obj.get("/Resources")
    if not resources:
        return
    xobjects = resources.get("/XObject")
    if not xobjects:
        return
    for value in xobjects.values():
        try:
            resolved = value.get_object()
        except Exception:
            continue
        if resolved.get("/Subtype") == "/Form":
            # Clean nested form content recursively.
            _strip_text_from_stream_object(resolved, reader)


def _sanitize_template_pages(reader: PdfReader) -> list:
    pages = []
    for page in reader.pages:
        try:
            contents = page.get_contents()
        except Exception:
            contents = None
        if contents is not None:
            stream = ContentStream(contents, reader)
            if _remove_text_operations(stream):
                page[NameObject("/Contents")] = stream
        resources = page.get("/Resources")
        if resources:
            xobjects = resources.get("/XObject")
            if xobjects:
                for item in xobjects.values():
                    try:
                        resolved = item.get_object()
                    except Exception:
                        continue
                    if resolved.get("/Subtype") == "/Form":
                        _strip_text_from_stream_object(resolved, reader)
        pages.append(page)
    return pages


def _load_template(template_path: Path) -> tuple[list, float, float]:
    reader = PdfReader(str(template_path))
    pages = _sanitize_template_pages(reader)
    if pages:
        first_page = pages[0]
        width = float(first_page.mediabox.width)
        height = float(first_page.mediabox.height)
    else:
        width, height = LETTER
    return pages, width, height


def _merge_with_template(content: bytes, template_pages: Sequence) -> bytes:
    content_reader = PdfReader(io.BytesIO(content))
    writer = PdfWriter()
    total_templates = len(template_pages)
    for idx, content_page in enumerate(content_reader.pages):
        template_index = idx if idx < total_templates else total_templates - 1
        template_page = template_pages[template_index]
        writer.add_page(template_page)
        writer.pages[-1].merge_page(content_page)
    merged = io.BytesIO()
    writer.write(merged)
    merged.seek(0)
    return merged.getvalue()


def generate_antenna_datasheet_pdf(antenna: Antenna) -> bytes:
    use_template = current_app.config.get("DATASHEET_USE_TEMPLATE", True)
    template_path = _resolve_template_path() if use_template else None
    if template_path:
        template_pages, page_width, page_height = _load_template(template_path)
    else:
        template_pages = None
        page_width, page_height = LETTER

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(page_width, page_height))
    margin = 0.7 * inch
    body_width = page_width - 2 * margin

    bg1 = _resolve_background("DATASHEET_BACKGROUND_PAGE1", "static/img/datasheet_bg_page1.png")
    if bg1:
        _draw_background(pdf, bg1, page_width, page_height)

    # Header
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(margin, page_height - margin, antenna.name or "EFTX Antenna")
    pdf.setFont("Helvetica", 12)
    subtitle = f"Model {antenna.model_number}" if antenna.model_number else "Datasheet"
    pdf.drawString(margin, page_height - margin - 22, subtitle)

    y_cursor = page_height - margin - 60

    marketing_image = antenna.marketing_image or antenna.thumbnail_image
    description = _truncate_text(antenna.description or "Description not provided.", max_words=80)

    marketing_top_base = page_height - margin - HEADER_OFFSET_PT
    marketing_width = MARKETING_WIDTH
    marketing_height = MARKETING_HEIGHT
    marketing_left = margin
    marketing_top = marketing_top_base

    description_left = marketing_left + marketing_width + DESCRIPTION_GAP
    description_top = marketing_top_base
    description_width = max(body_width - marketing_width - DESCRIPTION_GAP, body_width * 0.45)
    description_height = MARKETING_HEIGHT

    marketing_bottom = marketing_top
    if marketing_image:
        try:
            marketing_bottom = _draw_image(
                pdf,
                marketing_image,
                marketing_left,
                marketing_top,
                marketing_width,
                marketing_height,
            )
        except Exception:
            marketing_bottom = marketing_top - marketing_height
    else:
        marketing_bottom = marketing_top - marketing_height

    bottom_desc = _draw_text_block(
        pdf,
        description,
        description_left,
        description_top,
        description_width,
        description_height,
        style=BODY_STYLE,
    )
    y_cursor = min(marketing_bottom, bottom_desc) - SECTION_SPACING

    electric_items = _normalize_bullets(antenna.electrical_characteristics or "")
    mech_items = _normalize_bullets(antenna.mechanical_characteristics or "")
    mechanical_image = antenna.mechanical_image

    electric_rows: list[tuple[str, str | None]] = []
    base_electric = [
        (
            "Nominal gain",
            f"{_safe_float(antenna.nominal_gain_dbd):.2f} dBd" if antenna.nominal_gain_dbd is not None else "—",
        ),
        (
            "Frequency range",
            f"{_safe_float(antenna.frequency_min_mhz):.2f} – {_safe_float(antenna.frequency_max_mhz):.2f} MHz"
            if antenna.frequency_min_mhz and antenna.frequency_max_mhz
            else "—",
        ),
        ("Polarization", antenna.polarization or "—"),
        ("Manufacturer", antenna.manufacturer or "EFTX Broadcast & Telecom"),
        ("Category", antenna.category or "—"),
    ]
    for spec in base_electric:
        _append_or_update(electric_rows, spec)
    for item in electric_items:
        _append_or_update(electric_rows, _split_item(item))

    electric_table_width = body_width
    table_left = margin
    table_top = y_cursor

    table_top = _draw_characteristics_table(
        pdf,
        "Electrical Characteristics",
        electric_rows,
        table_left,
        table_top,
        electric_table_width,
    ) - 16

    mech_rows: list[tuple[str, str | None]] = []
    for item in mech_items:
        _append_or_update(mech_rows, _split_item(item))
    mechanical_table_width = max(body_width - MARKETING_WIDTH - DESCRIPTION_GAP, body_width * 0.55)
    mech_table_left = margin
    mech_table_top = table_top
    if mech_rows:
        mech_table_top = _draw_characteristics_table(
            pdf,
            "Mechanical Characteristics",
            mech_rows,
            mech_table_left,
            mech_table_top,
            mechanical_table_width,
        ) - 16

    image_bottom = mech_table_top
    if mechanical_image:
        try:
            mech_left = mech_table_left + mechanical_table_width + DESCRIPTION_GAP
            mech_width = MARKETING_WIDTH
            mech_height = MECHANICAL_IMAGE_HEIGHT
            image_bottom = _draw_image(
                pdf,
                mechanical_image,
                mech_left,
                table_top,
                mech_width,
                mech_height,
            )
        except Exception:
            image_bottom = mech_table_top

    y_cursor = min(mech_table_top, image_bottom) - SECTION_SPACING

    pdf.showPage()

    bg2 = _resolve_background("DATASHEET_BACKGROUND_PAGE2", "static/img/datasheet_bg_page2.png")
    if bg2:
        _draw_background(pdf, bg2, page_width, page_height)

    # Second page: diagrams + QR
    pdf.setFont("Helvetica-Bold", 20)
    diagram_title = antenna.name or "EFTX Antenna"
    pdf.drawString(margin, page_height - margin, f"Radiation Patterns – {diagram_title}")
    pdf.setFont("Helvetica", 11)

    hrp_pattern = antenna.pattern_for(PatternType.HRP)
    vrp_pattern = antenna.pattern_for(PatternType.VRP)
    hrp_plot = _generate_pattern_plot(hrp_pattern, "Azimuth Pattern", polar=True)
    vrp_plot = _generate_pattern_plot(vrp_pattern, "Elevation Pattern", polar=False)

    chart_top = page_height - margin - 70
    chart_width = (body_width - margin) / 2
    chart_height = 3.8 * inch

    hrp_bottom = chart_top
    if hrp_plot:
        hrp_bottom = _draw_image(pdf, hrp_plot, margin, chart_top, chart_width, chart_height)
    else:
        pdf.setFont("Helvetica", 11)
        pdf.drawString(margin, chart_top - 20, "Azimuth pattern unavailable")

    vrp_left = margin + chart_width + margin
    vrp_bottom = chart_top
    if vrp_plot:
        vrp_bottom = _draw_image(pdf, vrp_plot, vrp_left, chart_top, chart_width, chart_height)
    else:
        pdf.setFont("Helvetica", 11)
        pdf.drawString(margin + chart_width + margin, chart_top - 20, "Elevation pattern unavailable")

    hrp_stats = _pattern_stats(hrp_pattern)
    if hrp_stats:
        _draw_characteristics_table(
            pdf,
            "Azimuth Data",
            hrp_stats,
            margin,
            hrp_bottom - 10,
            chart_width,
        )

    vrp_stats = _pattern_stats(vrp_pattern)
    if vrp_stats:
        _draw_characteristics_table(
            pdf,
            "Elevation Data",
            vrp_stats,
            vrp_left,
            vrp_bottom - 10,
            chart_width,
        )

    qr_size = 1.2 * inch
    qr_left = page_width - margin - qr_size
    qr_bottom = margin
    _draw_qr(pdf, "https://wa.me/551941170270", qr_left, qr_bottom, qr_size)
    pdf.setFont("Helvetica", 9)
    pdf.drawCentredString(qr_left + qr_size / 2, qr_bottom - 10, "WhatsApp +55 19 4117-0270")

    pdf.save()
    buffer.seek(0)
    content_bytes = buffer.getvalue()

    if template_pages:
        return _merge_with_template(content_bytes, template_pages)
    return content_bytes
