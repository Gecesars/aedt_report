from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from flask import current_app
from reportlab.graphics.barcode import qr as reportlab_qr
from reportlab.graphics.shapes import Drawing
from reportlab.graphics import renderPDF
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, LEGAL, LETTER, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.platypus import Frame, Paragraph, Table, TableStyle

from ..models import DatasheetOrientation, DatasheetPageSize

MM_TO_PT = 72.0 / 25.4

PAGE_SIZE_MAP = {
    DatasheetPageSize.A4: A4,
    DatasheetPageSize.LETTER: LETTER,
    DatasheetPageSize.LEGAL: LEGAL,
}

ALIGN_MAP = {
    "left": TA_LEFT,
    "center": TA_CENTER,
    "right": TA_RIGHT,
    "justify": TA_JUSTIFY,
}


class RenderError(RuntimeError):
    """Raised when it is not possible to render the datasheet."""


@dataclass
class RenderOptions:
    design_id: str | None
    page: dict[str, Any] = field(default_factory=dict)
    grid: dict[str, Any] = field(default_factory=dict)
    guides: dict[str, Any] = field(default_factory=dict)
    pages: list[dict[str, Any]] = field(default_factory=list)
    margins_mm: dict[str, float] = field(default_factory=dict)
    bindings: dict[str, Any] = field(default_factory=dict)

    def page_size(self) -> DatasheetPageSize:
        size = (self.page or {}).get("size", DatasheetPageSize.A4)
        if isinstance(size, DatasheetPageSize):
            return size
        try:
            return DatasheetPageSize(size)
        except ValueError:
            return DatasheetPageSize.A4

    def orientation(self) -> DatasheetOrientation:
        orientation = (self.page or {}).get("orientation", DatasheetOrientation.PORTRAIT)
        if isinstance(orientation, DatasheetOrientation):
            return orientation
        try:
            return DatasheetOrientation(orientation)
        except ValueError:
            return DatasheetOrientation.PORTRAIT

    def margins_pt(self) -> dict[str, float]:
        margins = {
            "top": float(self.margins_mm.get("top", 0)) * MM_TO_PT,
            "right": float(self.margins_mm.get("right", 0)) * MM_TO_PT,
            "bottom": float(self.margins_mm.get("bottom", 0)) * MM_TO_PT,
            "left": float(self.margins_mm.get("left", 0)) * MM_TO_PT,
        }
        return margins


@dataclass
class FrameRect:
    x: float
    y: float
    width: float
    height: float


def _page_size_in_points(options: RenderOptions) -> tuple[float, float]:
    base = PAGE_SIZE_MAP.get(options.page_size(), A4)
    if options.orientation() == DatasheetOrientation.LANDSCAPE:
        return landscape(base)
    return base


def _resolve_frame(layer: dict[str, Any], page_height: float, margins: dict[str, float]) -> FrameRect:
    frame = layer.get("frame") or {}
    x_mm = float(frame.get("x", 0))
    y_mm = float(frame.get("y", 0))
    width_mm = float(frame.get("w", frame.get("width", 0)))
    height_mm = float(frame.get("h", frame.get("height", 0)))

    x_pt = margins.get("left", 0.0) + x_mm * MM_TO_PT
    width_pt = max(width_mm * MM_TO_PT, 0.1)
    height_pt = max(height_mm * MM_TO_PT, 0.1)

    # Y=0 na interface significa topo útil; converter para PDF com origem canto inferior
    top_offset = margins.get("top", 0.0) + y_mm * MM_TO_PT
    y_pt = page_height - top_offset - height_pt

    return FrameRect(x=x_pt, y=y_pt, width=width_pt, height=height_pt)


def _paragraph_style(props: dict[str, Any]) -> ParagraphStyle:
    font = props.get("font") or {}
    font_name = font.get("family", font.get("familyName", "Helvetica")) or "Helvetica"
    weight = font.get("weight") or props.get("fontWeight")
    if (weight in {"bold", 700, "700"}) and font_name == "Helvetica":
        font_name = "Helvetica-Bold"
    if (weight in {"bold", 700, "700"}) and font_name == "Source Sans Pro":
        font_name = "SourceSansPro-Bold"

    font_size = max(float(font.get("size", props.get("fontSize", 12))), 1.0)
    leading = float(font.get("lineHeight", props.get("lineHeight", font_size * 1.2)))
    align = props.get("align") or font.get("align") or "left"
    letter_spacing = float(font.get("letterSpacing", props.get("letterSpacing", 0)))
    text_color = colors.HexColor(props.get("color", font.get("color", "#111827")))

    style = ParagraphStyle(
        name="datasheet-text",
        fontName=font_name,
        fontSize=font_size,
        leading=leading,
        alignment=ALIGN_MAP.get(str(align).lower(), TA_LEFT),
        textColor=text_color,
        spaceAfter=0,
        spaceBefore=0,
    )
    if letter_spacing:
        style.wordSpace = letter_spacing
    return style


def _strip_contacts(text: str) -> str:
    if not text:
        return ""
    patterns = [
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        r"\+?\d{2,3}[\s-]?\(?\d{2,3}\)?[\s-]?\d{3,5}[\s-]?\d{4}",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "[contato removido]", text)
    return text


def _draw_text(pdf: pdf_canvas.Canvas, layer: dict[str, Any], frame: FrameRect) -> None:
    props = layer.get("props") or {}
    content = props.get("text") or props.get("content") or ""
    if props.get("stripContacts", True):
        content = _strip_contacts(content)
    style = _paragraph_style(props)
    box = Frame(frame.x, frame.y, frame.width, frame.height, showBoundary=0)
    paragraph = Paragraph(str(content).replace("\n", "<br/>"), style)
    try:
        box.addFromList([paragraph], pdf)
    except Exception as exc:  # pragma: no cover - defensive
        current_app.logger.warning("Falha ao renderizar texto: %s", exc)


def _resolve_image_path(src: str | None, asset_root: Path | None) -> Path | None:
    if not src:
        return None
    src = str(src)
    if src.startswith("http://") or src.startswith("https://"):
        return None
    candidate: Path | None = None
    if asset_root is not None:
        if src.startswith("/admin/datasheets/assets/"):
            try:
                _, _, rest = src.partition("/admin/datasheets/assets/")
                parts = Path(rest)
                if len(parts.parts) >= 1:
                    candidate = (asset_root / Path(*parts.parts[1:])).resolve()
            except Exception:  # pragma: no cover - defensive
                candidate = None
        elif src.startswith("/"):
            # Path absoluta dentro do export root? tentar resolver relativa ao projeto
            rel = src.lstrip("/")
            candidate = (asset_root / rel).resolve()
        else:
            candidate = (asset_root / src).resolve()
        if candidate is not None:
            try:
                candidate.relative_to(asset_root)
            except ValueError:
                candidate = None
            if candidate is not None and candidate.exists():
                return candidate
    # fallback: tentar resolver relativo ao projeto
    project_root = Path(current_app.config["PROJECT_ROOT"])
    candidate = (project_root / src.lstrip("/"))
    if candidate.exists():
        return candidate
    return None


def _draw_image(pdf: pdf_canvas.Canvas, layer: dict[str, Any], frame: FrameRect, asset_root: Path | None) -> None:
    props = layer.get("props") or {}
    src = props.get("src") or layer.get("src")
    image_path = _resolve_image_path(src, asset_root)
    if image_path is None:
        return
    preserve = props.get("objectFit", "contain") in {"contain", "scale-down"}
    try:
        pdf.drawImage(
            str(image_path),
            frame.x,
            frame.y,
            width=frame.width,
            height=frame.height,
            preserveAspectRatio=preserve,
            mask="auto",
        )
    except Exception as exc:  # pragma: no cover - defensive
        current_app.logger.warning("Falha ao desenhar imagem %s: %s", image_path, exc)


def _format_cell(value: Any, format_spec: str | None) -> str:
    if format_spec:
        suffix = ""
        core = format_spec
        if core and not core[-1].isalnum() and core[-1] not in {"%"}:
            suffix = core[-1]
            core = core[:-1]
        try:
            formatted = format(value, core) if core else str(value)
        except Exception:  # pragma: no cover - fallback
            formatted = str(value)
        return f"{formatted}{suffix}"
    return str(value)


def _resolve_path(data: Any, parts: Iterable[str]) -> Any:
    parts = list(parts)
    if not parts:
        return data
    head, *rest = parts
    if isinstance(data, dict):
        if head.endswith("[]"):
            key = head[:-2]
            items = []
            for item in data.get(key, []) or []:
                resolved = _resolve_path(item, rest)
                if isinstance(resolved, list):
                    items.extend(resolved)
                else:
                    items.append(resolved)
            return items
        return _resolve_path(data.get(head), rest)
    if isinstance(data, list):
        try:
            index = int(head)
        except ValueError:
            return None
        if 0 <= index < len(data):
            return _resolve_path(data[index], rest)
    return None


def _resolve_binding_rows(bindings: dict[str, Any], block_id: str) -> tuple[list[str], list[list[str]]]:
    blocks = bindings.get("blocks") if isinstance(bindings, dict) else {}
    block = blocks.get(block_id) if isinstance(blocks, dict) else None
    if not isinstance(block, dict):
        return [], []
    dataset_id = block.get("dataset")
    datasets = bindings.get("datasets") if isinstance(bindings, dict) else {}
    dataset = datasets.get(dataset_id) if isinstance(datasets, dict) else None
    if not isinstance(dataset, dict):
        return [], []
    payload = dataset.get("data")
    if not isinstance(payload, list):
        return [], []
    columns = block.get("map", {}).get("columns") or []
    headers = [col.get("label") or col.get("path") or "" for col in columns]
    rows: list[list[str]] = []
    for row_data in payload:
        row: list[str] = []
        for col in columns:
            path = col.get("path")
            fmt = col.get("format")
            resolved = _resolve_path(row_data, path.split(".")) if path else row_data
            if isinstance(resolved, list):
                resolved = ", ".join(str(item) for item in resolved)
            row.append(_format_cell(resolved, fmt))
        rows.append(row)
    if not columns and payload:
        first = payload[0]
        if isinstance(first, dict):
            headers = list(first.keys())
        for row_data in payload:
            if isinstance(row_data, dict):
                rows.append([str(row_data.get(key, "")) for key in headers])
            elif isinstance(row_data, (list, tuple)):
                rows.append([str(item) for item in row_data])
            else:
                rows.append([str(row_data)])
    return headers, rows


def _draw_table(pdf: pdf_canvas.Canvas, layer: dict[str, Any], frame: FrameRect, bindings: dict[str, Any]) -> None:
    props = layer.get("props") or {}
    headers = props.get("headers") or []
    rows = props.get("rows") or []
    binding_id = layer.get("bindingId") or props.get("bindingId")
    if binding_id:
        bound_headers, bound_rows = _resolve_binding_rows(bindings, binding_id)
        if bound_headers:
            headers = bound_headers
        if bound_rows:
            rows = bound_rows
    table_data = []
    if headers:
        table_data.append(headers)
    table_data.extend(rows)
    if not table_data:
        return
    table = Table(table_data, repeatRows=1 if headers else 0)
    base_style: list[tuple[str, tuple[int, int], tuple[int, int], Any]] = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#1f2937")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), props.get("fontSize", 9)),
    ]
    if headers:
        base_style.extend([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(props.get("headerBg", "#0a4e8b"))),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(props.get("headerColor", "#ffffff"))),
            ("FONTNAME", (0, 0), (-1, 0), props.get("headerFont", "Helvetica-Bold")),
        ])
    table.setStyle(TableStyle(base_style))
    available_height = frame.height
    _, table_height = table.wrap(frame.width, available_height)
    table.drawOn(pdf, frame.x, frame.y + (available_height - table_height))


def _draw_qr(pdf: pdf_canvas.Canvas, layer: dict[str, Any], frame: FrameRect) -> None:
    props = layer.get("props") or {}
    value = props.get("value") or layer.get("value") or "https://eftx.com.br"
    qrw = reportlab_qr.QrCodeWidget(str(value))
    bounds = qrw.getBounds()
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    size = min(frame.width, frame.height)
    scaling_x = size / max(width, 1)
    scaling_y = size / max(height, 1)
    drawing = Drawing(size, size, transform=[scaling_x, 0, 0, scaling_y, 0, 0])
    drawing.add(qrw)
    renderPDF.draw(drawing, pdf, frame.x, frame.y)


def _draw_line(pdf: pdf_canvas.Canvas, layer: dict[str, Any], frame: FrameRect) -> None:
    props = layer.get("props") or {}
    stroke = colors.HexColor(props.get("color", props.get("stroke", "#0a4e8b")))
    stroke_width = float(props.get("strokeWidth", 0.6))
    x1 = frame.x
    y1 = frame.y + frame.height
    x2 = x1 + frame.width
    y2 = frame.y
    if "x2" in props:
        x2 = frame.x + float(props.get("x2", 0)) * MM_TO_PT
    if "y2" in props:
        y2 = frame.y + frame.height - float(props.get("y2", 0)) * MM_TO_PT
    pdf.setStrokeColor(stroke)
    pdf.setLineWidth(stroke_width)
    pdf.line(x1, y1, x2, y2)


def _draw_rect(pdf: pdf_canvas.Canvas, layer: dict[str, Any], frame: FrameRect) -> None:
    props = layer.get("props") or {}
    fill = props.get("fill") or props.get("background")
    stroke = props.get("stroke")
    stroke_width = float(props.get("strokeWidth", 0.8))
    radius = float(props.get("radius", 0)) * mm

    if fill:
        pdf.setFillColor(colors.HexColor(fill))
    else:
        pdf.setFillColor(colors.transparent)
    if stroke:
        pdf.setStrokeColor(colors.HexColor(stroke))
    else:
        pdf.setStrokeColor(colors.transparent)
    pdf.setLineWidth(stroke_width)
    if radius:
        pdf.roundRect(frame.x, frame.y, frame.width, frame.height, radius, stroke=1 if stroke else 0, fill=1 if fill else 0)
    else:
        pdf.rect(frame.x, frame.y, frame.width, frame.height, stroke=1 if stroke else 0, fill=1 if fill else 0)


def _page_dimensions_for(template_sizes: list[tuple[float, float]] | None, idx: int, options: RenderOptions) -> tuple[float, float]:
    if template_sizes:
        return template_sizes[min(idx, len(template_sizes) - 1)]
    return _page_size_in_points(options)


def _margins_for(options: RenderOptions, page_entry: dict[str, Any]) -> dict[str, float]:
    base = dict(options.margins_mm or {})
    overrides = page_entry.get("margins_mm") if isinstance(page_entry, dict) else {}
    if isinstance(overrides, dict):
        base.update({k: float(v) for k, v in overrides.items() if v is not None})
    return {
        "top": float(base.get("top", 0)) * MM_TO_PT,
        "right": float(base.get("right", 0)) * MM_TO_PT,
        "bottom": float(base.get("bottom", 0)) * MM_TO_PT,
        "left": float(base.get("left", 0)) * MM_TO_PT,
    }


def _page_layers(page_entry: dict[str, Any]) -> list[dict[str, Any]]:
    layers = page_entry.get("layers") if isinstance(page_entry, dict) else []
    if not isinstance(layers, list):
        layers = []
    return sorted(
        (layer for layer in layers if isinstance(layer, dict) and layer.get("visible", True)),
        key=lambda item: item.get("zIndex", 0),
    )


def _draw_layer(pdf: pdf_canvas.Canvas, layer: dict[str, Any], page_height: float, margins: dict[str, float], asset_root: Path | None, bindings: dict[str, Any]) -> None:
    frame = _resolve_frame(layer, page_height, margins)
    ltype = layer.get("type")

    def _dispatch(target_frame: FrameRect) -> None:
        if ltype in {"title", "text", "textbox"}:
            if ltype == "title":
                props = layer.setdefault("props", {})
                font = props.setdefault("font", {})
                font.setdefault("size", 24)
                font.setdefault("weight", "bold")
            _draw_text(pdf, layer, target_frame)
        elif ltype == "table":
            _draw_table(pdf, layer, target_frame, bindings)
        elif ltype in {"image", "diagram"}:
            _draw_image(pdf, layer, target_frame, asset_root)
        elif ltype == "qr":
            _draw_qr(pdf, layer, target_frame)
        elif ltype in {"shape", "rect"}:
            _draw_rect(pdf, layer, target_frame)
        elif ltype == "line":
            _draw_line(pdf, layer, target_frame)
        else:
            current_app.logger.debug("Layer '%s' ignorada", ltype)

    rotation = 0.0
    if isinstance(layer.get("rotation"), (int, float)):
        rotation = float(layer.get("rotation") or 0.0)
    elif isinstance(layer.get("props"), dict) and isinstance(layer["props"].get("rotation"), (int, float)):
        rotation = float(layer["props"].get("rotation") or 0.0)

    if rotation:
        pdf.saveState()
        cx = frame.x + frame.width / 2
        cy = frame.y + frame.height / 2
        pdf.translate(cx, cy)
        pdf.rotate(-rotation)
        pdf.translate(-frame.width / 2, -frame.height / 2)
        local_frame = FrameRect(0, 0, frame.width, frame.height)
        _dispatch(local_frame)
        pdf.restoreState()
    else:
        _dispatch(frame)


def _build_overlay_pdf(options: RenderOptions, asset_root: Path | None, template_sizes: list[tuple[float, float]] | None) -> bytes:
    pages = options.pages or []
    if not pages:
        pages = [
            {
                "id": "page-1",
                "name": "Página 1",
                "order": 1,
                "layers": [],
            }
        ]
    buffer = io.BytesIO()
    default_size = _page_size_in_points(options)
    pdf = pdf_canvas.Canvas(buffer, pagesize=default_size)

    for idx, page_entry in enumerate(pages):
        width, height = _page_dimensions_for(template_sizes, idx, options)
        pdf.setPageSize((width, height))
        margins = _margins_for(options, page_entry)
        layers = _page_layers(page_entry)
        for layer in layers:
            _draw_layer(pdf, layer, height, margins, asset_root, options.bindings)
        pdf.showPage()

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


def render_pdf(
    options: RenderOptions,
    export_path: Path | None = None,
    asset_root: Path | None = None,
    template_path: Path | None = None,
) -> bytes:
    template_sizes: list[tuple[float, float]] | None = None
    template_reader = None
    project_root = Path(current_app.config.get("PROJECT_ROOT"))
    template_file = template_path or (project_root / "modelo.pdf")
    PdfReader = None
    PdfWriter = None
    if template_file.exists():
        try:
            from pypdf import PdfReader as _PdfReader, PdfWriter as _PdfWriter  # type: ignore

            PdfReader = _PdfReader
            PdfWriter = _PdfWriter
            template_reader = PdfReader(str(template_file))
            template_sizes = []
            for page in template_reader.pages:
                width = float(page.mediabox.width)
                height = float(page.mediabox.height)
                template_sizes.append((width, height))
        except Exception:
            template_reader = None
            template_sizes = None

    overlay_bytes = _build_overlay_pdf(options, asset_root, template_sizes)

    pdf_bytes = overlay_bytes
    if template_reader is not None and PdfReader and PdfWriter:
        try:
            overlay_reader = PdfReader(io.BytesIO(overlay_bytes))
            writer = PdfWriter()
            template_pages = list(template_reader.pages)
            if not template_pages:
                template_reader = None
            else:
                for idx, overlay_page in enumerate(overlay_reader.pages):
                    template_page = template_pages[min(idx, len(template_pages) - 1)].copy()
                    template_page.merge_page(overlay_page)
                    writer.add_page(template_page)
                buffer = io.BytesIO()
                writer.write(buffer)
                pdf_bytes = buffer.getvalue()
        except Exception as exc:
            current_app.logger.warning("Falha ao aplicar modelo.pdf: %s", exc)
            pdf_bytes = overlay_bytes

    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("wb") as handle:
            handle.write(pdf_bytes)
    return pdf_bytes
