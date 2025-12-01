"""Utilities to export designer layouts to SVG and PNG."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable

from PIL import Image, ImageDraw, ImageFont
import subprocess
import tempfile

from .datasheet_renderer import MM_TO_PT


MM_TO_PX_96 = 96.0 / 25.4
MM_TO_PX_300 = 300.0 / 25.4

PAGE_SIZES_MM = {
    "A4": (210, 297),
    "Letter": (215.9, 279.4),
    "Legal": (215.9, 355.6),
}


class DesignerExportError(RuntimeError):
    """Raised when rendering the designer output fails."""


def _page_dimensions_mm(design: Dict[str, Any]) -> tuple[float, float]:
    page = design.get("page") or {}
    size = page.get("size", "A4")
    dims = PAGE_SIZES_MM.get(size, PAGE_SIZES_MM["A4"])
    orientation = page.get("orientation", "portrait")
    if orientation == "landscape":
        return dims[1], dims[0]
    return dims


def _layer_rotation(layer: Dict[str, Any]) -> float:
    rotation = layer.get("rotation") or 0.0
    try:
        return float(rotation)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return 0.0


def render_svg(design: Dict[str, Any], asset_root: Path | None = None, template_image: Image.Image | None = None) -> str:
    width_mm, height_mm = _page_dimensions_mm(design)
    svg_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm}mm" height="{height_mm}mm" viewBox="0 0 {width_mm} {height_mm}">',
    ]

    if template_image is not None:
        buffer = BytesIO()
        template_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        svg_parts.append(
            f'<image href="data:image/png;base64,{encoded}" x="0" y="0" width="{width_mm}" height="{height_mm}" preserveAspectRatio="none" opacity="0.92" />'
        )

    for page in design.get("pages", [])[:1]:
        for layer in page.get("layers", []) or []:
            svg_parts.append(_layer_to_svg(layer, asset_root))

    svg_parts.append("</svg>")
    return "".join(filter(None, svg_parts))


def _layer_to_svg(layer: Dict[str, Any], asset_root: Path | None) -> str:
    layer_type = layer.get("type")
    if layer.get("hidden"):
        return ""
    x = layer.get("x", 0)
    y = layer.get("y", 0)
    width = layer.get("width", 0)
    height = layer.get("height", 0)
    rotation = _layer_rotation(layer)
    transform = ""
    if rotation:
        cx = x + width / 2
        cy = y + height / 2
        transform = f' transform="rotate({rotation} {cx} {cy})"'

    props = layer.get("props") or {}

    if layer_type in {"text", "title", "textbox"}:
        text_value = layer.get("text") or props.get("text") or ""
        fill = props.get("color", "#111827")
        font = props.get("font") or {}
        font_size = font.get("size", layer.get("fontSize", 12))
        anchor = "start"
        if props.get("align") == "center":
            anchor = "middle"
            x += width / 2
        elif props.get("align") == "right":
            anchor = "end"
            x += width
        lines = text_value.split("\n")
        line_height = font.get("lineHeight", font_size * 1.2)
        svg_lines = []
        for idx, line in enumerate(lines):
            dy = idx * (line_height / MM_TO_PT)
            svg_lines.append(f'<tspan x="{x}" dy="{idx and dy or 0}">{_escape_xml(line)}</tspan>')
        return (
            f'<text x="{x}" y="{y}" fill="{fill}" font-family="{font.get("family", "Helvetica")}" '
            f'font-size="{font_size}" text-anchor="{anchor}"{transform} dominant-baseline="hanging">{"".join(svg_lines)}</text>'
        )

    if layer_type == 'rect':
        fill = props.get("fill", "#1f2937")
        opacity = props.get("opacity", layer.get("opacity", 1))
        radius = props.get("borderRadius", 0)
        stroke = props.get("stroke")
        svg = f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" opacity="{opacity}" rx="{radius}" ry="{radius}"{transform}'
        if stroke:
            svg += f' stroke="{stroke}" stroke-width="0.8"'
        svg += ' />'
        return svg

    if layer_type == 'ellipse':
        fill = props.get("fill", "#334155")
        opacity = props.get("opacity", 1)
        stroke = props.get("stroke")
        rx = width / 2
        ry = height / 2
        cx = x + rx
        cy = y + ry
        svg = f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}" opacity="{opacity}"{transform}'
        if stroke:
            svg += f' stroke="{stroke}" stroke-width="0.8"'
        svg += ' />'
        return svg

    if layer_type == 'line':
        stroke = props.get("stroke", "#2563eb")
        stroke_width = props.get("strokeWidth", 1)
        x2 = x + width
        y2 = y + height
        return f'<line x1="{x}" y1="{y}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{stroke_width}" {transform}/>'

    if layer_type == 'image':
        src = layer.get("src") or props.get("src")
        if not src:
            return ""
        resolved = _resolve_asset_path(src, asset_root)
        href = resolved.as_uri() if resolved else src
        return f'<image href="{href}" x="{x}" y="{y}" width="{width}" height="{height}" preserveAspectRatio="xMidYMid meet"{transform} />'

    if layer_type in {'polygon', 'path'}:
        if layer_type == 'polygon' and props.get('points'):
            return f'<polygon points="{props["points"]}" fill="{props.get("fill", "rgba(59,130,246,0.3)")}" stroke="{props.get("stroke", "#2563eb")}" stroke-width="{props.get("strokeWidth", 1)}"{transform} />'
        if props.get('d'):
            return f'<path d="{props["d"]}" fill="{props.get("fill", "none")}" stroke="{props.get("stroke", "#2563eb")}" stroke-width="{props.get("strokeWidth", 1)}"{transform} />'
        return ""

    if layer_type == 'table':
        headers = props.get('headers') or []
        rows = props.get('rows') or []
        cell_height = height / max(1, len(rows) + (1 if headers else 0))
        group = [f'<g{transform}>']
        for idx, header in enumerate(headers):
            cell_y = y + idx * 0
            group.append(
                f'<rect x="{x}" y="{y}" width="{width}" height="{cell_height}" fill="{props.get("headerBg", "#0a4e8b")}" opacity="0.92" />'
            )
            group.append(
                f'<text x="{x + 2}" y="{y + 1}" fill="{props.get("headerColor", "#ffffff")}" font-size="{props.get("fontSize", 9)}" dominant-baseline="hanging">{_escape_xml(header)}</text>'
            )
        for row_idx, row in enumerate(rows):
            row_y = y + (headers and cell_height or 0) + row_idx * cell_height
            group.append(
                f'<rect x="{x}" y="{row_y}" width="{width}" height="{cell_height}" fill="rgba(15,28,46,0.05)" />'
            )
            col_width = width / max(1, len(row))
            for col_idx, cell in enumerate(row):
                cell_x = x + col_idx * col_width + 1
                group.append(
                    f'<text x="{cell_x}" y="{row_y + 1}" fill="#111827" font-size="{props.get("fontSize", 9)}" dominant-baseline="hanging">{_escape_xml(str(cell))}</text>'
                )
        group.append('</g>')
        return ''.join(group)

    return ""


def _escape_xml(value: str) -> str:
    return (
        value.replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
        .replace("'", '&#39;')
    )


def render_png(design: Dict[str, Any], asset_root: Path | None = None, template_image: Image.Image | None = None, dpi: int = 300) -> bytes:
    width_mm, height_mm = _page_dimensions_mm(design)
    width_px = int(width_mm * (dpi / 25.4))
    height_px = int(height_mm * (dpi / 25.4))
    image = Image.new('RGBA', (width_px, height_px), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)

    if template_image is not None:
        template_resized = template_image.resize((width_px, height_px))
        image = Image.alpha_composite(image, template_resized.convert('RGBA'))
        draw = ImageDraw.Draw(image)

    font_cache: Dict[str, ImageFont.FreeTypeFont] = {}
    page = design.get('pages', [])[0] if design.get('pages') else {}
    for layer in page.get('layers', []) or []:
        if layer.get('hidden'):
            continue
        _draw_layer_png(draw, image, layer, asset_root, font_cache, dpi)

    output = BytesIO()
    image.convert('RGB').save(output, format='PNG', dpi=(dpi, dpi))
    return output.getvalue()


def _load_font(family: str, size: float, font_cache: Dict[str, ImageFont.FreeTypeFont]) -> ImageFont.ImageFont:
    key = f"{family}|{size}"
    if key in font_cache:
        return font_cache[key]
    try:
        font = ImageFont.truetype(family, int(size))
    except Exception:  # pragma: no cover - fallback
        font = ImageFont.load_default()
    font_cache[key] = font
    return font


def _draw_layer_png(draw: ImageDraw.ImageDraw, canvas: Image.Image, layer: Dict[str, Any], asset_root: Path | None, font_cache: Dict[str, ImageFont.FreeTypeFont], dpi: int) -> None:
    layer_type = layer.get('type')
    x_mm = layer.get('x', 0)
    y_mm = layer.get('y', 0)
    width_mm = layer.get('width', 0)
    height_mm = layer.get('height', 0)
    x = int(x_mm * MM_TO_PX_300)
    y = int(y_mm * MM_TO_PX_300)
    width = max(1, int(width_mm * MM_TO_PX_300))
    height = max(1, int(height_mm * MM_TO_PX_300))
    props = layer.get('props') or {}

    if layer_type in {'text', 'title', 'textbox'}:
        text_value = layer.get('text') or props.get('text') or ''
        font_def = props.get('font') or {}
        font_size = font_def.get('size', 12)
        font = _load_font(font_def.get('family', 'DejaVuSans.ttf'), font_size * (dpi / 96), font_cache)
        color = props.get('color', '#111827')
        align = props.get('align', 'left')
        lines = text_value.split('\n')
        line_height = int(font_size * 1.2 * (dpi / 96))
        for idx, line in enumerate(lines):
            offset = idx * line_height
            text_width = draw.textlength(line, font=font)
            tx = x
            if align == 'center':
                tx = x + (width - text_width) // 2
            elif align == 'right':
                tx = x + width - text_width
            draw.text((tx, y + offset), line, fill=color, font=font)
        return

    if layer_type == 'rect':
        fill = props.get('fill', '#1f2937')
        opacity = int(255 * float(props.get('opacity', layer.get('opacity', 1))))
        color = _hex_to_rgba(fill, opacity)
        draw.rounded_rectangle([x, y, x + width, y + height], radius=int(props.get('borderRadius', 0) * MM_TO_PX_300), fill=color, outline=_hex_to_rgba(props.get('stroke', fill), opacity))
        return

    if layer_type == 'ellipse':
        fill = props.get('fill', '#334155')
        opacity = int(255 * float(props.get('opacity', 1)))
        draw.ellipse([x, y, x + width, y + height], fill=_hex_to_rgba(fill, opacity), outline=_hex_to_rgba(props.get('stroke', fill), opacity))
        return

    if layer_type == 'line':
        stroke = props.get('stroke', '#2563eb')
        draw.line([x, y, x + width, y + height], fill=stroke, width=int(props.get('strokeWidth', 2) * (dpi / 96)))
        return

    if layer_type == 'image':
        src = layer.get('src') or props.get('src')
        if not src:
            return
        resolved = _resolve_asset_path(src, asset_root)
        try:
            if resolved and resolved.exists():
                with Image.open(resolved) as im:
                    im = im.convert('RGBA')
                    im = im.resize((width, height))
                    canvas.alpha_composite(im, dest=(x, y))
        except Exception:  # pragma: no cover - fallback
            return
        return

    if layer_type == 'table':
        headers = props.get('headers') or []
        rows = props.get('rows') or []
        total_rows = len(rows) + (1 if headers else 0)
        if total_rows == 0:
            return
        row_height = height // total_rows
        font = _load_font('DejaVuSans.ttf', props.get('fontSize', 9) * (dpi / 96), font_cache)
        for idx, header in enumerate(headers):
            draw.rectangle([x, y, x + width, y + row_height], fill=_hex_to_rgba(props.get('headerBg', '#0a4e8b'), 220))
            draw.text((x + 4, y + 4), header, fill=props.get('headerColor', '#ffffff'), font=font)
        start_y = y + (row_height if headers else 0)
        for row in rows:
            draw.rectangle([x, start_y, x + width, start_y + row_height], outline='#1f2937', width=1)
            if row:
                col_width = width // len(row)
                for col_idx, cell in enumerate(row):
                    draw.text((x + col_idx * col_width + 4, start_y + 4), str(cell), fill='#111827', font=font)
            start_y += row_height


def _hex_to_rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join(ch * 2 for ch in hex_color)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b, alpha)


def _resolve_asset_path(src: str, asset_root: Path | None) -> Path | None:
    if not asset_root:
        return None
    candidate = (asset_root / src.lstrip('/')).resolve()
    try:
        candidate.relative_to(asset_root)
    except ValueError:
        return None
    return candidate if candidate.exists() else None


def load_template_image(template_path: Path, dpi: int = 300) -> Image.Image | None:
    try:
        if template_path.suffix.lower() == '.pdf':
            with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
                cmd = [
                    'gs',
                    '-dSAFER',
                    '-dBATCH',
                    '-dNOPAUSE',
                    '-sDEVICE=pngalpha',
                    f'-r{dpi}',
                    '-dFirstPage=1',
                    '-dLastPage=1',
                    f'-sOutputFile={tmp.name}',
                    str(template_path),
                ]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    with Image.open(tmp.name) as raster:
                        return raster.convert('RGBA')
                except (subprocess.CalledProcessError, FileNotFoundError):
                    return None
        with Image.open(template_path) as im:
            return im.convert('RGBA')
    except Exception:  # pragma: no cover - optional feature
        return None
