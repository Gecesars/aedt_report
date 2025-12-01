"""Utilities for templates, auto-fill and rendering metadata for the designer."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from zoneinfo import ZoneInfo

from flask import current_app, url_for

from ..schemas import EftxDatasheet, RangeFloat


EFTX_COLOR_TOKENS = {
    "primary": "#0a4e8b",
    "secondary": "#132a44",
    "accent": "#ff7a00",
    "accent_light": "#ffa94c",
    "bg_dark": "#050c16",
    "bg_panel": "#0f1c2e",
    "text_primary": "#f1f5ff",
    "text_body": "#d7e2f5",
}


DEFAULT_TEMPLATE_ID = "eftx-a4-dark"


def _format_range(range_value: RangeFloat, unit: str) -> str:
    return f"{range_value.min:.0f}–{range_value.max:.0f} {unit}".replace(".0", "")


def _format_value(value: Any, suffix: str | None = None, decimals: int = 1) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        fmt = f"{{:.{decimals}f}}".format(value)
        if fmt.endswith(".0"):
            fmt = fmt[:-2]
        return f"{fmt}{suffix or ''}"
    return f"{value}{suffix or ''}"


def _resolve_path(data: Any, path: str) -> Any:
    cursor = data
    for token in path.replace("[", ".").replace("]", "").split("."):
        if token == "":
            continue
        if isinstance(cursor, dict):
            cursor = cursor.get(token)
        elif isinstance(cursor, list) and token.isdigit():
            idx = int(token)
            cursor = cursor[idx] if 0 <= idx < len(cursor) else None
        else:
            return None
        if cursor is None:
            return None
    return cursor


def _build_highlights(summary: Dict[str, Any]) -> List[str]:
    highlights = summary.get("highlights") or []
    if not highlights and summary.get("description"):
        return [summary["description"]]
    return highlights


def _band_rows(datasheet: EftxDatasheet) -> List[List[str]]:
    rows: List[List[str]] = []
    for band in datasheet.bands:
        range_text = _format_range(band.freq_range_mhz, "MHz")
        gain_text = ""
        if band.gain_dbi:
            values = []
            if band.gain_dbi.typ is not None:
                values.append(f"typ {band.gain_dbi.typ:g} dBi")
            if band.gain_dbi.max is not None:
                values.append(f"max {band.gain_dbi.max:g} dBi")
            gain_text = ", ".join(values)
        beam_h = band.beamwidth_deg.h.typ if band.beamwidth_deg and band.beamwidth_deg.h else None
        beam_v = band.beamwidth_deg.v.typ if band.beamwidth_deg and band.beamwidth_deg.v else None
        beam_text = ""
        if beam_h is not None or beam_v is not None:
            beam_text = f"H {beam_h:g}° / V {beam_v:g}°"
        vswr = band.vswr_max or band.vswr_typ
        vswr_text = f"≤ {vswr:g}" if vswr else ""
        pol = band.polarization or ""
        rows.append([
            band.name or range_text,
            range_text,
            gain_text,
            beam_text,
            vswr_text,
            pol,
        ])
    return rows


def _mechanical_rows(datasheet: EftxDatasheet) -> List[List[str]]:
    mech = datasheet.mechanical
    if not mech:
        return []
    rows: List[List[str]] = []
    if mech.dimensions_mm:
        rows.append(["Dimensões (mm)", f"{mech.dimensions_mm.w:g} × {mech.dimensions_mm.h:g} × {mech.dimensions_mm.d:g}"])
    if mech.weight_kg is not None:
        rows.append(["Peso", f"{mech.weight_kg:g} kg"])
    if mech.tilt_range_deg:
        rows.append(["Tilt", f"{mech.tilt_range_deg.min:g}° a {mech.tilt_range_deg.max:g}°"])
    if mech.wind_operational_kmh is not None:
        rows.append(["Vento operacional", f"{mech.wind_operational_kmh:g} km/h"])
    if mech.wind_survival_kmh is not None:
        rows.append(["Vento sobrevivência", f"{mech.wind_survival_kmh:g} km/h"])
    if mech.temp_operating_c:
        rows.append([
            "Temperatura",
            f"{mech.temp_operating_c.min:g}°C a {mech.temp_operating_c.max:g}°C",
        ])
    if mech.connector:
        connector = mech.connector
        label = connector.type
        if connector.gender:
            label += f" {connector.gender}"
        if connector.count:
            label = f"{connector.count}× {label}"
        rows.append(["Conector", label])
    if mech.impedance_ohm is not None:
        rows.append(["Impedância", f"{mech.impedance_ohm:g} Ω"])
    return rows


def list_templates() -> List[Dict[str, Any]]:
    """Return the list of available layout templates."""

    blueprint = default_design_blueprint()
    return [
        {
            "id": DEFAULT_TEMPLATE_ID,
            "name": "A4 EFTX Dark",
            "description": "Layout padrão EFTX com cabeçalho, destaques e tabelas em tema escuro.",
            "thumbnail": "/static/img/designer/template_a4_dark.png",
            "page": {
                "size": "A4",
                "orientation": "portrait",
                "margins_mm": {"top": 18, "right": 14, "bottom": 18, "left": 14},
            },
            "blueprint": blueprint,
            "components": [
                "header",
                "summary",
                "bands_table",
                "mechanical_table",
                "radiation_slots",
                "footer",
            ],
        }
    ]


def default_design_blueprint() -> Dict[str, Any]:
    return {
        "id": DEFAULT_TEMPLATE_ID,
        "name": "EFTX Template A4",
        "page": {
            "size": "A4",
            "orientation": "portrait",
            "margins_mm": {"top": 18, "right": 14, "bottom": 18, "left": 14},
            "theme": "dark",
        },
        "grid": {"show": True, "size_mm": 5, "snap": True},
        "guides": {"smart": True, "ruler": True},
        "pages": [
            {
                "id": "page-1",
                "name": "Página 1",
                "order": 1,
                "zoom": 1,
                "layers": [
                    {
                        "id": "title",
                        "type": "text",
                        "x": 15,
                        "y": 12,
                        "width": 140,
                        "height": 16,
                        "props": {
                            "text": "Título principal",
                            "font": {"family": "Helvetica", "size": 22, "weight": "bold"},
                            "color": EFTX_COLOR_TOKENS["text_primary"],
                          },
                        "dataBind": "summary.title",
                    },
                    {
                        "id": "subtitle",
                        "type": "text",
                        "x": 15,
                        "y": 26,
                        "width": 140,
                        "height": 12,
                        "props": {
                            "text": "Subtítulo",
                            "font": {"family": "Helvetica", "size": 13},
                            "color": EFTX_COLOR_TOKENS["accent_light"],
                          },
                        "dataBind": "summary.subtitle",
                    },
                    {
                        "id": "revision",
                        "type": "text",
                        "x": 150,
                        "y": 12,
                        "width": 40,
                        "height": 10,
                        "props": {
                            "text": "Rev. 1.0",
                            "font": {"family": "Helvetica", "size": 10},
                            "color": EFTX_COLOR_TOKENS["text_body"],
                            "align": "right",
                          },
                        "dataBind": "meta.revision",
                    },
                    {
                        "id": "highlights",
                        "type": "text",
                        "x": 15,
                        "y": 38,
                        "width": 160,
                        "height": 36,
                        "props": {
                            "text": "• Destaque 1\n• Destaque 2",
                            "font": {"family": "Helvetica", "size": 11, "lineHeight": 14},
                            "color": EFTX_COLOR_TOKENS["text_body"],
                          },
                        "dataBind": "summary.highlights",
                    },
                    {
                        "id": "bands-table",
                        "type": "table",
                        "x": 15,
                        "y": 82,
                        "width": 160,
                        "height": 55,
                        "props": {
                            "headers": ["Faixa", "Frequência", "Ganho", "Polarização"],
                            "rows": [],
                            "headerBg": EFTX_COLOR_TOKENS["primary"],
                            "headerColor": "#ffffff",
                            "fontSize": 9,
                          },
                        "dataBind": "bands",
                    },
                    {
                        "id": "mechanical-table",
                        "type": "table",
                        "x": 15,
                        "y": 142,
                        "width": 160,
                        "height": 45,
                        "props": {
                            "headers": ["Item", "Valor"],
                            "rows": [],
                            "headerBg": EFTX_COLOR_TOKENS["secondary"],
                            "headerColor": "#ffffff",
                            "fontSize": 9,
                          },
                        "dataBind": "mechanical.rows",
                    },
                    {
                        "id": "pattern-slot",
                        "type": "image",
                        "x": 15,
                        "y": 190,
                        "width": 80,
                        "height": 60,
                        "props": {
                            "src": "/IMA/padrao.png",
                            "keepAspect": True,
                          },
                        "dataBind": "patterns[0].azimuth_image",
                    },
                    {
                        "id": "footer",
                        "type": "text",
                        "x": 15,
                        "y": 262,
                        "width": 170,
                        "height": 10,
                        "props": {
                            "text": "EFTX Broadcast & Telecom • www.eftx.com.br",
                            "font": {"family": "Helvetica", "size": 8},
                            "color": EFTX_COLOR_TOKENS["text_body"],
                            "align": "center",
                          },
                      },
                ],
            }
        ],
        "bindings": {},
    }


def _base_design_template(datasheet: EftxDatasheet) -> Dict[str, Any]:
    now = datetime.now(SAO_PAULO).isoformat()
    design_id = f"eftx-template-{datasheet.meta.model or 'datasheet'}".lower().replace(" ", "-")
    return {
        "id": design_id,
        "name": datasheet.summary.title or datasheet.meta.model or "Datasheet EFTX",
        "page": {
            "size": "A4",
            "orientation": "portrait",
            "margins_mm": {"top": 18, "right": 14, "bottom": 18, "left": 14},
            "theme": "dark",
        },
        "grid": {"show": True, "size_mm": 5, "snap": True},
        "guides": {"smart": True, "ruler": True},
        "meta": {
            "generated_at": now,
            "schema": "eftx.datasheet/v1",
        },
        "bindings": {
            "datasets": {
                "bands": {
                    "type": "table",
                    "data": [band.model_dump() for band in datasheet.bands],
                },
                "mechanical": {
                    "type": "table",
                    "data": _mechanical_rows(datasheet),
                },
            },
            "blocks": {
                "bands-table": {
                    "dataset": "bands",
                    "map": {
                        "columns": [
                            {"label": "Faixa", "path": "name"},
                            {"label": "Frequência", "path": "freq_range_mhz", "format": "range:MHz"},
                            {"label": "Ganho", "path": "gain_dbi", "format": "gain"},
                            {"label": "Beamwidth", "path": "beamwidth_deg", "format": "beam"},
                            {"label": "VSWR", "path": "vswr_max", "format": "number"},
                            {"label": "Polarização", "path": "polarization"},
                        ]
                    },
                },
            },
        },
        "pages": [
            {
                "id": "page-1",
                "name": "Página 1",
                "order": 1,
                "layers": [],
                "zoom": 1,
            }
        ],
    }


def _add_layer(page: Dict[str, Any], layer: Dict[str, Any]) -> None:
    page.setdefault("layers", []).append(layer)


def _slots_from_patterns(datasheet: EftxDatasheet) -> List[Dict[str, Any]]:
    slots: List[Dict[str, Any]] = []
    for idx, pattern in enumerate(datasheet.patterns):
        label = f"Padrão Banda {idx + 1}"
        slots.append(
            {
                "id": f"pattern-slot-{idx}",
                "type": "image",
                "x": 15 + (idx % 2) * 95,
                "y": 180 + (idx // 2) * 70,
                "width": 85,
                "height": 65,
                "props": {
                    "src": pattern.azimuth_image or pattern.elevation_image,
                    "keepAspect": True,
                },
                "dataBind": f"patterns[{idx}].azimuth_image",
            }
        )
    return slots


def build_design(datasheet: EftxDatasheet) -> Dict[str, Any]:
    """Generate a layout for the given datasheet using the default template."""

    design = _base_design_template(datasheet)
    page = design["pages"][0]

    summary = datasheet.summary
    _add_layer(
        page,
        {
            "id": "title",
            "type": "text",
            "x": 18,
            "y": 18,
            "width": 160,
            "height": 18,
            "props": {
                "text": summary.title,
                "font": {"family": "Helvetica", "size": 20, "weight": "bold"},
                "color": EFTX_COLOR_TOKENS["text_primary"],
            },
            "dataBind": "summary.title",
        },
    )

    if summary.subtitle:
        _add_layer(
            page,
            {
                "id": "subtitle",
                "type": "text",
                "x": 18,
                "y": 28,
                "width": 160,
                "height": 14,
                "props": {
                    "text": summary.subtitle,
                    "font": {"family": "Helvetica", "size": 14},
                    "color": EFTX_COLOR_TOKENS["accent_light"],
                },
                "dataBind": "summary.subtitle",
            },
        )

    highlights = _build_highlights(datasheet.summary.model_dump())
    if highlights:
        text = "\n".join(f"• {item}" for item in highlights)
        _add_layer(
            page,
            {
                "id": "highlights",
                "type": "text",
                "x": 18,
                "y": 42,
                "width": 160,
                "height": 40,
                "props": {
                    "text": text,
                    "font": {"family": "Helvetica", "size": 11, "lineHeight": 14},
                    "color": EFTX_COLOR_TOKENS["text_body"],
                },
                "dataBind": "summary.highlights",
            },
        )

    # Bands table block
    _add_layer(
        page,
        {
            "id": "bands-table",
            "type": "table",
            "bindingId": "bands-table",
            "x": 18,
            "y": 88,
            "width": 160,
            "height": 60,
            "props": {
                "headers": [
                    "Faixa",
                    "Frequência",
                    "Ganho",
                    "Beamwidth",
                    "VSWR",
                    "Polarização",
                ],
                "rows": _band_rows(datasheet),
                "fontSize": 9,
                "headerBg": EFTX_COLOR_TOKENS["primary"],
                "headerColor": "#ffffff",
            },
        },
    )

    mech_rows = _mechanical_rows(datasheet)
    if mech_rows:
        _add_layer(
            page,
            {
                "id": "mechanical-table",
                "type": "table",
                "x": 18,
                "y": 152,
                "width": 160,
                "height": 45,
                "props": {
                    "headers": ["Item", "Valor"],
                    "rows": mech_rows,
                    "fontSize": 9,
                    "headerBg": EFTX_COLOR_TOKENS["secondary"],
                    "headerColor": "#ffffff",
                },
                "dataBind": "mechanical",
            },
        )

    for slot in _slots_from_patterns(datasheet):
        _add_layer(page, slot)

    footer_text = "EFTX Broadcast & Telecom • www.eftx.com.br • comercial@eftx.com.br"
    _add_layer(
        page,
        {
            "id": "footer",
            "type": "text",
            "x": 18,
            "y": 262,
            "width": 170,
            "height": 12,
            "props": {
                "text": footer_text,
                "font": {"family": "Helvetica", "size": 8},
                "color": EFTX_COLOR_TOKENS["text_body"],
                "align": "center",
            },
            "dataBind": "meta.manufacturer",
        },
    )

    return design


def apply_data_bindings(design: Dict[str, Any], datasheet: EftxDatasheet) -> Dict[str, Any]:
    resolved = deepcopy(design)

    def resolve_formatter(path: str, formatter: str | None) -> str:
        value = _resolve_path(datasheet.model_dump(), path)
        if formatter == "range:MHz" and isinstance(value, dict) and {"min", "max"}.issubset(value):
            return _format_range(RangeFloat(**value), "MHz")
        if formatter == "number" and isinstance(value, (float, int)):
            return _format_value(value, "")
        if formatter == "gain" and isinstance(value, dict):
            parts = []
            typ = value.get("typ")
            mx = value.get("max")
            if typ is not None:
                parts.append(f"typ {typ:g} dBi")
            if mx is not None:
                parts.append(f"max {mx:g} dBi")
            return ", ".join(parts)
        if formatter == "beam" and isinstance(value, dict):
            h = value.get("h", {}).get("typ")
            v = value.get("v", {}).get("typ")
            parts = []
            if h is not None:
                parts.append(f"H {h:g}°")
            if v is not None:
                parts.append(f"V {v:g}°")
            return " / ".join(parts)
        if isinstance(value, (float, int)):
            return _format_value(value)
        if isinstance(value, list):
            return "\n".join(str(item) for item in value)
        return value or ""

    for page in resolved.get("pages", []):
        for layer in page.get("layers", []):
            binding_path = layer.get("dataBind")
            formatter = None
            if binding_path and "|" in binding_path:
                binding_path, formatter = [item.strip() for item in binding_path.split("|", 1)]
            if not binding_path:
                continue
            value = resolve_formatter(binding_path, formatter)
            if layer.get("type") == "text":
                layer.setdefault("props", {})
                layer["props"]["text"] = value
            elif layer.get("type") == "image":
                layer["props"] = {**layer.get("props", {}), "src": value}
            elif layer.get("type") == "table" and isinstance(value, list):
                layer.setdefault("props", {})
                layer["props"]["rows"] = value

    return resolved


def resolve_template(design: Dict[str, Any] | None, datasheet: EftxDatasheet) -> Dict[str, Any]:
    if design:
        return apply_data_bindings(design, datasheet)
    return build_design(datasheet)


def get_assets_library() -> List[Dict[str, Any]]:
    root = Path(current_app.config.get("PROJECT_ROOT")) / "IMA"
    root.mkdir(parents=True, exist_ok=True)
    allowed = {".png", ".jpg", ".jpeg", ".svg", ".webp"}
    assets: List[Dict[str, Any]] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed:
            rel = path.relative_to(root)
            stat = path.stat()
            asset_url = url_for("admin_datasheets.serve_global_asset", filename=rel.as_posix())
            assets.append(
                {
                    "name": path.stem,
                    "path": asset_url,
                    "ext": path.suffix.lower(),
                    "size": stat.st_size,
                    "modified_at": stat.st_mtime,
                }
            )
    assets.sort(key=lambda item: item["name"].lower())
    return assets
SAO_PAULO = ZoneInfo("America/Sao_Paulo")
