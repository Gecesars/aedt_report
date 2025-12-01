# app/services/exporters.py
from __future__ import annotations

import io
import json
import math
import re
import tempfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, LETTER, LEGAL, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib.utils import ImageReader
from reportlab.graphics.barcode import qr as rl_qr
from reportlab.graphics.shapes import Drawing
from reportlab.graphics import renderPDF

import google.generativeai as genai
from flask import current_app

from ..extensions import db
from ..models import Project, ProjectExport, PatternType
from .metrics import (
    directivity_2d_cut,
    estimate_gain_dbi,
    first_null_deg,
    front_to_back_db,
    hpbw_deg,
    lin_to_db,
    lin_to_att_db,
    peak_angle_deg,
    ripple_p2p_db,
    sidelobe_level_db,
)
from .pattern_composer import (
    get_composition,
    resample_pattern,
    resample_vertical,
)

BASE_DIR = Path(__file__).resolve().parents[2]


def resource_path(relative_path: str) -> Path:
    return (BASE_DIR / relative_path).resolve()


PAGE_SIZE_MAP = {
    "A4": A4,
    "LETTER": LETTER,
    "LEGAL": LEGAL,
}


PDF_THEMES = {
    "dark": {
        "title": colors.HexColor("#0A4E8B"),
        "text": colors.HexColor("#111827"),
        "muted": colors.HexColor("#4B5563"),
        "accent": colors.HexColor("#FF7A00"),
        "table_header_bg": colors.HexColor("#1F2A44"),
        "table_header_text": colors.white,
    },
    "light": {
        "title": colors.HexColor("#0B1B33"),
        "text": colors.black,
        "muted": colors.HexColor("#374151"),
        "accent": colors.HexColor("#FF7A00"),
        "table_header_bg": colors.HexColor("#0A4E8B"),
        "table_header_text": colors.white,
    },
}


PDF_FONT_MAP = {
    "Helvetica": ("Helvetica", "Helvetica-Bold"),
    "Source Sans Pro": ("Helvetica", "Helvetica-Bold"),  # fallback if font is not registered
}


SUMMARY_DENSITY_MAP = {
    "compact": {"font_size": 8, "padding": 2},
    "medium": {"font_size": 9, "padding": 3},
    "wide": {"font_size": 10, "padding": 5},
}


DEFAULT_PDF_LIMITS = {
    "min_font_pt": 6,
    "max_table_cols": 8,
    "pages": ["A4", "Letter", "Legal"],
    "orientations": ["portrait", "landscape"],
    "themes": ["dark", "light"],
    "summary_width_pct": {"min": 50, "max": 100},
}


class PDFConfigError(ValueError):
    """Raised when the PDF configuration payload is invalid."""


def _deep_update(target: dict, updates: dict) -> dict:
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{6,}\d")
ADDRESS_KEYWORDS = ("rua", "avenida", "av.", "street", "road", "endereço", "bairro", "cep")


def _strip_contacts(text: str) -> str:
    if not text:
        return ""

    cleaned: list[str] = []
    for original_line in text.splitlines():
        line = original_line.strip()
        if not line:
            cleaned.append("")
            continue

        if any(keyword in line.lower() for keyword in ADDRESS_KEYWORDS):
            continue

        line = EMAIL_RE.sub("", line)
        line = PHONE_RE.sub("", line)

        stripped = line.strip(" ,;|-/")
        if stripped:
            cleaned.append(stripped)

    return "\n".join(cleaned).strip()


def default_pdf_config(project: Project) -> dict:
    return {
        "title": project.name or "Relatório EFTX",
        "page": "A4",
        "orientation": "portrait",
        "margins_mm": {"top": 18, "right": 12, "bottom": 18, "left": 18},
        "theme": "dark",
        "font": "Helvetica",
        "qr": {"enabled": True, "url": "https://eftx.com.br"},
        "footer_corporate": True,
        "sections": {
            "p1_title": True,
            "p1_description": {
                "enabled": True,
                "source": "project",
                "custom_text": "",
                "strip_contacts": True,
            },
            "p1_summary_table": {
                "enabled": True,
                "max_width_pct": 90,
                "density": "compact",
                "auto_split": True,
            },
            "p2_hrp": {
                "enabled": True,
                "plot": "polar",
                "metrics": {"hpbw": True, "sll": True, "fb": True, "dir2d": True, "gain": True},
            },
            "p2_vrp": {
                "enabled": True,
                "plot": "planar",
                "metrics": {"hpbw": True, "null1": True, "ripple": True, "dir2d": True, "horizon": True},
            },
            "p2_comp_vertical": {"enabled": True, "legend": ["N", "Δv", "tilt", "β"], "layout": "side"},
            "p2_comp_horizontal": {"enabled": True, "legend": ["N", "R", "step", "β"], "layout": "side"},
            "tables": {
                "hrp": {"enabled": True, "columns": 6},
                "vrp": {"enabled": True, "columns": 6},
                "repeat_headers": True,
                "respect_footer": True,
                "paginate": True,
            },
        },
        "order": ["page1", "page2", "table_hrp", "table_vrp"],
    }


def pdf_config_limits() -> dict:
    return deepcopy(DEFAULT_PDF_LIMITS)


def resolve_pdf_config(project: Project, overrides: dict | None = None) -> dict:
    config = deepcopy(default_pdf_config(project))
    if overrides:
        _deep_update(config, deepcopy(overrides))
    return _validate_pdf_config(project, config)


def _validate_pdf_config(project: Project, config: dict) -> dict:
    limits = DEFAULT_PDF_LIMITS
    errors: list[str] = []

    title = str(config.get("title") or "").strip()
    if not title:
        title = project.name or "Relatório EFTX"
    config["title"] = title

    page_name = str(config.get("page", "A4")).upper()
    if page_name not in PAGE_SIZE_MAP:
        errors.append("Página inválida")
        page_name = "A4"
    config["page"] = "A4" if page_name == "A4" else page_name.title()

    orientation = str(config.get("orientation", "portrait")).lower()
    if orientation not in limits["orientations"]:
        errors.append("Orientação inválida")
        orientation = "portrait"
    config["orientation"] = orientation

    theme = str(config.get("theme", "dark")).lower()
    if theme not in limits["themes"]:
        errors.append("Tema inválido")
        theme = "dark"
    config["theme"] = theme

    font = config.get("font", "Helvetica")
    if font not in PDF_FONT_MAP:
        font = "Helvetica"
    config["font"] = font

    margins = config.get("margins_mm") or {}
    defaults = default_pdf_config(project)["margins_mm"]
    sanitized_margins = {}
    for side in ("top", "right", "bottom", "left"):
        value = margins.get(side, defaults[side])
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float(defaults[side])
        sanitized_margins[side] = max(numeric, 0.0)
    config["margins_mm"] = sanitized_margins

    qr_conf = config.get("qr") or {}
    qr_enabled = bool(qr_conf.get("enabled", True))
    qr_url = str(qr_conf.get("url") or "https://eftx.com.br").strip()
    config["qr"] = {"enabled": qr_enabled, "url": qr_url or "https://eftx.com.br"}

    config["footer_corporate"] = bool(config.get("footer_corporate", True))

    sections = config.get("sections") or {}
    config["sections"] = sections

    sections.setdefault("p1_title", True)

    # Page 1 description
    desc_conf = sections.get("p1_description") or {}
    desc_enabled = bool(desc_conf.get("enabled", True))
    desc_source = str(desc_conf.get("source", "project")).lower()
    if desc_source not in {"project", "custom"}:
        desc_source = "project"
    desc_text = str(desc_conf.get("custom_text") or "")
    if desc_enabled and desc_source == "custom" and not desc_text.strip():
        errors.append("Descrição customizada requer texto")
    desc_strip = bool(desc_conf.get("strip_contacts", True))
    sections["p1_description"] = {
        "enabled": desc_enabled,
        "source": desc_source,
        "custom_text": desc_text,
        "strip_contacts": desc_strip,
    }

    summary_conf = sections.get("p1_summary_table") or {}
    summary_enabled = bool(summary_conf.get("enabled", True))
    width_pct = summary_conf.get("max_width_pct", 90)
    try:
        width_pct = float(width_pct)
    except (TypeError, ValueError):
        width_pct = 90.0
    width_pct = max(limits["summary_width_pct"]["min"], min(width_pct, limits["summary_width_pct"]["max"]))
    density = str(summary_conf.get("density", "compact")).lower()
    if density not in SUMMARY_DENSITY_MAP:
        density = "compact"
    sections["p1_summary_table"] = {
        "enabled": summary_enabled,
        "max_width_pct": width_pct,
        "density": density,
        "auto_split": bool(summary_conf.get("auto_split", True)),
    }

    def _sanitize_plot_conf(sec_name: str, defaults: dict) -> dict:
        conf = sections.get(sec_name) or {}
        enabled = bool(conf.get("enabled", defaults["enabled"]))
        plot = str(conf.get("plot", defaults["plot"]).lower())
        metrics = {}
        for key, default_value in defaults["metrics"].items():
            metrics[key] = bool(conf.get("metrics", {}).get(key, default_value))
        return {"enabled": enabled, "plot": plot, "metrics": metrics}

    sections["p2_hrp"] = _sanitize_plot_conf(
        "p2_hrp",
        {"enabled": True, "plot": "polar", "metrics": {"hpbw": True, "sll": True, "fb": True, "dir2d": True, "gain": True}},
    )
    sections["p2_vrp"] = _sanitize_plot_conf(
        "p2_vrp",
        {"enabled": True, "plot": "planar", "metrics": {"hpbw": True, "null1": True, "ripple": True, "dir2d": True, "horizon": True}},
    )

    def _sanitize_comp(sec_name: str, defaults: dict) -> dict:
        conf = sections.get(sec_name) or {}
        enabled = bool(conf.get("enabled", defaults["enabled"]))
        legend = conf.get("legend", defaults["legend"])
        if not isinstance(legend, list):
            legend = defaults["legend"]
        layout_mode = str(conf.get("layout", defaults.get("layout", "side"))).lower()
        if layout_mode not in {"side", "stack"}:
            layout_mode = "side"
        return {"enabled": enabled, "legend": legend, "layout": layout_mode}

    sections["p2_comp_vertical"] = _sanitize_comp("p2_comp_vertical", {"enabled": True, "legend": ["N", "Δv", "tilt", "β"], "layout": "side"})
    sections["p2_comp_horizontal"] = _sanitize_comp("p2_comp_horizontal", {"enabled": True, "legend": ["N", "R", "step", "β"], "layout": "side"})

    tables_conf = sections.get("tables") or {}
    hrp_table = tables_conf.get("hrp") or {}
    vrp_table = tables_conf.get("vrp") or {}
    def _sanitize_table(conf: dict, default_cols: int) -> dict:
        enabled = bool(conf.get("enabled", True))
        cols = conf.get("columns", default_cols)
        try:
            cols = int(cols)
        except (TypeError, ValueError):
            cols = default_cols
        cols = max(1, min(cols, limits["max_table_cols"]))
        return {"enabled": enabled, "columns": cols}

    tables = {
        "hrp": _sanitize_table(hrp_table, 6),
        "vrp": _sanitize_table(vrp_table, 6),
        "repeat_headers": bool(tables_conf.get("repeat_headers", True)),
        "respect_footer": bool(tables_conf.get("respect_footer", True)),
        "paginate": bool(tables_conf.get("paginate", True)),
    }
    sections["tables"] = tables

    allowed_sections = ["page1", "page2", "table_hrp", "table_vrp"]
    order = config.get("order") or []
    sanitized_order: list[str] = []
    for item in order:
        if item in allowed_sections and item not in sanitized_order:
            sanitized_order.append(item)
    for default_section in allowed_sections:
        if default_section not in sanitized_order:
            sanitized_order.append(default_section)
    config["order"] = sanitized_order

    # Validate margins vs page size
    base_page = PAGE_SIZE_MAP.get(page_name, A4)
    page_size = landscape(base_page) if orientation == "landscape" else base_page
    width_mm = page_size[0] / mm
    height_mm = page_size[1] / mm
    h_margin = sanitized_margins["left"] + sanitized_margins["right"]
    v_margin = sanitized_margins["top"] + sanitized_margins["bottom"]
    if h_margin >= width_mm:
        errors.append("Margens horizontais excedem a largura da página")
    if v_margin >= height_mm:
        errors.append("Margens verticais excedem a altura da página")

    enabled_sections = []
    if sections.get("p1_title"):
        enabled_sections.append("p1_title")
    if sections.get("p1_description", {}).get("enabled"):
        enabled_sections.append("p1_description")
    if sections.get("p1_summary_table", {}).get("enabled"):
        enabled_sections.append("p1_summary_table")
    if sections.get("p2_hrp", {}).get("enabled") or sections.get("p2_vrp", {}).get("enabled"):
        enabled_sections.append("page2_plots")
    if sections.get("p2_comp_vertical", {}).get("enabled") or sections.get("p2_comp_horizontal", {}).get("enabled"):
        enabled_sections.append("page2_compositions")
    if tables["hrp"]["enabled"]:
        enabled_sections.append("table_hrp")
    if tables["vrp"]["enabled"]:
        enabled_sections.append("table_vrp")

    if not enabled_sections:
        errors.append("Habilite pelo menos uma seção para gerar o PDF")

    if errors:
        raise PDFConfigError("; ".join(errors))

    return config


def _draw_wrapped_text(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    max_width: float,
    line_height: float = 12,
    font_name: str = "Helvetica",
    font_size: int = 10,
) -> float:
    c.setFont(font_name, font_size)
    lines = []
    for paragraph in (text or "").splitlines():
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split()
        cur = ""
        for w in words:
            test = cur + (" " if cur else "") + w
            if c.stringWidth(test, font_name, font_size) <= max_width:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
    for ln in lines:
        c.drawString(x, y, ln)
        y -= line_height
    return y


def _project_summary_fallback(project: Project) -> str:
    antenna_name = project.antenna.name if project.antenna else "—"
    antenna_model = project.antenna.model_number if project.antenna else "—"
    frequency = project.frequency_mhz or 0.0
    tx_power = project.tx_power_w or 0.0
    tower_height = project.tower_height_m or 0.0
    v_spacing = project.v_spacing_m or 0.0
    v_tilt = project.v_tilt_deg or 0.0
    h_spacing = project.h_spacing_m or 0.0
    h_step = project.h_step_deg or 0.0
    cable_model = (project.cable.model_code if project.cable else project.cable_type) or "—"
    cable_length = project.cable_length_m or 0.0
    feeder_loss = project.feeder_loss_db
    if feeder_loss is None:
        feeder_loss = (project.splitter_loss_db or 0.0) + (project.connector_loss_db or 0.0)

    return (
        f"Projeto {project.name}: antena {antenna_name} ({antenna_model}), "
        f"{frequency:.2f} MHz, TX {tx_power:.1f} W, torre {tower_height:.2f} m. "
        f"Vertical: {project.v_count} elem, Δv={v_spacing:.3f} m, tilt={v_tilt:.2f}°. "
        f"Horizontal: {project.h_count} elem, raio={h_spacing:.3f} m, step={h_step:.2f}°. "
        f"Cabo {cable_model} ({cable_length:.2f} m), perdas totais {feeder_loss:.2f} dB."
    )


def _build_gemini_description(project: Project, metrics: dict | None, *, use_ai: bool = True) -> str:
    """Mantido para compatibilidade com views.py. **Não** é usado no PDF."""
    fallback_text = _project_summary_fallback(project)
    if not use_ai:
        return fallback_text
    try:
        genai.configure(api_key=current_app.config.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel(current_app.config.get("GEMINI_MODEL", "gemini-2.5-flash"))
    except Exception:
        model = None

    user = project.owner
    cable = project.cable
    payload = {
        "projeto": project.name,
        "usuario": {
            "nome": getattr(user, "full_name", None),
            "email": getattr(user, "email", None),
            "telefone": getattr(user, "phone", None),
            "endereco": {
                "linha": getattr(user, "address_line", None),
                "cidade": getattr(user, "city", None),
                "estado": getattr(user, "state", None),
                "cep": getattr(user, "postal_code", None),
                "pais": getattr(user, "country", None),
            },
        },
        "antena": {
            "nome": project.antenna.name,
            "modelo": project.antenna.model_number,
            "ganho_nominal_dbd": project.antenna.nominal_gain_dbd,
            "faixa_mhz": [project.antenna.frequency_min_mhz, project.antenna.frequency_max_mhz],
        },
        "sistema": {
            "frequencia_mhz": project.frequency_mhz,
            "potencia_tx_w": project.tx_power_w,
            "altura_torre_m": project.tower_height_m,
        },
        "composicao": {
            "vertical": {
                "elementos": project.v_count,
                "delta_v_m": project.v_spacing_m,
                "tilt_deg": project.v_tilt_deg,
                "beta_deg": project.v_beta_deg,
            },
            "horizontal": {
                "elementos": project.h_count,
                "raio_m": project.h_spacing_m,
                "step_deg": project.h_step_deg,
                "beta_deg": project.h_beta_deg,
            },
        },
        "cabo": {
            "modelo": (cable.model_code if cable else project.cable_type),
            "nome": (cable.display_name if cable else None),
            "fabricante": (cable.manufacturer if cable else None),
            "impedancia_ohms": (cable.impedance_ohms if cable else None),
            "comprimento_m": project.cable_length_m,
            "perda_splitter_db": project.splitter_loss_db,
            "perda_conectores_db": project.connector_loss_db,
            "perda_total_db": project.feeder_loss_db,
        },
        "metricas": metrics or {},
        "observacoes": project.notes,
    }

    prompt = (
        "Você é um engenheiro PhD em RF e antenas. Redija um DESCRITIVO TÉCNICO profissional (pt-BR) "
        "para o projeto a seguir. TOM: objetivo, impessoal, técnico (nível profissional). "
        "Inclua composição vertical/horizontal, frequência, potência e perdas. "
        "RESTRIÇÃO: não invente valores; use apenas os fornecidos.\n\n"
        f"DADOS EM JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )

    if model is None:
        return fallback_text
    try:
        resp = model.generate_content([prompt])
        text = getattr(resp, "text", None) or ""
        if not text and getattr(resp, "candidates", None):
            for cand in resp.candidates:
                parts = getattr(getattr(cand, "content", None), "parts", None)
                if parts:
                    for p in parts:
                        if getattr(p, "text", None):
                            text += p.text
        return (text or "").strip() or fallback_text
    except Exception:
        return fallback_text


def _build_project_description_no_contact(project: Project, metrics: dict | None) -> str:
    """Texto determinístico para o PDF, **sem** dados de contato do usuário."""
    antenna = project.antenna
    antenna_name = antenna.name if antenna and getattr(antenna, "name", None) else "—"
    antenna_model = antenna.model_number if antenna and getattr(antenna, "model_number", None) else "—"
    antenna_freq_min = getattr(antenna, "frequency_min_mhz", None)
    antenna_freq_max = getattr(antenna, "frequency_max_mhz", None)

    freq_min_text = f"{antenna_freq_min:.1f}" if isinstance(antenna_freq_min, (int, float)) else "N/A"
    freq_max_text = f"{antenna_freq_max:.1f}" if isinstance(antenna_freq_max, (int, float)) else "N/A"

    frequency_mhz = project.frequency_mhz if project.frequency_mhz is not None else 0.0
    tx_power = project.tx_power_w if project.tx_power_w is not None else 0.0
    tower_height = project.tower_height_m if project.tower_height_m is not None else 0.0

    v_count = project.v_count or 0
    v_spacing = project.v_spacing_m if project.v_spacing_m is not None else 0.0
    v_tilt = project.v_tilt_deg if project.v_tilt_deg is not None else 0.0
    v_beta = project.v_beta_deg if project.v_beta_deg is not None else 0.0

    h_count = project.h_count or 0
    h_radius = project.h_spacing_m if project.h_spacing_m is not None else 0.0
    h_step = project.h_step_deg if project.h_step_deg is not None else 0.0
    h_beta = project.h_beta_deg if project.h_beta_deg is not None else 0.0

    lines = []
    lines.append(
        f"Projeto {project.name}: antena {antenna_name} ({antenna_model}), "
        f"faixa {freq_min_text}–{freq_max_text} MHz."
    )
    lines.append(
        f"Operação em {frequency_mhz:.2f} MHz com potência de {tx_power:.1f} W e "
        f"altura de instalação de {tower_height:.2f} m."
    )
    lines.append(
        f"Composição vertical: {v_count} elementos, Δv={v_spacing:.3f} m, "
        f"tilt={v_tilt:.2f}°, β={v_beta:.2f}°."
    )
    lines.append(
        f"Composição horizontal: {h_count} elementos, raio={h_radius:.3f} m, "
        f"step={h_step:.2f}°, β={h_beta:.2f}°."
    )
    cable_name = project.cable.display_name if project.cable else (project.cable_type or "—")
    cable_length = project.cable_length_m if project.cable_length_m is not None else 0.0
    splitter_loss = project.splitter_loss_db if project.splitter_loss_db is not None else 0.0
    connector_loss = project.connector_loss_db if project.connector_loss_db is not None else 0.0
    feeder_loss = project.feeder_loss_db if project.feeder_loss_db is not None else 0.0
    lines.append(
        f"Alimentação: cabo {cable_name}, comprimento {cable_length:.2f} m; perdas adicionais "
        f"({splitter_loss:.2f} dB splitters, {connector_loss:.2f} dB conectores); "
        f"perda total estimada {feeder_loss:.2f} dB."
    )
    if metrics:
        hpbw_h = metrics.get("hrp_hpbw")
        hpbw_v = metrics.get("vrp_hpbw")
        front_back = metrics.get("front_to_back")
        ripple_db = metrics.get("ripple_db")
        sll_db = metrics.get("sll_db")
        dir_db = metrics.get("directivity_db")
        gain = metrics.get("gain_dbi")

        if hpbw_h is not None and hpbw_v is not None and gain is not None:
            metrics_line = (
                "Métricas do arranjo composto: "
                f"HPBW(H)={_format_value(hpbw_h, '°')}, "
                f"HPBW(V)={_format_value(hpbw_v, '°')}, "
                f"F/B={_format_value(front_back, ' dB')}, "
                f"Ripple={_format_value(ripple_db, ' dB')}, "
                f"SLL={_format_value(sll_db, ' dB')}, "
                f"Diretividade 2D={_format_value(dir_db, ' dB')}, "
                f"Ganho estimado={_format_value(gain, ' dBi')}"
            )
        else:
            metrics_line = "Métricas do arranjo composto disponíveis na seção de diagramas."
        lines.append(metrics_line)
    if project.notes:
        lines.append(f"Observações: {project.notes}")
    return "\n".join(lines)


def _project_summary_table_data(project: Project) -> list[list[str]]:
    left_rows = [
        ("Projeto", project.name),
        ("Antena", project.antenna.name),
        ("Modelo", project.antenna.model_number or "—"),
        ("Frequência", f"{project.frequency_mhz:.2f} MHz"),
        ("Potência TX", f"{project.tx_power_w:.1f} W"),
        ("Altura torre", f"{project.tower_height_m:.2f} m"),
        ("VSWR alvo", f"{project.vswr_target or 0:.2f}"),
    ]
    right_rows = [
        ("Elementos V", f"{project.v_count}"),
        ("Δv (m)", f"{project.v_spacing_m:.3f}"),
        ("Tilt V (°)", f"{project.v_tilt_deg or 0:.2f}"),
        ("Beta V (°)", f"{project.v_beta_deg or 0:.2f}"),
        ("Nível V", f"{project.v_level_amp or 0:.2f}"),
        ("Elementos H", f"{project.h_count}"),
        ("Raio arranjo (m)", f"{project.h_spacing_m:.3f}"),
        ("Step H (°)", f"{project.h_step_deg or 0:.2f}"),
        ("Beta H (°)", f"{project.h_beta_deg or 0:.2f}"),
        ("Nível H", f"{project.h_level_amp or 0:.2f}"),
    ]

    cable = project.cable
    if cable is not None:
        att_curve = getattr(cable, "attenuation_db_per_100m_curve", None)
        att_at_f = _interp_dict_num(att_curve, float(project.frequency_mhz)) if isinstance(att_curve, dict) else None
        cable_lines = [
            ("Cabo", cable.display_name or cable.model_code or "—"),
            ("Modelo", cable.model_code or "—"),
            ("Bitola", cable.size_inch or "—"),
            ("Impedância (Ω)", f"{cable.impedance_ohms or 50}"),
            ("Fabricante", cable.manufacturer or "—"),
            ("Comprimento (m)", f"{project.cable_length_m:.2f}"),
            ("Splitters (dB)", f"{project.splitter_loss_db:.2f}"),
            ("Conectores (dB)", f"{project.connector_loss_db:.2f}"),
            ("Perda total (dB)", f"{project.feeder_loss_db or 0:.2f}"),
            ("Aten. curva @f (dB/100m)", f"{att_at_f:.2f}" if att_at_f is not None else "—"),
        ]
    else:
        cable_lines = [
            ("Cabo", project.cable_type or "—"),
            ("Comprimento (m)", f"{(project.cable_length_m or 0.0):.2f}"),
            ("Splitters (dB)", f"{(project.splitter_loss_db or 0.0):.2f}"),
            ("Conectores (dB)", f"{(project.connector_loss_db or 0.0):.2f}"),
            ("Perda total (dB)", f"{(project.feeder_loss_db or 0.0):.2f}"),
        ]

    max_len = max(len(left_rows), len(right_rows), len(cable_lines))
    while len(left_rows) < max_len:
        left_rows.append(("", ""))
    while len(right_rows) < max_len:
        right_rows.append(("", ""))
    while len(cable_lines) < max_len:
        cable_lines.append(("", ""))

    header = ["Projeto", "Valor", "Composição", "Valor", "Cabo", "Valor"]
    table_data = [header]
    for i in range(max_len):
        table_data.append([
            left_rows[i][0], left_rows[i][1],
            right_rows[i][0], right_rows[i][1],
            cable_lines[i][0], cable_lines[i][1],
        ])
    return table_data
def resolve_antenna_image(antenna) -> Path | None:
    docs_dir = resource_path("docs")
    if not docs_dir.exists():
        return None

    name_tokens = re.split(r"[^a-z0-9]+", (antenna.model_number or antenna.name or "").lower())
    name_tokens = [token for token in name_tokens if token]
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}

    best_match = None
    for path in docs_dir.rglob("*"):
        if path.suffix.lower() not in image_exts:
            continue
        stem = path.stem.lower()
        if name_tokens and all(token in stem for token in name_tokens):
            best_match = path
            break
        if best_match is None:
            best_match = path
    return best_match


class ExportPaths:
    def __init__(self, root: Path, project: Project) -> None:
        timestamp = datetime.now(SAO_PAULO).strftime("%Y%m%d_%H%M%S")
        self.base_dir = root / str(project.id) / timestamp
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.pat = self.base_dir / "pattern.pat"
        self.prn = self.base_dir / "pattern.prn"
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", project.name).strip("-") or "relatorio"
        self.pdf = self.base_dir / f"{safe_name}.pdf"
        self.composition_vertical = self.base_dir / "composicao_vertical.png"
        self.composition_horizontal = self.base_dir / "composicao_horizontal.png"


def _save_vertical_composition(path: Path, v_count: int, v_spacing_m: float, v_tilt_deg: float) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots(figsize=(4.0, 5.0))
    ax.set_aspect('equal'); ax.axis('off')
    count = max(int(v_count or 1), 1)
    spacing = float(v_spacing_m or 0.0)
    ymin, ymax = 0.0, 8.0
    usable = max(ymax - ymin, 1e-6)
    pitch = usable / max(count - 1, 1)
    rect_h = 0.9; rect_w = 2.4; x0 = -rect_w/2
    for i in range(count):
        y = ymin + i * pitch
        rect = patches.Rectangle((x0, y - rect_h/2), rect_w, rect_h, linewidth=1,
                                 edgecolor='#cc0000', facecolor='#ffcccc')
        ax.add_patch(rect)
    if count >= 2:
        c0 = ymin; c1 = ymin + pitch
        ax.annotate('', xy=(rect_w/2 + 0.6, c0), xytext=(rect_w/2 + 0.6, c1),
                    arrowprops=dict(arrowstyle='<->', color='#444'))
        ax.text(rect_w/2 + 0.85, (c0+c1)/2, f"Δv = {spacing:.3f} m", rotation=90, va='center', fontsize=9)
    tilt = float(v_tilt_deg or 0.0)
    cy = ymin + (max(count - 1, 0) * pitch) / 2
    ax.annotate('', xy=(0.0 + 2.0*np.cos(-np.deg2rad(tilt)), cy + 2.0*np.sin(-np.deg2rad(tilt))),
                xytext=(0.0, cy), arrowprops=dict(arrowstyle='->', linewidth=2, color='#ff7a00'))
    ax.text(0.2, cy + 0.2, f"tilt {tilt:.1f}°", color='#ff7a00')
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(ymin - 0.8, ymax + 0.8)
    fig.savefig(path, dpi=220, bbox_inches='tight'); plt.close(fig)


def _save_horizontal_composition(path: Path, h_count: int, h_spacing_m: float, h_step_deg: float) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as patches
    from matplotlib.transforms import Affine2D
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.set_aspect('equal'); ax.axis('off')
    n = max(int(h_count or 1), 1)
    Rm = float(h_spacing_m or 0.0)
    step = float(h_step_deg or 0.0)
    R = 0.8 + 1.6 * (Rm / (Rm + 1.0))
    circ = patches.Circle((0,0), R, fill=False, linestyle='--', color='#8a8a8a')
    ax.add_patch(circ)
    ax.text(-0.2, R + 0.25, f"R = {Rm:.3f} m", fontsize=9, color='#555')
    el_w, el_h = 0.35, 0.25
    for i in range(n):
        ang_deg = i * (360.0 / n) + step
        ang = np.deg2rad(ang_deg)
        xc = (R + el_w/2) * np.cos(ang)
        yc = (R + el_w/2) * np.sin(ang)
        rect = patches.Rectangle((-el_w/2, -el_h/2), el_w, el_h,
                                 linewidth=1, edgecolor='#cc0000', facecolor='#ffcccc')
        t = Affine2D().rotate(ang).translate(xc, yc) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
        lx = (R + el_w + 0.35) * np.cos(ang)
        ly = (R + el_w + 0.35) * np.sin(ang)
        ax.text(lx, ly, f"{(ang_deg)%360:.0f}°", ha='center', va='center', fontsize=8)
    ax.set_xlim(-3.0, 3.0); ax.set_ylim(-3.0, 3.0)
    fig.savefig(path, dpi=220, bbox_inches='tight'); plt.close(fig)


def _format_value(value: float, suffix: str = "") -> str:
    if value is None or not np.isfinite(value):
        return f"N/A{suffix}"
    return f"{value:.2f}{suffix}"


def _safe_float(value) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def angles_to_full_circle(angles_deg: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base_angles = (angles_deg + 360.0) % 360.0
    order = np.argsort(base_angles)
    base_angles = base_angles[order]
    base_values = values[order]
    extended_angles = np.concatenate([base_angles, base_angles[:1] + 360.0])
    extended_values = np.concatenate([base_values, base_values[:1]])
    target = np.arange(0, 360, 1, dtype=float)
    interp_values = np.interp(target, extended_angles, extended_values)
    return target, interp_values


def vertical_to_full_circle(angles_deg: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted = angles_deg + 90.0
    base = np.clip(shifted, 0, 180)
    target = np.arange(0, 181, 1, dtype=float)
    interp = np.interp(target, base, values, left=values[0], right=values[-1])
    mirrored = interp[1:-1][::-1]
    angles = np.concatenate([target, np.arange(181, 360, 1, dtype=float)])
    amp = np.concatenate([interp, mirrored])
    return angles, amp


def _interp_dict_num(d: dict, x: float) -> float | None:
    try:
        items = sorted((float(k), float(v)) for k, v in d.items() if v is not None)
    except Exception:
        return None
    if not items:
        return None
    if len(items) == 1:
        return items[0][1]
    if x <= items[0][0]:
        x0, y0 = items[0]; x1, y1 = items[1]
        if x1 == x0: return y0
        t = (x - x0) / (x1 - x0); return y0 + t * (y1 - y0)
    if x >= items[-1][0]:
        x0, y0 = items[-2]; x1, y1 = items[-1]
        if x1 == x0: return y1
        t = (x - x0) / (x1 - x0); return y0 + t * (y1 - y0)
    for i in range(len(items) - 1):
        x0, y0 = items[i]; x1, y1 = items[i + 1]
        if x0 <= x <= x1:
            if x1 == x0: return y0
            t = (x - x0) / (x1 - x0); return y0 + t * (y1 - y0)
    return None


def _save_polar_plot(path: Path, angles: np.ndarray, values: np.ndarray, title: str) -> None:
    angles_mod = (angles + 360.0) % 360.0
    order = np.argsort(angles_mod)
    theta_sorted = angles_mod[order]
    values_sorted = values[order]
    theta_wrapped = np.concatenate([theta_sorted, [theta_sorted[0] + 360.0]])
    values_wrapped = np.concatenate([values_sorted, [values_sorted[0]]])
    theta_rad = np.deg2rad(theta_wrapped)

    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(theta_rad, values_wrapped, linewidth=1.6, color="#0A4E8B")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_planar_plot(
    path: Path,
    angles: np.ndarray,
    values: np.ndarray,
    title: str,
    highlight: tuple[float, float] | None = None,
) -> None:
    order = np.argsort(angles)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(angles[order], values[order], linewidth=1.6, color="#0A4E8B")
    ax.set_xlabel("Ângulo (deg)")
    ax.set_ylabel("Amplitude (linear)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if highlight is not None:
        angle, amplitude = highlight
        ax.scatter([angle], [amplitude], color="#D35400", s=40, zorder=5)
        ax.axvline(angle, color="#D35400", linestyle="--", linewidth=1, alpha=0.6)
        ax.axhline(amplitude, color="#D35400", linestyle="--", linewidth=1, alpha=0.6)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_table_chunks(
    angles: np.ndarray,
    values: np.ndarray,
    width: float,
    columns: int = 6,
    header_bg: colors.Color | None = None,
    header_text: colors.Color | None = None,
    header_font: str = "Helvetica-Bold",
    body_font: str = "Helvetica",
    body_font_size: int = 6,
    height_limit: float | None = None,
) -> Sequence[Table]:
    if len(angles) == 0:
        return []
    db_vals = lin_to_db(values)
    rows = [(f"{a:.1f}", f"{v:.4f}", f"{d:.2f}") for a, v, d in zip(angles, values, db_vals)]
    rows_per_col = math.ceil(len(rows) / columns)
    header = ["Ângulo", "Amplitude", "dB"]
    table_data = [header * columns]
    for i in range(rows_per_col):
        row_cells = []
        for c in range(columns):
            idx = i + c * rows_per_col
            if idx < len(rows):
                row_cells.extend(rows[idx])
            else:
                row_cells.extend(["", "", ""])
        table_data.append(row_cells)

    # colunas estreitas para caber no miolo
    base_ratios = [1.0, 1.0, 0.8]  # Ângulo, Amplitude, dB
    total_ratio = sum(base_ratios) * columns
    unit = width / total_ratio
    col_widths = [unit * r for r in base_ratios] * columns

    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    hdr_bg = header_bg or colors.HexColor("#1f2a44")
    hdr_text = header_text or colors.white
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.25, colors.gray),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 1),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("FONTNAME", (0, 0), (-1, 0), header_font),
        ("FONTNAME", (0, 1), (-1, -1), body_font),
        ("FONTSIZE", (0, 0), (-1, -1), body_font_size),
        ("BACKGROUND", (0, 0), (-1, 0), hdr_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), hdr_text),
    ]))
    usable_height = height_limit if height_limit is not None else (A4[1] - 60 * mm)
    return table.split(width, usable_height)


def _prepare_raw_pattern(project: Project, pattern_type: PatternType) -> tuple[np.ndarray, np.ndarray]:
    pattern = project.antenna.pattern_for(pattern_type)
    if not pattern:
        if pattern_type is PatternType.HRP:
            angles = np.arange(-180, 181, 1, dtype=float)
        else:
            angles = np.round(np.arange(-90.0, 90.0 + 0.0001, 0.1), 1)
        return angles, np.ones_like(angles, dtype=float)
    angles = np.asarray(pattern.angles_deg, dtype=float)
    values = np.asarray(pattern.amplitudes_linear, dtype=float)
    values = np.clip(values, 0.0, None)
    if pattern_type is PatternType.HRP:
        return resample_pattern(angles, values, -180, 180, 1)
    return resample_vertical(angles, values)


def _compute_export_metrics(
    project: Project,
    hrp_angles: np.ndarray,
    hrp_values: np.ndarray,
    vrp_angles: np.ndarray,
    vrp_values: np.ndarray,
) -> tuple[dict, float]:
    metrics: dict[str, float | None] = {
        "hrp_hpbw": _safe_float(hpbw_deg(hrp_angles, hrp_values)),
        "vrp_hpbw": _safe_float(hpbw_deg(vrp_angles, vrp_values)),
        "front_to_back": _safe_float(front_to_back_db(hrp_angles, hrp_values)),
        "ripple_db": _safe_float(ripple_p2p_db(hrp_angles, hrp_values)),
        "sll_db": _safe_float(sidelobe_level_db(hrp_angles, hrp_values)),
        "peak_angle": _safe_float(peak_angle_deg(hrp_angles, hrp_values)),
    }

    gain_est = _safe_float(estimate_gain_dbi(metrics["hrp_hpbw"], metrics["vrp_hpbw"])) if metrics["hrp_hpbw"] is not None and metrics["vrp_hpbw"] is not None else None
    metrics["gain_dbi"] = gain_est

    hrp_dir = directivity_2d_cut(hrp_angles, hrp_values) if hrp_values.size else float("nan")
    vrp_dir = directivity_2d_cut(vrp_angles, vrp_values) if vrp_values.size else float("nan")
    metrics["hrp_dir2d"] = _safe_float(10 * np.log10(hrp_dir) if np.isfinite(hrp_dir) and hrp_dir > 0 else float("nan"))
    metrics["vrp_dir2d"] = _safe_float(10 * np.log10(vrp_dir) if np.isfinite(vrp_dir) and vrp_dir > 0 else float("nan"))

    metrics["vrp_ripple_db"] = _safe_float(ripple_p2p_db(vrp_angles, vrp_values))
    metrics["vrp_first_null"] = _safe_float(first_null_deg(vrp_angles, vrp_values))

    horizon_idx = int(np.argmin(np.abs(vrp_angles))) if vrp_angles.size else 0
    horizon_value = float(vrp_values[horizon_idx]) if vrp_values.size else 0.0
    metrics["vrp_horizon"] = _safe_float(horizon_value)

    # Calibração de ganho por tabela (dBd -> dBi)
    try:
        base_nominal_dbi = float((project.antenna.nominal_gain_dbd or 0.0) + 2.15)
    except Exception:
        base_nominal_dbi = 0.0
    table = getattr(project.antenna, "gain_table", None)
    if isinstance(table, dict):
        table_dbd = _interp_dict_num(table, float(project.frequency_mhz))
        if table_dbd is not None:
            table_dbi = float(table_dbd) + 2.15
            if metrics["gain_dbi"] is not None:
                metrics["gain_dbi"] = float(metrics["gain_dbi"]) + (table_dbi - base_nominal_dbi)
            else:
                metrics["gain_dbi"] = table_dbi

    return metrics, horizon_value


def prepare_project_export_data(project: Project) -> dict:
    composition_arrays, composition_payload = get_composition(project, refresh=True)

    hrp_angles = np.asarray(composition_arrays.get("angles_deg", []), dtype=float)
    hrp_values = np.asarray(composition_arrays.get("hrp_linear", []), dtype=float)
    vrp_angles = np.asarray(composition_arrays.get("vrp_angles_deg", []), dtype=float)
    vrp_values = np.asarray(composition_arrays.get("vrp_linear", []), dtype=float)

    metrics, horizon_value = _compute_export_metrics(project, hrp_angles, hrp_values, vrp_angles, vrp_values)

    ang_full_hrp, val_full_hrp = angles_to_full_circle(hrp_angles, hrp_values)
    ang_full_vrp, val_full_vrp = vertical_to_full_circle(vrp_angles, vrp_values)

    raw_hrp_angles, raw_hrp_values = _prepare_raw_pattern(project, PatternType.HRP)
    raw_vrp_angles, raw_vrp_values = _prepare_raw_pattern(project, PatternType.VRP)

    return {
        "composition_arrays": composition_arrays,
        "composition_payload": composition_payload,
        "hrp_angles": hrp_angles,
        "hrp_values": hrp_values,
        "vrp_angles": vrp_angles,
        "vrp_values": vrp_values,
        "ang_full_hrp": ang_full_hrp,
        "val_full_hrp": val_full_hrp,
        "ang_full_vrp": ang_full_vrp,
        "val_full_vrp": val_full_vrp,
        "metrics": metrics,
        "horizon_value": horizon_value,
        "raw_hrp_angles": raw_hrp_angles,
        "raw_hrp_values": raw_hrp_values,
        "raw_vrp_angles": raw_vrp_angles,
        "raw_vrp_values": raw_vrp_values,
        "antenna_image": resolve_antenna_image(project.antenna),
    }


def create_pdf_report_with_config(
    project: Project,
    config: dict | None,
    composition_arrays: dict,
    metrics: dict | None,
    export_paths: ExportPaths | None = None,
    output_path: Path | None = None,
    prepared: dict | None = None,
) -> bytes:
    resolved_config = resolve_pdf_config(project, config or {})

    if prepared is None:
        hrp_angles = np.asarray(composition_arrays.get("angles_deg", []), dtype=float)
        hrp_values = np.asarray(composition_arrays.get("hrp_linear", []), dtype=float)
        vrp_angles = np.asarray(composition_arrays.get("vrp_angles_deg", []), dtype=float)
        vrp_values = np.asarray(composition_arrays.get("vrp_linear", []), dtype=float)
        computed_metrics, horizon_value = _compute_export_metrics(project, hrp_angles, hrp_values, vrp_angles, vrp_values)
        raw_hrp_angles, raw_hrp_values = _prepare_raw_pattern(project, PatternType.HRP)
        raw_vrp_angles, raw_vrp_values = _prepare_raw_pattern(project, PatternType.VRP)
        antenna_img = resolve_antenna_image(project.antenna)
    else:
        hrp_angles = prepared.get("hrp_angles", np.array([]))
        hrp_values = prepared.get("hrp_values", np.array([]))
        vrp_angles = prepared.get("vrp_angles", np.array([]))
        vrp_values = prepared.get("vrp_values", np.array([]))
        computed_metrics = dict(prepared.get("metrics", {}))
        horizon_value = float(prepared.get("horizon_value", 0.0) or 0.0)
        raw_hrp_angles = prepared.get("raw_hrp_angles", np.array([]))
        raw_hrp_values = prepared.get("raw_hrp_values", np.array([]))
        raw_vrp_angles = prepared.get("raw_vrp_angles", np.array([]))
        raw_vrp_values = prepared.get("raw_vrp_values", np.array([]))
        antenna_img = prepared.get("antenna_image") or resolve_antenna_image(project.antenna)

    merged_metrics = dict(computed_metrics)
    if metrics:
        for key, value in metrics.items():
            if value is not None:
                merged_metrics[key] = value

    pdf_bytes = _create_pdf_report(
        project,
        export_paths,
        hrp_angles,
        hrp_values,
        vrp_angles,
        vrp_values,
        raw_hrp_angles,
        raw_hrp_values,
        raw_vrp_angles,
        raw_vrp_values,
        merged_metrics,
        merged_metrics.get("vrp_horizon", horizon_value) or horizon_value,
        antenna_img,
        layout=resolved_config,
        output_path=output_path,
    )
    return pdf_bytes
def _diagram_metrics_block(
    cut: str,
    angles: np.ndarray,
    values: np.ndarray,
    include_fb: bool = False,
) -> list[str]:
    fb = front_to_back_db(angles, values) if include_fb else None
    lines = []
    lines.append(f"{cut}")
    lines.append(f"- HPBW: {_format_value(hpbw_deg(angles, values), ' deg')}")
    if include_fb:
        lines.append(f"- Front/Back: {_format_value(fb, ' dB')}")
    lines.append(f"- Ripple p2p: {_format_value(ripple_p2p_db(angles, values), ' dB')}")
    lines.append(f"- SLL: {_format_value(sidelobe_level_db(angles, values), ' dB')}")
    d2d = directivity_2d_cut(angles, values)
    d2d_db = 10 * np.log10(d2d) if np.isfinite(d2d) and d2d > 0 else float("nan")
    lines.append(f"- Diretividade 2D: {_format_value(d2d_db, ' dB')}")
    lines.append(f"- Pico @: {_format_value(peak_angle_deg(angles, values), ' deg')}")
    return lines



def _create_pdf_report(
    project: Project,
    paths: ExportPaths | None,
    hrp_angles: np.ndarray,
    hrp_values: np.ndarray,
    vrp_angles: np.ndarray,
    vrp_values: np.ndarray,
    raw_hrp_angles: np.ndarray,
    raw_hrp_values: np.ndarray,
    raw_vrp_angles: np.ndarray,
    raw_vrp_values: np.ndarray,
    array_metrics: dict,
    horizon_value: float,
    antenna_image: Path | None,
    layout: dict | None = None,
    output_path: Path | None = None,
) -> bytes:
    config = layout or default_pdf_config(project)
    theme = PDF_THEMES.get(config.get("theme", "dark"), PDF_THEMES["dark"])
    font_body, font_bold = PDF_FONT_MAP.get(config.get("font", "Helvetica"), ("Helvetica", "Helvetica-Bold"))

    page_key = str(config.get("page", "A4")).upper()
    base_page = PAGE_SIZE_MAP.get(page_key, A4)
    orientation = str(config.get("orientation", "portrait")).lower()
    page_size = landscape(base_page) if orientation == "landscape" else base_page

    margins = config.get("margins_mm", {})
    margin_top = float(margins.get("top", 18)) * mm
    margin_right = float(margins.get("right", 12)) * mm
    margin_bottom = float(margins.get("bottom", 18)) * mm
    margin_left = float(margins.get("left", 18)) * mm

    width, height = page_size
    available_width = width - margin_left - margin_right
    gutter = 10 * mm

    sections = config.get("sections", {})
    modelo_path = resource_path("modelo.pdf")
    template_exists = bool(config.get("footer_corporate", True)) and modelo_path.exists()

    hrp_full_angles, hrp_full_values = angles_to_full_circle(hrp_angles, hrp_values)
    vrp_full_angles, vrp_full_values = vertical_to_full_circle(vrp_angles, vrp_values)

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=page_size)

    tmp_dir = tempfile.TemporaryDirectory(prefix="eftx_pdf_")
    try:
        tmp_path = Path(tmp_dir.name)

        # Preparar ilustrações de composição
        if paths is not None:
            comp_h_path = Path(paths.composition_horizontal)
            comp_v_path = Path(paths.composition_vertical)
            comp_h_path.parent.mkdir(parents=True, exist_ok=True)
            comp_v_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            comp_h_path = tmp_path / "composition_horizontal.png"
            comp_v_path = tmp_path / "composition_vertical.png"

        _save_horizontal_composition(
            comp_h_path,
            int(project.h_count or 1),
            float(project.h_spacing_m or 0.0),
            float(project.h_step_deg or 0.0),
        )
        _save_vertical_composition(
            comp_v_path,
            int(project.v_count or 1),
            float(project.v_spacing_m or 0.0),
            float(project.v_tilt_deg or 0.0),
        )

        # Preparar diagramas HRP/VRP
        hrp_img_path: Path | None = None
        if sections.get("p2_hrp", {}).get("enabled"):
            hrp_img_path = tmp_path / "hrp_plot.png"
            plot_type = sections["p2_hrp"].get("plot", "polar")
            if plot_type == "planar":
                _save_planar_plot(hrp_img_path, hrp_angles, hrp_values, "Padrão Horizontal Composto (HRP)")
            else:
                _save_polar_plot(hrp_img_path, hrp_angles, hrp_values, "Padrão Horizontal Composto (HRP)")

        vrp_img_path: Path | None = None
        if sections.get("p2_vrp", {}).get("enabled"):
            vrp_img_path = tmp_path / "vrp_plot.png"
            _save_planar_plot(
                vrp_img_path,
                vrp_angles,
                vrp_values,
                "Padrão Vertical Composto (VRP)",
                highlight=(0.0, float(horizon_value or 0.0)),
            )

        def _build_plot_metrics_lines(plot_key: str, flags: dict) -> list[str]:
            lines: list[str] = []
            if plot_key == "hrp":
                if flags.get("hpbw"):
                    lines.append(f"HPBW: {_format_value(array_metrics.get('hrp_hpbw'), ' °')}")
                if flags.get("sll"):
                    lines.append(f"SLL: {_format_value(array_metrics.get('sll_db'), ' dB')}")
                if flags.get("fb"):
                    lines.append(f"F/B: {_format_value(array_metrics.get('front_to_back'), ' dB')}")
                if flags.get("dir2d"):
                    lines.append(f"Diretividade 2D: {_format_value(array_metrics.get('hrp_dir2d'), ' dB')}")
                if flags.get("gain"):
                    lines.append(f"Ganho estimado: {_format_value(array_metrics.get('gain_dbi'), ' dBi')}")
            else:
                if flags.get("hpbw"):
                    lines.append(f"HPBW: {_format_value(array_metrics.get('vrp_hpbw'), ' °')}")
                if flags.get("null1"):
                    lines.append(f"1º nulo: {_format_value(array_metrics.get('vrp_first_null'), ' °')}")
                if flags.get("ripple"):
                    lines.append(f"Ripple p2p: {_format_value(array_metrics.get('vrp_ripple_db'), ' dB')}")
                if flags.get("dir2d"):
                    lines.append(f"Diretividade 2D: {_format_value(array_metrics.get('vrp_dir2d'), ' dB')}")
                if flags.get("horizon"):
                    lines.append(f"E/Emax @ 0°: {_format_value(array_metrics.get('vrp_horizon'), '')}")
            return lines

        def _composition_lines(kind: str, legend: list[str]) -> list[str]:
            if kind == "horizontal":
                mapping = {
                    "N": f"Elementos H: {project.h_count}",
                    "R": f"Raio (m): {project.h_spacing_m:.3f}",
                    "step": f"Step H (°): {project.h_step_deg or 0:.2f}",
                    "β": f"Beta H (°): {project.h_beta_deg or 0:.2f}",
                    "amp": f"Nível H: {project.h_level_amp or 0:.2f}",
                }
            else:
                mapping = {
                    "N": f"Elementos V: {project.v_count}",
                    "Δv": f"Δv (m): {project.v_spacing_m:.3f}",
                    "tilt": f"Tilt V (°): {project.v_tilt_deg or 0:.2f}",
                    "β": f"Beta V (°): {project.v_beta_deg or 0:.2f}",
                    "amp": f"Nível V: {project.v_level_amp or 0:.2f}",
                }
            lines = []
            for token in legend:
                value = mapping.get(token)
                if value:
                    lines.append(value)
            return lines

        desc_conf = sections.get("p1_description", {})
        summary_conf = sections.get("p1_summary_table", {})
        page1_active = bool(sections.get("p1_title")) or desc_conf.get("enabled") or summary_conf.get("enabled")
        page2_active = (
            sections.get("p2_hrp", {}).get("enabled")
            or sections.get("p2_vrp", {}).get("enabled")
            or sections.get("p2_comp_horizontal", {}).get("enabled")
            or sections.get("p2_comp_vertical", {}).get("enabled")
        )
        table_hrp_conf = sections.get("tables", {}).get("hrp", {})
        table_vrp_conf = sections.get("tables", {}).get("vrp", {})
        table_hrp_active = bool(table_hrp_conf.get("enabled")) and hrp_full_angles.size > 0
        table_vrp_active = bool(table_vrp_conf.get("enabled")) and vrp_full_angles.size > 0

        active_map = {
            "page1": page1_active,
            "page2": page2_active,
            "table_hrp": table_hrp_active,
            "table_vrp": table_vrp_active,
        }
        raw_order = config.get("order", ["page1", "page2", "table_hrp", "table_vrp"])
        ordered_sections = [key for key in raw_order if active_map.get(key)]
        if not ordered_sections:
            ordered_sections = [key for key, value in active_map.items() if value]

        def render_page1() -> None:
            y = height - margin_top
            if sections.get("p1_title") and config.get("title"):
                c.setFillColor(theme["title"])
                c.setFont(font_bold, 20)
                c.drawCentredString((margin_left + width - margin_right) / 2, y - 6, config["title"])
                y -= 36
            else:
                y -= 12

            if desc_conf.get("enabled"):
                c.setFillColor(theme["text"])
                c.setFont(font_bold, 12)
                c.drawString(margin_left, y, "Descritivo técnico")
                y -= 18
                if desc_conf.get("source") == "custom":
                    desc_text = desc_conf.get("custom_text", "")
                    if desc_conf.get("strip_contacts", True):
                        desc_text = _strip_contacts(desc_text)
                else:
                    desc_text = _build_project_description_no_contact(project, array_metrics)
                    if desc_conf.get("strip_contacts", True):
                        desc_text = _strip_contacts(desc_text)
                c.setFillColor(theme["text"])
                y = _draw_wrapped_text(
                    c,
                    desc_text,
                    margin_left,
                    y,
                    available_width,
                    line_height=12,
                    font_name=font_body,
                    font_size=10,
                ) - 12
            else:
                y -= 6

            if summary_conf.get("enabled"):
                density_opts = SUMMARY_DENSITY_MAP.get(summary_conf.get("density", "compact"), SUMMARY_DENSITY_MAP["compact"])
                table_width = available_width * (float(summary_conf.get("max_width_pct", 90)) / 100.0)
                data = _project_summary_table_data(project)
                col_count = len(data[0])
                table = Table(data, colWidths=[table_width / col_count] * col_count)
                table.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.gray),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), font_bold),
                    ("FONTNAME", (0, 1), (-1, -1), font_body),
                    ("FONTSIZE", (0, 0), (-1, -1), density_opts["font_size"]),
                    ("LEFTPADDING", (0, 0), (-1, -1), density_opts["padding"]),
                    ("RIGHTPADDING", (0, 0), (-1, -1), density_opts["padding"]),
                    ("TOPPADDING", (0, 0), (-1, -1), density_opts["padding"]),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), density_opts["padding"]),
                    ("BACKGROUND", (0, 0), (-1, 0), theme["table_header_bg"]),
                    ("TEXTCOLOR", (0, 0), (-1, 0), theme["table_header_text"]),
                    ("TEXTCOLOR", (0, 1), (-1, -1), theme["text"]),
                ]))
                table.repeatRows = 1
                _, table_height = table.wrap(table_width, y - margin_bottom)
                y_table = max(margin_bottom + 36, y - table_height)
                table.drawOn(c, margin_left + (available_width - table_width) / 2, y_table)
                y = y_table - 24

            if config.get("qr", {}).get("enabled"):
                qr_url = config["qr"].get("url") or "https://eftx.com.br"
                try:
                    qrw = rl_qr.QrCodeWidget(qr_url)
                    bounds = qrw.getBounds()
                    size = 24 * mm
                    w = bounds[2] - bounds[0]
                    h = bounds[3] - bounds[1]
                    drawing = Drawing(size, size, transform=[size / w, 0, 0, size / h, 0, 0])
                    drawing.add(qrw)
                    renderPDF.draw(drawing, c, width - margin_right - size, margin_bottom + 6)
                except Exception:
                    c.setFillColor(theme["muted"])
                    c.setFont(font_body, 8)
                    c.drawRightString(width - margin_right, margin_bottom + 6, qr_url)

        def render_page2() -> None:
            y_top = height - margin_top
            column_defs = []
            if sections.get("p2_hrp", {}).get("enabled") and hrp_img_path is not None:
                column_defs.append(("HRP", hrp_img_path, sections["p2_hrp"], "hrp"))
            if sections.get("p2_vrp", {}).get("enabled") and vrp_img_path is not None:
                column_defs.append(("VRP", vrp_img_path, sections["p2_vrp"], "vrp"))

            col_count = max(len(column_defs), 1)
            col_width = (available_width - gutter * (col_count - 1)) / col_count
            chart_height = 65 * mm

            if column_defs:
                for idx, (label, _, _, key) in enumerate(column_defs):
                    c.setFillColor(theme["title"])
                    c.setFont(font_bold, 12)
                    title_text = "Diagrama de Azimute (HRP)" if key == "hrp" else "Diagrama de Elevação (VRP)"
                    x_center = margin_left + idx * (col_width + gutter) + col_width / 2
                    c.drawCentredString(x_center, y_top - 6, title_text)
                y_chart = y_top - 18
                lowest_metrics_y = y_chart
                for idx, (_, img_path, conf, key) in enumerate(column_defs):
                    x = margin_left + idx * (col_width + gutter)
                    c.drawImage(ImageReader(str(img_path)), x, y_chart - chart_height, width=col_width, height=chart_height, preserveAspectRatio=True, mask="auto")
                    metrics_lines = _build_plot_metrics_lines(key, conf.get("metrics", {}))
                    c.setFillColor(theme["text"])
                    c.setFont(font_body, 9)
                    y_metrics = y_chart - chart_height - 14
                    for line in metrics_lines:
                        if y_metrics <= margin_bottom + 36:
                            break
                        c.drawString(x, y_metrics, line)
                        y_metrics -= 12
                    lowest_metrics_y = min(lowest_metrics_y, y_metrics)
                y_cursor = lowest_metrics_y - 18
            else:
                y_cursor = y_top - 24

            comp_items = []
            if sections.get("p2_comp_horizontal", {}).get("enabled"):
                comp_items.append(("Composição Horizontal", comp_h_path, "horizontal", sections["p2_comp_horizontal"].get("legend", [])))
            if sections.get("p2_comp_vertical", {}).get("enabled"):
                comp_items.append(("Composição Vertical", comp_v_path, "vertical", sections["p2_comp_vertical"].get("legend", [])))

            if comp_items:
                layout_mode = sections.get("p2_comp_horizontal", {}).get("layout", "side")
                if len(comp_items) == 1 or layout_mode == "stack":
                    for title, img_path, kind, legend in comp_items:
                        c.setFillColor(theme["title"])
                        c.setFont(font_bold, 11)
                        c.drawString(margin_left, y_cursor, title)
                        y_cursor -= 14
                        comp_height = 48 * mm
                        c.drawImage(ImageReader(str(img_path)), margin_left, y_cursor - comp_height, width=available_width, height=comp_height, preserveAspectRatio=True, mask="auto")
                        y_cursor -= comp_height + 10
                        c.setFillColor(theme["text"])
                        c.setFont(font_body, 9)
                        for line in _composition_lines(kind, legend):
                            if y_cursor <= margin_bottom + 24:
                                break
                            c.drawString(margin_left, y_cursor, line)
                            y_cursor -= 12
                        y_cursor -= 12
                else:
                    comp_width = (available_width - gutter) / 2
                    comp_height = 48 * mm
                    titles_y = y_cursor
                    for idx, (title, _, _, _) in enumerate(comp_items[:2]):
                        x_center = margin_left + idx * (comp_width + gutter) + comp_width / 2
                        c.setFillColor(theme["title"])
                        c.setFont(font_bold, 11)
                        c.drawCentredString(x_center, titles_y, title)
                    y_cursor = titles_y - 14
                    for idx, (_, img_path, kind, legend) in enumerate(comp_items[:2]):
                        x = margin_left + idx * (comp_width + gutter)
                        c.drawImage(ImageReader(str(img_path)), x, y_cursor - comp_height, width=comp_width, height=comp_height, preserveAspectRatio=True, mask="auto")
                        text_y = y_cursor - comp_height - 10
                        c.setFillColor(theme["text"])
                        c.setFont(font_body, 9)
                        for line in _composition_lines(kind, legend):
                            if text_y <= margin_bottom + 24:
                                break
                            c.drawString(x, text_y, line)
                            text_y -= 12
                    y_cursor -= comp_height + 36

        def render_table(title: str, angles: np.ndarray, values: np.ndarray, config_table: dict) -> None:
            if not config_table.get("enabled") or angles.size == 0:
                return
            columns = int(config_table.get("columns", 6))
            chunks = _build_table_chunks(
                angles,
                values,
                width=available_width,
                columns=columns,
                header_bg=theme["table_header_bg"],
                header_text=theme["table_header_text"],
                header_font=font_bold,
                body_font=font_body,
                body_font_size=7,
                height_limit=height - margin_top - margin_bottom - 24,
            )
            if not chunks:
                return
            for idx, chunk in enumerate(chunks):
                c.setFillColor(theme["title"])
                c.setFont(font_bold, 14)
                label = title + (" (continuação)" if idx else "")
                c.drawCentredString((margin_left + width - margin_right) / 2, height - margin_top + 4, label)
                content_h = height - margin_top - margin_bottom - 24
                _, h = chunk.wrap(available_width, content_h)
                y = height - margin_top - h - 12
                y = max(y, margin_bottom + 6)
                chunk.drawOn(c, margin_left, y)
                if idx < len(chunks) - 1:
                    c.showPage()

        for index, section_key in enumerate(ordered_sections):
            is_last = index == len(ordered_sections) - 1
            if section_key == "page1":
                render_page1()
                if not is_last:
                    c.showPage()
            elif section_key == "page2":
                render_page2()
                if not is_last:
                    c.showPage()
            elif section_key == "table_hrp":
                render_table("Tabela HRP", hrp_full_angles, hrp_full_values, table_hrp_conf)
                if not is_last:
                    c.showPage()
            elif section_key == "table_vrp":
                render_table("Tabela VRP", vrp_full_angles, vrp_full_values, table_vrp_conf)
                if not is_last:
                    c.showPage()

        c.save()
        buffer.seek(0)
        content_reader = PdfReader(buffer)
        writer = PdfWriter()
        if template_exists:
            template_bytes = modelo_path.read_bytes()
            for page in content_reader.pages:
                template_reader = PdfReader(io.BytesIO(template_bytes))
                base_page = template_reader.pages[0]
                base_page.merge_page(page)
                writer.add_page(base_page)
        else:
            for page in content_reader.pages:
                writer.add_page(page)

        output_bytes = io.BytesIO()
        writer.write(output_bytes)
        pdf_bytes = output_bytes.getvalue()

        target_path = output_path
        if target_path is None and paths is not None:
            target_path = Path(paths.pdf)

        if target_path is not None:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("wb") as fh:
                fh.write(pdf_bytes)

        return pdf_bytes
    finally:
        tmp_dir.cleanup()


def write_pat_array(path: Path, description: str, gain: float, num_elems: int,
                    angles_0_359: np.ndarray, values_0_359: np.ndarray,
                    vertical_tail_angles: np.ndarray, vertical_tail_values: np.ndarray) -> None:
    # *** NÃO ALTERADO ***
    with path.open("w", encoding="utf-8") as f:
        f.write(f"'{description}', {gain:.2f}, {num_elems}\n")
        for ang in range(360):
            val = float(np.interp(ang, angles_0_359, values_0_359))
            f.write(f"{ang}, {val:.4f}\n")
        for ang in [356, 357, 358, 359]:
            val = float(np.interp(ang, angles_0_359, values_0_359))
            f.write(f"{ang}, {val:.4f}\n")
        f.write("999\n")
        f.write("1, 91\n")
        f.write("269,\n")
        # Cauda vertical já pronta (0, -1, ..., -90) -> escrever direto sem re-interpolar
        for ang, v in zip(vertical_tail_angles, vertical_tail_values):
            f.write(f"{ang:.1f}, {v:.4f}\n")


def write_prn(path: Path, name: str, make: str, frequency: float, freq_unit: str,
              h_width: float, v_width: float, front_to_back: float, gain: float,
              h_angles: np.ndarray, h_values: np.ndarray,
              v_angles: np.ndarray, v_values: np.ndarray) -> None:
    # *** NÃO ALTERADO ***
    with path.open("w", encoding="utf-8") as f:
        f.write(f"NAME {name}\n")
        f.write(f"MAKE {make}\n")
        f.write(f"FREQUENCY {frequency:.2f} {freq_unit}\n")
        f.write(f"H_WIDTH {h_width:.2f}\n")
        f.write(f"V_WIDTH {v_width:.2f}\n")
        f.write(f"FRONT_TO_BACK {front_to_back:.2f}\n")
        f.write(f"GAIN {gain:.2f} dBi\n")
        f.write("TILT MECHANICAL\n")
        f.write("HORIZONTAL 360\n")
        for ang in range(360):
            v = float(np.interp(ang, h_angles, h_values))
            f.write(f"{ang}\t{lin_to_att_db(v):.4f}\n")
        f.write("VERTICAL 360\n")
        for ang in range(360):
            source_angle = (ang + 90) % 360  # alinhado para que o pico (0° real) fique em 0° no PRN
            v = float(np.interp(source_angle, v_angles, v_values))
            f.write(f"{ang}\t{lin_to_att_db(v):.4f}\n")


def _prepare_pat_vertical_tail(vrp_angles: np.ndarray, vrp_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Gera a 'cauda vertical' do .PAT em ângulos 0, -1, ..., -90 (negativos),
    mapeando tilt positivo para ângulo negativo no arquivo: valor_pat(-θ) = valor_vrp(+θ).
    Garante xp crescente para a interp.
    """
    # Ângulos de saída do PAT: 0, -1, ..., -90
    tail_angles = np.arange(0.0, -91.0, -1.0, dtype=float)  # (0, -1, ..., -90)

    if vrp_angles.size == 0 or vrp_values.size == 0:
        return tail_angles, np.ones_like(tail_angles)

    # Ordena e deduplica xp para garantir monotonicidade crescente (requisito do np.interp)
    order = np.argsort(vrp_angles)
    xp = vrp_angles[order].astype(float)
    fp = vrp_values[order].astype(float)
    # remove duplicatas exatas em xp
    if xp.size >= 2:
        mask = np.r_[True, np.diff(xp) > 1e-12]
        xp, fp = xp[mask], fp[mask]

    # Converter -θ (saída) para +θ (lookup em VRP)
    lookup = np.abs(tail_angles)  # 0..90

    # Garantir limites (se VRP vier -90..+90, estamos dentro; se vier 0..90, também)
    x_min, x_max = float(xp[0]), float(xp[-1])
    lookup = np.clip(lookup, x_min, x_max)

    tail_values = np.interp(lookup, xp, fp, left=fp[0], right=fp[-1])
    return tail_angles, tail_values




def generate_project_export(
    project: Project,
    export_root: Path,
    pdf_config: dict | None = None,
) -> tuple[ProjectExport, ExportPaths]:
    export_root.mkdir(parents=True, exist_ok=True)
    prepared = prepare_project_export_data(project)

    composition_arrays = prepared["composition_arrays"]
    composition_payload = prepared["composition_payload"]
    hrp_angles = prepared["hrp_angles"]
    hrp_values = prepared["hrp_values"]
    vrp_angles = prepared["vrp_angles"]
    vrp_values = prepared["vrp_values"]
    ang_full_hrp = prepared["ang_full_hrp"]
    val_full_hrp = prepared["val_full_hrp"]
    ang_full_vrp = prepared["ang_full_vrp"]
    val_full_vrp = prepared["val_full_vrp"]
    metrics = prepared["metrics"]
    horizon_value = prepared["horizon_value"]

    export_paths = ExportPaths(export_root, project)

    # Arquivos (PAT/PRN) – **sem alterações**
    description = project.antenna.name or "EFTX"
    gain_metric = metrics.get("gain_dbi")
    gain = gain_metric if gain_metric is not None else float(project.antenna.nominal_gain_dbd or 0.0)
    num_elems = max(project.h_count or 1, 1) * max(project.v_count or 1, 1)

    # tail_angles = np.linspace(0, -90, 91)
    # if vrp_angles.size:
    #     tail_values = np.interp(tail_angles, vrp_angles[::-1], vrp_values[::-1], left=vrp_values[-1], right=vrp_values[0])
    # else:
    #     tail_values = np.ones_like(tail_angles)


    tail_angles, tail_values = _prepare_pat_vertical_tail(vrp_angles, vrp_values)


    write_pat_array(
        export_paths.pat,
        description,
        gain,
        num_elems,
        ang_full_hrp,
        val_full_hrp,
        tail_angles,
        tail_values,
    )

    prn_name = project.name
    prn_make = project.antenna.model_number or "EFTX"
    frequency = float(project.frequency_mhz)
    front_to_back = metrics["front_to_back"] if metrics["front_to_back"] is not None else 0.0
    write_prn(
        export_paths.prn,
        prn_name,
        prn_make,
        frequency,
        "MHz",
        metrics["hrp_hpbw"] if metrics["hrp_hpbw"] is not None else 0.0,
        metrics["vrp_hpbw"] if metrics["vrp_hpbw"] is not None else 0.0,
        front_to_back,
        gain,
        ang_full_hrp,
        val_full_hrp,
        ang_full_vrp,
        val_full_vrp,
    )

    # PDF com layout ajustado (config customizada ou padrão)
    create_pdf_report_with_config(
        project,
        pdf_config,
        composition_arrays,
        metrics,
        export_paths=export_paths,
        prepared=prepared,
    )

    export = ProjectExport(
        project=project,
        erp_metadata={**composition_payload, "metrics": metrics},
        pat_path=str(export_paths.pat.relative_to(export_root)),
        prn_path=str(export_paths.prn.relative_to(export_root)),
        pdf_path=str(export_paths.pdf.relative_to(export_root)),
    )
    db.session.add(export)
    db.session.commit()
    return export, export_paths
SAO_PAULO = ZoneInfo("America/Sao_Paulo")
