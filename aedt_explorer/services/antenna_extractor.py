from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Tuple

import google.generativeai as genai
from flask import current_app
from werkzeug.datastructures import FileStorage

try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore


EXTRACT_KEYS = [
    "name",
    "model_number",
    "description",
    "nominal_gain_dbd",
    "polarization",
    "frequency_min_mhz",
    "frequency_max_mhz",
    "manufacturer",
    "electrical_characteristics",
    "mechanical_characteristics",
]


def _save_upload(file: FileStorage) -> Tuple[str, str]:
    root = current_app.config.get("EXPORT_ROOT", "exports")
    target_dir = os.path.join(root, "uploads", "antennas")
    os.makedirs(target_dir, exist_ok=True)
    filename = file.filename or "datasheet.pdf"
    safe_name = filename.replace("/", "_").replace("\\", "_")
    path = os.path.join(target_dir, safe_name)
    file.stream.seek(0)
    file.save(path)
    return path, safe_name


def _extract_text_from_pdf(path: str) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n\n".join(t.strip() for t in texts if t and t.strip())
    except Exception:
        return ""


def _extract_first_image(path: str) -> str | None:
    """Best-effort: extract first embedded image from the first PDF pages.
    Returns saved path or None.
    """
    if PdfReader is None:
        return None
    try:
        reader = PdfReader(path)
        out_dir = os.path.join(current_app.config.get("EXPORT_ROOT", "exports"), "uploads", "antennas", "thumbs")
        os.makedirs(out_dir, exist_ok=True)
        for page_index, page in enumerate(reader.pages[:3]):
            images = getattr(page, "images", [])
            for img in images:
                img_name = getattr(img, "name", f"img_{page_index}")
                ext = ".png"
                if isinstance(img_name, str) and "." in img_name:
                    ext = os.path.splitext(img_name)[1] or ".png"
                out_path = os.path.join(out_dir, f"thumb_{os.path.basename(path)}_{page_index}{ext}")
                try:
                    with open(out_path, "wb") as fh:
                        fh.write(img.data)
                    return out_path
                except Exception:
                    continue
    except Exception:
        return None
    return None


def _coerce(payload: Dict[str, Any]) -> Dict[str, Any]:
    def _to_float(x):
        try:
            if x is None:
                return None
            if isinstance(x, (int, float)):
                return float(x)
            s = str(x).strip().replace(",", ".")
            for suf in ["dBd", "dBi", "MHz"]:
                s = s.replace(suf, "").strip()
            return float(s)
        except Exception:
            return None

    payload["nominal_gain_dbd"] = _to_float(payload.get("nominal_gain_dbd"))
    payload["frequency_min_mhz"] = _to_float(payload.get("frequency_min_mhz"))
    payload["frequency_max_mhz"] = _to_float(payload.get("frequency_max_mhz"))
    payload["description"] = _normalize_text_block(payload.get("description"))
    payload["electrical_characteristics"] = _normalize_characteristics_block(
        payload.get("electrical_characteristics")
    )
    payload["mechanical_characteristics"] = _normalize_characteristics_block(
        payload.get("mechanical_characteristics")
    )
    return payload


def _normalize_text_block(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


LABEL_KEYS = ("label", "item", "nome", "name", "titulo", "Etiqueta")
VALUE_KEYS = ("value", "valor", "detail", "dados")

LABEL_TRANSLATIONS = {
    "ganho nominal": "Nominal gain",
    "faixa de frequência": "Frequency range",
    "frequência": "Frequency",
    "polarização": "Polarization",
    "fabricante": "Manufacturer",
    "categoria": "Category",
    "impedância": "Nominal impedance",
    "impedancia": "Nominal impedance",
    "vswr": "VSWR",
    "frente/costas": "Front-to-back ratio",
    "peso": "Weight",
    "conector": "Connector",
    "área projetada": "Projected area",
    "area projetada": "Projected area",
    "dimensões": "Dimensions",
    "dimensoes": "Dimensions",
    "material": "Material",
}


def _normalize_characteristics_block(value: Any) -> str:
    def _extract_from_dict(item: Dict[str, Any]) -> str | None:
        label = None
        data = None
        for key in LABEL_KEYS:
            if item.get(key):
                label = str(item[key]).strip()
                break
        for key in VALUE_KEYS:
            if item.get(key):
                data = str(item[key]).strip()
                break
        if label and data:
            return f"{label}: {data}"
        if label:
            return label
        if data:
            return data
        return None

    lines: list[str] = []
    if value is None:
        return ""
    if isinstance(value, str):
        lines = value.splitlines()
    elif isinstance(value, list):
        for entry in value:
            if isinstance(entry, dict):
                extracted = _extract_from_dict(entry)
                if extracted:
                    lines.append(extracted)
            else:
                text = str(entry).strip()
                if text:
                    lines.append(text)
    else:
        text = str(value).strip()
        if text:
            lines.append(text)

    normalized: list[str] = []
    for raw in lines:
        cleaned = str(raw).strip()
        if not cleaned:
            continue
        cleaned = cleaned.strip("•●-–\u2022 \t")
        if ":" not in cleaned:
            parts = re.split(r"\s*[-–]\s*", cleaned, maxsplit=1)
            if len(parts) == 2 and all(parts):
                cleaned = f"{parts[0]}: {parts[1]}"
        if ":" not in cleaned and " " in cleaned:
            tokens = cleaned.split()
            cleaned = f"{tokens[0]}: {' '.join(tokens[1:])}"
        if ":" in cleaned:
            label_part, value_part = cleaned.split(":", 1)
            label_key = label_part.strip().lower()
            translated = LABEL_TRANSLATIONS.get(label_key)
            label_normalized = translated or label_part.strip()
            cleaned = f"{label_normalized}: {value_part.strip()}"
        else:
            label_key = cleaned.lower()
            cleaned = LABEL_TRANSLATIONS.get(label_key, cleaned)
        normalized.append(f"• {cleaned}")
    return "\n".join(normalized)


def extract_antenna_from_datasheet(file: FileStorage) -> Dict[str, Any]:
    path, _ = _save_upload(file)
    text = _extract_text_from_pdf(path)
    if not text:
        file.stream.seek(0)
        raw = file.stream.read(4000)
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            text = ""

    api_key = current_app.config.get("GEMINI_API_KEY")
    model_name = current_app.config.get("GEMINI_MODEL", "gemini-2.5-flash")
    if not api_key:
        return {"error": "GEMINI_API_KEY ausente", "datasheet_path": path}

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    prompt = (
        "You are an RF engineer reviewing an institutional antenna datasheet.\n"
        "Use the attached PDF as the primary source and fall back to the text excerpt below only if needed.\n"
        "Respond with JSON (English only) using the exact schema:\n"
        "{\n"
        '  "name": string | null,\n'
        '  "model_number": string | null,\n'
        '  "description": string | null,\n'
        '  "nominal_gain_dbd": number | null,\n'
        '  "polarization": string | null,\n'
        '  "frequency_min_mhz": number | null,\n'
        '  "frequency_max_mhz": number | null,\n'
        '  "manufacturer": string | null,\n'
        '  "electrical_characteristics": [{"label": string, "value": string}] | string | null,\n'
        '  "mechanical_characteristics": [{"label": string, "value": string}] | string | null\n'
        "}\n\n"
        "Rules:\n"
        "• description: single paragraph (≤1000 words), concise English.\n"
        "• electrical_characteristics / mechanical_characteristics: prefer arrays of objects with keys \"label\" and \"value\".\n"
        "  Use ≤20 words per value, normalize units (MHz, dBd, kg, m², etc.), and translate any terms to English.\n"
        "• Interpret abbreviations (e.g., FB → front-to-back), convert frequency spans to MHz.\n"
        "• Do not fabricate data; return null when information is missing.\n"
        "• Output must be pure JSON with no surrounding commentary.\n\n"
        "Reference text (fallback only):\n"
        f"{(text[:12000] if text else '')}"
    )

    upload = None
    upload_name = None
    try:
        upload = genai.upload_file(path=path, display_name=os.path.basename(path))
        upload_name = getattr(upload, "name", None)
        if upload_name:
            for _ in range(20):
                polled = genai.get_file(upload_name)
                state = getattr(polled, "state", None)
                state_name = getattr(state, "name", state)
                if state_name == "ACTIVE":
                    upload = polled
                    break
                time.sleep(0.5)
    except Exception as exc:  # pragma: no cover - depende da API
        current_app.logger.warning("Falha ao enviar PDF para o Gemini: %s", exc)
        upload = None
        upload_name = None

    try:
        parts: list[Any] = []
        if upload:
            parts.append(upload)
        elif path and os.path.exists(path):
            try:
                with open(path, "rb") as pdf_fh:
                    parts.append({"mime_type": "application/pdf", "data": pdf_fh.read()})
            except Exception:
                pass
        parts.append(prompt)
        response = model.generate_content(parts)
        content = getattr(response, "text", None) or ""
        if not content:
            candidates = getattr(response, "candidates", None) or []
            for c in candidates:
                parts = getattr(getattr(c, "content", None), "parts", None)
                if parts:
                    for p in parts:
                        if getattr(p, "text", None):
                            content += p.text
        content = (content or "").strip()
        start = content.find("{")
        end = content.rfind("}")
        json_text = content[start : end + 1] if start >= 0 and end >= start else "{}"
        data = json.loads(json_text)
    except Exception as exc:
        return {"error": f"Falha ao consultar Gemini: {exc}", "datasheet_path": path}
    finally:
        if upload_name:
            try:
                genai.delete_file(upload_name)
            except Exception:
                pass

    data = {k: data.get(k) for k in EXTRACT_KEYS if k in data}
    # Fabricante padrao fixo
    data["manufacturer"] = "EFTX Broadcast & Telecom"
    data["datasheet_path"] = path
    thumb = _extract_first_image(path)
    if thumb:
        data["thumbnail_path"] = thumb
    return _coerce(data)
