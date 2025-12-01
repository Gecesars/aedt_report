from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import google.generativeai as genai
from flask import current_app
from werkzeug.datastructures import FileStorage

try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover - optional import at runtime
    PdfReader = None  # type: ignore


EXTRACT_KEYS = [
    "display_name",
    "model_code",
    "size_inch",
    "impedance_ohms",
    "manufacturer",
    "notes",
    "frequency_min_mhz",
    "frequency_max_mhz",
    "velocity_factor",
    "max_power_w",
    "min_bend_radius_mm",
    "outer_diameter_mm",
    "weight_kg_per_km",
    "vswr_max",
    "shielding_db",
    "temperature_min_c",
    "temperature_max_c",
    "conductor_material",
    "dielectric_material",
    "jacket_material",
    "shielding_type",
    "conductor_diameter_mm",
    "dielectric_diameter_mm",
    "attenuation_db_per_100m_curve",
]


def _save_upload(file: FileStorage) -> Tuple[str, str]:
    root = current_app.config.get("EXPORT_ROOT", "exports")
    target_dir = os.path.join(root, "uploads", "cables")
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


def _coerce_numbers(payload: Dict[str, Any]) -> Dict[str, Any]:
    def _to_float(x):
        try:
            if x is None:
                return None
            if isinstance(x, (int, float)):
                return float(x)
            s = str(x).strip().replace(",", ".")
            # remove units
            for suf in ["dB/100m", "dB/100 m", "dB", "ohm", "Ω", "mm", "W", "°C", "C", "%", "MHz", "kg/km"]:
                s = s.replace(suf, "").strip()
            return float(s)
        except Exception:
            return None

    for key in list(payload.keys()):
        if key.endswith("_mhz") or key.endswith("_w") or key.endswith("_mm") or key.endswith("_db") or key.endswith("_ohms"):
            payload[key] = _to_float(payload.get(key))
        if key in {"velocity_factor", "vswr_max", "weight_kg_per_km", "temperature_min_c", "temperature_max_c"}:
            payload[key] = _to_float(payload.get(key))
    curve = payload.get("attenuation_db_per_100m_curve")
    if isinstance(curve, dict):
        clean = {}
        for k, v in curve.items():
            try:
                fk = float(str(k).replace(",", "."))
                clean[fk] = _to_float(v)
            except Exception:
                continue
        payload["attenuation_db_per_100m_curve"] = clean
    return payload


def extract_cable_from_datasheet(file: FileStorage) -> Dict[str, Any]:
    """Save datasheet, extract text (if PDF), and call Gemini to produce a strict JSON payload.

    Returns: dict with extracted fields and the key 'datasheet_path' when saved.
    """
    path, name = _save_upload(file)
    text = _extract_text_from_pdf(path)
    # Fallback to raw bytes (truncated) if no text could be extracted
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

    schema_hint = {
        "type": "object",
        "properties": {key: {"type": ["number", "string", "object", "null"]} for key in EXTRACT_KEYS},
        "required": ["display_name", "model_code"],
        "additionalProperties": True,
    }

    prompt = (
        "Voce recebe o texto de um datasheet de cabo coaxial. Extraia os campos padrao e responda APENAS com JSON valido, "
        "sem comentarios. Campos numericos devem ser numeros (ponto decimal). Chaves: "
        + ", ".join(EXTRACT_KEYS)
        + ". Se houver uma tabela de atenuacao por frequencia, preencha 'attenuation_db_per_100m_curve' como {MHz: dB/100m}.\n\n"
        "Se um valor nao existir, use null.\n\n"
        "Texto do datasheet:\n" + (text[:12000] if text else "")
    )

    try:
        response = model.generate_content([prompt])
        content = getattr(response, "text", None) or ""
        if not content:
            # try candidates
            candidates = getattr(response, "candidates", None) or []
            for c in candidates:
                parts = getattr(getattr(c, "content", None), "parts", None)
                if parts:
                    for p in parts:
                        if getattr(p, "text", None):
                            content += p.text
        content = (content or "").strip()
        # Attempt to locate a JSON object in the response
        start = content.find("{")
        end = content.rfind("}")
        json_text = content[start : end + 1] if start >= 0 and end >= start else "{}"
        data = json.loads(json_text)
    except Exception as exc:
        return {"error": f"Falha ao consultar Gemini: {exc}", "datasheet_path": path}

    data = {k: data.get(k) for k in EXTRACT_KEYS if k in data}
    data["datasheet_path"] = path
    return _coerce_numbers(data)
