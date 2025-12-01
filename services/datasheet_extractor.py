from __future__ import annotations

import json
import os
from typing import Any, Dict

import google.generativeai as genai
from flask import current_app
from werkzeug.datastructures import FileStorage

try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    PdfReader = None  # type: ignore


def _save_upload(file: FileStorage) -> tuple[str, str]:
    root = current_app.config.get("EXPORT_ROOT", "exports")
    target_dir = os.path.join(root, "uploads", "datasheets", "raw")
    os.makedirs(target_dir, exist_ok=True)
    filename = file.filename or "datasheet.pdf"
    safe_name = filename.replace("/", "_").replace("\\", "_")
    path = os.path.join(target_dir, safe_name)
    file.stream.seek(0)
    file.save(path)
    return path, safe_name


def _extract_text(path: str) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(path)
        chunks: list[str] = []
        for page in reader.pages:
            try:
                content = page.extract_text() or ""
            except Exception:
                content = ""
            if content:
                chunks.append(content.strip())
        return "\n\n".join(chunk for chunk in chunks if chunk)
    except Exception:
        return ""


def _normalize_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    payload = text[start : end + 1] if start >= 0 and end >= start else text
    try:
        return json.loads(payload)
    except Exception:
        # attempt to repair by wrapping
        try:
            return json.loads(f"{{\n" + payload + "\n}}")
        except Exception:
            return {"raw": text}


def extract_datasheet_intelligence(file: FileStorage) -> Dict[str, Any]:
    path, filename = _save_upload(file)
    text = _extract_text(path)
    if not text:
        file.stream.seek(0)
        snippet = file.stream.read(8192)
        try:
            text = snippet.decode("utf-8", errors="ignore")
        except Exception:
            text = ""

    api_key = current_app.config.get("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY ausente", "datasheet_path": path}

    model_name = "gemini-2.5-flash"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = (
            "Você é um analista de documentação técnica. Receba o texto integral de um datasheet (PDF convertido em texto) "
            "e extraia o máximo de informações estruturadas possível. Retorne SOMENTE JSON válido em português (pt-BR), "
            "com estrutura clara, mantendo números como números e representando tabelas como arrays de objetos. Inclua campos como "
            "metadata, especificacoes, tabelas, textos, notas, imagens (apenas referências se existirem) e qualquer outra seção relevante.\n\n"
            "Use chaves em snake_case, normalize as legendas para português e preserve o contexto técnico. Ao representar tabelas, mantenha "
            "as colunas com cabeçalhos coerentes em português.\n\nTexto do datasheet:\n"
            + (text[:24000] if text else "")
        )
        uploaded_file = None
        try:
            uploaded_file = genai.upload_file(path=path, display_name=filename)
        except Exception as exc:
            current_app.logger.warning("Falha ao enviar arquivo para Gemini: %s", exc)
            uploaded_file = None

        payload_parts: list[Any] = []
        if uploaded_file is not None:
            payload_parts.append(uploaded_file)
        payload_parts.append(prompt)

        response = model.generate_content(payload_parts)
        content = getattr(response, "text", None) or ""
        if not content:
            candidates = getattr(response, "candidates", None) or []
            for candidate in candidates:
                parts = getattr(getattr(candidate, "content", None), "parts", None)
                if not parts:
                    continue
                for part in parts:
                    chunk = getattr(part, "text", None)
                    if chunk:
                        content += chunk
        data = _normalize_json(content)
        if uploaded_file is not None:
            try:
                genai.delete_file(uploaded_file.name)
            except Exception:
                pass
    except Exception as exc:
        return {"error": f"Falha ao consultar Gemini: {exc}", "datasheet_path": path}

    return {
        "datasheet_path": path,
        "filename": filename,
        "extracted": data,
        "text_excerpt": text[:2000] if text else "",
    }
