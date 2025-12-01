from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from flask import current_app
from typing import Any as _Any
from werkzeug.local import LocalProxy
from pypdf import PdfReader


_MODEL_CACHE: dict[str, _Any] = {}
_EMBEDDINGS: dict[str, np.ndarray] = {}
_METADATA: dict[str, list[dict]] = {}


@dataclass(frozen=True)
class KnowledgeChunk:
    text: str
    source: str
    page: int | None = None


def _index_dir() -> Path:
    base = current_app.config.get("KNOWLEDGE_INDEX_DIR", "vector_store")
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _model():
    raw_name = current_app.config.get("KNOWLEDGE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    model_name = raw_name.strip()
    if model_name.startswith("sentence-transformers") and "/" not in model_name.split("sentence-transformers", 1)[-1]:
        model_name = model_name.replace("sentence-transformers", "sentence-transformers/", 1)

    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    token = (
        os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HF_HUB_TOKEN")
        or None
    )
    try:
        # Import tardio para evitar custo em caminhos como migrações
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer(model_name, use_auth_token=token)
    except OSError as exc:
        raise RuntimeError(
            "Falha ao baixar o modelo de embeddings. Configure HUGGINGFACEHUB_API_TOKEN ou disponibilize o modelo localmente."
        ) from exc

    _MODEL_CACHE[model_name] = model
    return model


def _chunk_text(text: str, *, size: int = 500, overlap: int = 100) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(start + size, length)
        chunk = cleaned[start:end]
        if chunk:
            chunks.append(chunk)
        start += max(size - overlap, 1)
    return chunks


def _extract_from_pdf(path: Path) -> Iterable[KnowledgeChunk]:
    reader = PdfReader(path)
    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for chunk in _chunk_text(text):
            yield KnowledgeChunk(text=chunk, source=str(path), page=page_index + 1)


def _extract_from_text(path: Path) -> Iterable[KnowledgeChunk]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for chunk in _chunk_text(text):
        yield KnowledgeChunk(text=chunk, source=str(path))


def build_index(source_dir: str | Path | None = None) -> int:
    source_path = Path(source_dir or "docs")
    if not source_path.exists():
        raise FileNotFoundError(f"Diretorio de fontes nao encontrado: {source_path}")

    chunks: list[KnowledgeChunk] = []
    for path in source_path.rglob("*"):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                chunks.extend(_extract_from_pdf(path))
            elif suffix in {".md", ".txt", ".rst"}:
                chunks.extend(_extract_from_text(path))
        except Exception as exc:
            current_app.logger.warning("knowledge.parse_failed", extra={"file": str(path), "error": str(exc)})

    if not chunks:
        return 0

    model = _model()
    embeddings = model.encode([chunk.text for chunk in chunks], convert_to_numpy=True, normalize_embeddings=True)
    metadata = [chunk.__dict__ for chunk in chunks]

    index_dir = _index_dir()
    np.save(index_dir / "embeddings.npy", embeddings)
    (index_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    _EMBEDDINGS.clear()
    _METADATA.clear()
    return len(chunks)


def _load_index() -> tuple[np.ndarray, list[dict]]:
    index_dir = _index_dir()
    embed_path = index_dir / "embeddings.npy"
    meta_path = index_dir / "metadata.json"
    if not embed_path.exists() or not meta_path.exists():
        return np.empty((0, 0)), []

    embeddings = _EMBEDDINGS.setdefault(str(embed_path), np.load(embed_path))
    metadata = _METADATA.setdefault(str(meta_path), json.loads(meta_path.read_text(encoding="utf-8")))
    return embeddings, metadata


def retrieve_contexts(query: str, top_k: int | None = None) -> list[str]:
    embeddings, metadata = _load_index()
    if embeddings.size == 0 or not metadata:
        return []

    model = _model()
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = embeddings @ query_vec
    k = top_k or current_app.config.get("KNOWLEDGE_TOPK", 3)
    top_indices = scores.argsort()[-k:][::-1]
    results = []
    for idx in top_indices:
        meta = metadata[idx]
        snippet = meta["text"] if "text" in meta else meta.get("chunk", "")
        if not snippet:
            continue
        source = meta.get("source")
        page = meta.get("page")
        header = f"Fonte: {source}" + (f", pagina {page}" if page else "")
        results.append(f"{header}\n{snippet}")
    return results
