from __future__ import annotations

import csv
import io
import re
from typing import Iterable, Tuple

import numpy as np

_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _normalise_token(token: str) -> str:
    return token.strip().replace(",", ".")


def _coerce_float(token: str) -> float | None:
    normalised = _normalise_token(token)
    if not normalised:
        return None
    if _FLOAT_RE.match(normalised):
        try:
            return float(normalised)
        except ValueError:
            return None
    return None


def _prepare_series(angles: Iterable[float], values: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    angles_arr = np.asarray(list(angles), dtype=float)
    values_arr = np.asarray(list(values), dtype=float)
    if angles_arr.size == 0:
        raise ValueError("Nenhum dado numerico encontrado no arquivo enviado.")
    mask = np.isfinite(angles_arr) & np.isfinite(values_arr)
    angles_arr = angles_arr[mask]
    values_arr = values_arr[mask]
    if angles_arr.size == 0:
        raise ValueError("Todos os valores encontrados sao invalidos ou nao numericos.")
    order = np.argsort(angles_arr)
    angles_arr = angles_arr[order]
    values_arr = values_arr[order]
    unique_angles, inverse = np.unique(angles_arr, return_inverse=True)
    accum = np.zeros_like(unique_angles, dtype=float)
    counts = np.zeros_like(unique_angles, dtype=float)
    np.add.at(accum, inverse, values_arr)
    np.add.at(counts, inverse, 1.0)
    averaged_values = accum / np.clip(counts, 1.0, None)
    return unique_angles, averaged_values


def _strip_header(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return rows
    first = "".join(cell.strip() for cell in rows[0])
    if any(ch.isalpha() for ch in first):
        return rows[1:]
    return rows


def _build_numeric_rows(rows: list[list[str]]) -> list[list[float | None]]:
    max_len = max(len(row) for row in rows)
    processed: list[list[float | None]] = []
    for row in rows:
        numeric_row: list[float | None] = []
        for idx in range(max_len):
            value = _coerce_float(row[idx]) if idx < len(row) else None
            numeric_row.append(value)
        processed.append(numeric_row)
    return processed


def _column_values(processed: list[list[float | None]], index: int) -> list[float]:
    return [row[index] for row in processed if row[index] is not None]


def _column_span(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(max(values) - min(values))


def _select_primary_phi_rows(processed: list[list[float | None]]) -> list[list[float | None]]:
    if not processed or len(processed[0]) <= 1:
        return processed
    phi_values = [row[1] for row in processed if row[1] is not None]
    if not phi_values:
        return processed
    counts: dict[float, int] = {}
    for value in phi_values:
        key = round(value, 4)
        counts[key] = counts.get(key, 0) + 1
    if len(counts) <= 1:
        return processed
    best_key = max(counts.items(), key=lambda item: (item[1], -abs(item[0])))[0]
    tolerance = 1e-3
    filtered = [row for row in processed if row[1] is not None and abs(row[1] - best_key) <= tolerance]
    return filtered if filtered else processed


def _pick_angle_column(columns: dict[int, list[float]], row_count: int) -> int:
    preferred = 2
    values = columns.get(preferred, [])
    if len(values) >= row_count * 0.6 and _column_span(values) >= 45.0:
        return preferred
    candidates = [
        idx
        for idx, vals in columns.items()
        if idx != preferred and len(vals) >= row_count * 0.6 and _column_span(vals) >= 45.0
    ]
    if candidates:
        return min(candidates, key=lambda idx: abs(idx - preferred))
    ranked = sorted(columns.items(), key=lambda item: _column_span(item[1]), reverse=True)
    for idx, vals in ranked:
        if len(vals) >= row_count * 0.4 and _column_span(vals) >= 1.0:
            return idx
    raise ValueError("Nao foi possivel identificar a coluna de angulos.")


def _pick_amplitude_column(columns: dict[int, list[float]], row_count: int, angle_idx: int) -> int:
    preferred = 3
    values = columns.get(preferred, [])
    if len(values) >= row_count * 0.6:
        return preferred
    candidates: list[tuple[int, list[float]]] = []
    for idx, vals in columns.items():
        if idx == angle_idx:
            continue
        if len(vals) < row_count * 0.6:
            continue
        if all(v >= 0 for v in vals):
            span = _column_span(vals)
            candidates.append((idx, vals, span))
    if candidates:
        candidates.sort(key=lambda item: (-item[2], item[0]))
        return candidates[0][0]
    for idx in sorted(columns.keys(), reverse=True):
        if idx != angle_idx and len(columns[idx]) >= row_count * 0.4:
            return idx
    raise ValueError("Nao foi possivel identificar a coluna de amplitudes.")


def parse_hfss_csv(text: str) -> Tuple[np.ndarray, np.ndarray]:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
    except csv.Error:
        dialect = csv.get_dialect("excel")
    reader = csv.reader(io.StringIO(text), dialect)
    rows = [row for row in reader if row and any(cell.strip() for cell in row)]
    rows = _strip_header(rows)
    if not rows:
        raise ValueError("Arquivo vazio ou sem linhas validas.")
    processed = _build_numeric_rows(rows)
    processed = _select_primary_phi_rows(processed)
    if not processed:
        raise ValueError("Nao foi possivel selecionar linhas validas a partir do arquivo.")
    max_len = len(processed[0])
    columns = {idx: _column_values(processed, idx) for idx in range(max_len)}
    row_count = len(processed)
    angle_idx = _pick_angle_column(columns, row_count)
    amplitude_idx = _pick_amplitude_column(columns, row_count, angle_idx)
    angles: list[float] = []
    amplitudes: list[float] = []
    for row in processed:
        angle = row[angle_idx]
        value = row[amplitude_idx]
        if angle is None or value is None:
            continue
        angles.append(float(angle))
        amplitudes.append(max(0.0, float(value)))
    if not angles:
        raise ValueError("Nao foi possivel extrair amostras de angulo e amplitude do arquivo.")
    return _prepare_series(angles, amplitudes)


def parse_generic_table(text: str) -> Tuple[np.ndarray, np.ndarray]:
    angles: list[float] = []
    values: list[float] = []
    splitter = re.compile(r"[\s;,]+")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [part for part in splitter.split(stripped) if part]
        numbers = []
        for token in parts:
            value = _coerce_float(token)
            if value is not None:
                numbers.append(value)
        if len(numbers) < 2:
            continue
        if len(numbers) >= 3:
            angle, amplitude = numbers[1], numbers[2]
        else:
            angle, amplitude = numbers[0], numbers[1]
        angles.append(angle)
        values.append(max(0.0, amplitude))
    if not angles:
        raise ValueError("Nenhuma linha com valores numericos suficientes foi encontrada.")
    return _prepare_series(angles, values)


def parse_pattern_bytes(data: bytes, filename: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    if not data:
        raise ValueError("O arquivo enviado esta vazio.")
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception as exc:
        raise ValueError("Nao foi possivel decodificar o arquivo como texto.") from exc
    if not text.strip():
        raise ValueError("Arquivo sem conteudo textual.")
    parsers = [parse_hfss_csv]
    if filename:
        lower = filename.lower()
        if lower.endswith((".txt", ".dat")):
            parsers.append(parse_generic_table)
    parsers.append(parse_generic_table)
    last_error: ValueError | None = None
    for parser in parsers:
        try:
            return parser(text)
        except ValueError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error