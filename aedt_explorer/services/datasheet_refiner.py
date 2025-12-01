"""Utilities to normalize raw IA extraction into the EFTX datasheet schema."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Dict, List, Tuple

from pydantic import ValidationError

from ..schemas import (
    Beamwidth,
    BeamwidthAxis,
    ConnectorInfo,
    DimensionsMm,
    EftxDatasheet,
    EftxDatasheetQuality,
    GainValue,
    MechanicalSection,
    MetaSection,
    NormalizationLogEntry,
    PatternEntry,
    RangeFloat,
    SummarySection,
    TemperatureRange,
    TiltRange,
)


_RANGE_RE = re.compile(r"(?P<min>-?\d+(?:[\.,]\d+)?)\s*(?:[-–]|a|to)\s*(?P<max>-?\d+(?:[\.,]\d+)?)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d+(?:[\.,]\d+)?")
_COUNT_RE = re.compile(r"(?P<count>\d+)\s*(?:x|×)\s*(?P<type>[0-9\.\-/]+)\s*(?P<gender>male|female)?", re.IGNORECASE)
_PIM_RE = re.compile(r"[>≥]?\s*(?P<value>\d+(?:[\.,]\d+)?)\s*dB", re.IGNORECASE)


class DatasheetRefineError(RuntimeError):
    """Raised when the incoming payload cannot be normalized."""


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        match = _NUMBER_RE.search(value)
        if match:
            return float(match.group(0).replace(",", "."))
    return None


def _parse_range(value: Any) -> RangeFloat | None:
    if isinstance(value, dict):
        min_val = _to_float(value.get("min"))
        max_val = _to_float(value.get("max"))
        if min_val is None or max_val is None:
            return None
        return RangeFloat(min=min_val, max=max_val)
    if isinstance(value, str):
        match = _RANGE_RE.search(value)
        if match:
            min_val = float(match.group("min").replace(",", "."))
            max_val = float(match.group("max").replace(",", "."))
            return RangeFloat(min=min_val, max=max_val)
    if isinstance(value, Iterable):
        values = list(value)
        if len(values) >= 2:
            first = _to_float(values[0])
            second = _to_float(values[1])
            if first is not None and second is not None:
                return RangeFloat(min=first, max=second)
    return None


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _normalize_summary(raw: Dict[str, Any], logs: List[NormalizationLogEntry]) -> SummarySection:
    title = raw.get("title") or raw.get("titulo") or raw.get("name") or "Datasheet EFTX"
    subtitle = raw.get("subtitle") or raw.get("subtitulo")
    description = raw.get("description") or raw.get("descricao") or raw.get("summary")
    highlights_raw = (
        raw.get("highlights")
        or raw.get("bullets")
        or raw.get("features")
        or raw.get("destaques")
        or []
    )
    highlights = [str(item).strip() for item in _ensure_list(highlights_raw) if str(item).strip()]
    if not highlights and description:
        logs.append(NormalizationLogEntry(field="summary.highlights", from_value="", to="derived from description", rule="fallback"))
    return SummarySection(title=title, subtitle=subtitle, description=description, highlights=highlights)


def _normalize_meta(raw: Dict[str, Any], logs: List[NormalizationLogEntry]) -> MetaSection:
    aliases = {
        "brand": ["brand", "marca"],
        "product_line": ["product_line", "linha", "linha_produto"],
        "model": ["model", "modelo", "name"],
        "sku": ["sku", "codigo", "code"],
        "revision": ["revision", "rev", "versao"],
        "release_date": ["release_date", "data_lancamento", "lançamento"],
    }
    resolved: Dict[str, Any] = {}
    for target, keys in aliases.items():
        for key in keys:
            if key in raw and raw[key]:
                resolved[target] = raw[key]
                if key != target:
                    logs.append(NormalizationLogEntry(field=f"meta.{target}", from_value=key, to=target, rule="alias"))
                break
    if "manufacturer" in raw and raw["manufacturer"] != "EFTX Broadcast & Telecom":
        logs.append(NormalizationLogEntry(field="meta.manufacturer", from_value=str(raw["manufacturer"]), to="EFTX Broadcast & Telecom", rule="override"))
    return MetaSection(**resolved)


def _normalize_gain(raw_band: Dict[str, Any]) -> GainValue | None:
    gain_fields = [
        ("typ", ["gain_typ", "gain", "typical_gain", "ganho"]),
        ("max", ["gain_max", "max_gain"]),
        ("min", ["gain_min"]),
    ]
    payload: Dict[str, float] = {}
    for target, keys in gain_fields:
        for key in keys:
            val = raw_band.get(key)
            if val is None:
                continue
            scalar = _to_float(val)
            if scalar is not None:
                payload[target] = scalar
                break
    if payload:
        return GainValue(**payload)
    return None


def _normalize_beamwidth(raw_band: Dict[str, Any]) -> Beamwidth | None:
    def build(axis: str) -> BeamwidthAxis | None:
        values = {}
        for kind in ("typ", "min", "max"):
            for key in (f"beamwidth_{axis}_{kind}", f"bw_{axis}_{kind}", f"beam_{axis}_{kind}"):
                if key in raw_band:
                    scalar = _to_float(raw_band[key])
                    if scalar is not None:
                        values[kind] = scalar
                        break
        if values:
            return BeamwidthAxis(**values)
        return None

    h_axis = build("h")
    v_axis = build("v")
    if h_axis or v_axis:
        return Beamwidth(h=h_axis or BeamwidthAxis(), v=v_axis or BeamwidthAxis())
    return None


def _normalize_vswr(raw_band: Dict[str, Any], logs: List[NormalizationLogEntry]) -> Tuple[float | None, float | None]:
    max_val = raw_band.get("vswr_max") or raw_band.get("vswr") or raw_band.get("vswr_maximo")
    typ_val = raw_band.get("vswr_typ") or raw_band.get("vswr_t") or raw_band.get("vswr_típico")
    extra_notes: List[str] = []

    def parse(value: Any) -> float | None:
        if value is None:
            return None
        text = str(value)
        if text.strip().startswith("<"):
            logs.append(NormalizationLogEntry(field="band.vswr_max", from_value=text, to=text.lstrip("<"), rule="upper-bound"))
            text = text.lstrip("<")
        return _to_float(text)

    return parse(max_val), parse(typ_val)


def _normalize_connector(raw_value: Any, logs: List[NormalizationLogEntry]) -> ConnectorInfo | None:
    if isinstance(raw_value, dict):
        payload: Dict[str, Any] = {
            "type": raw_value.get("type") or raw_value.get("tipo"),
            "gender": raw_value.get("gender") or raw_value.get("genero"),
            "count": raw_value.get("count") or raw_value.get("quantidade"),
        }
        if payload.get("count") is not None:
            payload["count"] = int(payload["count"])
        if payload.get("type"):
            return ConnectorInfo(**{k: v for k, v in payload.items() if v is not None})
        return None
    if isinstance(raw_value, str):
        match = _COUNT_RE.search(raw_value)
        if match:
            count = int(match.group("count"))
            ctype = match.group("type").replace("/", "-")
            gender = (match.group("gender") or "").lower() or None
            logs.append(NormalizationLogEntry(field="mechanical.connector", from_value=raw_value, to=f"{count}x {ctype} {gender or ''}".strip(), rule="parsed"))
            return ConnectorInfo(type=ctype, gender=gender, count=count)
        if raw_value:
            return ConnectorInfo(type=raw_value)
    return None


def _normalize_pim(value: Any, notes: List[str], logs: List[NormalizationLogEntry]) -> float | None:
    if value is None:
        return None
    text = str(value)
    match = _PIM_RE.search(text)
    if match:
        numeric = float(match.group("value").replace(",", "."))
        if text.strip().startswith((">", "≥")):
            notes.append(f"PIM informado como {text.strip()}")
            logs.append(NormalizationLogEntry(field="band.pim_dbc_at_2x43dbm", from_value=text, to=str(numeric), rule=">= normalization"))
        return numeric
    scalar = _to_float(text)
    if scalar is not None:
        return scalar
    return None


def _normalize_patterns(raw: Dict[str, Any], logs: List[NormalizationLogEntry]) -> List[PatternEntry]:
    patterns_raw = _ensure_list(raw.get("patterns") or raw.get("diagramas") or [])
    patterns: List[PatternEntry] = []
    for idx, entry in enumerate(patterns_raw):
        if not isinstance(entry, dict):
            continue
        band_index = entry.get("band_index")
        if band_index is None and idx < len(patterns_raw):
            band_index = idx
        samples = None
        if entry.get("samples"):
            samples = [
                {
                    "angle_deg": _to_float(sample.get("angle") or sample.get("angle_deg") or sample.get("theta")) or 0.0,
                    "level_db": _to_float(sample.get("level") or sample.get("db") or sample.get("gain")) or 0.0,
                }
                for sample in _ensure_list(entry.get("samples"))
                if isinstance(sample, dict)
            ]
        az = entry.get("azimuth_image") or entry.get("az_image") or entry.get("az")
        el = entry.get("elevation_image") or entry.get("el_image") or entry.get("el")
        if az or el or samples:
            patterns.append(
                PatternEntry(
                    band_index=int(band_index or 0),
                    azimuth_image=az,
                    elevation_image=el,
                    samples=samples,
                )
            )
    return patterns


def _normalize_mechanical(raw: Dict[str, Any], logs: List[NormalizationLogEntry], notes: List[str]) -> MechanicalSection | None:
    mech_raw = raw.get("mechanical") or raw.get("mecanico") or {}
    if not isinstance(mech_raw, dict):
        return None
    data: Dict[str, Any] = {}
    tilt = mech_raw.get("tilt_range_deg") or mech_raw.get("tilt")
    tilt_range = _parse_range(tilt)
    if tilt_range:
        data["tilt_range_deg"] = TiltRange(min=tilt_range.min, max=tilt_range.max)
    weight = mech_raw.get("weight_kg") or mech_raw.get("peso")
    weight_float = _to_float(weight)
    if weight_float is not None:
        data["weight_kg"] = weight_float
    op_wind = mech_raw.get("wind_operational_kmh") or mech_raw.get("vento_operacional")
    surv_wind = mech_raw.get("wind_survival_kmh") or mech_raw.get("vento_sobrevivencia")
    op_value = _to_float(op_wind)
    surv_value = _to_float(surv_wind)
    if op_value is not None:
        data["wind_operational_kmh"] = op_value
    if surv_value is not None:
        data["wind_survival_kmh"] = surv_value
    temp = mech_raw.get("temp_operating_c") or mech_raw.get("temperatura")
    temp_range = _parse_range(temp)
    if temp_range:
        data["temp_operating_c"] = TemperatureRange(min=temp_range.min, max=temp_range.max)
    dims = mech_raw.get("dimensions_mm") or mech_raw.get("dimensoes") or mech_raw.get("dimensions")
    if isinstance(dims, dict):
        dims_payload = {k: _to_float(v) for k, v in dims.items() if _to_float(v) is not None}
        if {"w", "h", "d"}.issubset(dims_payload):
            data["dimensions_mm"] = DimensionsMm(**{k: float(v) for k, v in dims_payload.items()})
    connector = mech_raw.get("connector") or mech_raw.get("conector")
    info = _normalize_connector(connector, logs)
    if info:
        data["connector"] = info
    impedance = mech_raw.get("impedance_ohm") or mech_raw.get("impedancia")
    imp_value = _to_float(impedance)
    if imp_value is not None:
        data["impedance_ohm"] = imp_value
    if not data:
        return None
    return MechanicalSection(**data)


def refine_datasheet(raw_payload: Dict[str, Any]) -> Tuple[EftxDatasheet, EftxDatasheetQuality]:
    if not isinstance(raw_payload, dict):
        raise DatasheetRefineError("Payload must be an object")

    logs: List[NormalizationLogEntry] = []
    notes: List[str] = list(_ensure_list(raw_payload.get("notes")))

    meta_raw = raw_payload.get("meta") or raw_payload.get("metadata") or {}
    summary_raw = raw_payload.get("summary") or raw_payload.get("resumo") or {}

    meta = _normalize_meta(meta_raw if isinstance(meta_raw, dict) else {}, logs)
    summary = _normalize_summary(summary_raw if isinstance(summary_raw, dict) else {}, logs)

    bands_raw = raw_payload.get("bands") or raw_payload.get("faixas") or raw_payload.get("frequency_bands") or []
    bands: List[Dict[str, Any]] = []
    for item in _ensure_list(bands_raw):
        if not isinstance(item, dict):
            continue
        freq_range = _parse_range(item.get("freq_range_mhz") or item.get("frequency") or item.get("faixa"))
        if freq_range is None:
            min_val = _to_float(item.get("freq_min"))
            max_val = _to_float(item.get("freq_max"))
            if min_val is not None and max_val is not None:
                freq_range = RangeFloat(min=min_val, max=max_val)
        if freq_range is None:
            continue
        band_data: Dict[str, Any] = {
            "name": item.get("name") or item.get("label") or item.get("faixa"),
            "freq_range_mhz": freq_range,
        }
        gain = _normalize_gain(item)
        if gain:
            band_data["gain_dbi"] = gain
        beamwidth = _normalize_beamwidth(item)
        if beamwidth:
            band_data["beamwidth_deg"] = beamwidth
        vswr_max, vswr_typ = _normalize_vswr(item, logs)
        if vswr_max is not None:
            band_data["vswr_max"] = vswr_max
        if vswr_typ is not None:
            band_data["vswr_typ"] = vswr_typ
        if item.get("polarization") or item.get("polarizacao"):
            band_data["polarization"] = item.get("polarization") or item.get("polarizacao")
        if item.get("port_isolation") or item.get("isolamento"):
            iso_val = _to_float(item.get("port_isolation") or item.get("isolamento"))
            if iso_val is not None:
                band_data["port_isolation_db"] = iso_val
        pim_val = item.get("pim") or item.get("pim_dbc_at_2x43dbm")
        pim_normalized = _normalize_pim(pim_val, notes, logs)
        if pim_normalized is not None:
            band_data["pim_dbc_at_2x43dbm"] = pim_normalized
        max_power = _to_float(item.get("max_input_power") or item.get("potencia_max"))
        if max_power is not None:
            band_data["max_input_power_w"] = max_power
        bands.append(band_data)

    if not bands:
        raise DatasheetRefineError("Nenhuma faixa de frequência válida encontrada.")

    mechanical = _normalize_mechanical(raw_payload, logs, notes)
    patterns = _normalize_patterns(raw_payload, logs)

    source_raw = raw_payload.get("source") or {}
    source_obj = None
    if isinstance(source_raw, dict) and any(source_raw.get(k) for k in ("origin", "file_name", "page_refs")):
        source_obj = {
            "origin": source_raw.get("origin") or source_raw.get("origem"),
            "file_name": source_raw.get("file_name") or source_raw.get("arquivo"),
            "page_refs": _ensure_list(source_raw.get("page_refs") or source_raw.get("paginas")),
        }

    features = [str(item).strip() for item in _ensure_list(raw_payload.get("features")) if str(item).strip()]
    compliance = [str(item).strip() for item in _ensure_list(raw_payload.get("compliance")) if str(item).strip()]
    ordering = [str(item).strip() for item in _ensure_list(raw_payload.get("ordering")) if str(item).strip()]
    notes = [str(item).strip() for item in notes if str(item).strip()]

    payload: Dict[str, Any] = {
        "meta": meta.model_dump(),
        "summary": summary.model_dump(),
        "bands": [
            {key: value.model_dump() if hasattr(value, "model_dump") else value for key, value in band.items()}
            for band in bands
        ],
        "mechanical": mechanical.model_dump() if mechanical else None,
        "patterns": [pattern.model_dump() for pattern in patterns],
        "features": features,
        "compliance": compliance,
        "ordering": ordering,
        "notes": notes,
        "source": source_obj,
    }

    try:
        datasheet = EftxDatasheet.model_validate(payload)
    except ValidationError as exc:
        raise DatasheetRefineError(f"Payload inválido: {exc}") from exc

    missing_fields: List[str] = []
    if not datasheet.summary.title:
        missing_fields.append("summary.title")
    if not datasheet.bands:
        missing_fields.append("bands")
    else:
        for idx, band in enumerate(datasheet.bands):
            if band.freq_range_mhz is None:
                missing_fields.append(f"bands[{idx}].freq_range_mhz")

    quality = EftxDatasheetQuality(
        confidence=min(1.0, 0.6 + 0.1 * len(datasheet.bands)),
        missing_fields=sorted(set(missing_fields)),
        ambiguous_fields=[],
        normalization_log=logs,
    )

    datasheet.quality = quality
    return datasheet, quality


def generate_schema() -> Dict[str, Any]:
    """Return the JSON schema for the EFTX datasheet."""

    return EftxDatasheet.model_json_schema(ref_template="#/$defs/{model}")
