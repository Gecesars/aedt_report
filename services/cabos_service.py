"""Serviços para consulta de cabos e cálculo de perdas."""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from flask import current_app

from ..extensions import db
from ..models import Cable
from ..rs_core import units


@dataclass(slots=True)
class CabosInterpolacao:
    frequencia_hz: float
    atenuacao_db_por_100m: float
    origem: str
    extrapolado: bool
    pontos_apoio: Sequence[tuple[float, float]]


class CaboNaoEncontradoError(LookupError):
    pass


def listar_cabos() -> list[Cable]:
    return Cable.query.order_by(Cable.display_name.asc()).all()


def obter_cabo_por_id(cabo_id: str | uuid.UUID) -> Cable:
    try:
        cab_uuid = uuid.UUID(str(cabo_id))
    except ValueError as exc:  # pragma: no cover - ids inválidos
        raise CaboNaoEncontradoError(f"Identificador de cabo inválido: {cabo_id}") from exc
    cable = db.session.get(Cable, cab_uuid)
    if not cable:
        raise CaboNaoEncontradoError(f"Cabo não encontrado para id {cabo_id}")
    return cable


def _normalizar_pontos(curve: object) -> list[tuple[float, float]]:
    if not curve:
        return []
    unit = "mhz"
    if isinstance(curve, dict) and "points" in curve:
        unit = str(curve.get("unit", unit)).lower().strip()
        payload = curve.get("points")
    else:
        payload = curve
    points: list[tuple[float, float]] = []
    if isinstance(payload, dict):
        iterable: Iterable = payload.items()
    else:
        iterable = payload or []
    for item in iterable:
        freq = None
        att = None
        if isinstance(item, tuple) and len(item) == 2:
            freq, att = item
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            freq, att = item[0], item[1]
        elif isinstance(item, dict):
            freq = item.get("frequency") or item.get("freq") or item.get("f") or item.get("mhz") or item.get("ghz")
            att = (
                item.get("attenuation")
                or item.get("loss")
                or item.get("db_per_100m")
                or item.get("attenuation_db_per_100m")
            )
            if freq is None:
                # fallback para chaves bem definidas
                if "frequency_mhz" in item:
                    freq = item["frequency_mhz"]
                elif "frequency_ghz" in item:
                    freq = item["frequency_ghz"]
                    unit = "ghz"
        if freq is None or att is None:
            continue
        try:
            freq = float(freq)
            att = float(att)
        except (TypeError, ValueError):
            continue
        if freq <= 0 or att <= 0:
            continue
        try:
            hz = units.to_hz(freq, unit)
        except ValueError:
            # heurística: assume MHz
            hz = units.to_hz(freq, "mhz")
        points.append((hz, att))
    points.sort(key=lambda x: x[0])
    return points


def _log_log_interpolar(pontos: Sequence[tuple[float, float]], freq_hz: float) -> CabosInterpolacao | None:
    if not pontos:
        return None
    if len(pontos) == 1:
        freq, att = pontos[0]
        return CabosInterpolacao(freq_hz, att, origem="ponto único", extrapolado=False, pontos_apoio=(pontos[0],))
    freq_hz = units.sanitize_positive(freq_hz, field="frequência")
    log_f = math.log(freq_hz)
    extrapolado = False

    if freq_hz < pontos[0][0]:
        p0, p1 = pontos[0], pontos[1]
        extrapolado = True
    elif freq_hz > pontos[-1][0]:
        p0, p1 = pontos[-2], pontos[-1]
        extrapolado = True
    else:
        p0, p1 = pontos[0], pontos[1]
        for i in range(len(pontos) - 1):
            if pontos[i][0] <= freq_hz <= pontos[i + 1][0]:
                p0, p1 = pontos[i], pontos[i + 1]
                break

    f0, a0 = p0
    f1, a1 = p1
    log_f0 = math.log(f0)
    log_f1 = math.log(f1)
    log_a0 = math.log(a0)
    log_a1 = math.log(a1)
    if log_f1 == log_f0:
        att = math.exp(log_a0)
    else:
        slope = (log_a1 - log_a0) / (log_f1 - log_f0)
        att = math.exp(log_a0 + slope * (log_f - log_f0))
    origem = "interpolação" if not extrapolado else "extrapolação"
    return CabosInterpolacao(freq_hz, att, origem=origem, extrapolado=extrapolado, pontos_apoio=(p0, p1))


def interpolar_atenuacao(cabo: Cable, frequencia_hz: float) -> CabosInterpolacao | None:
    pontos = _normalizar_pontos(getattr(cabo, "attenuation_db_per_100m_curve", None))
    if not pontos:
        return None
    return _log_log_interpolar(pontos, frequencia_hz)


def calcular_perda_total(
    cabo_id: str | uuid.UUID,
    frequencia_hz: float,
    comprimento_m: float,
    conectores_db: Sequence[float] | None = None,
) -> dict:
    cabo = obter_cabo_por_id(cabo_id)
    frequencia_hz = units.sanitize_positive(frequencia_hz, field="frequência")
    comprimento_m = units.sanitize_positive(comprimento_m, field="comprimento")
    conectores_db = [float(v) for v in (conectores_db or []) if v is not None]

    interpolacao = interpolar_atenuacao(cabo, frequencia_hz)
    if interpolacao is None:
        raise ValueError("Cabo não possui curva de atenuação cadastrada para cálculo preciso.")

    perda_cabo_db = (comprimento_m / 100.0) * interpolacao.atenuacao_db_por_100m
    perda_conectores_db = sum(x for x in conectores_db if x > 0)
    perda_total_db = perda_cabo_db + perda_conectores_db

    vf = getattr(cabo, "velocity_factor", None)
    velocidade = vf * units.C if vf else None

    detalhe = {
        "cabo": cabo,
        "frequencia_hz": frequencia_hz,
        "comprimento_m": comprimento_m,
        "perda_total_db": perda_total_db,
        "perda_cabo_db": perda_cabo_db,
        "perda_conectores_db": perda_conectores_db,
        "atenuacao_db_por_100m": interpolacao.atenuacao_db_por_100m,
        "origem": interpolacao.origem,
        "extrapolado": interpolacao.extrapolado,
        "pontos": [
            {"frequencia_hz": p[0], "atenuacao_db_por_100m": p[1]} for p in interpolacao.pontos_apoio
        ],
        "velocity_factor": vf,
        "propagation_velocity": velocidade,
    }
    return detalhe
