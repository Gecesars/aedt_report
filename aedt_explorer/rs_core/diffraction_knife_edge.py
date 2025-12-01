"""Difração por borda-de-faca conforme ITU-R P.526."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from . import units


@dataclass(slots=True)
class KnifeEdgeResult:
    v: float
    loss_db: float
    clearance_m: float
    fresnel_radius_m: float
    clearance_ratio: float
    obstructed: bool
    guidance: str
    plot_points: Sequence[tuple[float, float]]


def fresnel_radius(frequency_hz: float, d1_m: float, d2_m: float, zone: int = 1) -> float:
    frequency_hz = units.sanitize_positive(frequency_hz, field="frequência")
    d1_m = units.sanitize_positive(d1_m, field="d₁")
    d2_m = units.sanitize_positive(d2_m, field="d₂")
    if zone <= 0:
        raise ValueError("Zona de Fresnel deve ser >= 1.")
    wavelength = units.C / frequency_hz
    return math.sqrt(zone * wavelength * d1_m * d2_m / (d1_m + d2_m))


def knife_edge_v(height_m: float, frequency_hz: float, d1_m: float, d2_m: float) -> float:
    wavelength = units.C / units.sanitize_positive(frequency_hz, field="frequência")
    d1_m = units.sanitize_positive(d1_m, field="d₁")
    d2_m = units.sanitize_positive(d2_m, field="d₂")
    return height_m * math.sqrt(2.0 / wavelength * (d1_m + d2_m) / (d1_m * d2_m))


def knife_edge_loss_db(v: float) -> float:
    if v <= -0.7:
        return 0.0
    return 6.9 + 20.0 * math.log10(math.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)


def compute_knife_edge(
    frequency_hz: float,
    d1_m: float,
    d2_m: float,
    tx_height_m: float,
    rx_height_m: float,
    obstacle_height_m: float,
) -> KnifeEdgeResult:
    d1_m = units.sanitize_positive(d1_m, field="d₁")
    d2_m = units.sanitize_positive(d2_m, field="d₂")
    total = d1_m + d2_m
    los_height = tx_height_m + (rx_height_m - tx_height_m) * (d1_m / total)
    clearance = obstacle_height_m - los_height
    v = knife_edge_v(clearance, frequency_hz, d1_m, d2_m)
    loss = knife_edge_loss_db(v)
    r1 = fresnel_radius(frequency_hz, d1_m, d2_m, zone=1)
    clearance_ratio = clearance / r1 if r1 > 0 else math.inf
    obstructed = clearance > 0
    if clearance_ratio >= 1.0:
        guidance = "1ª zona de Fresnel totalmente obstruída. Expecte perdas elevadas."
    elif clearance_ratio >= 0.6:
        guidance = "Obstáculo invade parte significativa da 1ª zona de Fresnel. Ajuste alturas ou distâncias."
    elif clearance_ratio >= 0:
        guidance = "Obstáculo cruza a linha de visada, porém com invasão moderada da 1ª zona."
    else:
        guidance = "Obstáculo abaixo da linha de visada; margem de clearance disponível."
    plot_points = tuple((v_point, knife_edge_loss_db(v_point)) for v_point in _plot_range())
    return KnifeEdgeResult(
        v=v,
        loss_db=loss,
        clearance_m=clearance,
        fresnel_radius_m=r1,
        clearance_ratio=clearance_ratio,
        obstructed=obstructed,
        guidance=guidance,
        plot_points=plot_points,
    )


def _plot_range() -> Sequence[float]:
    values = [-3.0]
    step = 0.25
    current = -3.0
    while current < 6.0:
        current += step
        values.append(round(current, 4))
    return values
