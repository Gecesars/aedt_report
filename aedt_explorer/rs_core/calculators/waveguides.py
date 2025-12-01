"""Waveguide related calculators."""
from __future__ import annotations

from dataclasses import dataclass

from .. import units


@dataclass(frozen=True, slots=True)
class RectangularWaveguideCutoffResult:
    width_m: float
    cutoff_hz: float
    cutoff_ghz: float
    recommended_min_hz: float
    recommended_max_hz: float
    recommended_min_ghz: float
    recommended_max_ghz: float


RECOMMENDED_MIN_FACTOR = 1.25
RECOMMENDED_MAX_FACTOR = 1.89


def rectangular_waveguide_cutoff(width_m: float) -> RectangularWaveguideCutoffResult:
    width_m = units.sanitize_positive(width_m, field="Largura")
    cutoff_hz = units.C / (2.0 * width_m)
    recommended_min_hz = RECOMMENDED_MIN_FACTOR * cutoff_hz
    recommended_max_hz = RECOMMENDED_MAX_FACTOR * cutoff_hz
    return RectangularWaveguideCutoffResult(
        width_m=width_m,
        cutoff_hz=cutoff_hz,
        cutoff_ghz=cutoff_hz / 1e9,
        recommended_min_hz=recommended_min_hz,
        recommended_max_hz=recommended_max_hz,
        recommended_min_ghz=recommended_min_hz / 1e9,
        recommended_max_ghz=recommended_max_hz / 1e9,
    )
