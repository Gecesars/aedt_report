"""Antenna related calculators."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import units

EARTH_RADIUS_KM = 6371.0
EFFECTIVE_K_FACTOR = 4.0 / 3.0
KM_TO_MILES = 0.621371


@dataclass(frozen=True, slots=True)
class LineOfSightResult:
    height_m: float
    line_of_sight_km: float
    radio_horizon_km: float
    line_of_sight_miles: float
    radio_horizon_miles: float


@dataclass(frozen=True, slots=True)
class EffectiveApertureResult:
    gain_linear: float
    gain_dbi: float
    wavelength_m: float
    frequency_hz: float | None
    effective_area_m2: float
    effective_area_ft2: float


def line_of_sight(height_m: float) -> LineOfSightResult:
    """Calcula os horizontes geométrico e de rádio para uma dada altura."""
    height_m = units.sanitize_positive(height_m, strict=False, field="Altura")
    if height_m < 0:
        raise ValueError("Altura não pode ser negativa.")
    height_km = height_m / 1000.0
    base = math.sqrt(2.0 * EARTH_RADIUS_KM * height_km)
    radio = math.sqrt(2.0 * EFFECTIVE_K_FACTOR * EARTH_RADIUS_KM * height_km)
    return LineOfSightResult(
        height_m=height_m,
        line_of_sight_km=base,
        radio_horizon_km=radio,
        line_of_sight_miles=base * KM_TO_MILES,
        radio_horizon_miles=radio * KM_TO_MILES,
    )


def effective_aperture(
    gain_value: float,
    gain_is_dbi: bool,
    *,
    wavelength_m: float | None = None,
    frequency_hz: float | None = None,
) -> EffectiveApertureResult:
    if not gain_is_dbi:
        gain_linear = units.sanitize_positive(gain_value, field="Ganho (linear)")
        gain_dbi = 10.0 * math.log10(gain_linear)
    else:
        gain_dbi = gain_value
        gain_linear = 10.0 ** (gain_dbi / 10.0)

    resolved_wavelength: float | None = None
    resolved_frequency: float | None = None

    if wavelength_m is not None:
        resolved_wavelength = units.sanitize_positive(wavelength_m, field="Comprimento de onda (m)")
        resolved_frequency = units.C / resolved_wavelength
    elif frequency_hz is not None:
        resolved_frequency = units.sanitize_positive(frequency_hz, field="Frequência (Hz)")
        resolved_wavelength = units.C / resolved_frequency
    else:
        raise ValueError("Informe a frequência ou o comprimento de onda.")

    effective_area_m2 = (gain_linear * (resolved_wavelength ** 2)) / (4.0 * math.pi)
    meters_to_feet = units.from_meters(1.0, "ft")
    effective_area_ft2 = effective_area_m2 * (meters_to_feet ** 2)

    return EffectiveApertureResult(
        gain_linear=gain_linear,
        gain_dbi=gain_dbi,
        wavelength_m=resolved_wavelength,
        frequency_hz=resolved_frequency,
        effective_area_m2=effective_area_m2,
        effective_area_ft2=effective_area_ft2,
    )
