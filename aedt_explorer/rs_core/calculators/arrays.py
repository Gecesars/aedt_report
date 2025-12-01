"""Antenna array gain empirical formulas for bay stacks."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import units

HALF_WAVE_COEFF = 0.3047619
HALF_WAVE_INTERCEPT = 0.0989048

FULL_WAVE_LINEAR_COEFF = 0.60033
FULL_WAVE_LINEAR_INTERCEPT = -0.23823
FULL_WAVE_QUADRATIC_A = 0.00180571
FULL_WAVE_QUADRATIC_B = 0.57009336
FULL_WAVE_QUADRATIC_C = -0.15194641


@dataclass(frozen=True)
class ArrayGainResult:
    bays: int
    frequency_hz: float
    wavelength_m: float
    half_wave_gain: float
    half_wave_gain_db: float
    half_wave_gain_dbd: float
    full_wave_gain: float
    full_wave_gain_db: float
    full_wave_gain_dbd: float
    half_wave_length_m: float
    half_wave_length_ft: float
    full_wave_length_m: float
    full_wave_length_ft: float
    polarization: str
    polarization_label: str


def _validate_bays(bays: int) -> int:
    if bays < 1:
        raise ValueError("Número de bays deve ser pelo menos 1.")
    if bays > 32:
        raise ValueError("Número de bays muito alto para o modelo empírico (máx. 32).")
    return bays


def _half_wave_power(bays: int) -> float:
    return HALF_WAVE_COEFF * bays + HALF_WAVE_INTERCEPT


def _full_wave_power(bays: int) -> float:
    # Quadratic fit provides better fidelity; ensure positivity
    value = FULL_WAVE_QUADRATIC_A * (bays ** 2) + FULL_WAVE_QUADRATIC_B * bays + FULL_WAVE_QUADRATIC_C
    if value <= 0:
        # Fallback to linear approximation if the quadratic becomes negative for very low N
        value = FULL_WAVE_LINEAR_COEFF * bays + FULL_WAVE_LINEAR_INTERCEPT
    if value <= 0:
        raise ValueError("Ganho calculado não pode ser negativo. Verifique o número de bays.")
    return value

POLARIZATION_PROFILES = {
    "linear_vertical": {"label": "Linear (Vertical)", "bonus_db": 3.0},
    "circular": {"label": "Circular", "bonus_db": 0.0},
}


def array_gain(bays: int, frequency_hz: float, polarization: str = "linear_vertical") -> ArrayGainResult:
    bays = _validate_bays(int(bays))
    frequency_hz = units.sanitize_positive(frequency_hz, field="Frequência (Hz)")
    wavelength_m = units.C / frequency_hz

    profile = POLARIZATION_PROFILES.get(polarization, POLARIZATION_PROFILES["linear_vertical"])
    bonus_db = profile["bonus_db"]
    bonus_linear = 10 ** (bonus_db / 10.0)

    half_power = _half_wave_power(bays)
    full_power = _full_wave_power(bays)

    half_power *= bonus_linear
    full_power *= bonus_linear

    half_db = 10.0 * math.log10(half_power)
    full_db = 10.0 * math.log10(full_power)

    half_db_dbd = half_db - 2.15
    full_db_dbd = full_db - 2.15

    span = max(bays - 1, 0)
    half_length_m = span * 0.5 * wavelength_m
    full_length_m = span * 1.0 * wavelength_m

    half_length_ft = units.from_meters(half_length_m, "ft")
    full_length_ft = units.from_meters(full_length_m, "ft")

    return ArrayGainResult(
        bays=bays,
        frequency_hz=frequency_hz,
        wavelength_m=wavelength_m,
        half_wave_gain=half_power,
        half_wave_gain_db=half_db,
        half_wave_gain_dbd=half_db_dbd,
        full_wave_gain=full_power,
        full_wave_gain_db=full_db,
        full_wave_gain_dbd=full_db_dbd,
        half_wave_length_m=half_length_m,
        half_wave_length_ft=half_length_ft,
        full_wave_length_m=full_length_m,
        full_wave_length_ft=full_length_ft,
        polarization=polarization,
        polarization_label=profile["label"],
    )
