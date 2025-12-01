"""Atenuador Pi and related calculator helpers."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import units


@dataclass(frozen=True, slots=True)
class PiAttenuatorResult:
    """Resultados normalizados para o atenuador do tipo Pi."""

    attenuation_db: float
    characteristic_impedance_ohm: float
    voltage_ratio: float
    power_ratio: float
    shunt_resistance_ohm: float
    series_resistance_ohm: float


def pi_attenuator(attenuation_db: float, impedance_ohm: float) -> PiAttenuatorResult:
    """Calcula os resistores equivalentes de um atenuador Pi simétrico."""
    if attenuation_db < 0:
        raise ValueError("Atenuação deve ser maior ou igual a 0 dB.")
    z0 = units.sanitize_positive(impedance_ohm, field="Impedância característica")
    # Razão de tensão linear
    n = 10 ** (attenuation_db / 20.0)
    if math.isclose(n, 1.0, rel_tol=1e-12):
        shunt = math.inf
        series = 0.0
    else:
        shunt = z0 * ((n + 1.0) / (n - 1.0))
        series = z0 * ((n * n - 1.0) / (2.0 * n))
    power_ratio = 10 ** (attenuation_db / 10.0) if attenuation_db != 0 else 1.0
    return PiAttenuatorResult(
        attenuation_db=attenuation_db,
        characteristic_impedance_ohm=z0,
        voltage_ratio=n,
        power_ratio=power_ratio,
        shunt_resistance_ohm=shunt,
        series_resistance_ohm=series,
    )
