"""Coaxial cable models using empirical attenuation coefficients."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import units


@dataclass(frozen=True, slots=True)
class CoaxLossResult:
    """Aggregated attenuation data for a coaxial cable run."""

    coefficient_a: float
    coefficient_b: float
    frequency_hz: float
    frequency_mhz: float
    length_m: float
    length_ft: float
    attenuation_db_per_100m: float
    sqrt_component_db_per_100m: float
    linear_component_db_per_100m: float
    attenuation_db_per_m: float
    total_loss_db: float
    total_loss_linear: float


def coax_loss(
    frequency_hz: float,
    length_m: float,
    coefficient_a: float,
    coefficient_b: float,
) -> CoaxLossResult:
    """Compute coaxial attenuation using vendor coefficients.

    The common data-sheet model is ``α(f) = A·√f_MHz + B·f_MHz`` in dB/100 m.
    """
    frequency_hz = units.sanitize_positive(frequency_hz, field="Frequência (Hz)")
    length_m = units.sanitize_positive(length_m, field="Comprimento (m)")
    coefficient_a = float(coefficient_a)
    coefficient_b = float(coefficient_b)

    frequency_mhz = frequency_hz / 1_000_000.0
    length_ft = units.from_meters(length_m, "ft")

    sqrt_component = coefficient_a * math.sqrt(frequency_mhz)
    linear_component = coefficient_b * frequency_mhz
    attenuation_db_per_100m = sqrt_component + linear_component
    attenuation_db_per_m = attenuation_db_per_100m / 100.0
    total_loss_db = attenuation_db_per_m * length_m
    total_loss_linear = 10 ** (-total_loss_db / 10.0)

    return CoaxLossResult(
        coefficient_a=coefficient_a,
        coefficient_b=coefficient_b,
        frequency_hz=frequency_hz,
        frequency_mhz=frequency_mhz,
        length_m=length_m,
        length_ft=length_ft,
        attenuation_db_per_100m=attenuation_db_per_100m,
        sqrt_component_db_per_100m=sqrt_component,
        linear_component_db_per_100m=linear_component,
        attenuation_db_per_m=attenuation_db_per_m,
        total_loss_db=total_loss_db,
        total_loss_linear=total_loss_linear,
    )
