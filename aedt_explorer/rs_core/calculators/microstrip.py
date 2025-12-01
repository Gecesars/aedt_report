"""Microstrip characteristic impedance calculators."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import units


@dataclass(frozen=True, slots=True)
class MicrostripImpedanceResult:
    width_m: float
    height_m: float
    relative_permittivity: float
    effective_permittivity: float
    width_over_height: float
    impedance_ohm: float


def microstrip_impedance(width_m: float, height_m: float, eps_r: float) -> MicrostripImpedanceResult:
    """Calcula a impedância característica aproximada de uma microfita."""
    width_m = units.sanitize_positive(width_m, field="Largura")
    height_m = units.sanitize_positive(height_m, field="Altura")
    if eps_r <= 1.0:
        raise ValueError("ϵᵣ deve ser maior que 1.")
    ratio = width_m / height_m
    # Aproximação clássica de Hammerstad-Wheeler
    effective_eps = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 / math.sqrt(1.0 + 12.0 / ratio))
    if ratio <= 1.0:
        impedance = (60.0 / math.sqrt(effective_eps)) * math.log((8.0 / ratio) + 0.25 * ratio)
    else:
        impedance = (120.0 * math.pi / math.sqrt(effective_eps)) / (ratio + 1.393 + 0.667 * math.log(ratio + 1.444))
    return MicrostripImpedanceResult(
        width_m=width_m,
        height_m=height_m,
        relative_permittivity=eps_r,
        effective_permittivity=effective_eps,
        width_over_height=ratio,
        impedance_ohm=impedance,
    )
