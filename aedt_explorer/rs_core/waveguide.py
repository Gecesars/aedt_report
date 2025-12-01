"""Cálculos para guias de onda retangulares."""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import units


@dataclass(slots=True)
class CutoffResult:
    mode: str
    cutoff_hz: float
    cutoff_ghz: float
    propagates: bool
    guidance: str


@dataclass(slots=True)
class PropagationResult:
    frequency_hz: float
    cutoff_hz: float
    wavelength_free_space_m: float
    guide_wavelength_m: float | None
    phase_constant_rad_m: float | None


def _validate_mode(mode: str, m: int, n: int) -> None:
    if m < 0 or n < 0:
        raise ValueError("Índices m/n devem ser >= 0.")
    if mode not in {"TE", "TM"}:
        raise ValueError("Modo deve ser 'TE' ou 'TM'.")
    if mode == "TE" and (m == 0 and n == 0):
        raise ValueError("Para modos TE, m e n não podem ser simultaneamente zero.")
    if mode == "TM" and (m == 0 or n == 0):
        raise ValueError("Para modos TM, m e n devem ser >= 1.")


def cutoff_frequency_hz(mode: str, m: int, n: int, a_m: float, b_m: float) -> float:
    _validate_mode(mode, m, n)
    a_m = units.sanitize_positive(a_m, field="a")
    b_m = units.sanitize_positive(b_m, field="b")

    kx = m / a_m
    ky = n / b_m
    return units.C / 2.0 * math.sqrt(kx**2 + ky**2)


def cutoff_summary(mode: str, m: int, n: int, a_m: float, b_m: float, frequency_hz: float | None = None) -> CutoffResult:
    fc = cutoff_frequency_hz(mode, m, n, a_m, b_m)
    propagates = False
    guidance = ""
    if frequency_hz is not None:
        if frequency_hz > fc:
            propagates = True
            guidance = "Frequência acima do corte: modo propagante."
        else:
            guidance = "Frequência abaixo do corte: o modo é evanescente."
    return CutoffResult(
        mode=f"{mode}{m}{n}",
        cutoff_hz=fc,
        cutoff_ghz=fc / 1e9,
        propagates=propagates,
        guidance=guidance,
    )


def propagation_parameters(frequency_hz: float, cutoff_hz: float) -> PropagationResult:
    frequency_hz = units.sanitize_positive(frequency_hz, field="frequência")
    cutoff_hz = units.sanitize_positive(cutoff_hz, field="f_c")
    wavelength0 = units.C / frequency_hz
    if frequency_hz <= cutoff_hz:
        return PropagationResult(
            frequency_hz=frequency_hz,
            cutoff_hz=cutoff_hz,
            wavelength_free_space_m=wavelength0,
            guide_wavelength_m=None,
            phase_constant_rad_m=None,
        )
    ratio = math.sqrt(1 - (cutoff_hz / frequency_hz) ** 2)
    beta = 2 * math.pi * frequency_hz / units.C * ratio
    lambda_g = 2 * math.pi / beta
    return PropagationResult(
        frequency_hz=frequency_hz,
        cutoff_hz=cutoff_hz,
        wavelength_free_space_m=wavelength0,
        guide_wavelength_m=lambda_g,
        phase_constant_rad_m=beta,
    )
