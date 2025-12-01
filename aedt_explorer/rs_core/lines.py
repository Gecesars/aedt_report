"""Utilidades para linhas de transmissão."""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import units


@dataclass(slots=True)
class TransmissionLineResult:
    frequency_hz: float
    physical_length_m: float
    propagation_velocity_m_s: float
    guided_wavelength_m: float
    phase_deg: float
    beta_rad_m: float


def _resolve_velocity(velocity_factor: float | None, eps_eff: float | None) -> float:
    if velocity_factor is not None:
        velocity_factor = units.sanitize_positive(velocity_factor, field="vf")
        if velocity_factor > 1:
            raise ValueError("Fator de velocidade deve ser ≤ 1.")
        return velocity_factor * units.C
    if eps_eff is not None:
        eps_eff = units.sanitize_positive(eps_eff, field="ε_eff")
        return units.C / math.sqrt(eps_eff)
    raise ValueError("Informe fator de velocidade ou permissividade efetiva.")


def electrical_length(
    frequency_hz: float,
    physical_length_m: float,
    *,
    velocity_factor: float | None = None,
    eps_eff: float | None = None,
) -> TransmissionLineResult:
    frequency_hz = units.sanitize_positive(frequency_hz, field="frequência")
    physical_length_m = units.sanitize_positive(physical_length_m, field="comprimento")
    v = _resolve_velocity(velocity_factor, eps_eff)
    wavelength_g = v / frequency_hz
    beta = 2 * math.pi / wavelength_g
    phase = math.degrees(beta * physical_length_m)
    return TransmissionLineResult(
        frequency_hz=frequency_hz,
        physical_length_m=physical_length_m,
        propagation_velocity_m_s=v,
        guided_wavelength_m=wavelength_g,
        phase_deg=phase % 360.0,
        beta_rad_m=beta,
    )


def length_from_phase(
    frequency_hz: float,
    desired_phase_deg: float,
    *,
    velocity_factor: float | None = None,
    eps_eff: float | None = None,
) -> TransmissionLineResult:
    frequency_hz = units.sanitize_positive(frequency_hz, field="frequência")
    v = _resolve_velocity(velocity_factor, eps_eff)
    wavelength_g = v / frequency_hz
    phase_rad = math.radians(desired_phase_deg)
    phase_rad = phase_rad % (2 * math.pi)
    beta = 2 * math.pi / wavelength_g
    length = phase_rad / beta
    return TransmissionLineResult(
        frequency_hz=frequency_hz,
        physical_length_m=length,
        propagation_velocity_m_s=v,
        guided_wavelength_m=wavelength_g,
        phase_deg=desired_phase_deg % 360.0,
        beta_rad_m=beta,
    )
