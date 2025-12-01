"""Thermal noise and noise figure helpers."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

K_BOLTZMANN = 1.380649e-23  # J/K


@dataclass(slots=True)
class ThermalNoiseResult:
    temperature_k: float
    bandwidth_hz: float
    noise_figure_db: float
    noise_factor: float
    noise_power_watt: float
    noise_power_dbm: float
    density_dbm_hz: float


def thermal_noise(temperature_k: float, bandwidth_hz: float, noise_figure_db: float = 0.0) -> ThermalNoiseResult:
    if temperature_k <= 0:
        raise ValueError("Temperatura deve ser maior que zero Kelvin.")
    if bandwidth_hz <= 0:
        raise ValueError("Largura de banda deve ser maior que zero.")

    noise_factor = 10 ** (noise_figure_db / 10.0)
    noise_power_watt = K_BOLTZMANN * temperature_k * bandwidth_hz * noise_factor
    noise_power_dbm = 10.0 * math.log10(noise_power_watt / 1e-3)
    density_dbm_hz = -174.0 + 10.0 * math.log10(temperature_k / 290.0) + noise_figure_db

    return ThermalNoiseResult(
        temperature_k=temperature_k,
        bandwidth_hz=bandwidth_hz,
        noise_figure_db=noise_figure_db,
        noise_factor=noise_factor,
        noise_power_watt=noise_power_watt,
        noise_power_dbm=noise_power_dbm,
        density_dbm_hz=density_dbm_hz,
    )


@dataclass(slots=True)
class NoiseStage:
    index: int
    gain_db: float
    gain_linear: float
    noise_figure_db: float
    noise_factor: float
    cumulative_gain_linear: float
    cumulative_noise_factor: float
    cumulative_noise_figure_db: float


@dataclass(slots=True)
class CascadeNoiseResult:
    total_noise_factor: float
    total_noise_figure_db: float
    total_gain_db: float
    stages: list[NoiseStage]


def cascade_noise_figure(stages: Sequence[tuple[float, float]]) -> CascadeNoiseResult:
    """Apply Friis formula to a chain of gain/noise figure pairs."""

    if not stages:
        raise ValueError('Informe ao menos um estágio para o cálculo de ruído em cascata.')

    processed: list[NoiseStage] = []
    total_noise_factor = 0.0
    cumulative_gain = 1.0

    for idx, (gain_db, noise_figure_db) in enumerate(stages, start=1):
        gain_linear = 10 ** (gain_db / 10.0)
        noise_factor = 10 ** (noise_figure_db / 10.0)
        if idx == 1:
            total_noise_factor = noise_factor
            cumulative_gain = gain_linear
        else:
            total_noise_factor += (noise_factor - 1.0) / max(cumulative_gain, 1e-12)
            cumulative_gain *= gain_linear

        stage = NoiseStage(
            index=idx,
            gain_db=gain_db,
            gain_linear=gain_linear,
            noise_figure_db=noise_figure_db,
            noise_factor=noise_factor,
            cumulative_gain_linear=cumulative_gain,
            cumulative_noise_factor=total_noise_factor,
            cumulative_noise_figure_db=10.0 * math.log10(total_noise_factor),
        )
        processed.append(stage)

    total_gain_db = sum(gain_db for gain_db, _ in stages)
    total_noise_figure_db = processed[-1].cumulative_noise_figure_db

    return CascadeNoiseResult(
        total_noise_factor=total_noise_factor,
        total_noise_figure_db=total_noise_figure_db,
        total_gain_db=total_gain_db,
        stages=processed,
    )
