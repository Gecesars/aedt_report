"""Radar related calculators."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import units

METERS_PER_NAUTICAL_MILE = 1852.0


@dataclass(frozen=True, slots=True)
class RadarRangeResult:
    transmit_power_watt: float
    gain_tx_linear: float
    gain_rx_linear: float
    wavelength_m: float
    radar_cross_section_m2: float
    min_detectable_watt: float
    range_m: float
    range_km: float
    range_nautical_mile: float


def radar_range(
    transmit_power_watt: float,
    gain_tx_linear: float,
    gain_rx_linear: float,
    wavelength_m: float,
    radar_cross_section_m2: float,
    min_detectable_watt: float,
) -> RadarRangeResult:
    transmit_power_watt = units.sanitize_positive(transmit_power_watt, field="Potência transmitida")
    gain_tx_linear = units.sanitize_positive(gain_tx_linear, field="Ganho Tx")
    gain_rx_linear = units.sanitize_positive(gain_rx_linear, field="Ganho Rx")
    wavelength_m = units.sanitize_positive(wavelength_m, field="Comprimento de onda")
    radar_cross_section_m2 = units.sanitize_positive(radar_cross_section_m2, field="RCS")
    min_detectable_watt = units.sanitize_positive(min_detectable_watt, field="Sinal mínimo")

    numerator = transmit_power_watt * gain_tx_linear * gain_rx_linear * (wavelength_m ** 2) * radar_cross_section_m2
    denominator = ((4.0 * math.pi) ** 3) * min_detectable_watt
    if denominator == 0:
        raise ValueError("Sinal mínimo detectável não pode ser zero.")
    range_m = (numerator / denominator) ** 0.25
    return RadarRangeResult(
        transmit_power_watt=transmit_power_watt,
        gain_tx_linear=gain_tx_linear,
        gain_rx_linear=gain_rx_linear,
        wavelength_m=wavelength_m,
        radar_cross_section_m2=radar_cross_section_m2,
        min_detectable_watt=min_detectable_watt,
        range_m=range_m,
        range_km=range_m / 1000.0,
        range_nautical_mile=range_m / METERS_PER_NAUTICAL_MILE,
    )
