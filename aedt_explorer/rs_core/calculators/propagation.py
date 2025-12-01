"""Propagation-related calculators (FSPL, Fresnel zones, Friis equation)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import units


@dataclass(slots=True)
class FsplResult:
    distance_m: float
    distance_km: float
    frequency_hz: float
    frequency_mhz: float
    wavelength_m: float
    fspl_db: float
    fspl_db_practical: float
    delta_db: float


def free_space_path_loss(distance_m: float, frequency_hz: float) -> FsplResult:
    distance_m = units.sanitize_positive(distance_m, field="Distância (m)")
    frequency_hz = units.sanitize_positive(frequency_hz, field="Frequência (Hz)")

    wavelength_m = units.C / frequency_hz
    fspl_linear = (4.0 * math.pi * distance_m / wavelength_m) ** 2
    fspl_db = 10.0 * math.log10(fspl_linear)

    distance_km = distance_m / 1000.0
    frequency_mhz = frequency_hz / 1_000_000.0
    fspl_db_practical = 32.44 + 20.0 * math.log10(distance_km) + 20.0 * math.log10(frequency_mhz)
    delta_db = fspl_db - fspl_db_practical

    return FsplResult(
        distance_m=distance_m,
        distance_km=distance_km,
        frequency_hz=frequency_hz,
        frequency_mhz=frequency_mhz,
        wavelength_m=wavelength_m,
        fspl_db=fspl_db,
        fspl_db_practical=fspl_db_practical,
        delta_db=delta_db,
    )


@dataclass(slots=True)
class FresnelZoneResult:
    order: int
    d1_m: float
    d2_m: float
    total_distance_m: float
    frequency_hz: float
    wavelength_m: float
    radius_m: float
    radius_ft: float
    recommended_clearance_m: float
    recommended_clearance_ft: float


def fresnel_zone_radius(d1_m: float, d2_m: float, frequency_hz: float, order: int = 1) -> FresnelZoneResult:
    d1_m = units.sanitize_positive(d1_m, field="d1 (m)")
    d2_m = units.sanitize_positive(d2_m, field="d2 (m)")
    frequency_hz = units.sanitize_positive(frequency_hz, field="Frequência (Hz)")
    order = max(int(order), 1)

    wavelength_m = units.C / frequency_hz
    numerator = order * wavelength_m * d1_m * d2_m
    denominator = d1_m + d2_m
    radius_m = math.sqrt(numerator / denominator)

    radius_ft = units.from_meters(radius_m, "ft")
    first_zone_radius = math.sqrt(wavelength_m * d1_m * d2_m / denominator)
    clearance_m = 0.6 * first_zone_radius
    clearance_ft = units.from_meters(clearance_m, "ft")

    return FresnelZoneResult(
        order=order,
        d1_m=d1_m,
        d2_m=d2_m,
        total_distance_m=d1_m + d2_m,
        frequency_hz=frequency_hz,
        wavelength_m=wavelength_m,
        radius_m=radius_m,
        radius_ft=radius_ft,
        recommended_clearance_m=clearance_m,
        recommended_clearance_ft=clearance_ft,
    )


@dataclass(slots=True)
class FriisResult:
    tx_power_dbm: float
    tx_power_watt: float
    tx_gain_dbi: float
    rx_gain_dbi: float
    distance_m: float
    frequency_hz: float
    wavelength_m: float
    fspl_db: float
    rx_power_dbm: float
    rx_power_watt: float


def friis_rx_power(
    tx_power_dbm: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    distance_m: float,
    frequency_hz: float,
) -> FriisResult:
    distance_m = units.sanitize_positive(distance_m, field="Distância (m)")
    frequency_hz = units.sanitize_positive(frequency_hz, field="Frequência (Hz)")

    fspl = free_space_path_loss(distance_m, frequency_hz)
    rx_power_dbm = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - fspl.fspl_db
    rx_power_watt = 10 ** ((rx_power_dbm - 30.0) / 10.0)
    tx_power_watt = 10 ** ((tx_power_dbm - 30.0) / 10.0)

    return FriisResult(
        tx_power_dbm=tx_power_dbm,
        tx_power_watt=tx_power_watt,
        tx_gain_dbi=tx_gain_dbi,
        rx_gain_dbi=rx_gain_dbi,
        distance_m=distance_m,
        frequency_hz=frequency_hz,
        wavelength_m=fspl.wavelength_m,
        fspl_db=fspl.fspl_db,
        rx_power_dbm=rx_power_dbm,
        rx_power_watt=rx_power_watt,
    )
