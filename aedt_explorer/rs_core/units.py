"""Conversões de unidades auxiliares para calculadoras de RF."""
from __future__ import annotations

from dataclasses import dataclass

C = 299_792_458.0  # velocidade da luz no vácuo (m/s)


FREQUENCY_FACTORS = {
    "hz": 1.0,
    "khz": 1e3,
    "mhz": 1e6,
    "ghz": 1e9,
}

LENGTH_FACTORS = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 1e-3,
    "mil": 25.4e-6,
    "in": 0.0254,
    "ft": 0.3048,
    "km": 1000.0,
    "um": 1e-6,
}



def to_hz(value: float, unit: str) -> float:
    if unit not in FREQUENCY_FACTORS:
        raise ValueError(f"Unidade de frequência inválida: {unit}")
    return float(value) * FREQUENCY_FACTORS[unit]


def from_hz(value_hz: float, unit: str) -> float:
    if unit not in FREQUENCY_FACTORS:
        raise ValueError(f"Unidade de frequência inválida: {unit}")
    return float(value_hz) / FREQUENCY_FACTORS[unit]


def to_meters(value: float, unit: str) -> float:
    if unit not in LENGTH_FACTORS:
        raise ValueError(f"Unidade de comprimento inválida: {unit}")
    return float(value) * LENGTH_FACTORS[unit]


def from_meters(value_m: float, unit: str) -> float:
    if unit not in LENGTH_FACTORS:
        raise ValueError(f"Unidade de comprimento inválida: {unit}")
    return float(value_m) / LENGTH_FACTORS[unit]


def sanitize_positive(value: float, *, strict: bool = True, field: str | None = None) -> float:
    if strict and value <= 0:
        label = field or "valor"
        raise ValueError(f"{label.capitalize()} deve ser maior que zero.")
    if not strict and value < 0:
        label = field or "valor"
        raise ValueError(f"{label.capitalize()} não pode ser negativo.")
    return float(value)


@dataclass(slots=True)
class UnitValue:
    """Representa um valor físico com unidade para facilitar exibição."""

    value: float
    unit: str

    def as_unit(self, desired_unit: str) -> float:
        if self.unit in FREQUENCY_FACTORS:
            hz = to_hz(self.value, self.unit)
            return from_hz(hz, desired_unit)
        if self.unit in LENGTH_FACTORS:
            meters = to_meters(self.value, self.unit)
            return from_meters(meters, desired_unit)
        raise ValueError(f"Conversão não suportada para {self.unit}")
