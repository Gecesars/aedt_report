"""Cálculos de microfita (Hammerstad & Jensen)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import units


@dataclass(slots=True)
class MicrostripResult:
    width_m: float
    width_over_height: float
    effective_eps: float
    impedance_ohms: float
    warnings: list[str]


def _effective_eps(eps_r: float, u: float) -> float:
    u = max(u, 1e-12)
    if u <= 1:
        term = 1 / math.sqrt(1 + 12 / u) + 0.04 * (1 - u) ** 2
        return (eps_r + 1) / 2 + (eps_r - 1) / 2 * term
    return (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 / math.sqrt(1 + 12 / u))


def _delta_u(u: float, t_h: float, eps_eff: float) -> float:
    if t_h <= 0:
        return 0.0
    u = max(u, 1e-9)
    x = math.sqrt(6.517 * u)
    tanh_term = math.tanh(x) if x < 20 else 1.0  # evita overflow
    if tanh_term == 0.0:
        tanh_term = 1e-9
    return (t_h / math.pi) * (1 + 1 / eps_eff) * math.log(1 + (4 * math.e) / (t_h * (1 / tanh_term)))


def _characteristic_impedance(eps_r: float, u: float, t_h: float) -> tuple[float, float, float]:
    eps_eff = _effective_eps(eps_r, u)
    for _ in range(3):
        delta_u = _delta_u(u, t_h, eps_eff)
        u_eff = u + delta_u
        new_eps = _effective_eps(eps_r, u_eff)
        if abs(new_eps - eps_eff) < 1e-6:
            eps_eff = new_eps
            break
        eps_eff = new_eps
    else:
        u_eff = u + _delta_u(u, t_h, eps_eff)
    if u_eff <= 1:
        z0 = (60 / math.sqrt(eps_eff)) * math.log(8 / u_eff + 0.25 * u_eff)
    else:
        z0 = (120 * math.pi) / (
            math.sqrt(eps_eff) * (u_eff + 1.393 + 0.667 * math.log(u_eff + 1.444))
        )
    return z0, eps_eff, u_eff


def width_for_impedance(
    impedance_ohms: float,
    eps_r: float,
    substrate_height_m: float,
    conductor_thickness_m: float = 0.0,
) -> MicrostripResult:
    impedance_ohms = units.sanitize_positive(impedance_ohms, field="Z₀")
    eps_r = units.sanitize_positive(eps_r, field="εᵣ")
    substrate_height_m = units.sanitize_positive(substrate_height_m, field="h")
    conductor_thickness_m = max(conductor_thickness_m, 0.0)

    target = impedance_ohms
    h = substrate_height_m
    t_h = conductor_thickness_m / h if h > 0 else 0.0

    def z_from_u(u: float) -> tuple[float, float, float]:
        return _characteristic_impedance(eps_r, u, t_h)

    u_low = 1e-4
    z_low, eps_low, ueff_low = z_from_u(u_low)
    if z_low < target:
        # alvo muito baixo: aumentar largura ainda mais reduz impedância
        raise ValueError("Impedância alvo menor que limite atingível para dimensões físicas informadas.")

    u_high = 1.0
    z_high, eps_high, ueff_high = z_from_u(u_high)
    attempts = 0
    while z_high > target and attempts < 40:
        u_high *= 2
        z_high, eps_high, ueff_high = z_from_u(u_high)
        attempts += 1
        if u_high > 1e5:
            raise ValueError("Não foi possível encontrar largura para a impedância especificada.")

    # Busca binária
    solution_eps = eps_low
    solution_ueff = ueff_low
    u_mid = u_high
    for _ in range(80):
        u_mid = 0.5 * (u_low + u_high)
        z_mid, eps_mid, ueff_mid = z_from_u(u_mid)
        if abs(z_mid - target) / target < 1e-6:
            solution_eps = eps_mid
            solution_ueff = ueff_mid
            break
        if z_mid > target:
            u_low = u_mid
        else:
            u_high = u_mid
        solution_eps = eps_mid
        solution_ueff = ueff_mid

    width = u_mid * h
    warnings: list[str] = []
    if solution_ueff < 0.1 or solution_ueff > 20:
        warnings.append(
            "W/h fora da faixa típica de validade do modelo Hammerstad (0,1 a 20). Resultados podem ter maior erro."
        )
    return MicrostripResult(
        width_m=width,
        width_over_height=u_mid,
        effective_eps=solution_eps,
        impedance_ohms=target,
        warnings=warnings,
    )
