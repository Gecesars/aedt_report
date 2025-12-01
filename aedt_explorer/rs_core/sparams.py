"""Conversões e utilidades para parâmetros S."""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

from . import units


# Tolerância numérica para comparações (evita falsos "casamento perfeito")
EPS = 1e-9


class SParameterError(ValueError):
    """Erro de domínio para cálculos de S-parâmetros."""


@dataclass(slots=True)
class SParameterResult:
    magnitude_linear: float
    magnitude_db: float
    phase_deg: float
    gamma_complex: complex
    gamma_mag: float
    reflection_loss_db: float  # mismatch loss (−10·log10(1−|Γ|²))
    return_loss_db: float      # return loss (−20·log10|Γ|)
    vswr: float
    rho: float                 # |Γ| (alias)


# -------------------------
# Conversões em dB / linear
# -------------------------

def magnitude_db_to_linear(db_value: float) -> float:
    return pow(10.0, db_value / 20.0)


def magnitude_linear_to_db(magnitude: float) -> float:
    # Para amplitude em dB: requer magnitude > 0
    magnitude = units.sanitize_positive(magnitude, field="|S|")
    return 20.0 * math.log10(magnitude)


def power_db_to_linear(db_value: float) -> float:
    return pow(10.0, db_value / 10.0)


def power_linear_to_db(linear: float) -> float:
    linear = units.sanitize_positive(linear, field="potência")
    return 10.0 * math.log10(linear)


# -------------------------
# Relações VSWR / Γ / RL
# -------------------------

def gamma_from_vswr(vswr: float) -> float:
    vswr = units.sanitize_positive(vswr, field="VSWR")
    # Política: não aceitar casamento perfeito
    if vswr <= 1.0 + EPS:
        # Explicita para o usuário (não é erro 500 se tratado no handler)
        raise SParameterError("VSWR deve ser > 1.0 (casamento perfeito não é aceito).")
    return (vswr - 1.0) / (vswr + 1.0)


def vswr_from_gamma(gamma_mag: float) -> float:
    if not (0.0 <= gamma_mag < 1.0 + EPS):
        raise SParameterError("|Γ| deve estar no intervalo [0, 1).")
    if gamma_mag >= 1.0 - EPS:
        # Tendendo a reflexão total
        return math.inf
    return (1.0 + gamma_mag) / (1.0 - gamma_mag)


def return_loss_from_gamma(gamma_mag: float) -> float:
    if gamma_mag < 0:
        raise SParameterError("|Γ| não pode ser negativo.")
    # Política: não aceitar casamento perfeito (|Γ|≈0)
    if gamma_mag <= EPS:
        raise SParameterError("Casamento perfeito não é aceito (|Γ|≈0).")
    if gamma_mag < 1.0 - EPS:
        return -20.0 * math.log10(gamma_mag)
    if gamma_mag <= 1.0 + EPS:
        # Reflexão total → RL ≈ 0 dB
        return 0.0
    # |Γ|>1 implicaria rede ativa/instável (fora do escopo)
    raise SParameterError("|Γ| > 1 não suportado para redes passivas.")


def gamma_from_return_loss(return_loss_db: float) -> float:
    if return_loss_db is None or not math.isfinite(return_loss_db):
        raise SParameterError("RL inválido.")
    if return_loss_db < 0:
        raise SParameterError("RL negativo não suportado em redes passivas.")
    # RL=0 dB ⇒ |Γ|=1; RL>0 ⇒ 0<|Γ|<1; RL → +∞ ⇒ |Γ| → 0
    g = pow(10.0, -return_loss_db / 20.0)
    # Política: não aceitar casamento perfeito (RL→∞ ⇒ |Γ|→0)
    if g <= EPS:
        raise SParameterError("Casamento perfeito não é aceito (RL→∞).")
    if g > 1.0 + EPS:
        raise SParameterError("|Γ| > 1 derivado de RL — não suportado.")
    return min(g, 1.0)  # clamp seguro para tolerância


def vswr_from_return_loss(return_loss_db: float) -> float:
    gamma = gamma_from_return_loss(return_loss_db)
    return vswr_from_gamma(gamma)


def return_loss_from_vswr(vswr: float) -> float:
    gamma = gamma_from_vswr(vswr)
    return return_loss_from_gamma(gamma)


def mismatch_loss_db(gamma_mag: float) -> float:
    if gamma_mag < 0:
        raise SParameterError("|Γ| não pode ser negativo.")
    if gamma_mag >= 1.0 - EPS:
        # Reflexão total (ou muito próxima) ⇒ toda potência refletida ⇒ ML = +∞
        return math.inf
    # ML = -10*log10(1 - |Γ|^2); usar log1p para estabilidade numérica
    return -10.0 * (math.log1p(-gamma_mag * gamma_mag) / math.log(10.0))


# -------------------------
# Construção de resultados
# -------------------------

def sparameter_from_linear_phase(magnitude_linear: float, phase_deg: float) -> SParameterResult:
    # Aqui magnitude_linear é usado como |Γ| (módulo)
    if magnitude_linear < 0:
        raise SParameterError("|S|/|Γ| não pode ser negativo.")
    # Política: não aceitar casamento perfeito
    if magnitude_linear <= EPS:
        raise SParameterError("Casamento perfeito não é aceito (|Γ|≈0).")

    phase_rad = math.radians(phase_deg)
    gamma_complex = cmath.rect(magnitude_linear, phase_rad)
    gamma_mag = abs(gamma_complex)

    if gamma_mag < 1.0 - EPS:
        vswr = vswr_from_gamma(gamma_mag)
        rl = return_loss_from_gamma(gamma_mag)
        ml = mismatch_loss_db(gamma_mag)
    elif gamma_mag <= 1.0 + EPS:
        vswr = math.inf
        rl = 0.0
        ml = math.inf
    else:
        raise SParameterError("|Γ| > 1 não suportado para redes passivas.")

    # magnitude_db: para |Γ|→0 seria -inf, mas já bloqueamos casamento perfeito
    mag_db = magnitude_linear_to_db(magnitude_linear)

    return SParameterResult(
        magnitude_linear=magnitude_linear,
        magnitude_db=mag_db,
        phase_deg=normalized_phase(phase_deg),
        gamma_complex=gamma_complex,
        gamma_mag=gamma_mag,
        reflection_loss_db=ml,
        return_loss_db=rl,
        vswr=vswr,
        rho=gamma_mag,
    )


def sparameter_from_db_phase(magnitude_db: float, phase_deg: float) -> SParameterResult:
    magnitude_linear = magnitude_db_to_linear(magnitude_db)
    return sparameter_from_linear_phase(magnitude_linear, phase_deg)


def sparameter_from_vswr(vswr: float, phase_deg: float = 0.0) -> SParameterResult:
    gamma_mag = gamma_from_vswr(vswr)
    result = sparameter_from_linear_phase(gamma_mag, phase_deg)
    return result


# -------------------------
# Utilitário de fase
# -------------------------

def normalized_phase(angle_deg: float) -> float:
    """Normaliza fase para [-180, 180)."""
    x = (angle_deg + 180.0) % 360.0 - 180.0
    # Fix para -180 exato cair no intervalo
    return -180.0 if abs(x - 180.0) < 1e-12 else x



