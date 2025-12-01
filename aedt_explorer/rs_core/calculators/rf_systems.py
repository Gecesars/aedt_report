"""General RF system calculators."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import units
from .conversions import dbm_to_watt


@dataclass(frozen=True, slots=True)
class LinkBudgetResult:
    tx_power_dbm: float
    rx_power_dbm: float
    rx_power_watt: float
    total_gains_db: float
    total_losses_db: float
    link_margin_db: float | None


@dataclass(frozen=True, slots=True)
class EirpResult:
    tx_power_dbm: float
    tx_power_watt: float
    tx_gain_dbi: float
    losses_db: float
    eirp_dbm: float
    eirp_watt: float


@dataclass(frozen=True, slots=True)
class MismatchLossResult:
    reflection_coefficient: float
    vswr: float
    return_loss_db: float
    mismatch_loss_db: float
    delivered_power_ratio: float


def link_budget(
    tx_power_dbm: float,
    tx_gain_db: float,
    tx_loss_db: float,
    fspl_db: float,
    misc_loss_db: float,
    rx_gain_db: float,
    rx_loss_db: float,
    rx_sensitivity_dbm: float | None,
) -> LinkBudgetResult:
    """Resolve a soma logarítmica do orçamento de enlace."""
    tx_power_dbm = float(tx_power_dbm)
    gains = tx_gain_db + rx_gain_db
    losses = tx_loss_db + fspl_db + misc_loss_db + rx_loss_db
    rx_power_dbm = tx_power_dbm + gains - losses
    rx_power_watt = dbm_to_watt(rx_power_dbm).power_watt
    margin = None
    if rx_sensitivity_dbm is not None:
        margin = rx_power_dbm - rx_sensitivity_dbm
    return LinkBudgetResult(
        tx_power_dbm=tx_power_dbm,
        rx_power_dbm=rx_power_dbm,
        rx_power_watt=rx_power_watt,
        total_gains_db=gains,
        total_losses_db=losses,
        link_margin_db=margin,
    )


def eirp(tx_power_dbm: float, tx_gain_dbi: float, losses_db: float) -> EirpResult:
    tx_power_watt = dbm_to_watt(tx_power_dbm).power_watt
    eirp_dbm = tx_power_dbm + tx_gain_dbi - losses_db
    eirp_watt = dbm_to_watt(eirp_dbm).power_watt
    return EirpResult(
        tx_power_dbm=tx_power_dbm,
        tx_power_watt=tx_power_watt,
        tx_gain_dbi=tx_gain_dbi,
        losses_db=losses_db,
        eirp_dbm=eirp_dbm,
        eirp_watt=eirp_watt,
    )


def mismatch_loss_from_gamma(gamma: float) -> MismatchLossResult:
    if gamma < 0 or gamma >= 1:
        raise ValueError("|Γ| deve estar no intervalo [0, 1).")
    return_loss_db = -20.0 * math.log10(gamma) if gamma > 0 else math.inf
    vswr = (1 + gamma) / (1 - gamma) if gamma < 1 else math.inf
    mismatch_loss_db = -10.0 * math.log10(1 - gamma ** 2)
    delivered_ratio = 1 - gamma ** 2
    return MismatchLossResult(
        reflection_coefficient=gamma,
        vswr=vswr,
        return_loss_db=return_loss_db,
        mismatch_loss_db=mismatch_loss_db,
        delivered_power_ratio=delivered_ratio,
    )
