"""Core conversion calculators for RF domain."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import units

C0 = units.C


@dataclass(frozen=True)
class DBmToWattResult:
    power_watt: float
    power_milliwatt: float
    power_dbm: float


def dbm_to_watt(dbm: float) -> DBmToWattResult:
    """Converts power level given in dBm to watts and milliwatts."""
    power_watt = 10 ** ((dbm - 30) / 10)
    power_milliwatt = power_watt * 1000.0
    return DBmToWattResult(power_watt=power_watt, power_milliwatt=power_milliwatt, power_dbm=dbm)


def watt_to_dbm(power_watt: float) -> DBmToWattResult:
    """Converts power level given in watts to dBm, returning derived variants."""
    power_watt = units.sanitize_positive(power_watt, field="Potência")
    power_dbm = 10 * math.log10(power_watt / 0.001)
    power_milliwatt = power_watt * 1000.0
    return DBmToWattResult(power_watt=power_watt, power_milliwatt=power_milliwatt, power_dbm=power_dbm)


@dataclass(frozen=True)
class VswrToReturnLossResult:
    vswr: float
    reflection_coefficient: float
    return_loss_db: float


def vswr_to_return_loss(vswr: float) -> VswrToReturnLossResult:
    vswr = units.sanitize_positive(vswr, field="VSWR")
    if vswr < 1.0:
        raise ValueError("VSWR deve ser maior ou igual a 1.")
    reflection = (vswr - 1) / (vswr + 1)
    if reflection == 0:
        rl = math.inf
    else:
        rl = -20 * math.log10(reflection)
    return VswrToReturnLossResult(vswr=vswr, reflection_coefficient=reflection, return_loss_db=rl)


def return_loss_to_vswr(return_loss_db: float) -> VswrToReturnLossResult:
    return_loss_db = units.sanitize_positive(return_loss_db, field="Return Loss")
    reflection = 10 ** (-return_loss_db / 20)
    vswr = (1 + reflection) / (1 - reflection) if reflection < 1 else math.inf
    return VswrToReturnLossResult(vswr=vswr, reflection_coefficient=reflection, return_loss_db=return_loss_db)
