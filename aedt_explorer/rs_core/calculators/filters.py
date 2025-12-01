"""Passive filter order estimators and LC prototype scaling helpers."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .. import units


@dataclass(slots=True)
class ButterworthOrderResult:
    passband_ripple_db: float
    stopband_atten_db: float
    omega_p_hz: float
    omega_s_hz: float
    ratio: float
    order: int


def butterworth_minimum_order(ap_db: float, as_db: float, omega_p_hz: float, omega_s_hz: float) -> ButterworthOrderResult:
    omega_p_hz = units.sanitize_positive(omega_p_hz, field="Frequência de passagem (Hz)")
    omega_s_hz = units.sanitize_positive(omega_s_hz, field="Frequência de rejeição (Hz)")
    if omega_s_hz <= omega_p_hz:
        raise ValueError("A frequência de rejeição deve ser maior que a de passagem.")

    epsilon_p = max(10 ** (ap_db / 10.0) - 1.0, 1e-12)
    epsilon_s = 10 ** (as_db / 10.0) - 1.0
    if epsilon_s <= epsilon_p:
        raise ValueError("A atenuação de rejeição deve ser maior que a ondulação de passagem.")
    ratio = omega_s_hz / omega_p_hz

    numerator = math.log10(epsilon_s) - math.log10(epsilon_p)
    denominator = 2.0 * math.log10(ratio)
    order = math.ceil(numerator / denominator)

    return ButterworthOrderResult(
        passband_ripple_db=ap_db,
        stopband_atten_db=as_db,
        omega_p_hz=omega_p_hz,
        omega_s_hz=omega_s_hz,
        ratio=ratio,
        order=order,
    )


@dataclass(slots=True)
class ChebyshevOrderResult:
    passband_ripple_db: float
    stopband_atten_db: float
    omega_p_hz: float
    omega_s_hz: float
    ratio: float
    ripple_factor: float
    order: int


def chebyshev_type1_minimum_order(ap_db: float, as_db: float, omega_p_hz: float, omega_s_hz: float) -> ChebyshevOrderResult:
    omega_p_hz = units.sanitize_positive(omega_p_hz, field="Frequência de passagem (Hz)")
    omega_s_hz = units.sanitize_positive(omega_s_hz, field="Frequência de rejeição (Hz)")
    if omega_s_hz <= omega_p_hz:
        raise ValueError("A frequência de rejeição deve ser maior que a de passagem.")

    ratio = omega_s_hz / omega_p_hz
    epsilon = math.sqrt(max(10 ** (ap_db / 10.0) - 1.0, 1e-12))
    epsilon_s = math.sqrt(10 ** (as_db / 10.0) - 1.0)
    if epsilon_s <= epsilon:
        raise ValueError("A atenuação de rejeição deve ser maior que a ondulação de passagem.")

    ratio_term = epsilon_s / epsilon
    numerator = math.acosh(ratio_term)
    denominator = math.acosh(ratio)
    order = math.ceil(numerator / denominator)

    return ChebyshevOrderResult(
        passband_ripple_db=ap_db,
        stopband_atten_db=as_db,
        omega_p_hz=omega_p_hz,
        omega_s_hz=omega_s_hz,
        ratio=ratio,
        ripple_factor=epsilon,
        order=order,
    )


@dataclass(slots=True)
class CombinedOrderResult:
    """Wraps Butterworth and Chebyshev Type I minimum orders for quick comparison."""

    butterworth: ButterworthOrderResult
    chebyshev: ChebyshevOrderResult
    difference: float


@dataclass(slots=True)
class LcComponentResult:
    label: str
    index: int
    component: str
    configuration: str
    value: float
    unit: str
    notes: str | None = None


@dataclass(slots=True)
class LcPrototypeScalingResult:
    filter_type: str
    impedance_ohm: float
    cutoff_hz: float | None
    center_hz: float | None
    bandwidth_hz: float | None
    fractional_bandwidth: float | None
    g_values: list[float]
    components: list[LcComponentResult]


def combined_filter_orders(ap_db: float, as_db: float, omega_p_hz: float, omega_s_hz: float) -> CombinedOrderResult:
    butter = butterworth_minimum_order(ap_db, as_db, omega_p_hz, omega_s_hz)
    cheb = chebyshev_type1_minimum_order(ap_db, as_db, omega_p_hz, omega_s_hz)
    difference = float(cheb.order - butter.order)
    return CombinedOrderResult(butterworth=butter, chebyshev=cheb, difference=difference)


def _coerce_g_values(values: Iterable[float]) -> list[float]:
    coerced: list[float] = []
    for raw in values:
        value = float(raw)
        if value <= 0:
            continue
        coerced.append(value)
    if not coerced:
        raise ValueError('Forneça ao menos um valor g_k positivo.')
    return coerced


def lc_prototype_scaling(
    filter_type: str,
    g_values: Sequence[float],
    impedance_ohm: float,
    cutoff_frequency_hz: float | None = None,
    center_frequency_hz: float | None = None,
    bandwidth_hz: float | None = None,
) -> LcPrototypeScalingResult:
    """Scale normalized low-pass prototype coefficients to real component values."""

    normalized = _coerce_g_values(g_values)
    z0 = units.sanitize_positive(impedance_ohm, field='Impedância (Ω)')
    filter_type = filter_type.lower().strip()

    omega_c = None
    omega_0 = None
    fbw = None

    if filter_type in {'lowpass', 'highpass'}:
        if cutoff_frequency_hz is None:
            raise ValueError('Informe a frequência de corte para filtros passa-baixas/passa-altas.')
        omega_c = 2.0 * math.pi * units.sanitize_positive(cutoff_frequency_hz, field='Frequência de corte (Hz)')
    elif filter_type in {'bandpass', 'bandstop'}:
        if center_frequency_hz is None or bandwidth_hz is None:
            raise ValueError('Forneça a frequência central e a largura de banda para filtros passa-faixa/rejeita-faixa.')
        center = units.sanitize_positive(center_frequency_hz, field='Frequência central (Hz)')
        bandwidth = units.sanitize_positive(bandwidth_hz, field='Largura de banda (Hz)')
        if bandwidth >= center:
            raise ValueError('A largura de banda deve ser menor que a frequência central.')
        omega_0 = 2.0 * math.pi * center
        fbw = bandwidth / center
        if fbw <= 0:
            raise ValueError('A largura de banda fracionária deve ser positiva.')
    else:
        raise ValueError('Tipo de filtro inválido. Use lowpass, highpass, bandpass ou bandstop.')

    components: list[LcComponentResult] = []

    for idx, gk in enumerate(normalized, start=1):
        is_series = idx % 2 == 1
        if filter_type == 'lowpass':
            assert omega_c is not None
            if is_series:
                value = z0 * gk / omega_c
                components.append(
                    LcComponentResult(
                        label=f'g{idx}-L',
                        index=idx,
                        component='indutor',
                        configuration='série',
                        value=value,
                        unit='H',
                    )
                )
            else:
                value = gk / (z0 * omega_c)
                components.append(
                    LcComponentResult(
                        label=f'g{idx}-C',
                        index=idx,
                        component='capacitor',
                        configuration='shunt',
                        value=value,
                        unit='F',
                    )
                )
            continue

        if filter_type == 'highpass':
            assert omega_c is not None
            if is_series:
                base_l = z0 * gk / omega_c
                c_hp = 1.0 / (base_l * (omega_c ** 2))
                components.append(
                    LcComponentResult(
                        label=f'g{idx}-C',
                        index=idx,
                        component='capacitor',
                        configuration='série',
                        value=c_hp,
                        unit='F',
                        notes='Transformação de indutor série para capacitor série',
                    )
                )
            else:
                base_c = gk / (z0 * omega_c)
                l_hp = 1.0 / (base_c * (omega_c ** 2))
                components.append(
                    LcComponentResult(
                        label=f'g{idx}-L',
                        index=idx,
                        component='indutor',
                        configuration='shunt',
                        value=l_hp,
                        unit='H',
                        notes='Transformação de capacitor shunt para indutor shunt',
                    )
                )
            continue

        # Band transformations rely on omega_0 and fbw being defined
        assert omega_0 is not None and fbw is not None

        if filter_type == 'bandpass':
            if is_series:
                l_series = z0 * gk / (omega_0 * fbw)
                c_series = fbw / (z0 * gk * omega_0)
                components.extend(
                    [
                        LcComponentResult(
                            label=f'g{idx}-L',
                            index=idx,
                            component='indutor',
                            configuration='série',
                            value=l_series,
                            unit='H',
                            notes='Ramo ressonante série',
                        ),
                        LcComponentResult(
                            label=f'g{idx}-C',
                            index=idx,
                            component='capacitor',
                            configuration='série',
                            value=c_series,
                            unit='F',
                            notes='Ramo ressonante série',
                        ),
                    ]
                )
            else:
                c_parallel = gk / (z0 * omega_0 * fbw)
                l_parallel = (z0 * fbw) / (gk * omega_0)
                components.extend(
                    [
                        LcComponentResult(
                            label=f'g{idx}-C',
                            index=idx,
                            component='capacitor',
                            configuration='shunt',
                            value=c_parallel,
                            unit='F',
                            notes='Ramo ressonante paralelo',
                        ),
                        LcComponentResult(
                            label=f'g{idx}-L',
                            index=idx,
                            component='indutor',
                            configuration='shunt',
                            value=l_parallel,
                            unit='H',
                            notes='Ramo ressonante paralelo',
                        ),
                    ]
                )
            continue

        if filter_type == 'bandstop':
            if is_series:
                c_parallel = fbw / (z0 * gk * omega_0)
                l_parallel = z0 * gk / (omega_0 * fbw)
                components.extend(
                    [
                        LcComponentResult(
                            label=f'g{idx}-C',
                            index=idx,
                            component='capacitor',
                            configuration='shunt',
                            value=c_parallel,
                            unit='F',
                            notes='Ramo ressonante paralelo',
                        ),
                        LcComponentResult(
                            label=f'g{idx}-L',
                            index=idx,
                            component='indutor',
                            configuration='shunt',
                            value=l_parallel,
                            unit='H',
                            notes='Ramo ressonante paralelo',
                        ),
                    ]
                )
            else:
                l_series = (z0 * fbw) / (gk * omega_0)
                c_series = gk / (z0 * omega_0 * fbw)
                components.extend(
                    [
                        LcComponentResult(
                            label=f'g{idx}-L',
                            index=idx,
                            component='indutor',
                            configuration='série',
                            value=l_series,
                            unit='H',
                            notes='Ramo ressonante série',
                        ),
                        LcComponentResult(
                            label=f'g{idx}-C',
                            index=idx,
                            component='capacitor',
                            configuration='série',
                            value=c_series,
                            unit='F',
                            notes='Ramo ressonante série',
                        ),
                    ]
                )

    return LcPrototypeScalingResult(
        filter_type=filter_type,
        impedance_ohm=z0,
        cutoff_hz=omega_c / (2.0 * math.pi) if omega_c else None,
        center_hz=omega_0 / (2.0 * math.pi) if omega_0 else None,
        bandwidth_hz=bandwidth_hz if bandwidth_hz else None,
        fractional_bandwidth=fbw,
        g_values=list(normalized),
        components=components,
    )
