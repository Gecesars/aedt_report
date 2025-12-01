from __future__ import annotations

import math
from typing import Iterable

import numpy as np

EPSILON = 1e-12


def lin_to_db(values: np.ndarray | float) -> np.ndarray | float:
    arr = np.maximum(values, EPSILON)
    return 20.0 * np.log10(arr)


def lin_to_att_db(value: float) -> float:
    if value <= 0:
        return 100.0
    return max(0.0, -20.0 * math.log10(value))


def hpbw_deg(angles_deg: np.ndarray, values: np.ndarray) -> float:
    if len(angles_deg) < 3:
        return float("nan")
    normalized = values / np.max(values) if np.max(values) > 0 else values
    threshold = math.sqrt(0.5)
    idx_peak = int(np.argmax(normalized))
    left = None
    for i in range(idx_peak, 0, -1):
        if normalized[i] >= threshold and normalized[i - 1] < threshold:
            left = np.interp(
                threshold,
                [normalized[i - 1], normalized[i]],
                [angles_deg[i - 1], angles_deg[i]],
            )
            break
    right = None
    for i in range(idx_peak, len(normalized) - 1):
        if normalized[i] >= threshold and normalized[i + 1] < threshold:
            right = np.interp(
                threshold,
                [normalized[i], normalized[i + 1]],
                [angles_deg[i], angles_deg[i + 1]],
            )
            break
    if left is None or right is None:
        return float("nan")
    return float(right - left)


def front_to_back_db(angles_deg: np.ndarray, values: np.ndarray, window: float = 30.0) -> float:
    if len(angles_deg) == 0:
        return float("nan")
    normalized = values / np.max(values) if np.max(values) > 0 else values
    angles_norm = (angles_deg + 360.0) % 360.0
    mask = (angles_norm >= (180.0 - window)) & (angles_norm <= (180.0 + window))
    if not np.any(mask):
        return float("nan")
    back = np.max(normalized[mask])
    if back <= 0:
        return 100.0
    return max(0.0, -20.0 * math.log10(back))


def ripple_p2p_db(angles_deg: np.ndarray, values: np.ndarray, threshold_db: float = -6.0) -> float:
    if len(angles_deg) == 0:
        return float("nan")
    normalized = values / np.max(values) if np.max(values) > 0 else values
    db_vals = lin_to_db(normalized)
    peak_idx = int(np.argmax(normalized))
    left = peak_idx
    while left > 0 and db_vals[left - 1] >= threshold_db:
        left -= 1
    right = peak_idx
    while right < len(db_vals) - 1 and db_vals[right + 1] >= threshold_db:
        right += 1
    sector = db_vals[left:right + 1]
    if sector.size == 0:
        return float("nan")
    return float(np.max(sector) - np.min(sector))


def sidelobe_level_db(angles_deg: np.ndarray, values: np.ndarray, threshold_db: float = -6.0) -> float:
    if len(angles_deg) == 0:
        return float("nan")
    normalized = values / np.max(values) if np.max(values) > 0 else values
    db_vals = lin_to_db(normalized)
    peak_idx = int(np.argmax(normalized))
    left = peak_idx
    while left > 0 and db_vals[left - 1] >= threshold_db:
        left -= 1
    right = peak_idx
    while right < len(db_vals) - 1 and db_vals[right + 1] >= threshold_db:
        right += 1
    outside = np.concatenate([db_vals[:left], db_vals[right + 1:]]) if (left > 0 or right < len(db_vals) - 1) else np.array([])
    if outside.size == 0:
        return float("nan")
    return float(np.max(outside))


def peak_angle_deg(angles_deg: np.ndarray, values: np.ndarray) -> float:
    if len(angles_deg) == 0:
        return float("nan")
    return float(angles_deg[int(np.argmax(values))])


def estimate_gain_dbi(h_hpbw: float, v_hpbw: float) -> float:
    if not (math.isfinite(h_hpbw) and math.isfinite(v_hpbw) and h_hpbw > 0 and v_hpbw > 0):
        return float("nan")
    directivity = 41253.0 / (h_hpbw * v_hpbw)
    return 10.0 * math.log10(directivity)


def first_null_deg(angles_deg: np.ndarray, values: np.ndarray) -> float:
    if len(angles_deg) < 3:
        return float("nan")
    peak_idx = int(np.argmax(values))
    max_val = values[peak_idx]
    if max_val <= EPSILON:
        return float("nan")
    threshold = max_val * 0.05
    for idx in range(peak_idx + 1, len(values)):
        if values[idx] <= threshold:
            return float(angles_deg[idx])
    return float("nan")


def directivity_2d_cut(angles_deg: np.ndarray, values: np.ndarray) -> float:
    if len(angles_deg) < 2:
        return float("nan")
    peak = np.max(values)
    if peak <= EPSILON:
        return float("nan")
    normalized = values / peak
    power = normalized ** 2
    angles_rad = np.deg2rad(angles_deg)
    integral = np.trapz(power, angles_rad)
    span = float(angles_rad[-1] - angles_rad[0])
    if integral <= EPSILON or span <= EPSILON:
        return float("nan")
    return float(span / integral)