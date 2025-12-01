from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np

from ..extensions import db
from ..models import Antenna, PatternType, Project
from ..utils.calculations import vertical_beta_deg

C_LIGHT = 299_792_458.0
EPSILON = 1e-12


def resample_pattern(angles: Iterable[float], amplitudes: Iterable[float], start: int, stop: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    src_angles = np.asarray(list(angles), dtype=float)
    src_amp = np.asarray(list(amplitudes), dtype=float)
    dest_angles = np.arange(start, stop + step, step, dtype=float)
    sort_idx = np.argsort(src_angles)
    src_angles = src_angles[sort_idx]
    src_amp = src_amp[sort_idx]
    src_angles_ext = np.concatenate((src_angles - 360, src_angles, src_angles + 360))
    src_amp_ext = np.concatenate((src_amp, src_amp, src_amp))
    dest_amp = np.interp(dest_angles, src_angles_ext, src_amp_ext)
    return dest_angles, dest_amp


def resample_vertical(angles: Iterable[float], amplitudes: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    src_angles = np.asarray(list(angles), dtype=float)
    src_amp = np.asarray(list(amplitudes), dtype=float)
    dest_angles = np.round(np.arange(-90.0, 90.0 + 0.0001, 0.1), 1)
    sort_idx = np.argsort(src_angles)
    src_angles = src_angles[sort_idx]
    src_amp = src_amp[sort_idx]
    dest_amp = np.interp(dest_angles, src_angles, src_amp, left=src_amp[0], right=src_amp[-1])
    return dest_angles, dest_amp


def wavelength_m(frequency_mhz: float) -> float:
    return C_LIGHT / (frequency_mhz * 1_000_000)


def array_factor(
    count: int,
    spacing_m: float,
    beta_deg: float,
    level_amp: float,
    wave_number: float,
    angles_rad: np.ndarray,
    projection: str,
) -> np.ndarray:
    if count <= 1:
        return np.ones_like(angles_rad)
    beta = math.radians(beta_deg)
    indices = np.arange(count, dtype=float).reshape((-1, 1))
    weights = (level_amp ** indices) if level_amp not in (None, 0) else np.ones_like(indices)
    if projection == "horizontal":
        phase = wave_number * spacing_m * np.sin(angles_rad)
    else:
        phase = wave_number * spacing_m * np.cos(angles_rad)
    af = np.sum(weights * np.exp(1j * (indices * beta + phase)), axis=0)
    return np.abs(af)


def normalise(array: np.ndarray, mode: str) -> np.ndarray:
    if mode == "sum":
        total = np.sum(array)
        return array / total if total > EPSILON else array
    if mode == "first":
        base = array[0]
        return array / base if abs(base) > EPSILON else array
    peak = np.max(array)
    return array / peak if peak > EPSILON else array


def compose_horizontal_pattern(project: Project) -> tuple[np.ndarray, np.ndarray]:
    antenna: Antenna = project.antenna
    hrp = antenna.pattern_for(PatternType.HRP)
    if not hrp:
        angles = np.arange(-180, 181)
        return angles, np.ones_like(angles, dtype=float)

    base_angles = np.array(hrp.angles_deg, dtype=float)
    base_vals = np.array(hrp.amplitudes_linear, dtype=float)
    angles, element_pattern = resample_pattern(base_angles, base_vals, -180, 180, 1)

    count = max(int(project.h_count or 1), 1)
    beta_deg = project.h_beta_deg or 0.0
    spacing = max(project.h_spacing_m or 0.0, 0.0)
    level_amp = project.h_level_amp if project.h_level_amp not in (None, 0) else 1.0
    step_deg = project.h_step_deg or 0.0

    frequency_hz = max(project.frequency_mhz or 0.0, EPSILON) * 1e6
    wavelength = C_LIGHT / frequency_hz
    wave_number = 2.0 * math.pi / wavelength

    if count > 1 and spacing <= EPSILON:
        spacing = wavelength
        project.h_spacing_m = spacing

    angles_rad = np.radians(angles)
    composite = np.zeros_like(angles_rad, dtype=complex)

    if count == 1:
        composite = element_pattern.astype(complex)
    else:
        beta_rad = math.radians(beta_deg)
        if abs(step_deg) < 1e-6:
            offsets = (np.arange(count, dtype=float) - (count - 1) / 2.0) * spacing
            for idx, phi_rad in enumerate(angles_rad):
                total = 0.0 + 0.0j
                for m, offset in enumerate(offsets):
                    sample = element_pattern[idx]
                    phase_geom = wave_number * offset * math.sin(phi_rad)
                    phase_exc = beta_rad * m
                    total += level_amp * sample * np.exp(1j * (phase_geom + phase_exc))
                composite[idx] = total
        else:
            alpha_deg = np.arange(count, dtype=float) * step_deg
            alpha_rad = np.deg2rad(alpha_deg)
            radius = spacing / (2.0 * math.sin(math.pi / count)) if count > 1 else 0.0
            for idx, phi_deg in enumerate(angles):
                phi_rad = math.radians(phi_deg)
                ux = math.cos(phi_rad)
                uy = math.sin(phi_rad)
                total = 0.0 + 0.0j
                for m in range(count):
                    x_m = radius * math.cos(alpha_rad[m])
                    y_m = radius * math.sin(alpha_rad[m])
                    delta_r = x_m * ux + y_m * uy
                    phase_geom = wave_number * delta_r
                    phase_exc = beta_rad * m
                    rel_angle = (phi_deg - alpha_deg[m]) % 360.0
                    if rel_angle > 180.0:
                        rel_angle -= 360.0
                    sample = np.interp(
                        rel_angle,
                        angles,
                        element_pattern,
                        left=element_pattern[0],
                        right=element_pattern[-1],
                    )
                    total += level_amp * sample * np.exp(1j * (phase_geom + phase_exc))
                composite[idx] = total

    composite_mag = np.abs(composite)
    composite_mag = normalise(composite_mag, project.h_norm_mode or "max")
    return angles, composite_mag


def serialize_erp_payload(data: dict[str, np.ndarray]) -> dict[str, object]:
    serialised: dict[str, object] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            serialised[key] = value.tolist()
        else:
            serialised[key] = value
    return serialised


def get_composition(
    project: Project,
    *,
    refresh: bool = False,
    store: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    payload = project.composition_meta if not refresh else None
    if payload is None:
        payload = serialize_erp_payload(compute_erp(project))
        if store:
            project.composition_meta = payload
            try:
                db.session.add(project)
            except Exception:
                # Sessão pode não estar disponível (ex.: contexto read-only)
                pass

    arrays: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            arrays[key] = np.asarray(value, dtype=float)
        elif isinstance(value, np.ndarray):
            arrays[key] = value
        else:
            arrays[key] = value

    return arrays, payload


def compose_vertical_pattern(project: Project) -> tuple[np.ndarray, np.ndarray]:
    antenna: Antenna = project.antenna
    vrp = antenna.pattern_for(PatternType.VRP)
    if not vrp:
        angles = np.round(np.arange(-90.0, 90.0 + 0.0001, 0.1), 1)
        return angles, np.ones_like(angles, dtype=float)
    base_angles, base_amp = np.array(vrp.angles_deg, dtype=float), np.array(vrp.amplitudes_linear, dtype=float)
    angles, element_pattern = resample_vertical(base_angles, base_amp)

    count = max(int(project.v_count or 1), 1)
    spacing = project.v_spacing_m or 0.0
    beta_deg = vertical_beta_deg(project.frequency_mhz, spacing, project.v_tilt_deg or 0.0)
    project.v_beta_deg = beta_deg
    beta_rad = math.radians(beta_deg)
    level_amp = project.v_level_amp if project.v_level_amp not in (None, 0) else 1.0

    frequency_hz = max(project.frequency_mhz or 0.0, EPSILON) * 1e6
    wavelength = C_LIGHT / frequency_hz
    wave_number = 2.0 * math.pi / wavelength

    if count > 1 and spacing <= EPSILON:
        spacing = wavelength
        project.v_spacing_m = spacing

    theta_rad = np.deg2rad(angles)
    psi = wave_number * spacing * np.sin(theta_rad) + beta_rad
    indices = np.arange(count, dtype=float).reshape((-1, 1))
    weights = (level_amp ** indices) if level_amp not in (None, 0, 1) else np.ones_like(indices)
    af = np.sum(weights * np.exp(1j * indices * psi), axis=0)
    composite = element_pattern * np.abs(af)
    composite = normalise(composite, project.v_norm_mode or "max")
    return angles, composite


def compute_erp(project: Project) -> dict[str, np.ndarray]:
    angles_deg, hrp = compose_horizontal_pattern(project)
    v_angles_deg, vrp = compose_vertical_pattern(project)

    # Use the VRP value near 0° elevation to scale ERP radials (horizon)
    horizon_idx = int(np.argmin(np.abs(v_angles_deg)))
    vertical_scalar = max(vrp[horizon_idx], EPSILON)

    nominal_gain_dbd = project.antenna.nominal_gain_dbd or 0.0
    power_w = max(project.tx_power_w, EPSILON)
    feeder_loss_db = project.feeder_loss_db if project.feeder_loss_db is not None else 0.0

    hrp_composed = hrp * vertical_scalar
    hrp_linear = np.clip(hrp_composed, EPSILON, None)
    attenuation_db = 20 * np.log10(np.maximum(hrp_linear, EPSILON))
    gain_dbd = nominal_gain_dbd - attenuation_db
    erp_dbw = 10 * np.log10(power_w) - feeder_loss_db + gain_dbd
    erp_w = 10 ** (erp_dbw / 10)
    return {
        "angles_deg": angles_deg,
        "hrp_linear": hrp,
        "vrp_angles_deg": v_angles_deg,
        "vrp_linear": vrp,
        "attenuation_db": attenuation_db,
        "gain_dbd": gain_dbd,
        "erp_dbw": erp_dbw,
        "erp_w": erp_w,
        "vertical_scalar": vertical_scalar,
        "effective_h_spacing_m": project.h_spacing_m,
        "effective_v_spacing_m": project.v_spacing_m,
    }


def export_pat(file_path: Path, data: dict[str, np.ndarray]) -> None:
    angles = data["angles_deg"].astype(int)
    erp_dbw = data["erp_dbw"]
    with file_path.open("w", encoding="ascii") as fh:
        fh.write("# EFTX PAT export\n")
        for angle, value in zip(angles, erp_dbw):
            fh.write(f"{int(angle % 360):03d}\t{value:.3f}\n")


def export_prn(file_path: Path, data: dict[str, np.ndarray]) -> None:
    angles = data["angles_deg"].astype(int)
    erp_w = data["erp_w"]
    with file_path.open("w", encoding="ascii") as fh:
        fh.write("Angle,ERP_W\n")
        for angle, value in zip(angles, erp_w):
            fh.write(f"{int(angle % 360)},{value:.6f}\n")
