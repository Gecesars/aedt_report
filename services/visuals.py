from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import current_app, url_for

from ..models import Antenna, AntennaPattern, PatternType, Project
from .metrics import (
    directivity_2d_cut,
    estimate_gain_dbi,
    first_null_deg,
    front_to_back_db,
    hpbw_deg,
    lin_to_db,
    peak_angle_deg,
    ripple_p2p_db,
    sidelobe_level_db,
)
from .pattern_composer import get_composition, resample_pattern, resample_vertical
from .exporters import _save_vertical_composition, _save_horizontal_composition


def _preview_root() -> tuple[Path, Path]:
    static_dir = Path(current_app.static_folder)
    rel_root = Path(current_app.config.get("PREVIEW_IMAGE_ROOT", "generated/previews"))
    root = static_dir / rel_root
    root.mkdir(parents=True, exist_ok=True)
    return root, rel_root


def _write_and_url(file_path: Path, rel_path: Path) -> str:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    mtime = int(time.time())
    rel_posix = rel_path.as_posix()
    return url_for("static", filename=rel_posix, v=mtime)


def _format_value(value: float, suffix: str) -> str:
    if value is None or not np.isfinite(value):
        return f"N/A{suffix}"
    return f"{value:.2f}{suffix}"


def _compute_metrics(angles: np.ndarray, values: np.ndarray, *, include_front_to_back: bool = False) -> dict[str, float]:
    metrics: dict[str, float] = {}
    peak_level = float(np.max(values)) if values.size else float("nan")
    metrics["peak_angle"] = peak_angle_deg(angles, values)
    metrics["peak_linear"] = peak_level
    metrics["peak_db"] = float(lin_to_db(peak_level)) if np.isfinite(peak_level) and peak_level > 0 else float("nan")
    metrics["hpbw"] = hpbw_deg(angles, values)
    metrics["first_null"] = first_null_deg(angles, values)
    metrics["sll"] = sidelobe_level_db(angles, values)
    metrics["ripple"] = ripple_p2p_db(angles, values)
    metrics["directivity"] = directivity_2d_cut(angles, values)
    if include_front_to_back:
        metrics["front_to_back"] = front_to_back_db(angles, values)
    return metrics


def _save_polar_plot(path: Path, angles: np.ndarray, values: np.ndarray, title: str) -> None:
    angles = np.asarray(angles, dtype=float)
    values = np.asarray(values, dtype=float)
    theta_deg = (angles + 360.0) % 360.0
    order = np.argsort(theta_deg)
    theta_sorted = theta_deg[order]
    values_sorted = values[order]
    theta_rad = np.deg2rad(theta_sorted)
    theta_wrapped = np.append(theta_rad, theta_rad[0] + 2 * np.pi)
    values_wrapped = np.append(values_sorted, values_sorted[0])

    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(theta_wrapped, values_wrapped, linewidth=1.6, color="#0A4E8B")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_planar_plot(
    path: Path,
    angles: np.ndarray,
    values: np.ndarray,
    title: str,
    highlight: Optional[tuple[float, float]] = None,
) -> None:
    angles = np.asarray(angles, dtype=float)
    values = np.asarray(values, dtype=float)
    ordered_indices = np.argsort(angles)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(angles[ordered_indices], values[ordered_indices], linewidth=1.6, color="#0A4E8B")
    ax.set_xlabel("Angulo (deg)")
    ax.set_ylabel("Amplitude (linear)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axvline(0.0, color="#6b7280", linestyle="--", linewidth=0.8, alpha=0.4)
    if highlight is not None:
        ang, val = highlight
        ax.scatter([ang], [val], color="#FF6A3D", s=32, zorder=5)
        ax.annotate(
            f"{val:.3f}",
            xy=(ang, val),
            xytext=(ang + 4, val + 0.03),
            fontsize=8,
            color="#FF6A3D",
            arrowprops=dict(arrowstyle="->", color="#FF6A3D", linewidth=0.8),
        )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _prepare_pattern(pattern: AntennaPattern) -> tuple[np.ndarray, np.ndarray]:
    angles = np.asarray(pattern.angles_deg, dtype=float)
    values = np.asarray(pattern.amplitudes_linear, dtype=float)

    if angles.size == 0 or values.size == 0:
        legacy = getattr(pattern, "metadata_json", None) or {}
        legacy_angles = np.asarray(legacy.get("angles_deg", []), dtype=float)
        legacy_values = np.asarray(legacy.get("amplitudes_linear", []), dtype=float)
        if legacy_angles.size and legacy_values.size:
            angles = legacy_angles
            values = legacy_values

    if angles.size == 0 or values.size == 0:
        if pattern.pattern_type is PatternType.HRP:
            angles = np.linspace(-180.0, 180.0, num=361)
        else:
            angles = np.linspace(-90.0, 90.0, num=181)
        values = np.zeros_like(angles)

    values = np.clip(values, 0.0, None)
    if pattern.pattern_type is PatternType.HRP:
        return resample_pattern(angles, values, -180, 180, 1)
    return resample_vertical(angles, values)


def _metrics_to_lines(metrics: dict[str, float], *, include_front_to_back: bool = False, gain_dbi: float | None = None) -> list[str]:
    lines = [
        f"Pico: {_format_value(metrics.get('peak_db'), ' dB')} @ {_format_value(metrics.get('peak_angle'), ' deg')}",
        f"HPBW: {_format_value(metrics.get('hpbw'), ' deg')}",
        f"Ripple: {_format_value(metrics.get('ripple'), ' dB')}",
        f"SLL: {_format_value(metrics.get('sll'), ' dB')}",
        f"1o nulo: {_format_value(metrics.get('first_null'), ' deg')}",
    ]
    directivity = metrics.get("directivity")
    if directivity is not None and np.isfinite(directivity):
        lines.append(f"Diretividade 2D: {10 * np.log10(directivity):.2f} dB")
    else:
        lines.append("Diretividade 2D: N/A")
    if include_front_to_back:
        lines.append(f"Front/back: {_format_value(metrics.get('front_to_back'), ' dB')}")
    if gain_dbi is not None and np.isfinite(gain_dbi):
        directivity_total = 10 ** (gain_dbi / 10)
        lines.append(f"Ganho estimado: {gain_dbi:.2f} dBi ({directivity_total:.1f}x)")
    return lines


def generate_project_previews(project: Project, composition: Optional[dict[str, np.ndarray]] = None) -> dict[str, dict[str, object]]:
    root, rel_root = _preview_root()
    project_dir = root / "projects" / str(project.id)
    shutil.rmtree(project_dir, ignore_errors=True)
    project_dir.mkdir(parents=True, exist_ok=True)

    if composition is None:
        composition, _ = get_composition(project, refresh=True)

    hrp_angles = composition.get("angles_deg", np.array([]))
    hrp_values = composition.get("hrp_linear", np.array([]))
    vrp_angles = composition.get("vrp_angles_deg", np.array([]))
    vrp_values = composition.get("vrp_linear", np.array([]))

    previews: dict[str, dict[str, object]] = {}

    if hrp_angles.size and hrp_values.size:
        hrp_metrics = _compute_metrics(hrp_angles, hrp_values, include_front_to_back=True)
        hrp_path = project_dir / "hrp_composite.png"
        _save_polar_plot(hrp_path, hrp_angles, hrp_values, "Padrao Horizontal Composto")
        gain_dbi = None
        if vrp_angles.size and vrp_values.size:
            vrp_metrics = _compute_metrics(vrp_angles, vrp_values, include_front_to_back=False)
            gain_dbi = estimate_gain_dbi(
                hrp_metrics.get("hpbw", float("nan")),
                vrp_metrics.get("hpbw", float("nan")),
            )
        previews["azimuth"] = {
            "image": _write_and_url(hrp_path, rel_root / "projects" / str(project.id) / "hrp_composite.png"),
            "stats": _metrics_to_lines(hrp_metrics, include_front_to_back=True, gain_dbi=gain_dbi),
        }

    if vrp_angles.size and vrp_values.size:
        vrp_metrics = _compute_metrics(vrp_angles, vrp_values, include_front_to_back=False)
        vrp_path = project_dir / "vrp_composite.png"
        horizon_idx = int(np.argmin(np.abs(vrp_angles)))
        horizon_value = float(vrp_values[horizon_idx]) if vrp_values.size else float("nan")
        _save_planar_plot(
            vrp_path,
            vrp_angles,
            vrp_values,
            "Padrao Vertical Composto",
            highlight=(0.0, horizon_value),
        )
        previews["elevation"] = {
            "image": _write_and_url(vrp_path, rel_root / "projects" / str(project.id) / "vrp_composite.png"),
            "stats": _metrics_to_lines(vrp_metrics, include_front_to_back=False, gain_dbi=None)
            + [f"E/Emax @ 0°: {horizon_value:.4f}"],
        }

    # Ilustrações de composição (sempre atualizadas pelos parâmetros do projeto)
    comp_v_path = project_dir / "composition_vertical.png"
    comp_h_path = project_dir / "composition_horizontal.png"
    try:
        _save_vertical_composition(comp_v_path, int(project.v_count or 1), float(project.v_spacing_m or 0.0), float(project.v_tilt_deg or 0.0))
        _save_horizontal_composition(comp_h_path, int(project.h_count or 1), float(project.h_spacing_m or 0.0), float(project.h_step_deg or 0.0))
        previews["comp_vertical"] = {
            "image": _write_and_url(comp_v_path, rel_root / "projects" / str(project.id) / "composition_vertical.png"),
            "stats": [
                f"N = {int(project.v_count or 1)}",
                f"Δv = {float(project.v_spacing_m or 0.0):.3f} m",
                f"tilt = {float(project.v_tilt_deg or 0.0):.1f}°",
            ],
        }
        previews["comp_horizontal"] = {
            "image": _write_and_url(comp_h_path, rel_root / "projects" / str(project.id) / "composition_horizontal.png"),
            "stats": [
                f"N = {int(project.h_count or 1)}",
                f"R = {float(project.h_spacing_m or 0.0):.3f} m",
                f"step = {float(project.h_step_deg or 0.0):.1f}°",
            ],
        }
    except Exception:
        pass

    return previews


def generate_pattern_previews(antenna: Antenna) -> list[dict[str, object]]:
    root, rel_root = _preview_root()
    antenna_dir = root / "antennas" / str(antenna.id)
    shutil.rmtree(antenna_dir, ignore_errors=True)
    antenna_dir.mkdir(parents=True, exist_ok=True)

    previews: list[dict[str, object]] = []
    for pattern in sorted(antenna.patterns, key=lambda p: p.pattern_type.value):
        angles, values = _prepare_pattern(pattern)
        metrics = _compute_metrics(angles, values, include_front_to_back=(pattern.pattern_type is PatternType.HRP))
        filename = f"{pattern.pattern_type.value.lower()}_raw.png"
        file_path = antenna_dir / filename
        lines = _metrics_to_lines(metrics, include_front_to_back=(pattern.pattern_type is PatternType.HRP))
        if pattern.pattern_type is PatternType.HRP:
            _save_polar_plot(file_path, angles, values, f"{pattern.pattern_type.value} - Azimute")
        else:
            _save_planar_plot(file_path, angles, values, f"{pattern.pattern_type.value} - Elevacao")
        url = _write_and_url(file_path, rel_root / "antennas" / str(antenna.id) / filename)
        previews.append({"type": pattern.pattern_type.value, "image": url, "stats": lines})
    return previews
