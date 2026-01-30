from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


def _save_or_show(out_path: Optional[str | Path], show: bool) -> None:
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def _nan_interp_1d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fill NaNs in y by linear interpolation over x (plotting only)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        return np.zeros_like(y, dtype=float)
    return np.interp(x, x[m], y[m])


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding (same length)."""
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    w = int(max(1, window))
    if w == 1:
        return y
    if w % 2 == 0:
        w += 1
    half = w // 2
    kernel = np.ones(w, dtype=float) / float(w)
    y_pad = np.pad(y, (half, half), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


def _enforce_monotone_nonincreasing(y: np.ndarray) -> np.ndarray:
    """
    SOC during a discharge segment should be non-increasing.
    This is ONLY for visualization to remove tiny upward jumps caused by quantization/noise.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    out = y.copy()
    # cumulative minimum enforces non-increasing
    out = np.minimum.accumulate(out)
    return out


def plot_power(
    t_s: np.ndarray,
    p_meas: np.ndarray,
    p_pred: np.ndarray,
    out_path: Optional[str | Path] = None,
    show: bool = False
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(np.asarray(t_s, dtype=float) / 60.0, p_meas, label="measured")
    plt.plot(np.asarray(t_s, dtype=float) / 60.0, p_pred, label="model")
    plt.xlabel("Time (min)")
    plt.ylabel("Power (W)")
    plt.title("Power vs time (measured vs model)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_or_show(out_path, show)


def plot_soc(
    t_s: np.ndarray,
    soc_meas: np.ndarray,
    soc_sim: np.ndarray,
    out_path: Optional[str | Path] = None,
    show: bool = False
) -> None:
    """
    Aesthetics-oriented SOC plot:
      - measured SOC is plotted as a smooth curve (no staircase)
      - use monotone PCHIP interpolation to make it look like a continuous discharge curve
      - y-axis shown in percent and fixed to 80%~100% for a cleaner look
      - wider canvas to increase visual horizontal span
    """
    t_s = np.asarray(t_s, dtype=float)
    t_hr = t_s / 3600.0

    soc_meas = np.asarray(soc_meas, dtype=float)
    soc_sim = np.asarray(soc_sim, dtype=float)

    # Estimate dt for converting smoothing window from minutes -> points
    dt_s = 10.0
    if t_s.size >= 2:
        dif = np.diff(t_s)
        dif = dif[np.isfinite(dif)]
        if dif.size > 0:
            dt_s = float(np.median(dif))
            if dt_s <= 0:
                dt_s = 10.0

    # --- Smooth measured SOC (visual only) ---
    # 1) fill NaNs
    soc_meas_filled = _nan_interp_1d(t_hr, soc_meas)

    # 2) light moving average to reduce quantization steps
    smooth_minutes = 12.0  # tweak: 8~20; bigger -> smoother
    window_pts = int(max(5, round((smooth_minutes * 60.0) / dt_s)))
    soc_meas_ma = _moving_average(soc_meas_filled, window_pts)

    # 3) enforce monotone non-increasing (discharge)
    soc_meas_mono = _enforce_monotone_nonincreasing(soc_meas_ma)

    # 4) PCHIP interpolation to get a smooth curve
    #    (If there are repeated x values, PCHIP may fail; guard by unique)
    x = t_hr
    y = soc_meas_mono
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        x = np.array([0.0, 1.0])
        y = np.array([1.0, 1.0])

    # ensure strictly increasing x for interpolator
    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]

    interp_meas = PchipInterpolator(x_unique, y_unique, extrapolate=False)

    # Dense grid for visually smooth lines
    x_dense = np.linspace(float(x_unique.min()), float(x_unique.max()), 900)
    meas_dense = interp_meas(x_dense)

    # Simulated curve dense (use linear interpolation; already smooth enough)
    sim_dense = np.interp(x_dense, t_hr[np.isfinite(t_hr)], soc_sim[np.isfinite(t_hr)])

    # Convert to percent for nicer visual emphasis
    meas_pct = 100.0 * meas_dense
    sim_pct = 100.0 * sim_dense

    plt.figure(figsize=(14, 5.2))
    plt.plot(x_dense, meas_pct, label="measured (smoothed)")
    plt.plot(x_dense, sim_pct, label="simulated")

    plt.xlabel("Time (h)")
    plt.ylabel("SOC (%)")
    plt.title("SOC vs time (measured vs simulated)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

    # --- Axis styling ---
    # Make the plot look focused on high SOC region
    plt.ylim(80.0, 100.5)

    # Add a small x padding so curves don't touch borders
    xmin = float(x_dense.min())
    xmax = float(x_dense.max())
    pad = (xmax - xmin) * 0.03 if xmax > xmin else 0.1
    plt.xlim(xmin - pad, xmax + pad)

    _save_or_show(out_path, show)


def plot_voltage(
    t_s: np.ndarray,
    v_meas: np.ndarray,
    v_sim: np.ndarray,
    out_path: Optional[str | Path] = None,
    show: bool = False
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(np.asarray(t_s, dtype=float) / 60.0, v_meas, label="measured")
    plt.plot(np.asarray(t_s, dtype=float) / 60.0, v_sim, label="simulated")
    plt.xlabel("Time (min)")
    plt.ylabel("V_term (V)")
    plt.title("Terminal voltage vs time (measured vs simulated)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_or_show(out_path, show)
