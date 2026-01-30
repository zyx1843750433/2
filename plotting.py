from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def _save_or_show(out_path: Optional[str | Path], show: bool) -> None:
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_power(t_s: np.ndarray, p_meas: np.ndarray, p_pred: np.ndarray, out_path: Optional[str | Path] = None, show: bool = False) -> None:
    plt.figure()
    plt.plot(t_s / 60.0, p_meas, label="measured")
    plt.plot(t_s / 60.0, p_pred, label="model")
    plt.xlabel("Time (min)")
    plt.ylabel("Power (W)")
    plt.title("Power vs time (measured vs model)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_or_show(out_path, show)


def plot_soc(t_s: np.ndarray, soc_meas: np.ndarray, soc_sim: np.ndarray, out_path: Optional[str | Path] = None, show: bool = False) -> None:
    plt.figure()
    plt.plot(t_s / 60.0, soc_meas, label="measured")
    plt.plot(t_s / 60.0, soc_sim, label="simulated")
    plt.xlabel("Time (min)")
    plt.ylabel("SOC")
    plt.title("SOC vs time (measured vs simulated)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_or_show(out_path, show)


def plot_voltage(t_s: np.ndarray, v_meas: np.ndarray, v_sim: np.ndarray, out_path: Optional[str | Path] = None, show: bool = False) -> None:
    plt.figure()
    plt.plot(t_s / 60.0, v_meas, label="measured")
    plt.plot(t_s / 60.0, v_sim, label="simulated")
    plt.xlabel("Time (min)")
    plt.ylabel("V_term (V)")
    plt.title("Terminal voltage vs time (measured vs simulated)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_or_show(out_path, show)
