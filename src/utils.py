from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable
import numpy as np
import pandas as pd


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def celsius_to_kelvin(T_C: float) -> float:
    return float(T_C) + 273.15


def robust_to_datetime(series: pd.Series, dayfirst: bool = False) -> pd.Series:
    """
    Robust datetime parsing for mixed formats.

    Args:
        series: timestamp string series
        dayfirst: True for formats like '24-02-2023 00:00:00'

    Returns:
        pandas datetime64[ns] series (NaT on parse failure)
    """
    # Try fast parse first
    dt = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
    # If many NaT, try the opposite dayfirst
    if dt.isna().mean() > 0.3:
        dt2 = pd.to_datetime(series, errors="coerce", dayfirst=not dayfirst)
        if dt2.isna().mean() < dt.isna().mean():
            dt = dt2
    return dt


def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return a / (b + eps)


def isclose_series(a: pd.Series, b: pd.Series, tol: float = 1e-6) -> pd.Series:
    return (a - b).abs() <= tol


@dataclass
class ScaleInfo:
    feature_mins: dict[str, float]
    feature_maxs: dict[str, float]

    def transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            lo = self.feature_mins.get(c, 0.0)
            hi = self.feature_maxs.get(c, 1.0)
            if hi <= lo:
                out[c] = 0.0
            else:
                out[c] = (out[c] - lo) / (hi - lo)
                out[c] = out[c].clip(0.0, 1.0)

            # ✅ 最终兜底：不让 NaN/inf 流入 ML
            out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    @staticmethod
    def fit_from(df: pd.DataFrame, cols: list[str], q_low: float = 0.01, q_high: float = 0.99) -> "ScaleInfo":
        mins = {}
        maxs = {}
        for c in cols:
            s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if s.notna().sum() == 0:
                lo, hi = 0.0, 1.0
            else:
                lo = float(s.quantile(q_low))
                hi = float(s.quantile(q_high))
                if not np.isfinite(lo) or not np.isfinite(hi):
                    lo, hi = 0.0, 1.0
            mins[c] = lo
            maxs[c] = hi
        return ScaleInfo(mins, maxs)
