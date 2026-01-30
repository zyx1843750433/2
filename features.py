from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from src.utils import clamp, ScaleInfo


def infer_screen_on(df: pd.DataFrame) -> pd.Series:
    """
    screen_status coding differs across devices:
      - some use 0/1 (1=on)
      - some use 1/2 (2=on)

    We infer 'on' label by comparing mean screen_on_time in each status.
    """
    if "screen_on_time" in df.columns and df["screen_on_time"].notna().any():
        tmp = df[["screen_status", "screen_on_time"]].dropna()
        if len(tmp) > 0 and "screen_status" in tmp.columns:
            g = tmp.groupby("screen_status")["screen_on_time"].mean()
            if len(g) >= 2:
                on_label = float(g.idxmax())
                return (df["screen_status"] == on_label).astype(float)
    # fallback
    if "screen_on_time" in df.columns:
        return (pd.to_numeric(df["screen_on_time"], errors="coerce").fillna(0.0) > 0).astype(float)
    if "screen_status" in df.columns:
        # best effort: treat max value as "on"
        on_label = df["screen_status"].max()
        return (df["screen_status"] == on_label).astype(float)
    return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)


def compute_cpu_freq_mean(df: pd.DataFrame) -> pd.Series:
    freq_cols = [c for c in df.columns if c.startswith("frequency_core")]
    if len(freq_cols) == 0:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    freqs = df[freq_cols].apply(pd.to_numeric, errors="coerce")
    return freqs.mean(axis=1).fillna(0.0)


def compute_network_traffic(df: pd.DataFrame) -> pd.Series:
    """
    Network activity proxy: log1p(total bytes per sample).
    """
    cols = [c for c in ["wifi_rx", "wifi_tx", "mobile_rx", "mobile_tx"] if c in df.columns]
    if len(cols) == 0:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    x = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    total = x.sum(axis=1)
    return np.log1p(total)


def wifi_weakness_norm(wifi_intensity: pd.Series) -> pd.Series:
    """
    wifi_intensity looks like RSSI in dBm, usually negative:
      strong ~ -30, weak ~ -100.
    Map to [0,1] where 0=strong, 1=very weak.
    """
    rssi = pd.to_numeric(wifi_intensity, errors="coerce")
    # Some datasets use -1 as "missing"
    rssi = rssi.where(rssi <= -5, np.nan)
    weakness = (-rssi - 30.0) / 70.0
    weakness = weakness.clip(0.0, 1.0).fillna(0.0)
    return weakness


@dataclass
class FeatureSet:
    feature_cols: list[str]
    scaler: ScaleInfo

    def transform(self, df_feats: pd.DataFrame) -> pd.DataFrame:
        return self.scaler.transform(df_feats, self.feature_cols)

    @staticmethod
    def fit(df_feats: pd.DataFrame, feature_cols: list[str]) -> "FeatureSet":
        scaler = ScaleInfo.fit_from(df_feats, feature_cols, q_low=0.01, q_high=0.99)
        return FeatureSet(feature_cols=feature_cols, scaler=scaler)


def build_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """
    Build the key feature table used for power modeling.

    Output columns:
      timestamp, p_load_W_meas, soc_meas,
      screen_on, bright, cpu, cpu_freq, net, wifi_weak, gps, bg_count, collected
    """
    out = pd.DataFrame(index=df.index)
    out["timestamp"] = df["timestamp"]
    out["p_load_W_meas"] = df.get("p_load_W_meas", np.nan)
    out["soc_meas"] = df.get("soc_meas", np.nan)
    out["collected"] = pd.to_numeric(df.get("collected", 1), errors="coerce").fillna(1).astype(int)

    # Screen
    out["screen_on"] = infer_screen_on(df)
    # Brightness (normalize later)
    out["bright_level"] = pd.to_numeric(df.get("bright_level", 0.0), errors="coerce").fillna(0.0)

    # CPU
    out["cpu_usage"] = pd.to_numeric(df.get("cpu_usage", 0.0), errors="coerce").fillna(0.0)
    out["cpu_freq_mean"] = compute_cpu_freq_mean(df)

    # Network
    out["wifi_status"] = pd.to_numeric(df.get("wifi_status", 0.0), errors="coerce").fillna(0.0)
    out["wifi_weak"] = wifi_weakness_norm(df.get("wifi_intensity", np.nan))
    out["net_traffic_log"] = compute_network_traffic(df)

    # GPS
    # GPS: some devices record gps_status as a constant code (e.g., always 3), which would make
    # gps_active always 1 if we simply OR (gps_status>0). Prefer gps_activity when gps_status has no variation.
    gps_act = pd.to_numeric(df.get("gps_activity", 0.0), errors="coerce").fillna(0.0)
    gps_status = pd.to_numeric(df.get("gps_status", 0.0), errors="coerce").fillna(0.0)
    status_nunique = int(pd.Series(gps_status).nunique(dropna=True))
    if status_nunique >= 2:
        out["gps_active"] = ((gps_act > 0) | (gps_status > 0)).astype(float)
    else:
        out["gps_active"] = (gps_act > 0).astype(float)

    # Background
    out["bg_app_count"] = pd.to_numeric(df.get("bg_app_count", 0.0), errors="coerce").fillna(0.0)
    # Fallback proxy if bg_app_count is always 0 and ram_usage exists
    if out["bg_app_count"].max() == 0 and "ram_usage" in df.columns:
        out["bg_app_count"] = pd.to_numeric(df["ram_usage"], errors="coerce").fillna(0.0)

    # Create normalized feature columns in [0,1] (scaler fitted later)
    # We include interactions so that "screen_on" gates brightness.
    feats = pd.DataFrame(index=df.index)
    feats["screen_on"] = out["screen_on"]
    feats["screen_bright"] = out["screen_on"] * out["bright_level"]
    feats["cpu"] = out["cpu_usage"]
    feats["cpu_freq"] = out["cpu_freq_mean"]
    feats["net"] = out["net_traffic_log"]
    feats["wifi_weak"] = out["wifi_status"].clip(0, 1) * out["wifi_weak"]
    feats["gps"] = out["gps_active"]
    feats["bg"] = out["bg_app_count"]

    feature_cols = list(feats.columns)

    # Merge back
    for c in feature_cols:
        out[c] = feats[c]

    return out, feature_cols


def estimate_idle_baseline_power(df_feat: pd.DataFrame) -> float:
    """
    Estimate a constant baseline power P0 from "idle-like" samples.
    This helps when the dataset contains many low-quality/placeholder rows.

    We pick rows that look idle:
      screen_off, low cpu, gps off, low network.
    If not enough rows, use 10th percentile of measured load power.
    """
    p = pd.to_numeric(df_feat["p_load_W_meas"], errors="coerce").dropna()
    if len(p) == 0:
        return 0.5

    idle_mask = (
        (df_feat["screen_on"] < 0.5)
        & (pd.to_numeric(df_feat["cpu"], errors="coerce").fillna(0.0) < 5.0)
        & (df_feat["gps"] < 0.5)
        & (pd.to_numeric(df_feat["net"], errors="coerce").fillna(0.0) < 0.5)
    )
    p_idle = pd.to_numeric(df_feat.loc[idle_mask, "p_load_W_meas"], errors="coerce").dropna()
    if len(p_idle) >= 200:
        return float(p_idle.median())
    return float(p.quantile(0.10))