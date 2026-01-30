from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import zipfile
import re

from src.utils import robust_to_datetime, clamp


_DYNAMIC_RE = re.compile(r"/Dynamic data/(\d{8})/([^/]+)/.*_dynamic_processed\.csv$")
_BACKGROUND_RE = re.compile(r"/Background data/(\d{8})/([^/]+)/.*_backgroundAPPS\.csv$")
_STATIC_RE = re.compile(r"/Static data/(\d{8})/([^/]+)/.*_static\.csv$")


@dataclass
class DatasetIndex:
    zip_path: Path
    dynamic: pd.DataFrame
    background: pd.DataFrame
    static: pd.DataFrame

    @staticmethod
    def build(zip_path: str | Path) -> "DatasetIndex":
        zip_path = Path(zip_path)
        if not zip_path.is_file():
            raise FileNotFoundError(f"zip not found: {zip_path}")

        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]

        dyn_rows = []
        bg_rows = []
        st_rows = []

        for n in names:
            m = _DYNAMIC_RE.search(n)
            if m:
                dyn_rows.append({"date": m.group(1), "device_id": m.group(2), "path": n})
                continue
            m = _BACKGROUND_RE.search(n)
            if m:
                bg_rows.append({"date": m.group(1), "device_id": m.group(2), "path": n})
                continue
            m = _STATIC_RE.search(n)
            if m:
                st_rows.append({"date": m.group(1), "device_id": m.group(2), "path": n})
                continue

        dynamic = pd.DataFrame(dyn_rows).drop_duplicates()
        background = pd.DataFrame(bg_rows).drop_duplicates()
        static = pd.DataFrame(st_rows).drop_duplicates()

        return DatasetIndex(zip_path=zip_path, dynamic=dynamic, background=background, static=static)

    def find_dynamic(self, date: str, device_id: str) -> str:
        r = self.dynamic[(self.dynamic["date"] == date) & (self.dynamic["device_id"] == device_id)]
        if len(r) == 0:
            raise FileNotFoundError(f"Dynamic file not found for date={date}, device={device_id}")
        return str(r.iloc[0]["path"])

    def find_background(self, date: str, device_id: str) -> Optional[str]:
        r = self.background[(self.background["date"] == date) & (self.background["device_id"] == device_id)]
        if len(r) == 0:
            return None
        return str(r.iloc[0]["path"])

    def find_static(self, date: str, device_id: str) -> Optional[str]:
        r = self.static[(self.static["date"] == date) & (self.static["device_id"] == device_id)]
        if len(r) == 0:
            return None
        return str(r.iloc[0]["path"])


def _read_csv_from_zip(zf: zipfile.ZipFile, path: str, usecols=None) -> pd.DataFrame:
    with zf.open(path) as f:
        return pd.read_csv(f, usecols=usecols, low_memory=False)


def load_static_battery_capacity_mAh(index: DatasetIndex, date: str, device_id: str) -> Optional[float]:
    """
    static.csv contains 'battery_capacity' (mAh).

    Important: static data may not exist for the *same* date as the dynamic log.
    So we:
      1) try exact (date, device)
      2) otherwise pick the static file for this device whose date is closest to `date`
         (capacity is device-level and usually stable).
    """
    # 1) exact match
    static_path = index.find_static(date, device_id)
    chosen_date = date

    # 2) fallback: closest available static date for this device
    if static_path is None:
        cand = index.static[index.static["device_id"] == device_id].copy()
        if len(cand) == 0:
            return None
        # date strings are YYYYMMDD; compare as integers
        target = int(date)
        cand["date_int"] = cand["date"].astype(int)
        cand["abs_diff"] = (cand["date_int"] - target).abs()
        cand = cand.sort_values(["abs_diff", "date_int"])
        static_path = str(cand.iloc[0]["path"])
        chosen_date = str(cand.iloc[0]["date"])

    with zipfile.ZipFile(index.zip_path) as zf:
        st = _read_csv_from_zip(zf, static_path)
    if "battery_capacity" not in st.columns:
        return None
    cap = pd.to_numeric(st["battery_capacity"], errors="coerce").dropna()
    if len(cap) == 0:
        return None
    return float(cap.median())



def load_background_app_count(index: DatasetIndex, date: str, device_id: str) -> Optional[pd.DataFrame]:
    """
    Returns a DataFrame with columns:
      - timestamp (datetime64)
      - bg_app_count (float)

    Background file stores a semicolon-separated list of app IDs, or '-' when absent.
    """
    bg_path = index.find_background(date, device_id)
    if bg_path is None:
        return None
    with zipfile.ZipFile(index.zip_path) as zf:
        bg = _read_csv_from_zip(zf, bg_path)

    if "timestamp" not in bg.columns or "background_apps" not in bg.columns:
        return None

    # Background timestamps often come in 'dd-mm-yyyy HH:MM:SS'
    bg["timestamp_dt"] = robust_to_datetime(bg["timestamp"], dayfirst=True)
    bg = bg.dropna(subset=["timestamp_dt"]).copy()

    def count_apps(x: Any) -> float:
        if not isinstance(x, str):
            return np.nan
        x = x.strip()
        if x == "-" or x == "":
            return np.nan
        parts = [p.strip() for p in x.split(";") if p.strip() != ""]
        return float(len(parts))

    bg["bg_app_count"] = bg["background_apps"].apply(count_apps)

    bg = bg.sort_values("timestamp_dt")
    # Forward-fill missing counts (treat '-' as "no update")
    bg["bg_app_count"] = bg["bg_app_count"].ffill()
    # If still missing at start, set 0
    bg["bg_app_count"] = bg["bg_app_count"].fillna(0.0)

    return bg[["timestamp_dt", "bg_app_count"]].rename(columns={"timestamp_dt": "timestamp"})


def infer_current_unit_to_A(series: pd.Series) -> pd.Series:
    """
    battery_current can be in A (e.g., -0.3) or mA (e.g., -570).
    We infer by magnitude: if 95th percentile abs(current) > 20, assume mA -> convert to A.
    """
    s = pd.to_numeric(series, errors="coerce")
    q = float(s.abs().quantile(0.95))
    if q > 20.0:
        return s / 1000.0
    return s


def clean_dynamic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dynamic data and extract key columns needed for the modeling.

    Key columns:
      timestamp, battery_level, battery_current_A, battery_voltage_V, battery_power_W,
      screen_status, bright_level, screen_on_time,
      cpu_usage, frequency_core0..7,
      wifi_status, wifi_intensity, mobile_status, wifi_rx/tx, mobile_rx/tx,
      gps_status, gps_activity,
      ram_usage, foreground_app, collected
    """
    df = df.copy()

    # timestamp parsing (dynamic is usually ISO format)
    df["timestamp"] = robust_to_datetime(df["timestamp"], dayfirst=False)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # numeric conversions
    num_cols = [
        "battery_level", "battery_temperature", "battery_current", "battery_voltage", "battery_power",
        "battery_connection_status", "battery_charging_status",
        "cpu_usage", "wifi_intensity", "wifi_status", "mobile_status", "gps_status", "gps_activity",
        "bright_level", "screen_status", "screen_on_time", "ram_usage", "collected",
        "wifi_rx", "wifi_tx", "mobile_rx", "mobile_tx",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # unit fix for current
    if "battery_current" in df.columns:
        df["battery_current_A"] = infer_current_unit_to_A(df["battery_current"])
    else:
        df["battery_current_A"] = np.nan

    # voltage & power
    df["battery_voltage_V"] = pd.to_numeric(df.get("battery_voltage", np.nan), errors="coerce")
    df["battery_power_W"] = pd.to_numeric(df.get("battery_power", np.nan), errors="coerce")

    # SOC in [0,1]
    df["soc_meas"] = pd.to_numeric(df.get("battery_level", np.nan), errors="coerce") / 100.0

    # filter obvious invalid
    df = df[df["soc_meas"].between(-0.05, 1.05)]
    df = df[df["battery_voltage_V"].between(2.5, 5.0)]
    df = df[df["battery_current_A"].abs() < 20.0]  # 20A is far above phone range
    df = df[df["battery_power_W"].abs() < 200.0]   # sanity

    # discharge power (positive)
    # In the dataset, discharge current/power often appear as negative.
    df["p_load_W_meas"] = (-df["battery_power_W"]).clip(lower=0.0)

    return df


def merge_background(dynamic_df: pd.DataFrame, bg_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge background app count into dynamic df by timestamp (forward fill).
    If bg_df is None, create bg_app_count=0.
    """
    d = dynamic_df.sort_values("timestamp").copy()
    if bg_df is None or len(bg_df) == 0:
        d["bg_app_count"] = 0.0
        return d

    bg = bg_df.sort_values("timestamp").copy()
    merged = pd.merge_asof(d, bg, on="timestamp", direction="backward")

    # ✅ pandas 新版本不再支持 fillna(method="ffill")
    merged["bg_app_count"] = merged["bg_app_count"].ffill().fillna(0.0)

    return merged