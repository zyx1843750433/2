from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# --- Make imports robust when running this file directly in PyCharm ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.android_dataset import (
    DatasetIndex,
    load_background_app_count,
    load_static_battery_capacity_mAh,
    clean_dynamic,
    merge_background,
)
from src.features import build_feature_frame
from src.power_feature_model import fit_linear_power_model
from src.battery_model import BatterySOCModel, CapacityParams
from src.plotting import plot_power, plot_soc, plot_voltage


# --------------------------
# Auto-run helpers (no args)
# --------------------------

def _looks_like_android_dataset_zip(zip_path: Path) -> bool:
    """Quickly validate a zip is the expected Android dataset by checking it has dynamic csv entries."""
    try:
        idx = DatasetIndex.build(zip_path)
        return len(idx.dynamic) > 0
    except Exception:
        return False


def find_default_dataset_zip() -> Optional[Path]:
    """
    Find the dataset zip automatically so PyCharm users can "Run main.py" without parameters.

    Search order:
      1) Env var ANDROID_DATASET_ZIP
      2) Project folder: data/raw/*.zip, data/*.zip, project root/*.zip
      3) Common expected filename
    """
    # 1) environment variable
    env = os.environ.get("ANDROID_DATASET_ZIP", "").strip()
    if env:
        p = Path(env)
        if p.is_file() and _looks_like_android_dataset_zip(p):
            return p.resolve()

    # 2) common filename candidates
    candidates = [
        PROJECT_ROOT / "data" / "raw" / "A dataset from the daily use of features in Android devices.zip",
        PROJECT_ROOT / "data" / "A dataset from the daily use of features in Android devices.zip",
        PROJECT_ROOT / "data" / "raw" / "android_dataset.zip",
        PROJECT_ROOT / "data" / "android_dataset.zip",
        PROJECT_ROOT / "data" / "dataset.zip",
        PROJECT_ROOT / "android_dataset.zip",
        PROJECT_ROOT / "dataset.zip",
    ]
    for c in candidates:
        if c.is_file() and _looks_like_android_dataset_zip(c):
            return c.resolve()

    # 3) scan folders for any .zip (pick the first that looks right)
    for folder in [PROJECT_ROOT / "data" / "raw", PROJECT_ROOT / "data", PROJECT_ROOT]:
        if not folder.exists():
            continue
        zips = sorted(folder.glob("*.zip"), key=lambda p: p.stat().st_size, reverse=True)
        for z in zips:
            if _looks_like_android_dataset_zip(z):
                return z.resolve()

    return None


def auto_pick_date_device(zip_path: Path) -> Tuple[str, str]:
    """
    Pick a (date, device_id) pair automatically:
      Prefer ones that have BOTH background and static files (more complete),
      and among those, prefer later dates.
    """
    idx = DatasetIndex.build(zip_path)
    if len(idx.dynamic) == 0:
        raise ValueError("No dynamic data found in the dataset zip.")

    dyn = idx.dynamic.copy()
    bg_pairs = set(zip(idx.background.get("date", []), idx.background.get("device_id", [])))
    st_pairs = set(zip(idx.static.get("date", []), idx.static.get("device_id", [])))

    dyn["has_background"] = [(r.date, r.device_id) in bg_pairs for r in dyn.itertuples(index=False)]
    dyn["has_static"] = [(r.date, r.device_id) in st_pairs for r in dyn.itertuples(index=False)]
    dyn["score"] = dyn["has_background"].astype(int) * 2 + dyn["has_static"].astype(int)
    dyn["date_int"] = dyn["date"].astype(int)

    dyn = dyn.sort_values(["score", "date_int", "device_id"], ascending=[False, False, True])
    row = dyn.iloc[0]
    return str(row["date"]), str(row["device_id"])


def build_default_argv() -> Optional[list[str]]:
    """
    Construct argv for the 'run' subcommand when the user runs main.py without parameters.
    """
    zip_path = find_default_dataset_zip()
    if zip_path is None:
        return None

    # allow optional override by env vars
    date = os.environ.get("ANDROID_DATASET_DATE", "").strip()
    device = os.environ.get("ANDROID_DATASET_DEVICE", "").strip()

    if not date or not device:
        date, device = auto_pick_date_device(zip_path)

    # Default: dt=10s. You can change by setting env ANDROID_DT_S.
    dt_s = os.environ.get("ANDROID_DT_S", "10").strip() or "10"

    return [
        "run",
        "--zip", str(zip_path),
        "--date", date,
        "--device", device,
        "--dt", dt_s,
    ]


# --------------------------
# Core pipeline functions
# --------------------------

def find_discharge_segments(df: pd.DataFrame, gap_s: float = 120.0, min_len: int = 50) -> list[pd.DataFrame]:
    """
    Split the day into continuous segments and return those that look like DISCHARGE.

    Discharge heuristic (same as before):
      battery_connection_status == 0   (unplugged)
      p_load_W_meas > 0.02
      battery_current_A < 0          (discharge sign)

    gap_s: split segments when timestamp gap is larger than this.
    min_len: minimum raw rows in a segment.
    """
    d = df.sort_values("timestamp").copy()
    if "battery_connection_status" not in d.columns:
        d["battery_connection_status"] = 0
    if "battery_current_A" not in d.columns:
        d["battery_current_A"] = np.nan

    discharge = (
        (pd.to_numeric(d["battery_connection_status"], errors="coerce").fillna(0.0) == 0.0)
        & (pd.to_numeric(d["p_load_W_meas"], errors="coerce").fillna(0.0) > 0.02)
        & (pd.to_numeric(d["battery_current_A"], errors="coerce").fillna(-1.0) < 0.0)
    )

    t = d["timestamp"].astype("int64") / 1e9
    dt = t.diff().fillna(0.0)
    split = (dt > float(gap_s)) | (discharge != discharge.shift(1).fillna(False))
    seg_id = split.cumsum()

    segs: list[pd.DataFrame] = []
    for _, g in d.groupby(seg_id):
        if len(g) < int(min_len):
            continue
        if not bool(discharge.loc[g.index].iloc[0]):
            continue
        segs.append(g.copy())
    return segs


def _score_segment(g: pd.DataFrame) -> tuple[float, dict]:
    """Score a discharge segment: prefer usable SOC variation + some activity (screen/network)."""
    t0 = g["timestamp"].iloc[0]
    t1 = g["timestamp"].iloc[-1]
    dur_s = float((t1 - t0).total_seconds())

    soc = pd.to_numeric(g.get("soc_meas", np.nan), errors="coerce").dropna()
    if len(soc) >= 2:
        soc_start = float(soc.iloc[0])
        soc_end = float(soc.iloc[-1])
        soc_range = float(soc.max() - soc.min())
    elif len(soc) == 1:
        soc_start = float(soc.iloc[0]); soc_end = float(soc.iloc[0]); soc_range = 0.0
    else:
        soc_start = np.nan; soc_end = np.nan; soc_range = 0.0

    screen = pd.to_numeric(g.get("screen_on", 0.0), errors="coerce").fillna(0.0)
    screen_frac = float((screen > 0.5).mean()) if len(screen) else 0.0

    net = pd.to_numeric(g.get("net", 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    net_std = float(net.std()) if len(net) else 0.0

    cpu = pd.to_numeric(g.get("cpu", 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cpu_std = float(cpu.std()) if len(cpu) else 0.0

    # Main idea: if a segment starts at very low SOC (e.g. 7%) and SOC barely changes,
    # then the *measured* SOC line becomes almost flat (percentage quantization), which makes
    # the "measured vs simulated" plot misleading. Prefer segments with higher SOC and more SOC variation.
    soc_start_score = 0.0 if not np.isfinite(soc_start) else float(soc_start)
    score = (
        20.0 * float(soc_range)
        + 5.0 * soc_start_score
        + 2.0 * float(screen_frac)
        + 0.2 * float(net_std)
        + 0.02 * float(cpu_std)
        + 0.002 * (dur_s / 60.0)
    )

    info = {
        "t_start": t0,
        "t_end": t1,
        "dur_min": dur_s / 60.0,
        "soc_start": soc_start,
        "soc_end": soc_end,
        "soc_range": soc_range,
        "screen_frac": screen_frac,
        "net_std": net_std,
        "cpu_std": cpu_std,
        "score": score,
        "n_rows": int(len(g)),
    }
    return score, info


def select_best_discharge_segment(df: pd.DataFrame, gap_s: float = 120.0, verbose: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """
    Instead of always taking the *longest* discharge segment, choose a "best" segment
    for evaluation (plots/metrics) using a score that prefers:
      - higher starting SOC
      - more SOC variation (avoids a flat measured SOC line)
      - some usage activity (screen/network)
      - still keeps reasonable duration

    Returns (best_segment_df, segment_infos_sorted_by_score_desc).
    """
    segs = find_discharge_segments(df, gap_s=gap_s, min_len=50)
    if len(segs) == 0:
        d = df.sort_values("timestamp").copy()
        best = d[d["p_load_W_meas"] > 0.02].copy()
        return best, []

    scored: list[tuple[float, pd.DataFrame, dict]] = []
    for g in segs:
        sc, info = _score_segment(g)
        scored.append((sc, g, info))
    scored.sort(key=lambda x: x[0], reverse=True)

    infos = [x[2] for x in scored]
    best = scored[0][1]

    if verbose:
        print("[segment] Candidate discharge segments (top 5 by score):")
        show = infos[: min(5, len(infos))]
        # Pretty print without huge timestamps
        for k, inf in enumerate(show, start=1):
            print(
                f"  #{k} score={inf['score']:.3f} dur={inf['dur_min']:.1f}min "
                f"SOC_start={inf['soc_start']} SOC_range={inf['soc_range']:.3f} "
                f"screen_frac={inf['screen_frac']:.3f} net_std={inf['net_std']:.3f} "
                f"[{inf['t_start']} -> {inf['t_end']}] (rows={inf['n_rows']})"
            )
        print("[segment] Selected segment: #1 (best score)")

    return best, infos


def resample_regular(df: pd.DataFrame, dt_s: int) -> pd.DataFrame:
    """
    Resample time series to a regular dt_s seconds grid.

    - Most numeric columns -> mean
    - soc_meas -> last, then forward-fill
    - collected -> max (so it stays 1 if any sample in the bin was collected==1)

    Note: Using mean() for 'collected' can create many 0.9/0.8 values after resampling, which makes
    a strict (collected==1) filter overly aggressive.
    """
    d = df.sort_values("timestamp").copy()
    d = d.set_index("timestamp")

    num_cols = d.select_dtypes(include=[np.number]).columns.tolist()
    soc_col = "soc_meas" if "soc_meas" in d.columns else None
    if soc_col in num_cols:
        num_cols.remove(soc_col)

    out_num = d[num_cols].resample(f"{dt_s}s").mean()

    if "collected" in d.columns:
        out_num["collected"] = d["collected"].resample(f"{dt_s}s").max()

    out = out_num
    if soc_col is not None:
        soc = d[[soc_col]].resample(f"{dt_s}s").last().ffill()
        out = out.join(soc, how="left")

    out = out.reset_index()
    return out
def interp1d(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def f(t: float) -> float:
        return float(np.interp(float(t), x, y))
    return f


def cmd_list(args) -> int:
    idx = DatasetIndex.build(args.zip)
    dyn = idx.dynamic.copy()
    # mark availability of static/background
    bg_pairs = set(zip(idx.background["date"], idx.background["device_id"]))
    st_pairs = set(zip(idx.static["date"], idx.static["device_id"]))
    dyn["has_background"] = [(r.date, r.device_id) in bg_pairs for r in dyn.itertuples(index=False)]
    dyn["has_static"] = [(r.date, r.device_id) in st_pairs for r in dyn.itertuples(index=False)]

    dyn = dyn.sort_values(["date", "device_id"])
    if args.device:
        dyn = dyn[dyn["device_id"] == args.device]
    if args.date:
        dyn = dyn[dyn["date"] == args.date]

    print(dyn[["date", "device_id", "has_background", "has_static"]].drop_duplicates().to_string(index=False))
    return 0


def cmd_run(args) -> int:
    idx = DatasetIndex.build(args.zip)

    # load data
    dyn_path = idx.find_dynamic(args.date, args.device)
    import zipfile
    with zipfile.ZipFile(idx.zip_path) as zf:
        raw_dyn = pd.read_csv(zf.open(dyn_path), low_memory=False)

    dyn = clean_dynamic(raw_dyn)

    # merge background apps
    bg = load_background_app_count(idx, args.date, args.device)
    dyn = merge_background(dyn, bg)

    # build feature frame
    feat_df, feature_cols = build_feature_frame(dyn)
    # attach measured voltage/current/temp to feat_df for evaluation
    feat_df["v_meas_V"] = pd.to_numeric(dyn.get("battery_voltage_V", np.nan), errors="coerce").to_numpy()
    feat_df["i_meas_A"] = (-pd.to_numeric(dyn.get("battery_current_A", np.nan), errors="coerce")).to_numpy().clip(min=0.0)
    feat_df["T_C"] = pd.to_numeric(dyn.get("battery_temperature", np.nan), errors="coerce").to_numpy()
    feat_df["battery_connection_status"] = pd.to_numeric(dyn.get("battery_connection_status", 0), errors="coerce").fillna(0).to_numpy()

    # Select a discharge segment for evaluation.
    # IMPORTANT: do NOT always pick the *longest* segment. On some days, the longest segment may start at very low SOC
    # (e.g., 7%) where the reported SOC is quantized and nearly constant, causing a flat "measured" SOC curve.
    # We choose a "best" segment by a score that prefers higher SOC and more SOC variation.
    seg_best, _ = select_best_discharge_segment(feat_df, gap_s=args.gap, verbose=True)

    # Optionally limit evaluation to first N hours (faster demo)
    seg_eval_raw = seg_best
    if args.hours is not None:
        t0_eval = seg_eval_raw["timestamp"].iloc[0]
        seg_eval_raw = seg_eval_raw[seg_eval_raw["timestamp"] <= t0_eval + pd.Timedelta(hours=float(args.hours))].copy()

    # Resample evaluation segment to regular grid
    seg = resample_regular(seg_eval_raw, dt_s=args.dt)

    # Build training set for power weights: concatenate up to 3 best discharge segments (by the same score)
    # until we have enough rows. This helps avoid "all-zero" weights when one segment is too idle (screen always off, etc.).
    segs_all = find_discharge_segments(feat_df, gap_s=args.gap, min_len=50)
    scored_all: list[tuple[float, pd.DataFrame, dict]] = []
    for g in segs_all:
        sc, info = _score_segment(g)
        scored_all.append((sc, g, info))
    scored_all.sort(key=lambda x: x[0], reverse=True)

    train_parts: list[pd.DataFrame] = []
    total_rows = 0
    for sc, g, info in scored_all[:5]:  # consider top 5, but keep only first few needed
        rs = resample_regular(g, dt_s=args.dt)
        train_parts.append(rs)
        total_rows += len(rs)
        if total_rows >= 2000 or len(train_parts) >= 3:
            break
    train_df = pd.concat(train_parts, ignore_index=True) if len(train_parts) else seg

    print(f"[train] Using {len(train_parts)} segment(s), total train rows={len(train_df)} (dt={args.dt}s)")

    # Fit power model
    power_model = fit_linear_power_model(
        train_df,
        feature_cols=feature_cols,
        alpha=args.ridge_alpha,
        use_only_collected1=not args.use_all_rows,
        verbose=True,
    )
    # capacity from static
    cap_mAh = load_static_battery_capacity_mAh(idx, args.date, args.device)
    if cap_mAh is None:
        cap_mAh = args.default_capacity_mAh
        print(f"[WARN] static battery_capacity not found; using default {cap_mAh} mAh")
    C_nom_Ah = float(cap_mAh) / 1000.0

    # temperature
    T_C = float(pd.to_numeric(seg["T_C"], errors="coerce").dropna().mean()) if "T_C" in seg.columns else 25.0
    if not np.isfinite(T_C):
        T_C = 25.0

    # initial SOC
    soc0 = float(pd.to_numeric(seg["soc_meas"], errors="coerce").dropna().iloc[0])

    # time axis
    t0 = seg["timestamp"].iloc[0]
    t_s = (seg["timestamp"] - t0).dt.total_seconds().to_numpy(dtype=float)
    t_end = float(t_s[-1])

    # build power function for ODE
    P_fun = interp1d(t_s, p_pred.clip(min=0.0))

    # simulate SOC with coupled ODE
    model = BatterySOCModel(capacity=CapacityParams(C_nom_Ah=C_nom_Ah))
    model.cutoff_voltage_V = None if args.disable_vcut else float(args.vcut)
    sol = model.simulate_power_driven(
        t_span=(0.0, t_end),
        soc0=soc0,
        T_C=T_C,
        power_W=P_fun,
        t_eval=t_s,
    )

    soc_sim = sol.y[0]
    # If solver stopped early (event), align to its length
    t_sim = sol.t
    if len(t_sim) != len(t_s):
        keep = len(t_sim)
        seg = seg.iloc[:keep].copy()
        t_s = t_sim
        p_pred = p_pred[:keep]
        p_meas = p_meas[:keep]

    # simulated I and V
    I_sim = np.array([model.current_from_power(p_pred[i], soc_sim[i], T_C) for i in range(len(soc_sim))], dtype=float)
    V_sim = np.array([model.terminal_voltage(I_sim[i], soc_sim[i], T_C) for i in range(len(soc_sim))], dtype=float)

    # measured series
    soc_meas = pd.to_numeric(seg["soc_meas"], errors="coerce").ffill().fillna(soc0).to_numpy()
    v_meas = pd.to_numeric(seg["v_meas_V"], errors="coerce").ffill().fillna(np.nan).to_numpy()
    i_meas = pd.to_numeric(seg["i_meas_A"], errors="coerce").fillna(0.0).to_numpy()

    # metrics
    def rmse(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() == 0:
            return np.nan
        return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))

    def mae(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() == 0:
            return np.nan
        return float(np.mean(np.abs(a[m] - b[m])))

    metrics = {
        "power_rmse_W": rmse(p_meas, p_pred),
        "power_mae_W": mae(p_meas, p_pred),
        "soc_rmse": rmse(soc_meas, soc_sim),
        "soc_mae": mae(soc_meas, soc_sim),
        "voltage_rmse_V": rmse(v_meas, V_sim),
        "current_rmse_A": rmse(i_meas, I_sim),
        "T_C_mean": T_C,
        "C_nom_Ah": C_nom_Ah,
        "P0_W": power_model.P0_W,
    }

    # save results
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        "timestamp": seg["timestamp"],
        "t_s": t_s,
        "soc_meas": soc_meas,
        "soc_sim": soc_sim,
        "p_meas_W": p_meas,
        "p_pred_W": p_pred,
        "v_meas_V": v_meas,
        "v_sim_V": V_sim,
        "i_meas_A": i_meas,
        "i_sim_A": I_sim,
    })
    out.to_csv(out_csv, index=False)

    # plots
    plot_power(t_s, p_meas, p_pred, out_path=args.out_plot_power, show=args.show)
    plot_soc(t_s, soc_meas, soc_sim, out_path=args.out_plot_soc, show=args.show)
    plot_voltage(t_s, v_meas, V_sim, out_path=args.out_plot_voltage, show=args.show)

    # print summary
    print("\n[OK] Finished end-to-end run.")
    print(f"     zip={idx.zip_path}")
    print(f"     device={args.device}, date={args.date}")
    print(f"     segment_duration_s={float(t_s[-1]):.1f}, dt_s={args.dt}")
    print(f"     saved model: {model_path}")
    print(f"     saved csv:   {out_csv}")
    print(f"     plots:       {args.out_plot_power}, {args.out_plot_soc}, {args.out_plot_voltage}")
    print("     metrics:")
    for k, v in metrics.items():
        print(f"        {k:16s} {v}")

    # also save metrics JSON
    import json
    metrics_path = Path(args.out_metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Battery SOC model with Android daily-use dataset: "
            "P(t)=P0 + feature contributions, I(t) solved by coupled equations.\n\n"
            "Tip: If you run this file with NO parameters, it will auto-locate the dataset zip in ./data and run a default example."
        )
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List available (date, device) in the dataset zip.")
    p_list.add_argument("--zip", required=True, help="Path to the dataset zip file")
    p_list.add_argument("--date", default=None)
    p_list.add_argument("--device", default=None)
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="End-to-end: load data -> clean -> fit power model -> simulate SOC -> output plots.")
    p_run.add_argument("--zip", required=True, help="Path to the dataset zip file")
    p_run.add_argument("--date", required=True, help="YYYYMMDD")
    p_run.add_argument("--device", required=True, help="device_id string (folder name)")
    p_run.add_argument("--dt", type=int, default=10, help="Resample step in seconds (default 10)")
    p_run.add_argument("--gap", type=float, default=120.0, help="Split discharge segments when time gap > gap seconds")
    p_run.add_argument("--hours", type=float, default=None, help="Optional: only use the first N hours of the selected discharge segment")

    p_run.add_argument("--ridge-alpha", type=float, default=0.5, help="Ridge regularization alpha for power model")
    p_run.add_argument("--use-all-rows", action="store_true", help="Use collected==0 rows too when fitting power weights")
    p_run.add_argument("--default-capacity-mAh", type=float, default=4000.0, help="Fallback capacity if static.csv missing")

    p_run.add_argument("--vcut", type=float, default=3.0, help="Cutoff voltage (V)")
    p_run.add_argument("--disable-vcut", action="store_true", help="Disable cutoff voltage event (simulate full duration)")

    p_run.add_argument("--out-model", default="results/power_model.json")
    p_run.add_argument("--out-csv", default="results/sim_results.csv")
    p_run.add_argument("--out-metrics", default="results/metrics.json")
    p_run.add_argument("--out-plot-power", default="results/power.png")
    p_run.add_argument("--out-plot-soc", default="results/soc.png")
    p_run.add_argument("--out-plot-voltage", default="results/voltage.png")
    p_run.add_argument("--show", action="store_true")
    p_run.set_defaults(func=cmd_run)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    # If user runs with no args: auto-build default argv and run.
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) == 0:
        default_argv = build_default_argv()
        if default_argv is None:
            # dataset not found -> print friendly guidance
            print("[ERROR] You ran src/main.py without parameters, but the dataset zip was not found.")
            print("\nHow to fix:")
            print("  1) Create folder: <project_root>/data/raw/")
            print("  2) Put your dataset zip there (do NOT unzip). For example rename it to:")
            print("        data/raw/android_dataset.zip")
            print("  3) Run again: python src/main.py")
            print("\nOr run explicitly:")
            print('  python src/main.py run --zip "path/to/dataset.zip" --date YYYYMMDD --device <device_id>')
            print("\nOr set environment variable ANDROID_DATASET_ZIP to the zip path.")
            return 2

        print("[INFO] No arguments provided -> auto-run with:")
        print("       ", " ".join(default_argv))
        argv = default_argv

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
