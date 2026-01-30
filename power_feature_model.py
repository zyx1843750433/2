from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.features import FeatureSet, estimate_idle_baseline_power


@dataclass
class LinearPowerModel:
    """
    Interpretable additive power model (as suggested in your PDF):

      P(t) = P0 + sum_i w_i * f_i(t)

    where f_i are normalized features in [0,1], and w_i >= 0.

    We fit w_i from the dataset using Ridge regression with positive coefficients.
    """
    P0_W: float
    feature_cols: list[str]
    weights: dict[str, float]
    feature_set: FeatureSet

    def predict_power(self, df_feat: pd.DataFrame) -> np.ndarray:
        X = df_feat[self.feature_cols].copy()
        Xs = self.feature_set.transform(X)
        M = Xs[self.feature_cols].to_numpy(dtype=float)
        w = np.array([self.weights[c] for c in self.feature_cols], dtype=float)
        return self.P0_W + M.dot(w)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "P0_W": float(self.P0_W),
            "feature_cols": list(self.feature_cols),
            "weights": {k: float(v) for k, v in self.weights.items()},
            "scaler": {
                "feature_mins": self.feature_set.scaler.feature_mins,
                "feature_maxs": self.feature_set.scaler.feature_maxs,
            },
        }

    @staticmethod
    def from_json_dict(d: Dict[str, Any]) -> "LinearPowerModel":
        feature_cols = list(d["feature_cols"])
        from src.utils import ScaleInfo
        scaler = ScaleInfo(
            feature_mins={k: float(v) for k, v in d["scaler"]["feature_mins"].items()},
            feature_maxs={k: float(v) for k, v in d["scaler"]["feature_maxs"].items()},
        )
        feature_set = FeatureSet(feature_cols=feature_cols, scaler=scaler)
        weights = {k: float(v) for k, v in d["weights"].items()}
        return LinearPowerModel(P0_W=float(d["P0_W"]), feature_cols=feature_cols, weights=weights, feature_set=feature_set)

    def save(self, path: str | Path) -> None:
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "LinearPowerModel":
        import json
        path = Path(path)
        return LinearPowerModel.from_json_dict(json.loads(path.read_text(encoding="utf-8")))


def fit_linear_power_model(
    df_feat: pd.DataFrame,
    feature_cols: list[str],
    *,
    p0_W: Optional[float] = None,
    alpha: float = 0.5,
    use_only_collected1: bool = True,
    verbose: bool = True,
) -> LinearPowerModel:
    """
    Fit P(t)=P0+sum w_i f_i(t) from measured load power.

    Args:
        df_feat: output of build_feature_frame (must contain p_load_W_meas and feature_cols)
        feature_cols: list of feature column names
        p0_W: optional fixed baseline. If None, estimate from idle samples.
        alpha: Ridge regularization (bigger -> smoother weights).
        use_only_collected1: if True, fit weights using collected==1 rows (features more reliable).
    """
    df = df_feat.copy()

    # baseline
    if p0_W is None:
        p0_W = estimate_idle_baseline_power(df)

    df["y"] = pd.to_numeric(df["p_load_W_meas"], errors="coerce") - float(p0_W)
    df["y"] = df["y"].clip(lower=0.0)

    # training mask
    mask = df["p_load_W_meas"].notna()
    mask &= df["p_load_W_meas"] > 0.02
    if use_only_collected1 and "collected" in df.columns:
        mask &= (df["collected"] == 1)

    # drop extreme outliers in y
    y_all = df.loc[mask, "y"]
    if len(y_all) < 1000:
        # if too few, relax collected filter
        mask = df["p_load_W_meas"].notna() & (df["p_load_W_meas"] > 0.02)
        y_all = df.loc[mask, "y"]

    lo = float(y_all.quantile(0.001))
    hi = float(y_all.quantile(0.999))
    mask &= df["y"].between(lo, hi)

    train = df.loc[mask].copy()
    if len(train) < 500:
        raise ValueError("Too few valid training rows after cleaning. Try a different day/device.")

    if verbose:
        # Diagnostics: if a feature is (almost) constant in the selected discharge segment(s),
        # the regression cannot identify its contribution and the optimal weight will be ~0.
        print("[fit] feature activity in training window (raw, before scaling):")
        for c in feature_cols:
            s = pd.to_numeric(train[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            nz = float((s != 0).mean())
            sd = float(s.std())
            mn = float(s.mean())
            flag = ""
            if sd < 1e-6 or nz < 0.01:
                flag = "  (low-variation -> weight may be 0)"
            print(f"       {c:12s} mean={mn:.4g} std={sd:.4g} nonzero%={100*nz:5.1f}%{flag}")

    # feature scaling to [0,1] using quantile range
    fs = FeatureSet.fit(train, feature_cols)
    Xs = fs.transform(train[feature_cols])
    X = Xs[feature_cols].to_numpy(dtype=float)
    y = train["y"].to_numpy(dtype=float)

    # fit ridge with positive coefficients
    reg = Ridge(alpha=float(alpha), fit_intercept=False, positive=True, random_state=0)
    reg.fit(X, y)

    weights = {c: float(w) for c, w in zip(feature_cols, reg.coef_)}

    if verbose:
        y_pred = reg.predict(X)
        rmse = float(np.sqrt(np.mean((y_pred - y) ** 2)))
        mae = float(np.mean(np.abs(y_pred - y)))
        print("[fit] baseline P0_W =", float(p0_W))
        print("[fit] train rows =", len(train), "RMSE(y)=", rmse, "MAE(y)=", mae)
        print("[fit] weights:")
        for k, v in sorted(weights.items(), key=lambda kv: -kv[1]):
            print(f"       {k:12s} {v:.4f} W")

    return LinearPowerModel(P0_W=float(p0_W), feature_cols=feature_cols, weights=weights, feature_set=fs)