# forecast_and_scale.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import joblib
import yaml
import numpy as np
import pandas as pd
import torch


# ==============================
# 0) FIND PROJECT ROOT (ROBUST)
# ==============================
def find_project_root(start: Path) -> Path:
    """
    Tự dò từ vị trí file hiện tại đi lên cho tới khi thấy:
      - config/config.yaml
      - hoặc thư mục data
    Tránh lỗi ROOT bị trỏ vào .../src
    """
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "config" / "config.yaml").exists():
            return p
        if (p / "data").exists():
            return p
    return cur.parents[-1]


ROOT = find_project_root(Path(__file__).resolve())

# đảm bảo ROOT nằm trong sys.path để import package
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ==============================
# 1) IMPORT LSTM FORECASTER SAFELY
# ==============================
try:
    # nếu models.py nằm ở src/models.py
    from src.models import LSTMForecaster  # type: ignore
except Exception:
    # fallback nếu models.py nằm ở root/models.py
    from models import LSTMForecaster  # type: ignore


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS: List[str] = [
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "is_weekend",
    "lag_1step",
    "lag_1h",
    "lag_24h",
    "lag_7d",
    "roll_mean_4h",
    "roll_std_4h",
    "roll_max_4h",
]
TARGET_COL = "intensity"


# ==============================
# 2) CONFIG
# ==============================
def load_full_config() -> Dict[str, Any]:
    cfg_path = ROOT / "config" / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    raise FileNotFoundError(f"Không tìm thấy config/config.yaml tại: {cfg_path}")


def get_data_dir(cfg: Dict[str, Any]) -> Path:
    """
    Ưu tiên cfg['paths']['input_dir'] (nếu có và tồn tại),
    fallback chắc chắn về ROOT/data.
    """
    input_dir = cfg.get("paths", {}).get("input_dir")
    if input_dir:
        p = ROOT / input_dir
        if p.exists():
            return p
    return ROOT / "data"


def get_models_dir(cfg: Optional[Dict[str, Any]] = None) -> Path:
    candidates = [
        ROOT / "saved_models",
        ROOT / "models",
    ]
    for p in candidates:
        if p.exists():
            return p
    # fallback về ROOT/models (dù chưa tồn tại)
    return ROOT / "models"


# ==============================
# 3) DATA LOADING (prepared/processed fallback)
# ==============================
def load_prepared(interval: str, mode: str = "test", cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    if cfg is None:
        cfg = load_full_config()

    data_dir = get_data_dir(cfg)

    candidates = [
        data_dir / f"prepared_{mode}_{interval}.csv",
        data_dir / f"processed_{mode}_{interval}.csv",
    ]

    fp: Optional[Path] = None
    for c in candidates:
        if c.exists():
            fp = c
            break

    if fp is None:
        raise FileNotFoundError(
            "Không tìm thấy file data. Đã thử:\n" + "\n".join(str(x) for x in candidates)
        )

    df = pd.read_csv(fp)
    if "ds" not in df.columns:
        raise ValueError(f"File {fp} thiếu cột 'ds'.")

    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").dropna().reset_index(drop=True)
    return df


def _choose_feat_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in FEATURE_COLS if c in df.columns]


# ==============================
# 4) FORECAST
# ==============================
def forecast(interval: str, model_name: str, horizon: Optional[int] = None) -> pd.DataFrame:
    """
    Trả về DataFrame: ds, yhat (đã inverse-scale cho XGB/LSTM).
    Forecast trên prepared/processed_test_{interval}.csv

    model_name: "prophet" | "xgboost" | "lstm"
    """
    cfg = load_full_config()
    models_dir = get_models_dir(cfg)

    df_test = load_prepared(interval, "test", cfg=cfg)
    feat_cols = _choose_feat_cols(df_test)

    if horizon is None or horizon > len(df_test):
        horizon = len(df_test)

    df_slice = df_test.iloc[:horizon].copy()

    # ===== Prophet =====
    if model_name == "prophet":
        pkl = models_dir / f"prophet_{interval}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"Missing model: {pkl}")
        model_p = joblib.load(pkl)
        yhat = model_p.predict(df_slice)
        return pd.DataFrame({"ds": df_slice["ds"], "yhat": yhat.astype(float)})

    # ===== XGBoost =====
    if model_name == "xgboost":
        pkl = models_dir / f"xgboost_{interval}.pkl"
        scaler_X_pkl = models_dir / f"scaler_X_{interval}.pkl"
        scaler_y_pkl = models_dir / f"scaler_y_{interval}.pkl"

        for f in [pkl, scaler_X_pkl, scaler_y_pkl]:
            if not f.exists():
                raise FileNotFoundError(f"Missing file: {f}")

        model_xgb = joblib.load(pkl)
        scaler_X = joblib.load(scaler_X_pkl)
        scaler_y = joblib.load(scaler_y_pkl)

        if len(feat_cols) == 0:
            raise ValueError("Không tìm thấy feature columns phù hợp trong file data.")

        X = scaler_X.transform(df_slice[feat_cols].values)
        pred_sc = model_xgb.predict(X).reshape(-1, 1)
        yhat = scaler_y.inverse_transform(pred_sc).flatten()
        yhat = np.maximum(yhat, 0)

        return pd.DataFrame({"ds": df_slice["ds"], "yhat": yhat.astype(float)})

    # ===== LSTM =====
    if model_name == "lstm":
        pth = models_dir / f"lstm_{interval}.pth"
        scaler_X_pkl = models_dir / f"scaler_X_{interval}.pkl"
        scaler_y_pkl = models_dir / f"scaler_y_{interval}.pkl"

        for f in [pth, scaler_X_pkl, scaler_y_pkl]:
            if not f.exists():
                raise FileNotFoundError(f"Missing file: {f}")

        scaler_X = joblib.load(scaler_X_pkl)
        scaler_y = joblib.load(scaler_y_pkl)

        if len(feat_cols) == 0:
            raise ValueError("Không tìm thấy feature columns phù hợp trong file data.")

        X_scaled = scaler_X.transform(df_slice[feat_cols].values)

        input_dim = len(feat_cols)
        forecaster = LSTMForecaster(load_full_config(), input_dim)

        forecaster.model.load_state_dict(torch.load(pth, map_location=DEVICE))
        forecaster.model.to(DEVICE)
        forecaster.model.eval()

        n_lags = load_full_config().get("models", {}).get("lstm", {}).get("n_lags", 30)
        if len(X_scaled) <= n_lags:
            raise ValueError(f"Horizon quá ngắn cho LSTM (need > n_lags={n_lags}).")

        X_seq = np.array([X_scaled[i : i + n_lags] for i in range(len(X_scaled) - n_lags)])
        inp = torch.from_numpy(X_seq).float().to(DEVICE)

        with torch.no_grad():
            pred_sc = forecaster.model(inp).detach().cpu().numpy().flatten()

        yhat = scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).flatten()
        yhat = np.maximum(yhat, 0)

        ds = df_slice["ds"].iloc[n_lags:].reset_index(drop=True)
        return pd.DataFrame({"ds": ds, "yhat": yhat.astype(float)})

    raise ValueError("model_name must be one of: prophet, xgboost, lstm")


# ==============================
# 5) SCALING
# ==============================
def load_scaling_specs(policy_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = load_full_config()
    specs = (cfg.get("scaling_simulation") or {}).copy()
    if not specs:
        raise KeyError("Thiếu scaling_simulation trong config.yaml")

    if policy_overrides:
        for k, v in policy_overrides.items():
            if v is not None:
                specs[k] = v

    return specs


def recommend_scaling(forecast_df: pd.DataFrame, specs: Dict[str, Any]) -> pd.DataFrame:
    cap = float(specs["server_capacity"])
    min_rep = int(specs["min_replicas"])
    max_rep = int(specs["max_replicas"])
    buffer_ratio = float(specs.get("buffer_ratio", 0.2))
    cooldown_period = int(specs.get("cooldown_period", 3))

    current = min_rep
    cooldown = 0
    rows: List[Dict[str, Any]] = []

    for _, r in forecast_df.iterrows():
        yhat = float(r["yhat"])

        target = int(np.ceil((yhat / cap) * (1 + buffer_ratio)))
        target = int(np.clip(target, min_rep, max_rep))

        final_rep = current
        action = "hold"
        reason = "stable"

        if target > current:
            final_rep = target
            action = "scale_out"
            reason = "forecast above capacity threshold"
            cooldown = cooldown_period

        elif target < current:
            if cooldown <= 0:
                final_rep = target
                action = "scale_in"
                reason = "forecast below threshold and cooldown passed"
            else:
                final_rep = current
                action = "hold"
                reason = f"cooldown active ({cooldown})"
                cooldown -= 1
        else:
            cooldown = max(0, cooldown - 1)

        current = final_rep

        rows.append(
            {
                "ds": r["ds"],
                "yhat": yhat,
                "recommended_replicas": int(final_rep),
                "action": action,
                "reason": reason,
            }
        )

    return pd.DataFrame(rows)


# ==============================
# 6) CLI / EXAMPLE RUN
# ==============================
def main():
    # Ví dụ chạy:
    # python forecast_and_scale.py 5min xgboost 1000
    interval = sys.argv[1] if len(sys.argv) > 1 else "5min"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "xgboost"
    horizon = int(sys.argv[3]) if len(sys.argv) > 3 else None

    fc = forecast(interval=interval, model_name=model_name, horizon=horizon)
    specs = load_scaling_specs()
    plan = recommend_scaling(fc, specs)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fc_path = out_dir / f"forecast_{model_name}_{interval}.csv"
    plan_path = out_dir / f"scaling_plan_{model_name}_{interval}.csv"

    fc.to_csv(fc_path, index=False)
    plan.to_csv(plan_path, index=False)

    print(f"[OK] Saved forecast: {fc_path}")
    print(f"[OK] Saved scaling plan: {plan_path}")


if __name__ == "__main__":
    main()
