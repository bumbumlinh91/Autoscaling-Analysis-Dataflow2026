from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import joblib
import yaml
import numpy as np
import pandas as pd
import torch

# --- 1. SETUP ROOT PATH CHUẨN ---
# Tự động tìm về thư mục gốc chứa file config/config.yaml
def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "config" / "config.yaml").exists():
            return parent
    return current.parents[1] # Fallback: Lên 2 cấp (src/api -> src -> root)

ROOT = get_project_root()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import Model an toàn
try:
    from src.models import LSTMForecaster
except ImportError:
    # Fallback cho trường hợp chạy script lẻ
    try:
        from models import LSTMForecaster
    except ImportError:
        print("⚠️ Cảnh báo: Không tìm thấy class LSTMForecaster. LSTM sẽ lỗi nếu gọi.")
        LSTMForecaster = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. CONFIG & UTILS ---
def load_full_config() -> Dict[str, Any]:
    cfg_path = ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy config tại: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_models_dir() -> Path:
    # Ưu tiên folder models ở root
    return ROOT / "models"

def get_data_dir() -> Path:
    return ROOT / "data"

# --- 3. CORE LOGIC ---
def forecast(interval: str, model_name: str, horizon: Optional[int] = None) -> pd.DataFrame:
    cfg = load_full_config()
    models_dir = get_models_dir()
    data_dir = get_data_dir()
    
    # Load data test
    csv_path = data_dir / f"prepared_test_{interval}.csv"
    if not csv_path.exists():
        # Fallback thử file processed
        csv_path = data_dir / f"processed_test_{interval}.csv"
        if not csv_path.exists():
             raise FileNotFoundError(f"Missing data file: {csv_path}")

    df_test = pd.read_csv(csv_path)
    df_test["ds"] = pd.to_datetime(df_test["ds"])
    
    # Cắt đúng horizon
    if horizon is None or horizon > len(df_test):
        horizon = len(df_test)
    df_slice = df_test.iloc[:horizon].copy()

    # === PROPHET (ĐÃ FIX LỖI) ===
    if model_name == "prophet":
        pkl = models_dir / f"prophet_{interval}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"Missing model: {pkl}")
        
        model_p = joblib.load(pkl)
        # FIX: Lấy đúng cột yhat
        forecast_df = model_p.predict(df_slice)
        return pd.DataFrame({"ds": df_slice["ds"], "yhat": forecast_df["yhat"].values.astype(float)})

    # === XGBOOST ===
    if model_name == "xgboost":
        pkl = models_dir / f"xgboost_{interval}.pkl"
        scaler_X_path = models_dir / f"scaler_X_{interval}.pkl"
        scaler_y_path = models_dir / f"scaler_y_{interval}.pkl"
        
        for p in [pkl, scaler_X_path, scaler_y_path]:
            if not p.exists(): raise FileNotFoundError(f"Missing: {p}")
            
        model_xgb = joblib.load(pkl)
        sc_X = joblib.load(scaler_X_path)
        sc_y = joblib.load(scaler_y_path)
        
        # Chọn feature cols (Hardcode hoặc lấy từ config nếu muốn)
        feat_cols = [c for c in df_slice.columns if c not in ["ds", "y", "intensity"]]
        if not feat_cols:
             raise ValueError("Data file không có cột features (lag/rolling)")
             
        X = sc_X.transform(df_slice[feat_cols].values)
        pred = model_xgb.predict(X).reshape(-1, 1)
        yhat = sc_y.inverse_transform(pred).flatten()
        return pd.DataFrame({"ds": df_slice["ds"], "yhat": np.maximum(yhat, 0)})

    # === LSTM ===
    if model_name == "lstm":
        if LSTMForecaster is None:
            raise ImportError("Chưa import được class LSTMForecaster")
            
        pth = models_dir / f"lstm_{interval}.pth"
        scaler_X_path = models_dir / f"scaler_X_{interval}.pkl"
        scaler_y_path = models_dir / f"scaler_y_{interval}.pkl"
        
        for p in [pth, scaler_X_path, scaler_y_path]:
            if not p.exists(): raise FileNotFoundError(f"Missing: {p}")

        sc_X = joblib.load(scaler_X_path)
        sc_y = joblib.load(scaler_y_path)
        
        feat_cols = [c for c in df_slice.columns if c not in ["ds", "y", "intensity"]]
        X_scaled = sc_X.transform(df_slice[feat_cols].values)
        
        # Init Model
        input_dim = len(feat_cols)
        # Lưu ý: Cần truyền đúng tham số init cho LSTMForecaster của mày
        forecaster = LSTMForecaster(cfg, input_dim=input_dim) 
        forecaster.model.load_state_dict(torch.load(pth, map_location=DEVICE))
        forecaster.model.to(DEVICE)
        forecaster.model.eval()
        
        # Predict
        with torch.no_grad():
            inp = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            out = forecaster.model(inp).cpu().numpy().flatten()
            
        yhat = sc_y.inverse_transform(out.reshape(-1, 1)).flatten()
        return pd.DataFrame({"ds": df_slice["ds"], "yhat": np.maximum(yhat, 0)})

    raise ValueError(f"Unknown model: {model_name}")

# ==============================
# 4) FORECAST
# ==============================
def load_predictions_from_results(interval: str, model_col: str) -> pd.DataFrame:
    """
    Đọc file kết quả tĩnh từ thư mục results/predictions_{interval}.csv
    File này được tạo ra bởi scripts/evaluate.py, chứa các cột: ds, Actual, Prophet, XGBoost...
    """
    results_dir = ROOT / "results"
    csv_path = results_dir / f"predictions_{interval}.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Chưa có kết quả dự báo tại: {csv_path}. Hãy chạy scripts/evaluate.py trước.")
        
    df = pd.read_csv(csv_path)
    
    # Kiểm tra cột model
    if model_col not in df.columns:
        raise KeyError(f"Model '{model_col}' không có trong file kết quả. Các model có sẵn: {list(df.columns)}")
        
    # Trả về định dạng chuẩn: ds, yhat, y (nếu có)
    res = pd.DataFrame({
        "ds": df["ds"],
        "yhat": df[model_col]
    })
    # Xử lý NaN
    res['yhat'] = res['yhat'].fillna(0)
    # [FIX] Cắt bỏ giá trị âm (giống logic trong evaluate.py)
    res["yhat"] = res["yhat"].clip(lower=0)

    if "Actual" in df.columns:
        res["y"] = df["Actual"]
    return res

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
    forecast_df['yhat'] = forecast_df['yhat'].fillna(0)
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
