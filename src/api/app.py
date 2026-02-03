from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from .schema import ForecastRequest, ForecastResponse, RecommendRequest, RecommendResponse
from .services import load_predictions_from_results, load_scaling_specs, recommend_scaling

app = FastAPI(title="Dataflow 2026 Autoscaling API", version="1.0")
@app.get("/")
def trang_chu():
    return {
        "thong_bao": "Hệ thống API Dự báo tải & Autoscaling đang hoạt động",
        "huong_dan": "Vui lòng truy cập trang web sau để sử dụng API",
        "trang_su_dung_api": "http://127.0.0.1:8000/docs"
    }
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}
@app.get("/config")
def get_config_defaults():
    """Trả về cấu hình mặc định từ config.yaml cho Dashboard"""
    return load_scaling_specs()
@app.post("/forecast", response_model=ForecastResponse)
def api_forecast(req: ForecastRequest):
    try:
        # [FIX] Map model name to CSV column format (Capitalized)
        model_map = {
            "prophet": "Prophet",
            "xgboost": "XGBoost",
            "lstm": "LSTM"
        }
        model_col = model_map.get(req.model.lower(), req.model)

        df = load_predictions_from_results(req.interval, model_col)
        
        if req.horizon is not None and int(req.horizon) > 0:
            df = df.head(int(req.horizon))
            
        points = [{
                "ds": str(row["ds"]), 
                "yhat": float(row["yhat"]),
                "y": float(row["y"]) if "y" in row and pd.notnull(row["y"]) else None
            }
            for _, row in df.iterrows()
        ]
        return {"interval": req.interval, "model": req.model, "points": points}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recommend-scaling", response_model=RecommendResponse)
def api_recommend(req: RecommendRequest):
    try:
        # [FIX] Map model name
        model_map = {
            "prophet": "Prophet",
            "xgboost": "XGBoost",
            "lstm": "LSTM"
        }
        model_col = model_map.get(req.model.lower(), req.model)

        df_fc = load_predictions_from_results(req.interval, model_col)
        
        if req.horizon is not None and int(req.horizon) > 0:
            df_fc = df_fc.head(int(req.horizon))

        overrides = (req.policy_params.model_dump() if req.policy_params else None)
        specs = load_scaling_specs(overrides)

        df_plan = recommend_scaling(df_fc, specs)
        points = [
            {
                "ds": str(r["ds"]),
                "yhat": float(r["yhat"]),
                "recommended_replicas": int(r["recommended_replicas"]),
                "action": str(r["action"]),
                "reason": str(r["reason"]),
            }
            for _, r in df_plan.iterrows()
        ]

        return {
            "interval": req.interval,
            "model": req.model,
            "points": points,
            "policy_used": specs
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))