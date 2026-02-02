from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schema import (
    ForecastRequest, ForecastResponse,
    RecommendRequest, RecommendResponse
)

from .services import (
    load_predictions_from_results,
    load_scaling_specs,
    recommend_scaling
)

app = FastAPI(title="Dataflow 2026 Autoscaling API", version="1.0")

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


@app.post("/forecast", response_model=ForecastResponse)
def api_forecast(req: ForecastRequest):
    """
    Không chạy model forecast realtime.
    Chỉ đọc forecast có sẵn từ: results/predictions_{interval}.xlsx
    Cột model phải đúng tên cột trong Excel: Prophet / XGBoost / LSTM
    """
    try:
        df = load_predictions_from_results(
            interval=req.interval,
            model_col=req.model
        )

        # nếu muốn cắt theo horizon (lấy N điểm đầu)
        if req.horizon is not None and int(req.horizon) > 0:
            df = df.head(int(req.horizon))

        points = [
            {"ds": str(x), "yhat": float(y)}
            for x, y in zip(df["ds"], df["yhat"])
        ]

        return {"interval": req.interval, "model": req.model, "points": points}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/recommend-scaling", response_model=RecommendResponse)
def api_recommend(req: RecommendRequest):
    """
    1) Load forecast có sẵn từ Excel
    2) Load scaling specs (config + overrides)
    3) Recommend scaling plan
    """
    try:
        df_fc = load_predictions_from_results(
            interval=req.interval,
            model_col=req.model
        )

        # nếu muốn cắt theo horizon (lấy N điểm đầu)
        if req.horizon is not None and int(req.horizon) > 0:
            df_fc = df_fc.head(int(req.horizon))

        overrides = req.policy_params.model_dump() if req.policy_params else None
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
            "policy_used": specs,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
