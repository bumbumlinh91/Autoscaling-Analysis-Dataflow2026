from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schema import ForecastRequest, ForecastResponse, RecommendRequest, RecommendResponse
from .services import forecast, load_scaling_specs, recommend_scaling

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

@app.post("/forecast", response_model=ForecastResponse)
def api_forecast(req: ForecastRequest):
    try:
        df = forecast(req.interval, req.model, req.horizon)
        points = [{"ds": str(x), "yhat": float(y)} for x, y in zip(df["ds"], df["yhat"])]
        return {"interval": req.interval, "model": req.model, "points": points}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recommend-scaling", response_model=RecommendResponse)
def api_recommend(req: RecommendRequest):
    try:
        df_fc = forecast(req.interval, req.model, req.horizon)

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