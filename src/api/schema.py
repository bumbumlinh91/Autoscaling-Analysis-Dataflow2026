from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

Interval = Literal["1min", "5min", "15min"]
ModelName = Literal["prophet", "xgboost", "lstm"]
TargetName = Literal["intensity"]  # bạn đang dùng intensity làm target

class ForecastRequest(BaseModel):
    interval: Interval = "15min"
    model: ModelName = "xgboost"
    horizon: Optional[int] = Field(default=None, description="Số bước forecast (default: full test length)")
    target: TargetName = "intensity"

class ForecastPoint(BaseModel):
    ds: str
    yhat: float

class ForecastResponse(BaseModel):
    interval: Interval
    model: ModelName
    points: List[ForecastPoint]

class PolicyParams(BaseModel):
    min_replicas: Optional[int] = None
    max_replicas: Optional[int] = None
    buffer_ratio: Optional[float] = None
    cooldown_period: Optional[int] = None
    server_capacity: Optional[float] = None

class RecommendRequest(BaseModel):
    interval: Interval = "15min"
    model: ModelName = "xgboost"
    horizon: Optional[int] = None
    policy_params: Optional[PolicyParams] = None

class ScalePoint(BaseModel):
    ds: str
    yhat: float
    recommended_replicas: int
    action: str
    reason: str

class RecommendResponse(BaseModel):
    interval: Interval
    model: ModelName
    points: List[ScalePoint]
    policy_used: Dict[str, Any]