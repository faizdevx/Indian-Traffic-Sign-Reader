from pydantic import BaseModel


class TopKItem(BaseModel):
    label: str
    probability: float


class PredictionResponse(BaseModel):
    model: str
    prediction: str
    confidence: float  # raw softmax probability of the top class, not a guarantee of correctness
    calibrated_confidence: float | None = None  # present only if a validation-fitted temperature exists
    top_k: list[TopKItem]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    models: dict[str, bool]  # model name -> checkpoint present
