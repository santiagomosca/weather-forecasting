# api/schemas.py

from pydantic import BaseModel, field_validator
from typing import List

NUM_FEATURES  = 19
WINDOW_SIZE   = 144


class PredictRequest(BaseModel):
    window: List[List[float]]

    @field_validator('window')
    @classmethod
    def validate_window(cls, v):
        if len(v) != WINDOW_SIZE:
            raise ValueError(f"Se esperan {WINDOW_SIZE} pasos temporales, se recibieron {len(v)}")
        for i, step in enumerate(v):
            if len(step) != NUM_FEATURES:
                raise ValueError(
                    f"Paso {i}: se esperan {NUM_FEATURES} features, se recibieron {len(step)}"
                )
        return v


class PredictResponse(BaseModel):
    predicted_temperature_celsius: float
    model                        : str
    horizon_minutes              : int


class InfoResponse(BaseModel):
    model        : str
    window_size  : int
    num_features : int
    horizon      : int
    units        : str
    test_mae     : float
    test_rmse    : float


class HealthResponse(BaseModel):
    status: str
