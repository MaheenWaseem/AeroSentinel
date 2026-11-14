from pydantic import BaseModel, Field


class EngineRecord(BaseModel):
    cycle: int = Field(gt=0)

    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_17: float
    sensor_20: float
    sensor_21: float


class PredictionRequest(BaseModel):
    engine_id: int = Field(gt=0)
    history: list[EngineRecord] = Field(min_length=1)


class PredictionResponse(BaseModel):
    engine_id: int
    last_observed_cycle: int
    predicted_rul: float