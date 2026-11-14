from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes.health import router as health_router
from src.api.routes.predict import router as predict_router
from src.inference.rul_inference_service import RULInferenceService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rul_service = RULInferenceService()

    yield

    app.state.rul_service = None


app = FastAPI(
    title="NASA C-MAPSS RUL Prediction API",
    description=(
        "Production inference API for predicting aircraft engine "
        "Remaining Useful Life using the NASA C-MAPSS FD001 dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.include_router(health_router)
app.include_router(predict_router)