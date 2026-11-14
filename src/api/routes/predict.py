import pandas as pd

from fastapi import APIRouter, HTTPException, Request, status

from src.api.schemas.predict import (
    PredictionRequest,
    PredictionResponse,
)


router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
def predict_rul(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    try:
        history_records = [
            record.model_dump()
            for record in payload.history
        ]

        engine_history_df = pd.DataFrame(history_records)

        engine_history_df.insert(
            0,
            "engine_id",
            payload.engine_id,
        )

        result = request.app.state.rul_service.predict(
            engine_history_df
        )

        return PredictionResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to an internal server error.",
        ) from exc