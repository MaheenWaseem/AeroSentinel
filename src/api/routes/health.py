from fastapi import APIRouter, status

from src.api.schemas.health import HealthResponse


router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["System"],
)
def health() -> HealthResponse:
    return HealthResponse(status="ok")