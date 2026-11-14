# Base Linux image with Python 3.12
FROM python:3.12-slim

# Copy uv into the image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Configure Python and uv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Set the application's working directory
WORKDIR /app

# Copy dependency configuration
COPY pyproject.toml uv.lock ./

# Install locked production dependencies
RUN uv sync --locked --no-dev --no-install-project

# Copy the existing application source
COPY src/ ./src/

# Copy only the final inference artifacts
COPY models/saved_artifacts/fd001_final_sequence_sensor_scaler.joblib \
     models/saved_artifacts/

COPY models/saved_artifacts/fd001_final_cycle_feature_scaler.joblib \
     models/saved_artifacts/

COPY models/saved_artifacts/fd001_final_cycle_aware_lstm.pt \
     models/saved_artifacts/

# Document the application port
EXPOSE 8000

# Start the existing FastAPI application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]