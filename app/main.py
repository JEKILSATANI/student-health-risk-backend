"""Student Health Risk Prediction FastAPI Application.

Provides RESTful endpoints for student health risk prediction, health checking,
and root status. Built for the ICT654 IT Capstone Project.

Note:
    This system is an academic capstone prototype for early-warning and decision
    support. It is NOT a medical diagnosis system.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.ml_service import model_service
from app.schemas import (
    HealthCheckResponse,
    PredictionRequest,
    PredictionResponse,
    RootResponse,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler verifying ML model readiness on startup."""
    logger.info("FastAPI backend initializing...")
    if model_service.pipeline is None:
        raise RuntimeError("ML model pipeline was not initialized.")
    logger.info(
        "ML model ready. Detected classes: %s",
        model_service.classes,
    )
    yield
    logger.info("FastAPI backend shutting down...")


app = FastAPI(
    title="Student Health Risk Prediction API",
    description=(
        "An API for predicting student health risk using a trained Gradient Boosting "
        "machine-learning model.\n\n"
        "**Responsible Use Disclaimer**:\n"
        "This system is an academic capstone prototype engineered for early-warning "
        "and decision support. It is **NOT** a medical diagnosis system and should "
        "not be used as a substitute for professional clinical advice."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware Configuration
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get(
    "/",
    response_model=RootResponse,
    summary="Root Endpoint",
    description="Returns service status confirmation indicating the API is running.",
    tags=["General"],
)
def root() -> RootResponse:
    """Return welcome status message."""
    return RootResponse(message="Student Health Risk Prediction API is running.")


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Returns the operational status of the prediction API.",
    tags=["Monitoring"],
)
def health_check() -> HealthCheckResponse:
    """Return healthy status."""
    return HealthCheckResponse(status="healthy")


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Student Health Risk",
    description=(
        "Evaluates 13 multi-dimensional lifestyle and physiological student indicators "
        "using a trained HistGradientBoostingClassifier pipeline.\n\n"
        "Returns the exact predicted class, user-friendly risk category, confidence score, "
        "and class probabilities distribution."
    ),
    tags=["Prediction"],
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Execute risk prediction on validated input payload.

    Args:
        request: Validated student health data and optional student_id.

    Returns:
        PredictionResponse with predicted risk, confidence, and calibrated probabilities.
    """
    try:
        payload = request.model_dump()
        result = model_service.predict(payload)
        return PredictionResponse(**result)
    except Exception as exc:
        logger.exception("Error executing prediction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction execution failed: {str(exc)}",
        ) from exc