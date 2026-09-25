"""Pydantic schemas for the Student Health Risk Prediction API.

Defines request and response schemas with strict validation, case-insensitive
categorical normalization, and comprehensive OpenAPI documentation examples.
"""

from typing import Dict, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


# Allowed categorical values (lowercase)
ALLOWED_DIET_TYPES = {"veg", "balanced", "non-veg"}
ALLOWED_STRESS_LEVELS = {"low", "medium", "high"}
ALLOWED_SLEEP_QUALITIES = {"poor", "average", "good"}
ALLOWED_ACTIVITY_LEVELS = {"sedentary", "moderate", "active"}
ALLOWED_SMOKING_ALCOHOL = {"yes", "no", "occasional"}
ALLOWED_GENDERS = {"female", "male", "other"}


class PredictionRequest(BaseModel):
    """Input payload containing student lifestyle, activity, and physiological features.

    Note:
        student_id is metadata only and is excluded from the machine learning model.
    """

    student_id: Optional[str] = Field(
        default=None,
        description="Optional student identifier for record-keeping. Excluded from ML features.",
        max_length=50,
        examples=["S10245"],
    )

    # 7 Numerical features
    sleep_duration: float = Field(
        ...,
        ge=0.0,
        le=24.0,
        description="Average daily sleep duration in hours (0 - 24)",
        examples=[7.2],
    )
    heart_rate: float = Field(
        ...,
        ge=30.0,
        le=250.0,
        description="Resting heart rate in beats per minute (30 - 250)",
        examples=[72.0],
    )
    bmi: float = Field(
        ...,
        ge=10.0,
        le=70.0,
        description="Body Mass Index in kg/m² (10 - 70)",
        examples=[22.4],
    )
    calorie_expenditure: float = Field(
        ...,
        ge=0.0,
        le=15000.0,
        description="Estimated daily calorie expenditure in kcal (0 - 15000)",
        examples=[2240.0],
    )
    step_count: float = Field(
        ...,
        ge=0.0,
        le=100000.0,
        description="Average daily step count (0 - 100000)",
        examples=[8500.0],
    )
    exercise_duration: float = Field(
        ...,
        ge=0.0,
        le=1440.0,
        description="Daily exercise duration in minutes (0 - 1440)",
        examples=[45.0],
    )
    water_intake: float = Field(
        ...,
        ge=0.0,
        le=20.0,
        description="Daily water intake in liters (0 - 20)",
        examples=[2.5],
    )

    # 6 Categorical features (case-insensitive validation, normalized to lowercase)
    diet_type: str = Field(
        ...,
        description="Dietary habits: 'veg', 'balanced', or 'non-veg'",
        examples=["balanced"],
    )
    stress_level: str = Field(
        ...,
        description="Self-reported stress level: 'low', 'medium', or 'high'",
        examples=["medium"],
    )
    sleep_quality: str = Field(
        ...,
        description="Perceived sleep quality: 'poor', 'average', or 'good'",
        examples=["good"],
    )
    physical_activity_level: str = Field(
        ...,
        description="Routine physical activity: 'sedentary', 'moderate', or 'active'",
        examples=["active"],
    )
    smoking_alcohol: str = Field(
        ...,
        description="Substance use frequency: 'no', 'occasional', or 'yes'",
        examples=["no"],
    )
    gender: str = Field(
        ...,
        description="Student gender: 'female', 'male', or 'other'",
        examples=["male"],
    )

    @field_validator("diet_type", mode="before")
    @classmethod
    def validate_diet_type(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("diet_type must be a string")
        norm = value.strip().lower()
        if norm not in ALLOWED_DIET_TYPES:
            raise ValueError(
                f"Invalid diet_type '{value}'. Expected one of: {sorted(ALLOWED_DIET_TYPES)}"
            )
        return norm

    @field_validator("stress_level", mode="before")
    @classmethod
    def validate_stress_level(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("stress_level must be a string")
        norm = value.strip().lower()
        if norm not in ALLOWED_STRESS_LEVELS:
            raise ValueError(
                f"Invalid stress_level '{value}'. Expected one of: {sorted(ALLOWED_STRESS_LEVELS)}"
            )
        return norm

    @field_validator("sleep_quality", mode="before")
    @classmethod
    def validate_sleep_quality(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("sleep_quality must be a string")
        norm = value.strip().lower()
        if norm not in ALLOWED_SLEEP_QUALITIES:
            raise ValueError(
                f"Invalid sleep_quality '{value}'. Expected one of: {sorted(ALLOWED_SLEEP_QUALITIES)}"
            )
        return norm

    @field_validator("physical_activity_level", mode="before")
    @classmethod
    def validate_activity_level(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("physical_activity_level must be a string")
        norm = value.strip().lower()
        if norm not in ALLOWED_ACTIVITY_LEVELS:
            raise ValueError(
                f"Invalid physical_activity_level '{value}'. Expected one of: {sorted(ALLOWED_ACTIVITY_LEVELS)}"
            )
        return norm

    @field_validator("smoking_alcohol", mode="before")
    @classmethod
    def validate_smoking_alcohol(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("smoking_alcohol must be a string")
        norm = value.strip().lower()
        if norm not in ALLOWED_SMOKING_ALCOHOL:
            raise ValueError(
                f"Invalid smoking_alcohol '{value}'. Expected one of: {sorted(ALLOWED_SMOKING_ALCOHOL)}"
            )
        return norm

    @field_validator("gender", mode="before")
    @classmethod
    def validate_gender(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("gender must be a string")
        norm = value.strip().lower()
        if norm not in ALLOWED_GENDERS:
            raise ValueError(
                f"Invalid gender '{value}'. Expected one of: {sorted(ALLOWED_GENDERS)}"
            )
        return norm

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "student_id": "S10245",
                "sleep_duration": 7.2,
                "heart_rate": 72,
                "bmi": 22.4,
                "calorie_expenditure": 2240,
                "step_count": 8500,
                "exercise_duration": 45,
                "water_intake": 2.5,
                "diet_type": "balanced",
                "stress_level": "medium",
                "sleep_quality": "good",
                "physical_activity_level": "active",
                "smoking_alcohol": "no",
                "gender": "male",
            }
        }
    )


class PredictionResponse(BaseModel):
    """Output prediction response generated by the trained Gradient Boosting model."""

    student_id: Optional[str] = Field(
        default=None,
        description="Echoed student identifier from the request.",
        examples=["S10245"],
    )
    prediction: str = Field(
        ...,
        description="Exact predicted class label from the trained model pipeline ('at-risk', 'fit', 'unhealthy').",
        examples=["at-risk"],
    )
    risk_category: str = Field(
        ...,
        description="User-friendly Title-Case risk category ('At-Risk', 'Fit', 'Unhealthy').",
        examples=["At-Risk"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence calculated from the predicted class probability.",
        examples=[0.7475],
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Calibrated probability distribution across exact model classes ('at-risk', 'fit', 'unhealthy').",
        examples=[
            {
                "at-risk": 0.7475,
                "fit": 0.2443,
                "unhealthy": 0.0082,
            }
        ],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "student_id": "S10245",
                "prediction": "at-risk",
                "risk_category": "At-Risk",
                "confidence": 0.7475,
                "probabilities": {
                    "at-risk": 0.7475,
                    "fit": 0.2443,
                    "unhealthy": 0.0082,
                },
            }
        }
    )


class HealthCheckResponse(BaseModel):
    """Health check status response."""

    status: str = Field(
        default="healthy",
        description="Operational health status of the API.",
        examples=["healthy"],
    )


class RootResponse(BaseModel):
    """Root endpoint welcome response."""

    message: str = Field(
        ...,
        description="Service welcome message.",
        examples=["Student Health Risk Prediction API is running."],
    )
