"""Machine learning service for loading and running the Student Health Risk model.

Loads the pre-trained HistGradientBoostingClassifier joblib pipeline once,
dynamically discovers classes, and performs vectorized inference with pandas.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import pandas as pd

logger = logging.getLogger(__name__)

# Exact 13 features expected by the trained scikit-learn pipeline
FEATURE_NAMES: List[str] = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]

# Formatting mapping for the user-facing risk_category field
LABEL_TITLE_MAP: Dict[str, str] = {
    "at-risk": "At-Risk",
    "fit": "Fit",
    "unhealthy": "Unhealthy",
}


class ModelService:
    """Singleton service for managing model loading and inference."""

    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            # Default to backend/models/student_health_risk_gradient_boosting.joblib
            base_dir = Path(__file__).resolve().parent.parent
            model_path = base_dir / "models" / "student_health_risk_gradient_boosting.joblib"

        self.model_path = model_path
        self.pipeline: Any = None
        self.classes: List[str] = []
        self._load_model()

    def _load_model(self) -> None:
        """Load the joblib pipeline from disk and verify its structure."""
        if not self.model_path.exists():
            error_msg = f"Trained model file not found at: {self.model_path}"
            logger.critical(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info("Loading pre-trained pipeline from: %s", self.model_path)
        self.pipeline = joblib.load(self.model_path)

        # Retrieve classes dynamically from the pipeline/model
        if hasattr(self.pipeline, "classes_"):
            self.classes = [str(c) for c in self.pipeline.classes_]
        elif hasattr(self.pipeline, "named_steps") and "model" in self.pipeline.named_steps:
            self.classes = [str(c) for c in self.pipeline.named_steps["model"].classes_]
        elif hasattr(self.pipeline, "named_steps") and "classifier" in self.pipeline.named_steps:
            self.classes = [str(c) for c in self.pipeline.named_steps["classifier"].classes_]
        else:
            # Fallback if classes_ is not exposed as top-level attribute
            self.classes = ["at-risk", "fit", "unhealthy"]

        logger.info("Model loaded successfully. Classes detected: %s", self.classes)

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform health risk prediction for a single validated student record.

        Args:
            input_data: Dictionary containing student features and optional student_id.

        Returns:
            Dictionary containing student_id, prediction, risk_category, confidence,
            and class probabilities.
        """
        # Extract exclusively the 13 ML features
        features = {}
        for feat in FEATURE_NAMES:
            val = input_data.get(feat)
            if isinstance(val, str):
                features[feat] = val.strip().lower()
            else:
                features[feat] = val

        # Construct single-row DataFrame with explicit feature ordering
        df = pd.DataFrame([features], columns=FEATURE_NAMES)

        # Run inference through the scikit-learn pipeline
        raw_pred = self.pipeline.predict(df)[0]
        raw_probs = self.pipeline.predict_proba(df)[0]

        prediction = str(raw_pred)

        # Compute probabilities rounded to 4 decimals containing only exact model class labels
        probabilities: Dict[str, float] = {}
        for cls_name, prob in zip(self.classes, raw_probs):
            probabilities[str(cls_name)] = round(float(prob), 4)

        # Calculate confidence as the probability associated with the predicted class
        confidence = probabilities.get(prediction, round(float(max(raw_probs)), 4))

        # Format user-friendly Title-Case risk category
        risk_category = LABEL_TITLE_MAP.get(prediction, prediction.capitalize())

        return {
            "student_id": input_data.get("student_id"),
            "prediction": prediction,
            "risk_category": risk_category,
            "confidence": confidence,
            "probabilities": probabilities,
        }


# Global singleton instance loaded at module import time
model_service = ModelService()
