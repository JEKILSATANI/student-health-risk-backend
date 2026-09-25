"""Automated tests for the Student Health Risk Prediction FastAPI backend.

Verifies model loading, endpoint responses, case-insensitive categorical normalization,
schema validation, and mathematical integrity of prediction probabilities.
"""

import sys
import unittest
from pathlib import Path

# Ensure backend root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient

from app.main import app
from app.ml_service import model_service


class TestStudentHealthRiskAPI(unittest.TestCase):
    """Test suite verifying backend endpoints and ML inference."""

    @classmethod
    def setUpClass(cls):
        """Initialize FastAPI test client with lifespan context."""
        cls.client = TestClient(app)

    def test_01_model_loading(self):
        """Verify the trained joblib model loaded and discovered expected classes."""
        self.assertIsNotNone(model_service.pipeline, "Pipeline should be loaded.")
        self.assertEqual(
            set(model_service.classes),
            {"at-risk", "fit", "unhealthy"},
            "Model classes must exactly match ['at-risk', 'fit', 'unhealthy'].",
        )

    def test_02_get_root(self):
        """Verify GET / endpoint returns running confirmation."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertEqual(data["message"], "Student Health Risk Prediction API is running.")

    def test_03_get_health(self):
        """Verify GET /health returns healthy status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data, {"status": "healthy"})

    def test_04_post_predict_valid_sample(self):
        """Verify POST /predict with standard benchmark payload from Section 7."""
        payload = {
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

        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # 1. student_id verification
        self.assertEqual(data["student_id"], "S10245")

        # 2. prediction verification
        self.assertIn(data["prediction"], {"at-risk", "fit", "unhealthy"})

        # 3. risk_category verification (Title-Case)
        self.assertIn(data["risk_category"], {"At-Risk", "Fit", "Unhealthy"})

        # 4. confidence verification
        self.assertIsInstance(data["confidence"], (int, float))
        self.assertGreaterEqual(data["confidence"], 0.0)
        self.assertLessEqual(data["confidence"], 1.0)

        # 5. probabilities keys: MUST contain ONLY the exact model class labels
        self.assertIn("probabilities", data)
        probs = data["probabilities"]
        self.assertEqual(
            set(probs.keys()),
            {"at-risk", "fit", "unhealthy"},
            "Probabilities must contain only exact model labels without duplicates.",
        )

        # 6. probabilities values within [0, 1]
        for cls_name, p_val in probs.items():
            self.assertGreaterEqual(p_val, 0.0, f"Probability for {cls_name} < 0")
            self.assertLessEqual(p_val, 1.0, f"Probability for {cls_name} > 1")

        # 7. probabilities sum to approximately 1.0
        prob_sum = sum(probs.values())
        self.assertAlmostEqual(
            prob_sum,
            1.0,
            places=2,
            msg=f"Probabilities must sum to ~1.0, got {prob_sum}",
        )

        # 8. confidence matches the probability of the predicted class
        self.assertAlmostEqual(data["confidence"], probs[data["prediction"]], places=4)

    def test_05_case_insensitive_categorical_inputs(self):
        """Verify categorical inputs are validated case-insensitively and normalized."""
        payload = {
            "student_id": "TEST_CASE_INSENSITIVE",
            "sleep_duration": 6.5,
            "heart_rate": 80,
            "bmi": 24.0,
            "calorie_expenditure": 2100,
            "step_count": 6000,
            "exercise_duration": 30,
            "water_intake": 2.0,
            "diet_type": "  BALANCED  ",  # Uppercase + whitespace
            "stress_level": "High",       # Title-case
            "sleep_quality": "POOR",      # Uppercase
            "physical_activity_level": "Sedentary",
            "smoking_alcohol": "Occasional",
            "gender": "Female",
        }

        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["student_id"], "TEST_CASE_INSENSITIVE")
        self.assertIn(data["prediction"], {"at-risk", "fit", "unhealthy"})
        self.assertIn(data["risk_category"], {"At-Risk", "Fit", "Unhealthy"})

    def test_06_validation_error_invalid_categorical(self):
        """Verify invalid categorical input returns 422 Unprocessable Entity."""
        payload = {
            "student_id": "S10999",
            "sleep_duration": 7.0,
            "heart_rate": 70,
            "bmi": 22.0,
            "calorie_expenditure": 2000,
            "step_count": 8000,
            "exercise_duration": 30,
            "water_intake": 2.0,
            "diet_type": "keto_junk",  # Invalid categorical value
            "stress_level": "medium",
            "sleep_quality": "good",
            "physical_activity_level": "active",
            "smoking_alcohol": "no",
            "gender": "male",
        }

        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)
        errors = response.json()
        self.assertIn("detail", errors)

    def test_07_validation_error_numeric_out_of_bounds(self):
        """Verify numeric feature exceeding boundary returns 422 Unprocessable Entity."""
        payload = {
            "student_id": "S10999",
            "sleep_duration": -4.0,  # Negative sleep duration is impossible
            "heart_rate": 70,
            "bmi": 22.0,
            "calorie_expenditure": 2000,
            "step_count": 8000,
            "exercise_duration": 30,
            "water_intake": 2.0,
            "diet_type": "balanced",
            "stress_level": "medium",
            "sleep_quality": "good",
            "physical_activity_level": "active",
            "smoking_alcohol": "no",
            "gender": "male",
        }

        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_08_optional_student_id(self):
        """Verify prediction succeeds even when student_id is omitted."""
        payload = {
            "sleep_duration": 8.0,
            "heart_rate": 65,
            "bmi": 21.5,
            "calorie_expenditure": 2300,
            "step_count": 10000,
            "exercise_duration": 60,
            "water_intake": 3.0,
            "diet_type": "veg",
            "stress_level": "low",
            "sleep_quality": "good",
            "physical_activity_level": "active",
            "smoking_alcohol": "no",
            "gender": "female",
        }

        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsNone(data["student_id"])
        self.assertIn(data["prediction"], {"at-risk", "fit", "unhealthy"})


if __name__ == "__main__":
    unittest.main()
