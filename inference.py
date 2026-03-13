"""
inference.py — Load saved model + scaler and run predictions.
"""

import numpy as np
import joblib
from pathlib import Path

from settings import MODEL_PATH, SCALER_PATH, SAVED_MODELS_DIR

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

AVAILABLE_MODELS = ["random_forest", "logistic_regression", "svm"]


class IrisClassifier:
    def __init__(self, model_path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.class_names = CLASS_NAMES

    def predict(self, features: list[float]) -> dict:
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        class_idx = int(self.model.predict(X_scaled)[0])
        class_name = self.class_names[class_idx]

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)[0].tolist()
            probabilities = {
                name: round(p, 4)
                for name, p in zip(self.class_names, probs)
            }
        else:
            probabilities = {
                name: (1.0 if i == class_idx else 0.0)
                for i, name in enumerate(self.class_names)
            }

        return {
            "predicted_class": class_name,
            "class_index": class_idx,
            "probabilities": probabilities,
        }

    def predict_batch(self, batch: list[list[float]]) -> list[dict]:
        return [self.predict(features) for features in batch]


def load_classifier(model_name: str) -> IrisClassifier:
    """Load a classifier by model name (random_forest, logistic_regression, svm)."""
    model_path = SAVED_MODELS_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model '{model_name}' not found. Run train.py first.")
    return IrisClassifier(model_path=model_path)