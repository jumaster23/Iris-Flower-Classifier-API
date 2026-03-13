"""
train.py — Train and save the Iris classification model.

Usage:
    uv run train.py
"""

import json
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

from settings import SAVED_MODELS_DIR, SCALER_PATH, METADATA_PATH

SAVED_MODELS_DIR.mkdir(exist_ok=True)

MODELS = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5),
    "logistic_regression": LogisticRegression(max_iter=200, random_state=42),
    "svm": SVC(kernel="rbf", probability=True, random_state=42),
}


def train():
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    for name, model in MODELS.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        acc = accuracy_score(y_test, preds)
        cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5)
        results[name] = {
            "accuracy": round(acc, 4),
            "cv_mean": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
        }
        path = SAVED_MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, path)
        print(f"  [{name}] accuracy={acc:.4f}  cv={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        print(classification_report(y_test, preds, target_names=iris.target_names))

    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved → {SCALER_PATH}")

    best = "random_forest"
    metadata = {
        "model": "RandomForestClassifier",
        "filename": f"{best}.joblib",
        "accuracy": results[best]["accuracy"],
        "cv_mean": results[best]["cv_mean"],
        "cv_std": results[best]["cv_std"],
        "features": iris.feature_names,
        "classes": iris.target_names.tolist(),
        "all_models": results,
        "dataset": "Iris (UCI)",
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll models saved to {SAVED_MODELS_DIR}/")
    print(f"Metadata saved → {METADATA_PATH}")


if __name__ == "__main__":
    train()