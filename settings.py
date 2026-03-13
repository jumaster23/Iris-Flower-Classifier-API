import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
SAVED_MODELS_DIR = BASE_DIR / "saved_models"

MODEL_FILENAME = os.getenv("MODEL_FILENAME", "random_forest.joblib")
SCALER_FILENAME = os.getenv("SCALER_FILENAME", "scaler.joblib")

MODEL_PATH = SAVED_MODELS_DIR / MODEL_FILENAME
SCALER_PATH = SAVED_MODELS_DIR / SCALER_FILENAME
METADATA_PATH = SAVED_MODELS_DIR / "metadata.json"

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8001))
API_KEY = os.getenv("API_KEY", "")