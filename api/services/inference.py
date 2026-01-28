import json
from pathlib import Path
import joblib
import pandas as pd


# -------------------------------------------------------------------
# Resolve artifact paths
# -------------------------------------------------------------------
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "model"

MODEL_PATH = ARTIFACTS_DIR / "gradient_boosting_model.joblib"
THRESHOLD_PATH = ARTIFACTS_DIR / "decision_threshold.json"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"


# -------------------------------------------------------------------
# Load artifacts ONCE at startup
# -------------------------------------------------------------------
model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as f:
    _thresholds = json.load(f)

with open(METADATA_PATH, "r") as f:
    _metadata = json.load(f)


# Expected structure in decision_threshold.json
APPROVE_THRESHOLD = _thresholds.get("approve", 0.3)
CONDITIONAL_THRESHOLD = _thresholds.get("conditional", 0.6)


# -------------------------------------------------------------------
# Public inference function
# -------------------------------------------------------------------
def run_inference(features: pd.DataFrame) -> dict:
    """
    Run model inference on preprocessed features.

    Args:
        features (pd.DataFrame): Output of preprocess_request()

    Returns:
        dict: prediction results + business decision
    """

    # Safety check
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")

    # Model prediction
    prediction = int(model.predict(features)[0])

    # Probability of default (assumes binary classifier)
    if hasattr(model, "predict_proba"):
        probability_of_default = float(model.predict_proba(features)[0][1])
    else:
        probability_of_default = None

    # Business decision logic
    if probability_of_default is None:
        decision = "UNKNOWN"
    elif probability_of_default < APPROVE_THRESHOLD:
        decision = "APPROVE"
    elif probability_of_default < CONDITIONAL_THRESHOLD:
        decision = "CONDITIONAL_APPROVAL"
    else:
        decision = "REJECT"

    return {
        "decision": decision,
        "prediction": prediction,
        "probability_of_default": probability_of_default,
        "model_version": _metadata.get("model_version"),
        "model_name": _metadata.get("model_name"),
    }


def run_inference_batch(features: pd.DataFrame) -> list[dict]:
    """
    Run batch inference on preprocessed features.

    Args:
        features (pd.DataFrame): Output of preprocess_request_batch()

    Returns:
        list[dict]: Prediction results per record
    """

    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")

    # Predict labels
    predictions = model.predict(features)

    # Predict probabilities (binary classifier assumed)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[:, 1]
    else:
        probabilities = [None] * len(predictions)

    results = []

    for pred, prob in zip(predictions, probabilities):
        # Decision logic
        if prob is None:
            decision = "UNKNOWN"
        elif prob < APPROVE_THRESHOLD:
            decision = "APPROVE"
        elif prob < CONDITIONAL_THRESHOLD:
            decision = "CONDITIONAL_APPROVAL"
        else:
            decision = "REJECT"

        results.append({
            "decision": decision,
            "prediction": int(pred),
            "probability_of_default": float(prob) if prob is not None else None,
            "model_version": _metadata.get("model_version"),
            "model_name": _metadata.get("model_name"),
        })

    return results
