from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.services.preprocessing import preprocess_request
from api.services.inference import run_inference
from api.services.preprocessing import preprocess_request_batch
from api.services.inference import run_inference_batch


app = FastAPI(
    title="Credit Risk Scoring API",
    version="1.0.0",
    description="Production-ready Credit Risk Scoring Service",
)


# -----------------------------
# Request schema
# -----------------------------
class CreditApplicationRequest(BaseModel):
    person_age: int = Field(..., example=35)
    person_income: float = Field(..., example=75000)
    person_home_ownership: str = Field(..., example="RENT")
    person_emp_length: int = Field(..., example=5)
    loan_intent: str = Field(..., example="PERSONAL")
    loan_grade: str = Field(..., example="B")
    loan_amnt: float = Field(..., example=15000)
    loan_int_rate: float = Field(..., example=12.5)
    loan_percent_income: float = Field(..., example=0.2)
    cb_person_default_on_file: str = Field(..., example="N")
    cb_person_cred_hist_length: int = Field(..., example=8)


# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


# -----------------------------
# Single prediction
# -----------------------------
@app.post("/predict")
def predict_credit_risk(payload: CreditApplicationRequest):
    try:
        features = preprocess_request(payload.dict())
        result = run_inference(features)
        return result

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        raise HTTPException(
            status_code=500, detail="Internal prediction error")


# -----------------------------
# Batch prediction (MAX 500)
# -----------------------------
@app.post("/predict/batch")
def predict_credit_risk_batch(payloads: List[CreditApplicationRequest]):
    batch_size = len(payloads)

    if batch_size == 0:
        raise HTTPException(status_code=400, detail="Batch is empty")

    if batch_size > 500:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds maximum limit of 500 records"
        )

    try:
        # Convert list of payloads to list of dicts
        raw_records = [p.dict() for p in payloads]

        # Preprocess entire batch at once
        features = preprocess_request_batch(raw_records)

        # Run vectorized inference
        results = run_inference_batch(features)

        return {
            "batch_size": batch_size,
            "results": results
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        raise HTTPException(
            status_code=500, detail="Internal batch prediction error")
