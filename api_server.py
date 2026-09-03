#!/usr/bin/env python3
"""Private FastAPI service for the trained jailbreak detector.

The service binds only to 127.0.0.1. Nginx is the only public entry point.
"""
import os
import re
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "output", "models", "logistic_regression.joblib")
VECTORIZER_PATH = os.path.join(BASE, "output", "models", "vectorizer.joblib")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

app = FastAPI(title="Jailbreak Detector API", version="1.0.0", docs_url=None, redoc_url=None)

class PredictRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=20000)


def clean_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def predict(prompt: str) -> dict[str, Any]:
    x = vectorizer.transform([clean_prompt(prompt)])
    probabilities = model.predict_proba(x)[0]
    classes = model.classes_
    idx = int(np.argmax(probabilities))
    label = str(classes[idx])
    return {
        "label": label,
        "confidence": float(probabilities[idx]),
        "probabilities": {
            "benign": float(probabilities[list(classes).index("benign")]),
            "jailbreak": float(probabilities[list(classes).index("jailbreak")]),
        },
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "logistic_regression"}


@app.post("/api/predict")
def api_predict(request: PredictRequest) -> dict[str, Any]:
    try:
        return predict(request.prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
