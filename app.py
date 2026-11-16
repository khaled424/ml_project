from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from model_pipeline import load_model

MODEL_PATH = Path(__file__).resolve().parent / "classifier.joblib"
model = load_model(str(MODEL_PATH))

FEATURE_COLUMNS = [
    "CreditScore",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

app = FastAPI(title="API de prédiction du churn", version="1.0.0")


class ChurnFeatures(BaseModel):
    CreditScore: int
    Gender: Literal["Female", "Male"]
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


@app.get("/")
def healthcheck():
    return {"status": "online"}


@app.post("/predict")
def predict_churn(features: ChurnFeatures):
    values = features.model_dump()
    values["Gender"] = {"Female": 0, "Male": 1}[values["Gender"]]
    input_data = pd.DataFrame([values], columns=FEATURE_COLUMNS)

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    return {
        "prediction": prediction,
        "interpretation": (
            "Client à risque de départ (churn)"
            if prediction == 1
            else "Client fidèle"
        ),
        "probability_churn": probability,
    }
