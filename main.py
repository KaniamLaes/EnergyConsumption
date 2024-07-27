# main.py

import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

with open('xgboost_model.sav', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

# Define the request model
class PredictionRequest(BaseModel):
    day_of_year: float
    hour: float
    day_of_week: float
    quarter: float
    month: float
    year: float

# Define the response model
class PredictionResponse(BaseModel):
    prediction: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    features = [[
        request.day_of_year,
        request.hour,
        request.day_of_week,
        request.quarter,
        request.month,
        request.year
    ]]
    prediction = model.predict(features)[0]
    return PredictionResponse(prediction=prediction)
