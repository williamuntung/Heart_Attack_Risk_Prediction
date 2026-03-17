from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

scaler = joblib.load("artifacts/preprocessor.pkl")
model = joblib.load("artifacts/model.pkl")

class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int
    
@app.get("/")
def read_root():
       return {"message": "Welcome to the ML Model API"}

@app.post('/predict')

def predict(data: HeartDiseaseInput):

    input_data = data.dict()

    features = [[
        input_data['age'],
        input_data['sex'],
        input_data['cp'],
        input_data['trestbps'],
        input_data['chol'],
        input_data['fbs'],
        input_data['restecg'],
        input_data['thalach'],
        input_data['exang'],
        input_data['oldpeak'],
        input_data['slope'],
        input_data['ca'],
        input_data['thal']
    ]]

    prediction = model.predict(features)

    return {"prediction": int(prediction[0])}