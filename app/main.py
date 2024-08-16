"""This module contains the code for the Flask app that serves the model."""
import pickle

import pandas as pd
from fastapi import FastAPI

with open('model.pkl', 'rb') as f_in:
    model = pickle.load(f_in)

app = FastAPI(title="Indicators of Heart Disease",
              openapi_tags=[
                  {
                      "name": "Health",
                      "description": "Get API health"
                  },
                  {
                      "name": "Prediction",
                      "description": "Model prediction"
                  }
              ])


@app.get(path='/', tags=['Health'])
def api_health():
    """
    A function that represents the health endpoint of the API.

    Returns:
        dict: A dictionary containing the status of the API, with the key
        "status" and the value "healthy".
    """
    return {"status": "healthy"}


@app.post('/predict', tags=['Prediction'])
def predict(request: dict):
    """Receives a POST request with a JSON payload, runs the model, and returns
    the prediction."""
    input_data = request
    df_input = pd.DataFrame.from_dict(input_data, orient='index').T

    # For sample inputs, the target column is included in the input data.
    if "HadHeartAttack" in df_input.columns:
        df_input = df_input.drop(columns=["HadHeartAttack"])

    pred = model.predict(df_input)[0]
    result = {'HadHeartAttack': pred}

    return result
