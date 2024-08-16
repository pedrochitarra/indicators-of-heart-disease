"""Integration test for the model created with MLFlow."""
import pickle
import numpy as np
import pandas as pd


def test_model():
    """Function to test the model created with MLFlow."""
    data = {
        "State": "Colorado",
        "Sex": "Male",
        "GeneralHealth": "Good",
        "PhysicalHealthDays": 0.0,
        "MentalHealthDays": 0.0,
        "LastCheckupTime": ("Within past year (anytime "
                            "less than 12 months ago)"),
        "PhysicalActivities": "Yes",
        "SleepHours": 7.0,
        "RemovedTeeth": "None of them",
        "HadHeartAttack": "No",
        "HadAngina": "No",
        "HadStroke": "No",
        "HadAsthma": "No",
        "HadSkinCancer": "No",
        "HadCOPD": "No",
        "HadDepressiveDisorder": "No",
        "HadKidneyDisease": "No",
        "HadArthritis": "No",
        "HadDiabetes": "No",
        "DeafOrHardOfHearing": "No",
        "BlindOrVisionDifficulty": "No",
        "DifficultyConcentrating": "No",
        "DifficultyWalking": "No",
        "DifficultyDressingBathing": "No",
        "DifficultyErrands": "No",
        "SmokerStatus": "Never smoked",
        "ECigaretteUsage": "Never used e-cigarettes in my entire life",
        "ChestScan": "No",
        "RaceEthnicityCategory": "Hispanic",
        "AgeCategory": "Age 65 to 69",
        "HeightInMeters": 1.85,
        "WeightInKilograms": 122.47,
        "BMI": 35.62,
        "AlcoholDrinkers": "Yes",
        "HIVTesting": "No",
        "FluVaxLast12": "Yes",
        "PneumoVaxEver": "No",
        "TetanusLast10Tdap": ("Yes, received tetanus shot "
                              "but not sure what type"),
        "HighRiskLastYear": "No",
        "CovidPos": "No"
    }

    df_inputs = pd.DataFrame(data, index=[0])
    df_inputs = df_inputs.drop(columns=["HadHeartAttack"])

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    prediction = model.predict(df_inputs)
    assert isinstance(prediction, np.ndarray)
