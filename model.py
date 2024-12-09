import mlflow
import numpy as np
import os
from pydantic import BaseModel, ConfigDict

from custom_threshold_model import CustomThresholdModel

class InputData(BaseModel):
    PAYMENT_RATE: float
    EXT_SOURCE_3: float
    EXT_SOURCE_2: float
    DAYS_BIRTH: float
    EXT_SOURCE_1: float
    DAYS_EMPLOYED_PERC: float
    ANNUITY_INCOME_PERC: float
    INSTAL_DBD_MEAN: float
    DAYS_LAST_PHONE_CHANGE: float
    REGION_POPULATION_RELATIVE: float
    ACTIVE_DAYS_CREDIT_UPDATE_MEAN: float

model_path = os.path.join(os.getcwd(), "models", "latest")

# Charger le modèle
loaded_model = mlflow.sklearn.load_model(model_path)

def predict_score(data_list):
    # Convert list of dictionaries to a numpy array
    data_array = np.array([list(data.model_dump().values()) for data in data_list])
    predictions = loaded_model.predict(data_array)
    return predictions