import mlflow
import numpy as np
import os
from pydantic import BaseModel, ConfigDict

from custom_threshold_model import CustomThresholdModel

class InputData(BaseModel):
    AMT_ANNUITY: float
    AMT_CREDIT: float
    AMT_GOODS_PRICE: float
    CODE_GENDER: int
    DAYS_EMPLOYED: float
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    NAME_EDUCATION_TYPE_Highereducation: bool
    OWN_CAR_AGE: float
    PAYMENT_RATE: float

model_path = os.path.join(os.getcwd(), "models", "latest")

# Charger le modèle
loaded_model = mlflow.sklearn.load_model(model_path)

def predict_score(data_list):
    # Convert list of dictionaries to a numpy array
    data_array = np.array([list(data.model_dump().values()) for data in data_list])
    predictions = loaded_model.predict(data_array)
    return predictions