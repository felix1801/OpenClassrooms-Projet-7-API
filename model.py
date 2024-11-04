import mlflow
import numpy as np
import os

from pydantic import BaseModel, ConfigDict

from sklearn.base import BaseEstimator, ClassifierMixin

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

class CustomThresholdModel(BaseEstimator, ClassifierMixin):
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def fit(self, X, y):
        # Pas besoin de ré-entraîner le modèle
        return self

    def predict(self, X):
        probas = self.model.predict_proba(X)[:, 1]
        scores = self.predict_score(X, probas)
        return scores, probas
    
    def predict_score(self, X, probas):
        return (probas >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

model_path = os.path.join(os.getcwd(), "models", "latest")

# Charger le modèle
loaded_model = mlflow.sklearn.load_model(model_path)

def predict_score(data_list):
    # Convert list of dictionaries to a numpy array
    data_array = np.array([list(data.model_dump().values()) for data in data_list])
    predictions = loaded_model.predict(data_array)
    return predictions