import mlflow
import numpy as np
import os

from sklearn.base import BaseEstimator, ClassifierMixin

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


# Local version
"""
import sys
# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add project root to Python path
if project_root not in sys.path:
    sys.path.append(project_root)

# Add modeling directory to Python path
modeling_path = os.path.join(project_root, 'modeling')
if modeling_path not in sys.path:
    sys.path.append(modeling_path)

from pipelines.utils.custom_threshold_model import CustomThresholdModel

model_path = os.path.join(os.path.expanduser("~"), "Desktop", "Ecole", "OpenClassrooms-Projet-7", "modeling", "data", "06_models", "latest")
"""

# Remote version
model_path = os.path.join(os.getcwd(), "models", "latest")
print(model_path)

# Charger le modèle
loaded_model = mlflow.sklearn.load_model(model_path)

def predict_score(data_list):
    # Convert list of dictionaries to a numpy array
    data_array = np.array([list(data.values()) for data in data_list])
    predictions = loaded_model.predict(data_array)
    return predictions