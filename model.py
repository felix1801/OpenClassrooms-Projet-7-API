import numpy as np
import os
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

import mlflow
from pipelines.utils.custom_threshold_model import CustomThresholdModel

model_path = os.path.join(os.path.expanduser("~"), "Desktop", "Ecole", "OpenClassrooms-Projet-7", "modeling", "data", "06_models", "latest")

# Charger le modèle
loaded_model = mlflow.sklearn.load_model(model_path)

print(f"Modèle chargé depuis : {model_path}")


def predict_score(data_list):
    # Convert list of dictionaries to a numpy array
    data_array = np.array([list(data.values()) for data in data_list])
    predictions = loaded_model.predict(data_array)
    return predictions