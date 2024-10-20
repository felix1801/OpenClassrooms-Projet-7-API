import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Ajouter le chemin du dossier 'modeling' au PYTHONPATH
modeling_path = os.path.join(project_root, 'modeling')
if modeling_path not in sys.path:
    sys.path.append(modeling_path)


import mlflow
from pipelines.utils.custom_threshold_model import CustomThresholdModel

model_path = os.path.join(os.path.expanduser("~"), "Desktop", "Ecole", "OpenClassrooms-Projet-7", "modeling", "data", "06_models", "latest")

# Charger le modèle
loaded_model = mlflow.sklearn.load_model(model_path)

print(f"Modèle chargé depuis : {model_path}")


def predict_score(data):
    # reshape request json data to fit into my model input and predict
    data = np.array([list(data.values())])
    prediction = loaded_model.predict(data)

    return prediction