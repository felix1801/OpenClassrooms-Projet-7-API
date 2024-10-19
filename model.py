import mlflow.pyfunc
import numpy as np
import os

# THRESHOLD = 0.7 # charger le seuil en même temps que je charge le modèle avec mlflow

# # Dans une vraie application, vous chargeriez ici votre modèle entraîné
# model_uri = os.path.join("..", "modeling", "data", "06_models", "model.pkl")  # Replace with your model URI

# Get the absolute path of your project directory
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Construct the absolute path to your model
model_path = os.path.join(project_dir, "modeling", "data", "06_models", "model.pkl")
# Convert the path to a URI
model_uri = "file:///" + model_path.replace("\\", "/")
print(model_uri)
print(os.getcwd())
model = mlflow.pyfunc.load_model(model_uri)

print(model.metadata)
THRESHOLD = model.metadata.get_metric("optimal_threshold")
print(THRESHOLD)

# Ou load avec pickle
# import pickle
# with open('models/model.pkl', 'rb') as f:
#     model = pickle.load(f)

def predict_score(data):
    # Ici, vous utiliseriez normalement votre modèle pour faire la prédiction
    # reshape request json data to fit into my model input and predict
    data = np.array([list(data.values())])

    prediction = model.predict(data)
    probability = prediction[0]
    print(f"Probability: {probability}")

    # probability can be higher than 0.5 to give the loan to the client (depending of false positive cost)
    # score is do I give the loan to the client or not? decision
    score = 1 if probability >= THRESHOLD else 0

    return score, probability