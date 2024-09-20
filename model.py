import mlflow.pyfunc
import numpy as np

THRESHOLD = 0.7 # charger le seuil en même temps que je charge le modèle avec mlflow

# # Dans une vraie application, vous chargeriez ici votre modèle entraîné
# model_uri = "path/to/your/mlflow/model"  # Replace with your model URI
# model = mlflow.pyfunc.load_model(model_uri)

# Ou load avec pickle
import pickle
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

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