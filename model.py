import mlflow.pyfunc
import numpy as np

# Dans une vraie application, vous chargeriez ici votre modèle entraîné
model_uri = "path/to/your/mlflow/model"  # Replace with your model URI
model = mlflow.pyfunc.load_model(model_uri)

# Ou load avec pickle
# import pickle
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)

def predict_score(data):
    # Ici, vous utiliseriez normalement votre modèle pour faire la prédiction
    # Preprocess the input data as needed, for example, convert it to a numpy array or DataFrame
    input_data = np.array(data).reshape(1, -1)

    # Ceci est une simulation pour l'exemple
    prediction = model.predict(input_data)
    probability = prediction[0][1]

    # probability can be higher than 0.5 to give the loan to the client (depending of false positive cost)
    # score is do I give the loan to the client or not? decision
    score = 1 if probability > 0.5 else 0

    return score, probability