from model import predict_score, InputData
from flask import Flask, request, jsonify
import os
from pydantic import ValidationError

# Initialize the Flask application
app = Flask(__name__)

# Defines a route /predict that accepts POST requests and returns a JSON object with the score and probability
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data_list = [InputData(**data) for data in request.json]
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400

    scores, probas = predict_score(data_list)
    return jsonify({
        'scores': scores.tolist(),  # Convert numpy array to list for JSON serialization
        'probas': probas.tolist()
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", debug=True, port=port)

# TO DO: 
# - [X] Réaliser 2 tests unitaires
# - [X] Déployer avec GitHub Action en réalisant les tests unitaires
# - [X] En entrée dans l'API, vérifier que c'est bien le bon type de données (dict et vars dedans du bon type?) avec les dataclass de pydantic
