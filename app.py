from flask import Flask, request, jsonify
from model import predict_score

# Initialize the Flask application
app = Flask(__name__)

# Defines a route /predict that accepts POST requests and returns a JSON object with the score and probability
@app.route('/predict', methods=['POST'])
def predict():
    # TO DO: vérifier que c'est bien le bon type de données (dict) avec les dataclass de pydantic

    data_list = request.json
    scores, probas = predict_score(data_list)
    return jsonify({
        'scores': scores.tolist(),  # Convert numpy array to list for JSON serialization
        'probas': probas.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True, port=8001)
