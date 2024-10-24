from flask import Flask, request, jsonify
# from flask_cors import CORS # Handle Cross Origin Resource Sharing and Request 
from model import predict_score

# Initialize the Flask application
app = Flask(__name__)

# Enable CORS to allow requests from the frontend to the backend server 
# CORS(app)

# Defines a route /predict that accepts POST requests and returns a JSON object with the score and probability
@app.route('/predict', methods=['POST'])
def predict():
    data_list = request.json
    scores, probas = predict_score(data_list)
    return jsonify({
        'scores': scores.tolist(),  # Convert numpy array to list for JSON serialization
        'probas': probas.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True, port=8001)
