from flask import Flask, request, jsonify
from flask_cors import CORS
from prediction import HeartDiseasePredictor
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the predictor
predictor = HeartDiseasePredictor()

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'message': 'Heart Disease Prediction API',
        'endpoints': {
            '/predict': 'POST - Single patient prediction',
            '/health': 'GET - Health check status'
        }
    })

import traceback

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(f"Received prediction request: {data}")
        model_choice = data.get('model_choice', 'SVM')
        input_data = data.get('input_data')
        
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
            
        result = predictor.predict_single(input_data, model_choice)
        print(f"Prediction result: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Ensure we are in the correct directory to load models
    app.run(debug=True, port=5000)
