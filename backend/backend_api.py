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
            '/models': 'GET - Model metadata and accuracies',
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

@app.route('/models', methods=['GET'])
def get_models():
    models_metadata = [
        {'id': 'Logistic Regression', 'name': 'Logistic Regression', 'accuracy': 0.8689},
        {'id': 'SVM', 'name': 'SVM (Support Vector Machine)', 'accuracy': 0.8852},
        {'id': 'KNN', 'name': 'KNN (K-Nearest Neighbors)', 'accuracy': 0.9016},
        {'id': 'Random Forest', 'name': 'Random Forest', 'accuracy': 0.9016}
    ]
    return jsonify(models_metadata)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Get port from environment variable for Render compatibility
    port = int(os.environ.get('PORT', 5000))
    # Listen on 0.0.0.0 to be accessible from outside the container
    app.run(host='0.0.0.0', port=port, debug=True)
