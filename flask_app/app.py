import os
import sys
PROJECT_ROOT = os.path.abspath("./")
sys.path.append(PROJECT_ROOT)
from src.data_processing.image_reader import get_image_to_mnist_dataframe
from src.evaluation.model_loader import load_rf_model_and_pca_decomposer
from src.evaluation.model_prediction import get_predictions


#Load model in global memory
pca, model = load_rf_model_and_pca_decomposer()

from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def return_healthcheck():
    return jsonify({"Status": "API is working"})

@app.route('/predict', methods=['POST'])
def identify_digit():
    file = request.files['file']
    file_df = get_image_to_mnist_dataframe(file)
    prediction = get_predictions(input_df=file_df, pca_decomposer=pca, rf_model=model)

    return jsonify({"Digit in Image": prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)